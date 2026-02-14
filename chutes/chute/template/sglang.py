import asyncio
import json
import os
import re
import sys
import uuid
import aiohttp
from enum import Enum
from loguru import logger
from pydantic import BaseModel, Field
from typing import Dict, Callable, Literal, Optional, Union, List
from chutes.image import Image
from chutes.image.standard.sglang import SGLANG
from chutes.chute import Chute, ChutePack, NodeSelector
from chutes.chute.template.helpers import (
    set_default_cache_dirs,
    set_nccl_flags,
    monitor_engine,
    generate_mtls_certs,
    build_client_ssl_context,
    build_wrong_client_ssl_context,
    validate_mtls,
    force_exit,
    mtls_enabled,
    set_encrypted_env_var,
)


class DefaultRole(Enum):
    user = "user"
    assistant = "assistant"
    developer = "developer"


class ChatMessage(BaseModel):
    role: str
    content: str


class Logprob(BaseModel):
    logprob: float
    rank: Optional[int] = None
    decoded_token: Optional[str] = None


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[Dict] = None


class BaseRequest(BaseModel):
    model: str
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = 0
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = Field(None, ge=0, le=9223372036854775807)
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    best_of: Optional[int] = None
    use_beam_search: bool = False
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    prompt_logprobs: Optional[int] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionRequest(BaseRequest):
    messages: List[ChatMessage]


class CompletionRequest(BaseRequest):
    prompt: str


class ChatCompletionLogProb(BaseModel):
    token: str
    logprob: float = -9999.0
    bytes: Optional[List[int]] = None


class ChatCompletionLogProbsContent(ChatCompletionLogProb):
    top_logprobs: List[ChatCompletionLogProb] = Field(default_factory=list)


class ChatCompletionLogProbs(BaseModel):
    content: Optional[List[ChatCompletionLogProbsContent]] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    logprobs: Optional[ChatCompletionLogProbs] = None
    finish_reason: Optional[str] = "stop"
    stop_reason: Optional[Union[int, str]] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo
    prompt_logprobs: Optional[List[Optional[Dict[int, Logprob]]]] = None


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    logprobs: Optional[ChatCompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


class CompletionLogProbs(BaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[CompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = Field(
        default=None,
        description=(
            "The stop string or token id that caused the completion "
            "to stop, None if the completion finished for some other reason "
            "including encountering the EOS token"
        ),
    )
    prompt_logprobs: Optional[List[Optional[Dict[int, Logprob]]]] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[CompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = Field(
        default=None,
        description=(
            "The stop string or token id that caused the completion "
            "to stop, None if the completion finished for some other reason "
            "including encountering the EOS token"
        ),
    )


class CompletionStreamResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


class SGLangChute(ChutePack):
    chat: Callable
    completion: Callable
    chat_stream: Callable
    completion_stream: Callable
    models: Callable


def build_sglang_chute(
    username: str,
    model_name: str,
    node_selector: NodeSelector,
    image: str | Image = SGLANG,
    tagline: str = "",
    readme: str = "",
    concurrency: int = 32,
    engine_args: str = None,
    revision: str = None,
    max_instances: int = 1,
    scaling_threshold: float = 0.75,
    shutdown_after_seconds: int = 300,
    allow_external_egress: bool = False,
    tee: bool = False,
):
    # Prevent revision in the code, must be in the top level helper args.
    m = re.search(r"--revision\s*=?\s*([^ ]+)", engine_args or "", re.I)
    if m:
        raise ValueError("Revision is now a top-level argument to build_sglang_chute!")

    if not revision:
        from chutes.chute.template.helpers import get_current_hf_commit

        suggested_commit = None
        try:
            suggested_commit = get_current_hf_commit(model_name)
        except Exception:
            ...
        suggestion = (
            "Unable to fetch the current refs/heads/main commit from HF, please check the model name."
            if not suggested_commit
            else f"The current refs/heads/main commit is: {suggested_commit}"
        )
        raise ValueError(
            f"You must specify revision= to properly lock a model to a given huggingface revision. {suggestion}"
        )

    chute = Chute(
        username=username,
        name=model_name,
        tagline=tagline,
        readme=readme,
        image=image,
        node_selector=node_selector,
        concurrency=concurrency,
        standard_template="vllm",
        revision=revision,
        shutdown_after_seconds=shutdown_after_seconds,
        max_instances=max_instances,
        scaling_threshold=scaling_threshold,
        allow_external_egress=allow_external_egress,
        tee=tee,
    )

    # Minimal input schema with defaults.
    class MinifiedMessage(BaseModel):
        role: DefaultRole = DefaultRole.user
        content: str = Field("")

    class MinifiedStreamChatCompletion(BaseModel):
        messages: List[MinifiedMessage] = [MinifiedMessage()]
        temperature: float = Field(0.7)
        seed: int = Field(42)
        stream: bool = Field(True)
        max_tokens: int = Field(1024)
        model: str = Field(model_name)

    class MinifiedChatCompletion(MinifiedStreamChatCompletion):
        stream: bool = Field(False)

    # Minimal completion input.
    class MinifiedStreamCompletion(BaseModel):
        prompt: str
        temperature: float = Field(0.7)
        seed: int = Field(42)
        stream: bool = Field(True)
        max_tokens: int = Field(1024)
        model: str = Field(model_name)

    class MinifiedCompletion(MinifiedStreamCompletion):
        stream: bool = Field(False)

    @chute.on_startup()
    async def initialize_sglang(self):
        nonlocal engine_args
        nonlocal model_name

        # Imports here to avoid needing sglang installed to build the chutes remotely.
        import torch
        import multiprocessing
        import subprocess
        from chutes.util.hf import verify_cache, purge_model_cache, CacheVerificationError
        from huggingface_hub import snapshot_download
        from chutes.chute.template.helpers import warmup_model, validate_auth

        # Enable NCCL for multi-GPU on some chips by default.
        gpu_count = int(os.getenv("CUDA_DEVICE_COUNT", str(torch.cuda.device_count())))
        gpu_model = torch.cuda.get_device_name(0)
        set_nccl_flags(gpu_count, gpu_model)

        # Download the model.
        download_path = None
        for attempt in range(5):
            download_kwargs = {}
            if self.revision:
                download_kwargs["revision"] = self.revision
            try:
                logger.info(f"Attempting to download {model_name} to cache...")
                download_path = await asyncio.to_thread(
                    snapshot_download, repo_id=model_name, **download_kwargs
                )
                logger.success(f"Successfully downloaded {model_name} to {download_path}")
                break
            except Exception as exc:
                logger.warning(f"Failed downloading {model_name} {download_kwargs or ''}: {exc}")
            await asyncio.sleep(60)
        if not download_path:
            raise Exception(f"Failed to download {model_name} after 5 attempts.")

        # Verify the cache.
        try:
            await verify_cache(repo_id=model_name, revision=revision)
        except CacheVerificationError:
            purge_model_cache(repo_id=model_name)
            raise

        # Set torch inductor, flashinfer, etc., cache directories.
        set_default_cache_dirs(download_path)
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["SGL_MODEL_NAME"] = self.name
        os.environ["SGL_REVISION"] = revision

        # Reset torch.
        torch.cuda.empty_cache()
        torch.cuda.init()
        torch.cuda.set_device(0)
        multiprocessing.set_start_method("spawn", force=True)

        # Configure engine arguments
        if not engine_args:
            engine_args = ""
        if "--tp=" not in engine_args and "--tp " not in engine_args:
            engine_args += f" --tp {gpu_count}"
        if "--enable-cache-report" not in engine_args:
            engine_args += " --enable-cache-report"
        if "--enable-return-hidden-states" not in engine_args:
            engine_args += " --enable-return-hidden-states"
        if self.revision:
            engine_args += f" --revision {self.revision}"
        if "--api-key" in engine_args:
            raise ValueError("You may not override api key!")
        api_key = str(uuid.uuid4())
        use_mtls = mtls_enabled()
        ssl_ctx = None
        wrong_ssl_ctx = None

        if use_mtls:
            # Generate ephemeral mTLS certificates.
            certs = generate_mtls_certs()
            ssl_ctx = build_client_ssl_context(
                certs["ca_cert_file"],
                certs["client_cert_file"],
                certs["client_key_file"],
                certs["password"],
            )
            wrong_ssl_ctx = build_wrong_client_ssl_context(
                certs["ca_cert_file"],
                certs["wrong_client_cert_file"],
                certs["wrong_client_key_file"],
                certs["password"],
            )
            self.passthrough_ssl_context = ssl_ctx
            self._wrong_ssl_context = wrong_ssl_ctx
            logger.info("mTLS enabled for SGLang engine communication")
        else:
            logger.warning("mTLS disabled (LLM_ENGINE_MTLS_ENABLE not set)")

        startup_command = f"{sys.executable} -m sglang.launch_server --host 127.0.0.1 --port 10101 --model-path {model_name} {engine_args} --api-key {api_key}"
        if use_mtls:
            startup_command += " --ssl-cert-reqs 2"
        command = startup_command.replace("\\\n", " ").replace("\\", " ")
        parts = command.split()
        display_cmd = startup_command.replace(api_key, "*" * len(api_key))
        logger.info(f"Launching SGLang with command: {display_cmd}")
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        if use_mtls:
            env["SGLANG_SSL_KEYFILE_PEM"] = certs["server_key_pem"].decode()
            env["SGLANG_SSL_CERTFILE_PEM"] = certs["server_cert_pem"].decode()
            env["SGLANG_SSL_CA_CERTS_PEM"] = certs["ca_cert_pem"].decode()
            set_encrypted_env_var(env, "SGLANG_SSL_KEYFILE_PASSWORD", certs["password"])
        self._sglang_process = subprocess.Popen(parts, text=True, stderr=subprocess.STDOUT, env=env)

        server_ready = asyncio.Event()
        self._monitor_task = asyncio.create_task(
            monitor_engine(
                self._sglang_process,
                api_key,
                server_ready,
                model_name=self.name,
                ssl_context=ssl_ctx,
                wrong_ssl_context=wrong_ssl_ctx,
            )
        )

        def _on_monitor_done(t):
            if t.cancelled():
                return
            exc = t.exception()
            if exc:
                logger.error("SGLang monitor task failed, terminating: {}", exc)
                force_exit(1)

        self._monitor_task.add_done_callback(_on_monitor_done)

        # Poll for server readiness.
        base_url = "https://127.0.0.1:10101" if use_mtls else "http://127.0.0.1:10101"
        while True:
            if self._sglang_process.poll() is not None:
                raise RuntimeError(
                    f"SGLang subprocess exited before readiness check (exit={self._sglang_process.returncode})"
                )
            try:
                connector = aiohttp.TCPConnector(ssl=ssl_ctx) if ssl_ctx else None
                async with aiohttp.ClientSession(connector=connector) as session:
                    async with session.get(
                        f"{base_url}/v1/models",
                        headers={"Authorization": f"Bearer {api_key}"},
                    ) as resp:
                        if resp.status == 200:
                            logger.success("SGLang engine /v1/models endpoint ping success!")
                            break
            except Exception:
                pass
            await asyncio.sleep(1)

        self.passthrough_headers["Authorization"] = f"Bearer {api_key}"
        await warmup_model(self, base_url=base_url, api_key=api_key, ssl_context=ssl_ctx)
        await validate_auth(self, base_url=base_url, api_key=api_key, ssl_context=ssl_ctx)
        if use_mtls:
            await validate_mtls(self.name, api_key, ssl_ctx, wrong_ssl_ctx)
        server_ready.set()

    def _parse_stream_chunk(encoded_chunk):
        chunk = encoded_chunk if isinstance(encoded_chunk, str) else encoded_chunk.decode()
        if "data: {" in chunk:
            return json.loads(chunk[6:])
        return None

    @chute.cord(
        passthrough_path="/v1/chat/completions",
        passthrough_port=10101,
        public_api_path="/v1/chat/completions",
        method="POST",
        passthrough=True,
        stream=True,
        input_schema=ChatCompletionRequest,
        minimal_input_schema=MinifiedStreamChatCompletion,
        sglang_passthrough=True,
    )
    async def chat_stream(encoded_chunk) -> ChatCompletionStreamResponse:
        return _parse_stream_chunk(encoded_chunk)

    @chute.cord(
        passthrough_path="/v1/completions",
        passthrough_port=10101,
        public_api_path="/v1/completions",
        method="POST",
        passthrough=True,
        stream=True,
        input_schema=CompletionRequest,
        minimal_input_schema=MinifiedStreamCompletion,
        sglang_passthrough=True,
    )
    async def completion_stream(encoded_chunk) -> CompletionStreamResponse:
        return _parse_stream_chunk(encoded_chunk)

    @chute.cord(
        passthrough_path="/v1/chat/completions",
        passthrough_port=10101,
        public_api_path="/v1/chat/completions",
        method="POST",
        passthrough=True,
        input_schema=ChatCompletionRequest,
        minimal_input_schema=MinifiedChatCompletion,
        sglang_passthrough=True,
    )
    async def chat(data) -> ChatCompletionResponse:
        return data

    @chute.cord(
        passthrough_path="/v1/completions",
        passthrough_port=10101,
        public_api_path="/v1/completions",
        method="POST",
        passthrough=True,
        input_schema=CompletionRequest,
        minimal_input_schema=MinifiedCompletion,
        sglang_passthrough=True,
    )
    async def completion(data) -> CompletionResponse:
        return data

    @chute.cord(
        passthrough_path="/v1/models",
        passthrough_port=10101,
        public_api_path="/v1/models",
        public_api_method="GET",
        method="GET",
        passthrough=True,
        sglang_passthrough=True,
    )
    async def get_models(data):
        return data

    return SGLangChute(
        chute=chute,
        chat=chat,
        chat_stream=chat_stream,
        completion=completion,
        completion_stream=completion_stream,
        models=get_models,
    )
