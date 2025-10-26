import asyncio
import json
import os
import re
from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Callable, Literal, Optional, Union, List
from chutes.image import Image
from chutes.image.standard.sglang import SGLANG
from chutes.chute import Chute, ChutePack, NodeSelector
from chutes.chute.template.helpers import set_default_cache_dirs, set_nccl_flags


class DefaultRole(Enum):
    user = "user"
    assistant = "assistant"


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
        from sglang.utils import (
            execute_shell_command,
            wait_for_server,
        )

        from huggingface_hub import snapshot_download
        from chutes.chute.template.helpers import warmup_model

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
                print(f"Attempting to download {model_name} to cache...")
                download_path = await asyncio.to_thread(
                    snapshot_download, repo_id=model_name, **download_kwargs
                )
                print(f"Successfully downloaded {model_name} to {download_path}")
                break
            except Exception as exc:
                print(f"Failed downloading {model_name} {download_kwargs or ''}: {exc}")
            await asyncio.sleep(60)
        if not download_path:
            raise Exception(f"Failed to download {model_name} after 5 attempts.")

        # Set torch inductor, flashinfer, etc., cache directories.
        set_default_cache_dirs(download_path)

        # Reset torch.
        torch.cuda.empty_cache()
        torch.cuda.init()
        torch.cuda.set_device(0)
        multiprocessing.set_start_method("spawn", force=True)

        # Configure engine arguments
        if not engine_args:
            engine_args = ""
        engine_args += f" --tp {gpu_count}"
        if "--enable-cache-report" not in engine_args:
            engine_args += " --enable-cache-report"
        if "--enable-return-hidden-states" not in engine_args:
            engine_args += " --enable-return-hidden-states"
        if self.revision:
            engine_args += f" --revision {self.revision}"
        startup_command = f"python -m sglang.launch_server --host 127.0.0.1 --port 10101 --model-path {model_name} {engine_args}"
        _ = execute_shell_command(startup_command)
        wait_for_server("http://127.0.0.1:10101")
        await warmup_model(self)

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
