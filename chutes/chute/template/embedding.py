import asyncio
import os
import re
import json
import sys
import uuid
import shlex
import aiohttp
import subprocess
from loguru import logger
from pydantic import BaseModel
from typing import Dict, Callable, List, Optional, Literal
from chutes.image import Image
from chutes.image.standard.vllm import VLLM
from chutes.chute import Chute, ChutePack, NodeSelector
from chutes.chute.template.helpers import set_default_cache_dirs, set_nccl_flags, monitor_engine

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def get_optimal_pooling_type(model_name: str) -> str:
    model_lower = model_name.lower()
    if "e5-" in model_lower or "multilingual-e5" in model_lower:
        return "MEAN"
    elif "bge-" in model_lower:
        return "CLS"
    elif "gte-" in model_lower:
        return "LAST"
    elif "sentence-t5" in model_lower or "st5" in model_lower:
        return "MEAN"
    elif "jina-embeddings" in model_lower:
        return "MEAN"
    elif "qwen" in model_lower and "embedding" in model_lower:
        return "LAST"
    else:
        return "MEAN"


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int
    completion_tokens: Optional[int] = 0
    prompt_tokens_details: Optional[Dict] = None


class EmbeddingData(BaseModel):
    index: int
    object: str = "embedding"
    embedding: List[float]


class EmbeddingRequest(BaseModel):
    model: str
    input: str | List[str]
    encoding_format: Optional[Literal["float", "base64"]] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None
    truncate_prompt_tokens: Optional[int] = None


class EmbeddingResponse(BaseModel):
    id: str
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


class MinifiedEmbeddingRequest(BaseModel):
    input: str | List[str]
    model: Optional[str] = None


class EmbeddingChutePack(ChutePack):
    embed: Callable


def build_embedding_chute(
    username: str,
    model_name: str,
    node_selector: NodeSelector,
    image: str | Image = VLLM,
    tagline: str = "",
    readme: str = "",
    concurrency: int = 32,
    engine_args: str = None,
    revision: str = None,
    max_instances: int = 1,
    scaling_threshold: float = 0.75,
    shutdown_after_seconds: int = 300,
    pooling_type: str = "auto",
    max_embed_len: int = 3072000,
    enable_chunked_processing: bool = True,
    allow_external_egress: bool = False,
    tee: bool = False,
):
    if engine_args and "--revision" in engine_args:
        raise ValueError("Revision is now a top-level argument to build_embedding_chute!")

    if not revision:
        from chutes.chute.template.helpers import get_current_hf_commit

        suggested_commit = None
        try:
            suggested_commit = get_current_hf_commit(model_name)
        except Exception:
            pass
        suggestion = (
            "Unable to fetch the current refs/heads/main commit from HF, please check the model name."
            if not suggested_commit
            else f"The current refs/heads/main commit is: {suggested_commit}"
        )
        raise ValueError(
            f"You must specify revision= to properly lock a model to a given huggingface revision. {suggestion}"
        )

    if pooling_type == "auto":
        pooling_type = get_optimal_pooling_type(model_name)
        logger.info(f"🔍 Auto-detected pooling type: {pooling_type} for model {model_name}")

    chute = Chute(
        username=username,
        name=model_name,
        tagline=tagline,
        readme=readme,
        image=image,
        node_selector=node_selector,
        concurrency=concurrency,
        standard_template="embedding",
        revision=revision,
        shutdown_after_seconds=shutdown_after_seconds,
        max_instances=max_instances,
        scaling_threshold=scaling_threshold,
        allow_external_egress=allow_external_egress,
        tee=tee,
    )

    @chute.on_startup()
    async def initialize_vllm_embedding(self):
        nonlocal engine_args
        nonlocal model_name
        nonlocal pooling_type
        nonlocal max_embed_len
        nonlocal enable_chunked_processing

        import torch
        import multiprocessing
        from chutes.util.hf import verify_cache
        from huggingface_hub import snapshot_download

        if enable_chunked_processing:
            os.environ["VLLM_ENABLE_CHUNKED_PROCESSING"] = "true"

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
                logger.info(f"Successfully downloaded {model_name} to {download_path}")
                break
            except Exception as exc:
                logger.info(f"Failed downloading {model_name} {download_kwargs or ''}: {exc}")
            await asyncio.sleep(60)
        if not download_path:
            raise Exception(f"Failed to download {model_name} after 5 attempts")

        # Verify the cache contents.
        await verify_cache(repo_id=model_name, revision=revision)

        set_default_cache_dirs(download_path)

        torch.cuda.empty_cache()
        torch.cuda.init()
        torch.cuda.set_device(0)
        multiprocessing.set_start_method("spawn", force=True)

        gpu_count = int(os.getenv("CUDA_DEVICE_COUNT", str(torch.cuda.device_count())))
        gpu_model = torch.cuda.get_device_name(0)
        set_nccl_flags(gpu_count, gpu_model)

        pooler_config = {
            "pooling_type": pooling_type,
            "normalize": True,
        }
        if enable_chunked_processing:
            pooler_config["enable_chunked_processing"] = True
            pooler_config["max_embed_len"] = max_embed_len

        logger.info("📋 Embedding Configuration:")
        logger.info(f"   - Model: {model_name}")
        logger.info(f"   - GPU Count: {gpu_count}")
        logger.info(f"   - Pooling Type: {pooling_type}")
        logger.info(f"   - Chunked Processing: {enable_chunked_processing}")
        if enable_chunked_processing:
            logger.info(f"   - Max Embed Length: {max_embed_len} tokens")

        api_key = str(uuid.uuid4())
        engine_args = engine_args or ""
        if "--tensor-parallel-size" not in engine_args:
            engine_args += f" --tensor-parallel-size {gpu_count}"

        if len(re.findall(r"(?:^|\s)--(?:tensor-parallel-size|tp)[=\s]", engine_args)) > 1:
            raise ValueError(
                "Please use only --tensor-parallel-size (or omit and let gpu_count set it automatically)"
            )
        if "--served-model-name" in engine_args:
            raise ValueError("You may not override served model name!")
        if "--api-key" in engine_args:
            raise ValueError("You may not override api key!")

        env = os.environ.copy()
        if enable_chunked_processing:
            env["VLLM_ENABLE_CHUNKED_PROCESSING"] = "true"

        env["HF_HUB_OFFLINE"] = "1"
        env["SGL_MODEL_NAME"] = self.name
        env["SGL_REVISION"] = revision

        pooler_config_arg = shlex.quote(json.dumps(pooler_config))
        startup_command = (
            f"{sys.executable} -m vllm.entrypoints.openai.api_server "
            f"--model {model_name} --served-model-name {self.name} "
            f"--revision {revision} --pooler-config {pooler_config_arg} "
            f"--port 10101 --host 127.0.0.1 --api-key {api_key} {engine_args}"
        )
        display_cmd = startup_command.replace(api_key, "*" * len(api_key))
        parts = shlex.split(startup_command)

        logger.info(f"Launching vllm embedding server with command: {display_cmd}")

        self._vllm_process = subprocess.Popen(parts, text=True, stderr=subprocess.STDOUT, env=env)

        server_ready = asyncio.Event()
        self._monitor_task = asyncio.create_task(
            monitor_engine(self._vllm_process, api_key, server_ready, model_name=self.name)
        )

        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        "http://127.0.0.1:10101/v1/models",
                        headers={"Authorization": f"Bearer {api_key}"},
                    ) as resp:
                        if resp.status == 200:
                            logger.success("vllm engine /v1/models endpoint ping success!")
                            break
            except Exception:
                pass
            await asyncio.sleep(1)

        self.passthrough_headers["Authorization"] = f"Bearer {api_key}"
        server_ready.set()
        logger.info("✅ Embedding server initialized successfully!")

    @chute.cord(
        passthrough_path="/v1/embeddings",
        passthrough_port=10101,
        public_api_path="/v1/embeddings",
        method="POST",
        passthrough=True,
        input_schema=EmbeddingRequest,
        minimal_input_schema=MinifiedEmbeddingRequest,
    )
    async def embed(data) -> EmbeddingResponse:
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

    return EmbeddingChutePack(
        chute=chute,
        embed=embed,
    )
