import asyncio
import os
from loguru import logger
from pydantic import BaseModel
from typing import Dict, Any, Callable, List, Optional, Literal
from chutes.image import Image
from chutes.image.standard.vllm import VLLM
from chutes.chute import Chute, ChutePack, NodeSelector

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def get_optimal_pooling_type(model_name: str) -> str:
    """
    Determine optimal pooling type for known embedding models.
    """
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
    engine_args: Dict[str, Any] = {},
    revision: str = None,
    max_instances: int = 1,
    scaling_threshold: float = 0.75,
    shutdown_after_seconds: int = 300,
    pooling_type: str = "auto",
    max_embed_len: int = 3072000,
    enable_chunked_processing: bool = True,
    allow_external_egress: bool = False,
):
    """
    Build a vLLM embedding chute with enhanced chunked processing support.

    Args:
        username: Chute username
        model_name: HuggingFace model name
        node_selector: Node selector for deployment
        image: Docker image to use (default: VLLM)
        tagline: Chute tagline
        readme: Chute readme
        concurrency: Max concurrent requests
        engine_args: Additional vLLM engine arguments
        revision: HuggingFace model revision/commit
        max_instances: Maximum number of instances
        scaling_threshold: Scaling threshold (0-1)
        shutdown_after_seconds: Idle shutdown time
        pooling_type: Pooling type (auto, MEAN, CLS, LAST)
        max_embed_len: Maximum embedding length in tokens
        enable_chunked_processing: Enable chunked processing for long texts
    """
    if engine_args.get("revision"):
        raise ValueError("revision is now a top-level argument to build_embedding_chute!")

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
    )

    @chute.on_startup()
    async def initialize_vllm_embedding(self):
        nonlocal engine_args
        nonlocal model_name
        nonlocal pooling_type
        nonlocal max_embed_len
        nonlocal enable_chunked_processing

        # Imports here to avoid needing torch/vllm/etc. to just perform inference/build remotely
        import torch
        import multiprocessing
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        import vllm.entrypoints.openai.api_server as vllm_api_server
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

        try:
            from vllm.entrypoints.openai.serving_engine import BaseModelPath
        except Exception:
            from vllm.entrypoints.openai.serving_models import (
                BaseModelPath,
                OpenAIServingModels,
            )

        from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding

        torch.cuda.empty_cache()
        torch.cuda.init()
        torch.cuda.set_device(0)
        multiprocessing.set_start_method("spawn", force=True)

        # Configure GPU count
        gpu_count = int(os.getenv("CUDA_DEVICE_COUNT", str(torch.cuda.device_count())))

        # Build pooler config
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
            logger.info("   - Cross-chunk Aggregation: MEAN (automatic)")

        # Configure engine arguments
        engine_args_config = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=gpu_count,
            override_pooler_config=pooler_config,
            **engine_args,
        )

        # Initialize engine
        self.engine = AsyncLLMEngine.from_engine_args(engine_args_config)
        model_config = await self.engine.get_model_config()

        base_model_paths = [
            BaseModelPath(name=chute.name, model_path=chute.name),
        ]

        self.include_router(vllm_api_server.router)

        # Check vLLM version for compatibility
        import vllm.version as vv

        version_parts = vv.__version__.split(".")
        old_vllm = False
        if (
            not vv.__version__.startswith("0.1.dev")
            and int(version_parts[0]) == 0
            and int(version_parts[1]) < 7
        ):
            old_vllm = True

        # Set up serving models
        if old_vllm:
            models_arg = base_model_paths
        else:
            models_arg = OpenAIServingModels(
                engine_client=self.engine,
                model_config=model_config,
                base_model_paths=base_model_paths,
                lora_modules=[],
            )
            self.state.openai_serving_models = models_arg

        # Initialize embedding serving
        vllm_api_server.embedding = lambda s: OpenAIServingEmbedding(
            self.engine,
            model_config,
            models_arg,
            request_logger=None,
            chat_template=None,
            chat_template_content_format="string",
        )

        logger.info("✅ Embedding server initialized successfully!")

    @chute.cord(
        passthrough_path="/v1/embeddings",
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
