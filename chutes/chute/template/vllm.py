import json
from typing import Dict, Any, Callable
from chutes.image import Image
from chutes.image.standard.vllm import VLLM
from chutes.chute import Chute, ChutePack, NodeSelector
import torch
from vllm import AsyncEngineArgs, AsyncLLMEngine
import vllm.entrypoints.openai.api_server as vllm_api_server
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_engine import BaseModelPath


class VLLMChute(ChutePack):
    chat: Callable
    completion: Callable
    chat_stream: Callable
    completion_stream: Callable
    models: Callable


def build_vllm_chute(
    model_name: str,
    node_selector: NodeSelector,
    image: str | Image = VLLM,
    engine_args: Dict[str, Any] = {},
):
    chute = Chute(
        name=model_name,
        image=image,
        node_selector=node_selector,
        standard_template="vllm",
    )

    # Semi-optimized defaults.
    if not engine_args:
        engine_args.update(
            {
                "num_scheduler_steps": 16,
                "multi_step_stream_outputs": True,
                "max_logprobs": 5,
                "enforce_eager": False,
            }
        )

    @chute.on_startup()
    async def initialize_vllm(self):
        nonlocal engine_args
        nonlocal model_name
        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=torch.cuda.device_count(),
            **engine_args,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        model_config = await self.engine.get_model_config()
        request_logger = RequestLogger(max_log_len=1024)
        base_model_paths = [
            BaseModelPath(name=chute.name, model_path=chute.name),
        ]
        self.include_router(vllm_api_server.router)
        vllm_api_server.chat = lambda s: OpenAIServingChat(
            self.engine,
            model_config=model_config,
            base_model_paths=base_model_paths,
            chat_template=None,
            response_role="assistant",
            lora_modules=[],
            prompt_adapters=[],
            request_logger=request_logger,
            return_tokens_as_token_ids=True,
        )
        vllm_api_server.completion = lambda s: OpenAIServingCompletion(
            self.engine,
            model_config=model_config,
            base_model_paths=base_model_paths,
            lora_modules=[],
            prompt_adapters=[],
            request_logger=request_logger,
            return_tokens_as_token_ids=True,
        )

    def _parse_stream_chunk(encoded_chunk):
        chunk = (
            encoded_chunk if isinstance(encoded_chunk, str) else encoded_chunk.decode()
        )
        if "data: {" in chunk:
            return json.loads(chunk[6:])
        return None

    @chute.cord(
        passthrough_path="/v1/chat/completions",
        method="POST",
        passthrough=True,
        stream=True,
    )
    async def chat_stream(encoded_chunk):
        return _parse_stream_chunk(encoded_chunk)

    @chute.cord(
        passthrough_path="/v1/chat/completions", method="POST", passthrough=True
    )
    async def chat(data):
        return data

    @chute.cord(
        passthrough_path="/v1/completions", method="POST", passthrough=True, stream=True
    )
    async def completion_stream(encoded_chunk):
        return _parse_stream_chunk(encoded_chunk)

    @chute.cord(passthrough_path="/v1/completions", method="POST", passthrough=True)
    async def completion(data):
        return data

    @chute.cord(passthrough_path="/v1/models", method="GET", passthrough=True)
    async def get_models(data):
        return data

    return VLLMChute(
        chute=chute,
        chat=chat,
        chat_stream=chat_stream,
        completion=completion,
        completion_stream=completion_stream,
        models=get_models,
    )
