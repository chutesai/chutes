from chutes.chute import NodeSelector
from chutes.chute.template.vllm import build_vllm_chute
from chutes.image import Image

image = (
    Image(
        username="chutes",
        name="vllm_gemma",
        tag="0.8.1",
        readme="## vLLM - fast, flexible llm inference",
    )
    .from_base("parachutes/base-python:3.12.9")
    .run_command(
        "pip install --no-cache wheel packaging git+https://github.com/huggingface/transformers.git qwen-vl-utils[decord]==0.0.8"
    )
    .run_command("pip install --upgrade vllm==0.8.1")
    .run_command("pip install --no-cache flash-attn")
    .add("gemma_chat_template.jinja", "/app/gemma_chat_template.jinja")
)

chute = build_vllm_chute(
    username="chutes",
    readme="Gemma 3 1B IT",
    model_name="unsloth/gemma-3-1b-it",
    image=image,
    node_selector=NodeSelector(
        gpu_count=8,
        min_vram_gb_per_gpu=48,
    ),
    concurrency=8,
    engine_args=dict(
        revision="284477f075e7d8bfa2c7e2e0131c3fe4055baa7f",
        num_scheduler_steps=8,
        enforce_eager=False,
        max_num_seqs=8,
        tool_call_parser="pythonic",
        enable_auto_tool_choice=True,
        chat_template="/app/gemma_chat_template.jinja",
    ),
)
