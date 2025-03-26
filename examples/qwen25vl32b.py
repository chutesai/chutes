import os
from chutes.chute import NodeSelector
from chutes.chute.template.vllm import build_vllm_chute

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["VLLM_USE_V1"] = "0"

chute = build_vllm_chute(
    username="chutes",
    readme="Qwen/Qwen2.5-VL-32B-Instruct",
    model_name="Qwen/Qwen2.5-VL-32B-Instruct",
    image="chutes/vllm:0.8.1.p2",
    concurrency=12,
    node_selector=NodeSelector(
        gpu_count=8,
        min_vram_gb_per_gpu=48,
    ),
    engine_args=dict(
        max_model_len=128000,
        trust_remote_code=True,
        num_scheduler_steps=1,
        enforce_eager=False,
        revision="6bcf1c9155874e6961bcf82792681b4f4421d2f7",
        limit_mm_per_prompt={
            "image": 8,
        },
    ),
)
