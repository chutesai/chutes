import os
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
chute = build_sglang_chute(
    username="chutes",
    readme="QwQ-32b",
    model_name="Qwen/QwQ-32B",
    image="chutes/sglang:0.4.3.post4",
    concurrency=12,
    node_selector=NodeSelector(
        gpu_count=8,
        min_vram_gb_per_gpu=40,
        exclude=["a6000", "l40"],
    ),
    engine_args=(
        "--trust-remote-code "
        "--revision 7c0a8dc0ac2eef85a227942ad8daeabe9f3ad709 "
        "--enable-torch-compile "
        "--torch-compile-max-bs 8 "
        "--dist-timeout 3600 "
        "--schedule-policy fcfs"
    ),
)
