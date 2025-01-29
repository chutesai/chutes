import os
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# The image was built as:
from chutes.image import Image
image = (
    Image(
        username="chutes",
        name="sglang",
        tag="0.4.2",
        readme="SGLang is a fast serving framework for large language models and vision language models. It makes your interaction with models faster and more controllable by co-designing the backend runtime and frontend language."
    )
    .from_base("parachutes/base-python:3.12.7")
    .run_command("pip install --upgrade pip")
    .run_command("pip install 'sglang[all]==0.4.2' --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/")
)

chute = build_sglang_chute(
    username="chutes",
    readme="DeepSeek-R1, which incorporates initial training data before reinforcement learning, achieves performance comparable to OpenAI-o1 across math, code, and reasoning tasks.",
    model_name="deepseek-ai/DeepSeek-R1",
    image=image,
    concurrency=5,
    node_selector=NodeSelector(
        gpu_count=8,
        min_vram_gb_per_gpu=140,
    ),
    engine_args="--trust-remote-code",
)
chute.chute._standard_template ="vllm"
