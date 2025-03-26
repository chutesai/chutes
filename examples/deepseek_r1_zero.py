import os
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute
from chutes.image import Image

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
NEXTN = os.path.join(os.environ.get("HF_HOME", "/cache"), "hub", "DeepSeek-R1-Zero-NextN")
if os.getenv("CHUTES_EXECUTION_CONTEXT") == "REMOTE":
    from huggingface_hub import snapshot_download

    os.makedirs(os.path.dirname(NEXTN), exist_ok=True)
    snapshot_download(repo_id="chutesai/DeepSeek-R1-Zero-NextN", local_dir=NEXTN)

image = (
    Image(
        username="chutes",
        name="sglang",
        tag="0.4.4.post1a",
        readme="SGLang is a fast serving framework for large language models and vision language models. It makes your interaction with models faster and more controllable by co-designing the backend runtime and frontend language.",
    )
    .from_base("parachutes/base-python:3.12.9")
    .run_command("pip install --upgrade pip")
    .run_command(
        "pip install --upgrade 'sglang[all]>=0.4.4.post1' --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/"
    )
    .with_env("SGL_ENABLE_JIT_DEEPGEMM", "1")
    .run_command("pip install 'transformers<4.49.0' datasets blobfile")
)

chute = build_sglang_chute(
    username="chutes",
    readme="DeepSeek-R1-Zero",
    model_name="deepseek-ai/DeepSeek-R1-Zero",
    image=image,
    concurrency=12,
    node_selector=NodeSelector(
        gpu_count=8,
        min_vram_gb_per_gpu=140,
    ),
    engine_args=(
        "--trust-remote-code "
        "--revision b3c33b0822efb24324ad2d384c645d9aabab6da3 "
        "--enable-torch-compile "
        "--torch-compile-max-bs 1 "
        "--enable-flashinfer-mla "
        "--speculative-algo EAGLE "
        f"--speculative-draft {NEXTN} "
        "--speculative-num-steps 3 "
        "--speculative-eagle-topk 1 "
        "--speculative-num-draft-tokens 4"
    ),
)
