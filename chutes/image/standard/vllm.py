VLLM = "chutes/vllm:nightly-2025111700"

# Example...
# from chutes.image import Image
# image = (
#     Image(
#         username="chutes",
#         name="vllm",
#         tag="nightly-2025100200",
#         readme="## vLLM - fast, flexible llm inference",
#     )
#     .from_base("parachutes/python:3.12")
#     .run_command("pip install --no-cache wheel packaging blobfile tiktoken 'qwen-vl-utils==0.0.14'")
#     .run_command("pip install -U vllm --torch-backend=cu128 --extra-index-url https://wheels.vllm.ai/nightly flashinfer-python accelerate")
#     .run_command("pip install -U git+https://github.com/huggingface/transformers.git 'numpy<2.3'")
# )
