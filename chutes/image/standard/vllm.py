VLLM = "chutes/vllm:0.9.1.dev"

# To build this yourself, you can use something like:
# image = (
#     Image(
#         username="chutes",
#         name="vllm",
#         tag="0.9.1.dev",
#         readme="## vLLM - fast, flexible llm inference",
#     )
#     .from_base("parachutes/python:3.12.9")
#     .run_command("pip install --no-cache wheel packaging qwen-vl-utils[decord]")
#     .run_command("pip install --upgrade vllm --pre --extra-index-url https://wheels.vllm.ai/nightly")
#     .run_command("pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl")
#     .run_command("pip install --no-cache blobfile datasets")
# )
