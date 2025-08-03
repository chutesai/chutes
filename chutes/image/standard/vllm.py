VLLM = "chutes/vllm:0.9.2"

# Example...
# image = (
#     Image(
#         username="chutes",
#         name="vllm",
#         tag="0.9.2",
#         readme="## vLLM - fast, flexible llm inference",
#     )
#     .from_base("parachutes/python:3.12")
#     .run_command("pip install --no-cache wheel packaging qwen-vl-utils[decord]")
#     .run_command("pip install --upgrade vllm==0.9.2")
#     .run_command("pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.6.post1%2Bcu128torch2.7-cp39-abi3-linux_x86_64.whl")
#     .run_command("pip install --no-cache blobfile datasets accelerate")
# )
