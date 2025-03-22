VLLM = "chutes/vllm:0.8.1.p0"

# To build this yourself, you can use something like:
# image = (
#     Image(
#         username="chutes",
#         name="vllm",
#         tag="0.8.1",
#         readme="## vLLM - fast, flexible llm inference",
#     )
#     .from_base("parachutes/base-python:3.12.9")
#     .run_command(
#         "pip install --no-cache wheel packaging git+https://github.com/huggingface/transformers.git qwen-vl-utils[decord]==0.0.8"
#     )
#     .run_command("pip install --upgrade vllm==0.8.1")
#     .run_command("pip install --no-cache flash-attn")
# )
