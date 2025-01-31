VLLM = "chutes/vllm:0.7.0"

# To build this yourself, you can use something like:
# image = (
#     Image(
#         username="chutes", name="vllm", tag="0.7.0", readme="## vLLM - fast, flexible llm inference"
#     )
#     .from_base("parachutes/base-python:3.12.7")
#     .run_command("pip install --no-cache 'vllm==0.7.0' wheel packaging git+https://github.com/huggingface/transformers.git@ec7afad60909dd97d998c1f14681812d69a15728 qwen-vl-utils[decord]==0.0.8")
#     .run_command("pip install --no-cache flash-attn")
# )
