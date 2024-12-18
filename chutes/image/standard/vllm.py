VLLM = "chutes/vllm:0.6.4"

# To build this yourself, you can use something like:
# from chutes.image import Image  # noqa: E402
# image = (
#     Image(username="chutes", name="vllm", tag="0.6.4", readme="## vLLM - fast, flexible llm inference")
#     .from_base("parachutes/base-python:3.12.7")
#     .run_command("pip install --no-cache 'vllm<0.6.5' wheel packaging")
#     .run_command("pip install --no-cache flash-attn")
#     .run_command("pip uninstall -y xformers")
# )
