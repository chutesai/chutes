from chutes.chute import NodeSelector
from chutes.chute.template.vllm import build_vllm_chute

# If you wanted to build the image yourself, it would like something like:
#
# from chutes.image import Image
# image = (
#     Image(
#         username="chutes", name="vllm", tag="0.6.4", readme="## vLLM - fast, flexible llm inference"
#     )
#     .from_base("parachutes/base-python:3.12.7")
#     .run_command("pip install --no-cache 'vllm<0.6.5' wheel packaging")
#     .run_command("pip install --no-cache flash-attn")
#     .run_command("pip uninstall -y xformers")
# )
#
# Then, update the below with image=image

chute = build_vllm_chute(
    username="chutes",
    readme="## Meta Llama 3.2 1B Instruct\n### Hello.",
    model_name="unsloth/Llama-3.2-1B-Instruct",
    image="chutes/vllm:0.6.4",
    # The smallest supported GPU on chutes is 16GB VRAM, so we don't need
    # any significant detail here in the node selector, any node should work.
    node_selector=NodeSelector(
        gpu_count=1,
    ),
)
