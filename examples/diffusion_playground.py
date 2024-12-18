from chutes.chute import NodeSelector
from chutes.chute.template.diffusion import build_diffusion_chute

# If you wanted to build the image yourself, it would like something like:
#
# from chutes.image import Image
# image = (
#     Image(username="chutes", name="diffusion", tag="0.31.0", readme="## Diffusion pipelines")
#     .from_base("parachutes/base-python:3.12.7")
#     .run_command("pip install diffusers==0.31.0 transformers accelerate safetensors xformers")
# )
#
# Then, update the below with image=image

chute = build_diffusion_chute(
    username="chutes",
    name="playground-v2.5",
    readme="## Playground V2.5 Aesthetic",
    model_name_or_url="playgroundai/playground-v2.5-1024px-aesthetic",
    image="chutes/diffusion:0.31.0",
    node_selector=NodeSelector(
        gpu_count=1,
    ),
)
