DIFFUSION = "chutes/diffusion:0.32.1"

# To build this yourself, you can use something like:
# image = (
#     Image(username="chutes", name="diffusion", tag="0.32.1", readme="## Diffusion pipelines")
#     .from_base("parachutes/base-python:3.12.7")
#     .run_command(
#         "pip install diffusers==0.32.1 transformers accelerate safetensors xformers protobuf sentencepiece"
#     )
# )
