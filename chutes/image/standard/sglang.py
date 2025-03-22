SGLANG = "chutes/sglang:0.4.4.post1"

# To build this yourself, you can use something like:
# from chutes.image import Image
# image = (
#     Image(
#         username="chutes",
#         name="sglang",
#         tag="0.4.4.post1",
#         readme="SGLang is a fast serving framework for large language models and vision language models. It makes your interaction with models faster and more controllable by co-designing the backend runtime and frontend language.",
#     )
#     .from_base("parachutes/base-python:3.12.9")
#     .run_command("pip install --upgrade pip")
#     .run_command(
#         "pip install --upgrade 'sglang[all]>=0.4.4.post1' --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/"
#     )
#     .with_env("SGL_ENABLE_JIT_DEEPGEMM", "1")
#     .run_command("pip install git+https://github.com/huggingface/transformers.git datasets")
# )
