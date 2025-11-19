SGLANG = "chutes/sglang:nightly-2025111800"

# Example...
# from chutes.image import Image
#
# image = (
#     Image(
#         username="chutes",
#         name="sglang",
#         tag="nightly-2025101900",
#         readme="SGLang is a fast serving framework for large language models and vision language models. It makes your interaction with models faster and more controllable by co-designing the backend runtime and frontend language.",
#     )
#     .from_base("parachutes/python:3.12")
#     .run_command("pip install --upgrade datasets blobfile tiktoken setuptools wheel qwen-vl-utils xgrammar llguidance")
#     .run_command("git clone -b chutes https://github.com/chutesai/sglang sglang_src && cd sglang_src && pip install -e python[all] ")
#     .run_command(
#         "pip install tilelang --torch-backend=cu128 "
#         "git+https://github.com/Dao-AILab/fast-hadamard-transform.git "
#         "git+https://github.com/deepseek-ai/DeepGEMM "
#         "https://github.com/sgl-project/whl/releases/download/v0.3.16.post2/sgl_kernel-0.3.16.post2-cp310-abi3-manylinux2014_x86_64.whl "
#         "--no-build-isolation"
#     )
#     .with_env("FLASH_MLA_DISABLE_SM100", "1")
#     .run_command(
#         "git clone https://github.com/deepseek-ai/FlashMLA.git flashmla_src && cd flashmla_src && git submodule update --init --recursive && pip install -v -e . --no-build-isolation"
#     )
#     .with_env("SGLANG_ENABLE_JIT_DEEPGEMM", "1")
#     .with_env("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")
# )
