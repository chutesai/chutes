SGLANG = "chutes/sglang:0.4.10.post1"

# Example...
# image = (
#     Image(
#         username="chutes",
#         name="sglang",
#         tag="0.4.10.post1",
#         readme="SGLang is a fast serving framework for large language models and vision language models. It makes your interaction with models faster and more controllable by co-designing the backend runtime and frontend language.",
#     )
#     .from_base("parachutes/python:3.12")
#     .run_command("pip install --upgrade pip")
#     .run_command("pip install --upgrade 'sglang[all]==0.4.10.post1' datasets blobfile tiktoken")
#     .with_env("SGL_ENABLE_JIT_DEEPGEMM", "1")
#     .add("fix_logit_bias.patch", "/tmp/fix_logit_bias.patch")
#     .run_command("cd /home/chutes/.local/lib/python3.12/site-packages/sglang/srt/sampling && patch -p1 < /tmp/fix_logit_bias.patch")
# )
