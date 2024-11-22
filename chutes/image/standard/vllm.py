VLLM = "parachutes/vllm:0.6.2"

# To build this yourself, you can use something like:
# from chutes.image import Image  # noqa: E402

# image = (
#     Image("username", "vllm-custom", "0.6.2")
#     .with_python("3.12.7")
#     .apt_install(["git"])
#     .run_command("useradd vllm -s /sbin/nologin")
#     .run_command(
#         "mkdir -p /app /home/vllm && chown vllm:vllm /app /home/vllm"
#     )
#     .set_user("vllm")
#     .set_workdir("/app")
#     .with_env("PATH", "/opt/python/bin:$PATH")
#     .run_command("/opt/python/bin/pip install --no-cache 'vllm<0.6.4' wheel packaging")
#     .run_command("/opt/python/bin/pip install --no-cache flash-attn")
#     .run_command("/opt/python/bin/pip uninstall -y xformers")
# )
