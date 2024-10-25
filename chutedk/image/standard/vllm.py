from chutedk.image import Image

VLLM = (
    Image()
    .with_python("3.12.7")
    .apt_install(["google-perftools", "git"])
    .run_command("useradd vllm -s /sbin/nologin")
    .run_command("/opt/python/bin/pip install vllm==0.6.3 wheel packaging")
    .run_command("/opt/python/bin/pip install flash-attn==2.6.3")
    .run_command("/opt/python/bin/pip uninstall -y xformers")
    .run_command("mkdir -p /workspace && chown vllm:vllm /workspace")
    .set_user("vllm")
    .set_workdir("/workspace")
)
