import os
import re
import io
import asyncio
import uuid
import hashlib
import json
import time
import tempfile
from copy import deepcopy
from loguru import logger
from fastapi import Response
from pydantic import BaseModel, Field
from typing import Optional
from chutes.image import Image as ChuteImage
from chutes.chute import Chute, NodeSelector

OUTPUT_RE = re.compile(r"Outputs:\s*\n(.*?)\n", re.MULTILINE)

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
chute_image = (
    ChuteImage(
        username="chutes",
        name="comfyui",
        tag="0.0.1",
        readme="comfyui base image",
    )
    .from_base("parachutes/base-python:3.12.7")
    .set_user("root")
    .run_command("apt -y update && apt -y install git unzip libgl1-mesa-glx")
    .set_user("chutes")
    .run_command("pip install --upgrade pip")
    .run_command("pip install comfy-cli==1.3.8")
    .run_command("pip install huggingface_hub[hf_transfer]")
    .run_command(
        "pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"
    )
    .run_command("comfy --skip-prompt install --nvidia --version 0.3.26")
    .set_workdir("/home/chutes/comfy/ComfyUI/custom_nodes")
    .run_command(
        "git clone https://github.com/ltdrdata/ComfyUI-Inspire-Pack && cd ComfyUI-Inspire-Pack && git checkout 985f6a239b1aed0c67158f64bf579875ec292cb2 && cd -"
    )
    .run_command(
        "git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus && cd ComfyUI_IPAdapter_plus && git checkout 0d0a7b3693baf8903fe2028ff218b557d619a93d && cd -"
    )
    .run_command(
        "git clone https://github.com/cubiq/ComfyUI_InstantID && cd ComfyUI_InstantID && git checkout 50445991e2bd1d5ec73a8633726fe0b33a825b5b && cd -"
    )
    .set_workdir("/home/chutes/comfy/ComfyUI")
    .run_command(
        "mkdir -p models/insightface/models && "
        "cd models/insightface/models && "
        "wget -O antelopev2.zip https://huggingface.co/tau-vision/insightface-antelopev2/resolve/main/antelopev2.zip && "
        "unzip antelopev2.zip && rm -f antelopev2.zip && "
        "mkdir -p /home/chutes/comfy/ComfyUI/models/instantid && cd /home/chutes/comfy/ComfyUI/models/instantid && "
        "wget -O ip-adapter.bin 'https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin?download=true' && "
        "mkdir -p /home/chutes/comfy/ComfyUI/models/controlnet && cd /home/chutes/comfy/ComfyUI/models/controlnet && "
        "wget -O diffusion_pytorch_model.safetensors 'https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors?download=true'"
    )
    .run_command("pip install opencv-python insightface 'uvicorn[standard]'")
    .run_command(
        "git clone https://github.com/lodestone-rock/ComfyUI_FluxMod.git /home/chutes/comfy/ComfyUI/custom_nodes/ComfyUI_FluxMod"
    )
    .add("chroma_workflow_api.json", "/app/workflow.json")
    .with_env("PYTHONPATH", "/home/chutes/comfy/ComfyUI")
    .set_workdir("/app")
)


class TextToImagePayload(BaseModel):
    prompt: str = Field(...)
    seed: Optional[int] = Field(0, title="Seed", description="Seed for text generation.", gte=0)
    steps: Optional[int] = Field(30, title="Steps", description="Steps for text generation.", gte=5, lte=50)
    cfg: Optional[float] = Field(
        4.5, title="CFG Scale", description="CFG Scale for text generation.", gte=1.0, lte=7.5
    )
    width: Optional[int] = Field(
        1024, title="Width", description="Width for text generation.", lte=2048, ge=200
    )
    height: Optional[int] = Field(
        1024, title="Height", description="Height for text generation.", lte=2048, ge=200
    )


readme = "Chroma text-to-image"
chute = Chute(
    username="chutes",
    name="chroma",
    readme=readme,
    image=chute_image,
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=24,
    ),
    concurrency=2,
)


def check_integrity(path, expected_digest):
    """
    Check the last chunk of a local file to make sure the entire file was downloaded.
    """
    if not os.path.exists(path):
        logger.debug(f"Cache path does not exist: {path}")
        return False
    file_size = os.path.getsize(path)
    hash_obj = hashlib.sha256()
    with open(path, "rb") as f:
        if file_size > 1024:
            f.seek(-1024, os.SEEK_END)
            data = f.read(1024)
        hash_obj.update(data)
    digest = hash_obj.hexdigest()
    if digest == expected_digest:
        return True
    logger.warning(f"Found cache file {path} but digest does not match!")
    return False


async def download_file(url, destination, expected_digest):
    """
    Download a file.
    """
    logger.info(f"Downloading {url} to {destination}")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if os.path.exists(f"{destination}.tmp"):
        os.remove(f"{destination}.tmp")
    process = await asyncio.create_subprocess_exec(
        "wget",
        url,
        "-O",
        f"{destination}.tmp",
    )
    stdout, stderr = await process.communicate()
    if process.returncode == 0:
        if check_integrity(f"{destination}.tmp", expected_digest):
            logger.success(f"Finished downloading {url} to {destination}")
            if os.path.exists(destination):
                os.remove(destination)
            os.rename(f"{destination}.tmp", destination)
            return
    raise Exception(f"Failed downloading {url} or corrupt file detected.")


async def ensure_downloaded_and_linked(url, destination, expected_digest):
    """
    Download a file from URL to destination.
    """
    filename = os.path.basename(destination)
    cache_path = os.path.join(
        os.getenv("HF_HOME", "/cache"),
        "chroma-comfy",
        filename,
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    logger.info(f"Ensuring {url} is available at {cache_path=} with {destination=}")
    if os.path.exists(cache_path) and check_integrity(cache_path, expected_digest):
        if os.path.exists(destination):
            os.remove(destination)
        os.symlink(cache_path, destination)
        return
    await download_file(url, cache_path, expected_digest)
    if os.path.exists(destination):
        os.remove(destination)
    os.symlink(cache_path, destination)


@chute.on_startup()
async def initialize_pipeline(self):
    """
    Ensure model files are downloaded, start comfyui.
    """
    await asyncio.gather(
        *[
            ensure_downloaded_and_linked(
                "https://huggingface.co/lodestones/Chroma/resolve/main/chroma-unlocked-v13.safetensors",
                "/home/chutes/comfy/ComfyUI/models/diffusion_models/chroma.safetensors",
                "95d0a549e7980f0b50371d0706ef434f98d961a1f0ea54999aaa55328d7f1d09",
            ),
            ensure_downloaded_and_linked(
                "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
                "/home/chutes/comfy/ComfyUI/models/clip/t5xxl.safetensors",
                "25d0c23316153b5feb539e2d077ec319a6c632311bfd06cc8766f726bc421007",
            ),
            ensure_downloaded_and_linked(
                "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors",
                "/home/chutes/comfy/ComfyUI/models/vae/ae.safetensors",
                "dc8c8a1b8d6a47004aea7029057e4a97aadebeece938d81f19d4603dc2d88f39",
            ),
        ]
    )

    # Start comfyui.
    process = await asyncio.create_subprocess_exec(
        "comfy",
        "launch",
        "--background",
        "--",
        "--port",
        "12345",
        "--listen",
        "127.0.0.1",
    )
    stdout, stderr = await process.communicate()
    assert process.returncode == 0
    logger.success("ComfyUI successfully started!")
    with open("/app/workflow.json") as infile:
        self.template = json.load(infile)


@chute.cord(
    public_api_path="/generate",
    method="POST",
    input_schema=TextToImagePayload,
    output_content_type="image/jpeg",
)
async def generate(self, args: TextToImagePayload) -> Response:
    """
    Generate an image from text.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as workflow_f:
        workflow = deepcopy(self.template)
        workflow["4"]["inputs"]["text"] = args.prompt
        workflow["9"]["inputs"].update(
            {
                "seed": args.seed,
                "steps": args.steps,
                "cfg": args.cfg,
            }
        )
        workflow["14"]["inputs"].update(
            {
                "width": args.width,
                "height": args.height,
            }
        )
        workflow["19"]["inputs"]["filename_prefix"] = str(uuid.uuid4())
        workflow_f.write(json.dumps(workflow, indent=2).encode())
        workflow_f.close()

        try:
            started_at = time.time()
            process = await asyncio.create_subprocess_exec(
                "comfy",
                "run",
                "--workflow",
                workflow_f.name,
                "--wait",
                "--timeout",
                "180",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            stdout = stdout.decode("utf-8").strip()
            stderr = stderr.decode("utf-8").strip()
            if process.returncode == 0:
                output_match = OUTPUT_RE.search(stdout)
                if not output_match:
                    raise Exception(f"No output file created: {stdout}")
                output_path = output_match.group(1)
                delta = time.time() - started_at
                logger.success(f"Generated prompt in {delta} seconds: {output_path=}")
                try:
                    buffer = io.BytesIO()
                    with open(output_path, "rb") as infile:
                        buffer.write(infile.read())
                    buffer.seek(0)
                    return Response(
                        content=buffer.getvalue(),
                        media_type="image/jpeg",
                        headers={
                            "Content-Disposition": f'attachment; filename="{uuid.uuid4()}.png"'
                        },
                    )
                finally:
                    os.remove(output_path)
            else:
                raise Exception(f"Unhandled exception executing workflow: {stderr}")
        finally:
            os.remove(workflow_f.name)
