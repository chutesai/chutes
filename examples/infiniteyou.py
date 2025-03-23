import uuid
import os
import random
import base64
from enum import Enum
from io import BytesIO
from PIL import Image
from fastapi import HTTPException, status, Response
from pydantic import BaseModel, Field
from typing import Optional
from chutes.chute import Chute, NodeSelector
from chutes.image import Image as ChutesImage


class GenerationInput(BaseModel):
    prompt: str
    id_image_b64: str
    control_image_b64: Optional[str] = None
    seed: Optional[int] = Field(None, gt=0.0, le=1000000000)
    guidance_scale: float = Field(3.5, gt=1.0, le=10.0)
    steps: int = Field(30, gt=10, le=50)
    infusenet_conditioning_scale: float = Field(1.0, ge=0.0, le=1.0)
    infusenet_guidance_start: float = Field(0.0, ge=0.0, le=1.0)
    infusenet_guidance_end: float = Field(1.0, ge=0.0, le=1.0)


readme = "InfiniteYou: Flexible Photo Recrafting While Preserving Your Identity"
image = (
    ChutesImage(
        username="chutes",
        name="infiniteyou",
        tag="0.0.1",
        readme=readme,
    )
    .from_base("parachutes/flux.1-dev:latest")
    .set_user("root")
    .run_command("apt -y update && apt -y install git unzip libgl1-mesa-glx")
    .set_user("chutes")
    .run_command("git clone https://github.com/bytedance/InfiniteYou")
    .run_command("mv -f InfiniteYou/* /app/ && pip install -r /app/requirements.txt")
    .add("const.png", "/app/const.png")
)

chute = Chute(
    username="chutes",
    name="infiniteyou",
    readme=readme,
    image=image,
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=80,
    ),
    concurrency=2,
)


@chute.on_startup()
async def initialize_pipeline(self):
    import torch
    from pipelines.pipeline_infu_flux import InfUFluxPipeline
    from huggingface_hub import snapshot_download

    self.torch = torch
    torch.cuda.empty_cache()
    torch.cuda.init()

    # Load the pipelines.
    cache_path = os.path.join(
        os.getenv("HF_HOME", "/cache"),
        "InfiniteYou",
    )
    snapshot_path = snapshot_download(repo_id="ByteDance/InfiniteYou", local_dir=cache_path)
    self.lora_dir = os.path.join(snapshot_path, "supports/optional_loras")
    self.torch.cuda.set_device(0)
    infu_model_path = os.path.join(snapshot_path, "infu_flux_v1.0", "aes_stage2")
    insightface_root_path = os.path.join(snapshot_path, "supports", "insightface")
    self.pipe = InfUFluxPipeline(
        base_model_path="/home/chutes/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44",
        infu_model_path=infu_model_path,
        insightface_root_path=insightface_root_path,
        infu_flux_version="v1.0",
        model_version="aes_stage2",
    )

    # With loras.
    loras = [
        [os.path.join(self.lora_dir, "flux_realism_lora.safetensors"), "realism", 1.0],
        [os.path.join(self.lora_dir, "flux_anti_blur_lora.safetensors"), "anti_blur", 1.0],
    ]
    self.pipe.load_loras(loras)

    # Perform a single edit in the warmup phase to make sure the model is compiled before serving requests.
    image = self.pipe(
        id_image=Image.open("/app/const.png").convert("RGB"),
        prompt="An old man, happy with his life's work",
        control_image=None,
        seed=torch.seed() & 0xFFFFFFFF,
        guidance_scale=3.5,
        num_steps=30,
        infusenet_conditioning_scale=1.0,
        infusenet_guidance_start=0.0,
        infusenet_guidance_end=1.0,
    )
    image.save(f"/app/warmup.png")


def prepare_input_images(args):
    """
    Decode and prepare input image(s).
    """
    id_image = None
    control_image = None
    try:
        id_image = Image.open(BytesIO(base64.b64decode(args.id_image_b64))).convert("RGB")
        if args.control_image_b64:
            control_image = Image.open(BytesIO(base64.b64decode(args.control_image_b64))).convert(
                "RGB"
            )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image input! {exc}",
        )
    return id_image, control_image


@chute.cord(
    public_api_path="/generate",
    method="POST",
    input_schema=GenerationInput,
)
async def generate(self, args: GenerationInput):
    """
    Generate an image based on an input ID image, an optional control image, and a prompt.
    """
    with self.torch.no_grad():
        id_image, control_image = prepare_input_images(args)
        if args.seed is None or not args.seed:
            args.seed = random.randint(0, 1000000000)

        # Generate the image.
        image = self.pipe(
            id_image=id_image,
            prompt=args.prompt,
            control_image=control_image,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            num_steps=args.steps,
            infusenet_conditioning_scale=args.infusenet_conditioning_scale,
            infusenet_guidance_start=args.infusenet_guidance_start,
            infusenet_guidance_end=args.infusenet_guidance_end,
        )

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        return Response(
            content=buffer.getvalue(),
            media_type="image/jpeg",
            headers={"Content-Disposition": f'attachment; filename="{uuid.uuid4()}.jpg"'},
        )
