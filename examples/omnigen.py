import os
import uuid
import tempfile
import pybase64 as base64
from io import BytesIO
from fastapi import Response
from pydantic import BaseModel, Field
from typing import Optional
from chutes.chute import Chute, NodeSelector
from chutes.image import Image

# Docker image.
image = (
    Image(username="chutes", name="omnigen", tag="0.0.1", readme="OmniGen diffusion multi-modal image library")
    .from_base("parachutes/base-python:3.12.7")
    .run_command(
        "pip install diffusers==0.32.1 transformers accelerate safetensors xformers protobuf sentencepiece"
    )
    .run_command("pip install git+https://github.com/staoxiao/OmniGen.git")
)


# Define an input schema so we can properly handle HTTP invocation via the chutes subdomain.
# The other benefit is the chute will use this input schema via function type hints to
# automatically generate JSONSchema objects which can be used to automagically build UIs.
class GenerationInput(BaseModel):
    prompt: str
    image_b64: Optional[list[str]] = Field(
        default=None, description="Base64 encoded images for image-to-image pipelines."
    )
    height: int = Field(default=1024, ge=128, le=2048)
    width: int = Field(default=1024, ge=128, le=2048)
    num_inference_steps: int = Field(default=15, ge=1, le=50)
    guidance_scale: float = Field(default=2.5, ge=1.0, le=20.0)
    img_guidance_scale: Optional[float] = Field(default=1.6, ge=1.0, le=20.0)
    seed: Optional[int] = Field(default=None, ge=0, le=2**32 - 1)


# Minimal input is just a prompt, the other params can be defaults.
class MinifiedGenerationInput(BaseModel):
    prompt: str = "a beautiful mountain landscape"


readme = "OmniGen is a unified image generation model that can generate a wide range of images from multi-modal prompts. It is designed to be simple, flexible, and easy to use."

# Define the chute.
chute = Chute(
    username="chutes",
    name="Shitao/OmniGen-v1",
    readme=readme,
    image=image,
    # This model is quite large, so we'll require GPUs with at least 48GB VRAM to run it.
    node_selector=NodeSelector(
        gpu_count=1,
        include=["4090", "l40s", "a6000_ada", "h100", "h100_sxm"],
    ),
    # Limit one request at a time.
    concurrency=4,
)


@chute.on_startup()
async def initialize_pipeline(self):
    """
    Initialize the pipeline, download model if necessary.

    This code never runs on your machine directly, it runs on the GPU nodes
    powering chutes.
    """
    import torch
    from OmniGen import OmniGenPipeline

    self.torch = torch
    torch.cuda.empty_cache()
    torch.cuda.init()
    torch.cuda.set_device(0)

    self.pipeline = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")


@chute.cord(
    # Expose this function via the subdomain-based chutes.ai HTTP invocation, e.g.
    # this becomes https://{username}-{chute slug}.chutes.ai/generate
    public_api_path="/generate",
    # The function is invoked in the subdomain-based system via POSTs.
    method="POST",
    # Input/minimal input schemas.
    input_schema=GenerationInput,
    minimal_input_schema=MinifiedGenerationInput,
    # Set output content type header to image/jpeg so we can return the raw image.
    output_content_type="image/jpeg",
)
async def generate(self, params: GenerationInput) -> Response:
    """
    Generate an image.
    """
    image_paths = []
    try:
        if params.image_b64:
            for img in params.image_b64:
                with tempfile.TemporaryFile(mode="wb", delete=False) as outfile:
                    outfile.write(base64.b64decode(img))
                    image_paths.append(outfile.name)
        with self.torch.inference_mode():
            images = self.pipeline(
                prompt=params.prompt,
                input_images=image_paths,
                height=params.height,
                width=params.width,
                guidance_scale=params.guidance_scale,
                img_guidance_scale=params.img_guidance_scale,
                seed=params.seed,
            )
        buffer = BytesIO()
        images[0].save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        return Response(
            content=buffer.getvalue(),
            media_type="image/jpeg",
            headers={"Content-Disposition": f'attachment; filename="{uuid.uuid4()}.jpg"'},
        )
    finally:
        for path in image_paths:
            os.remove(path)
