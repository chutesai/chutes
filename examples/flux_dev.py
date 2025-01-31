import uuid
from io import BytesIO
from fastapi import Response
from pydantic import BaseModel, Field, ValidationError
from typing import Optional
from chutes.chute import Chute, NodeSelector
from chutes.image import Image


# Define an input schema so we can properly handle HTTP invocation via the chutes subdomain.
# The other benefit is the chute will use this input schema via function type hints to
# automatically generate JSONSchema objects which can be used to automagically build UIs.
class GenerationInput(BaseModel):
    prompt: str
    height: int = Field(default=1024, ge=128, le=2048)
    width: int = Field(default=1024, ge=128, le=2048)
    num_inference_steps: int = Field(default=10, ge=1, le=30)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    seed: Optional[int] = Field(default=None, ge=0, le=2**32 - 1)


# Minimal input is just a prompt, the other params can be defaults.
class MinifiedGenerationInput(BaseModel):
    prompt: str = "a beautiful mountain landscape"


# Create a markdown readme (pulled from model page on huggingface).
readme = """`FLUX.1 [dev]` is a 12 billion parameter rectified flow transformer capable of generating images from text descriptions.
For more information, please read our [blog post](https://blackforestlabs.ai/announcing-black-forest-labs/).

# Key Features
1. Cutting-edge output quality, second only to our state-of-the-art model `FLUX.1 [pro]`.
2. Competitive prompt following, matching the performance of closed source alternatives .
3. Trained using guidance distillation, making `FLUX.1 [dev]` more efficient.
4. Open weights to drive new scientific research, and empower artists to develop innovative workflows.
5. Generated outputs can be used for personal, scientific, and commercial purposes as described in the [`FLUX.1 [dev]` Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).
"""

image = (
    Image(
        username="chutes", name="flux.1-dev", tag="0.0.2", readme=readme,
    )
    .from_base("parachutes/flux.1-dev:latest")
)


# Define the chute.
chute = Chute(
    username="chutes",
    name="FLUX.1-dev",
    readme=readme,
    image=image,
    # This model is quite large, so we'll require GPUs with at least 48GB VRAM to run it.
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=80,
    ),
    # Limit one request at a time.
    concurrency=1,
)


@chute.on_startup()
async def initialize_pipeline(self):
    """
    Initialize the pipeline, download model if necessary.

    This code never runs on your machine directly, it runs on the GPU nodes
    powering chutes.
    """
    import torch
    from diffusers import FluxPipeline

    self.torch = torch
    torch.cuda.empty_cache()
    torch.cuda.init()
    torch.cuda.set_device(0)

    self.pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        cache_dir="/home/chutes/.cache/huggingface/hub",
    ).to("cuda")


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
    generator = None
    if params.seed is not None:
        generator = self.torch.Generator(device="cuda").manual_seed(params.seed)
    with self.torch.inference_mode():
        result = self.pipeline(
            prompt=params.prompt,
            height=params.height,
            width=params.width,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            max_sequence_length=256,
            generator=generator,
        )
    image = result.images[0]
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return Response(
        content=buffer.getvalue(),
        media_type="image/jpeg",
        headers={"Content-Disposition": f'attachment; filename="{uuid.uuid4()}.jpg"'},
    )
