import os
import uuid
from io import BytesIO
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Callable, Optional, Union
from chutes.chute import Chute, ChutePack, NodeSelector
from chutes.image import Image


class GenerationInput(BaseModel):
    prompt: str
    negative_prompt: str = ""
    height: int = Field(default=1024, ge=128, le=2048)
    width: int = Field(default=1024, ge=128, le=2048)
    num_inference_steps: int = Field(default=25, ge=1, le=50)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    seed: Optional[int] = Field(default=None, ge=0, le=2**32 - 1)


class MinifiedGenerationInput(BaseModel):
    prompt: str = "a beautiful mountain landscape"


class DiffusionChute(ChutePack):
    generate: Callable


def build_diffusion_chute(
    username: str,
    name: str,
    model_name_or_url: str,
    node_selector: NodeSelector,
    image: Union[str, Image],
    readme: str = "",
    xl: Optional[bool] = True,
):
    chute = Chute(
        username=username,
        name=name,
        readme=readme,
        image=image,
        node_selector=node_selector,
        concurrency=1,
    )

    @chute.on_startup()
    async def initialize_pipeline(self):
        """
        Initialize the pipeline, download model if necessary.
        """
        import torch
        import aiohttp
        from urllib.parse import urlparse
        from diffusers import (
            StableDiffusionPipeline,
            StableDiffusionXLPipeline,
        )

        self.torch = torch
        torch.cuda.empty_cache()
        torch.cuda.init()
        torch.cuda.set_device(0)

        # Initialize cache dir.
        hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        os.makedirs(hf_home, exist_ok=True)
        civitai_home = os.getenv("CIVITAI_HOME", os.path.expanduser("~/.cache/civitai"))
        os.makedirs(civitai_home, exist_ok=True)

        # Handle civitai models/cache.
        model_identifier = model_name_or_url
        if model_name_or_url.lower().startswith("https://civitai.com"):
            model_id = urlparse(model_name_or_url).path.rstrip("/").split("/")[-1]
            api_url = f"https://civitai.com/api/v1/models/{model_id}"
            async with aiohttp.ClientSession(raise_for_status=True) as session:
                async with session.get(api_url) as resp:
                    model_info = await resp.json()
                    download_url = model_info["modelVersions"][0]["downloadUrl"]
                    model_path = os.path.join(civitai_home, f"{model_id}.safetensors")
                    model_identifier = model_path
                if not os.path.exists(model_path):
                    print(f"Downloading model: {download_url}")
                    async with session.get(download_url) as response:
                        with open(model_path, "wb") as outfile:
                            while chunk := await response.content.read(8192):
                                outfile.write(chunk)

        # Initialize the pipeline.
        pipeline_class = StableDiffusionXLPipeline if xl else StableDiffusionPipeline
        self.pipeline = pipeline_class.from_pretrained(
            model_identifier, torch_dtype=torch.float16, safety_checker=None, use_safetensors=True
        )
        self.pipeline.to("cuda")

    @chute.cord(
        public_api_path="/generate",
        method="POST",
        input_schema=GenerationInput,
        minimal_input_schema=MinifiedGenerationInput,
        output_content_type="image/jpeg",
        pass_chute=True,
    )
    async def generate(self, params: GenerationInput) -> FileResponse:
        """
        Generate an image.
        """
        generator = None
        if params.seed is not None:
            generator = self.torch.Generator(device=self.device).manual_seed(params.seed)
        with self.torch.inference_mode():
            result = self.pipeline(
                prompt=params.prompt,
                negative_prompt=params.negative_prompt,
                height=params.height,
                width=params.width,
                num_inference_steps=params.num_inference_steps,
                num_images=1,
                guidance_scale=params.guidance_scale,
                generator=generator,
            )
        image = result.images[0]
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        return FileResponse(buffer, media_type="image/jpeg", filename=f"{uuid.uuid4()}.jpg")

    return DiffusionChute(
        chute=chute,
        generate=generate,
    )
