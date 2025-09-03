import os
import re
import uuid
import json
import asyncio
from loguru import logger
from io import BytesIO
from fastapi import Response
from pydantic import BaseModel, Field
from typing import Callable, Optional, Union
from chutes.chute import Chute, ChutePack, NodeSelector
from chutes.image import Image
from chutes.image.standard.diffusion import DIFFUSION


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


def single_file_pipeline(model_path, model_type="auto"):
    """
    Load either SD or SDXL models with fallback components.
    """
    import torch
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        AutoencoderKL,
        UNet2DConditionModel,
    )
    from transformers import (
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPTokenizer,
    )
    import safetensors.torch

    # Load model metadata to check architecture
    primary = "sdxl"
    secondary = "sd"
    if model_type == "auto":
        try:
            tensors = safetensors.torch.load_file(model_path, device="cpu")
            if (
                model_type == "auto"
                and any(k.startswith("conditioner.embedders.1") for k in tensors.keys())
                or any("text_model_2" in k for k in tensors.keys())
            ):
                print("Appears to be SDXL architecture.")
                ...
            else:
                primary, secondary = "sd", "sdxl"
                print("Appears to be SD architecture.")
        except Exception:
            ...

    if model_type == "auto":
        try:
            return single_file_pipeline(model_path, primary)
        except Exception:
            return single_file_pipeline(model_path, secondary)

    pipeline_class = StableDiffusionXLPipeline if model_type == "sdxl" else StableDiffusionPipeline
    base_model = (
        "stabilityai/stable-diffusion-xl-base-1.0"
        if model_type == "sdxl"
        else "runwayml/stable-diffusion-v1-5"
    )
    try:
        pipeline = pipeline_class.from_single_file(
            model_path, torch_dtype=torch.float16, use_safetensors=True
        )
    except Exception:
        try:
            components = {}
            components["unet"] = UNet2DConditionModel.from_pretrained(
                base_model,
                subfolder="unet",
                torch_dtype=torch.float16,
                variant="fp16",
            )
            if model_type == "sdxl":
                components["tokenizer"] = CLIPTokenizer.from_pretrained(
                    base_model, subfolder="tokenizer"
                )
                components["tokenizer_2"] = CLIPTokenizer.from_pretrained(
                    base_model, subfolder="tokenizer_2"
                )
                components["text_encoder"] = CLIPTextModel.from_pretrained(
                    base_model, subfolder="text_encoder", torch_dtype=torch.float16
                )
                components["text_encoder_2"] = CLIPTextModelWithProjection.from_pretrained(
                    base_model,
                    subfolder="text_encoder_2",
                    torch_dtype=torch.float16,
                )
            else:
                components["tokenizer"] = CLIPTokenizer.from_pretrained(
                    base_model, subfolder="tokenizer"
                )
                components["text_encoder"] = CLIPTextModel.from_pretrained(
                    base_model, subfolder="text_encoder", torch_dtype=torch.float16
                )
            pipeline = pipeline_class.from_single_file(
                model_path,
                **components,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
        except Exception:
            vae_model = (
                "madebyollin/sdxl-vae-fp16-fix"
                if model_type == "sdxl"
                else "stabilityai/sd-vae-ft-mse"
            )
            components["vae"] = AutoencoderKL.from_pretrained(vae_model, torch_dtype=torch.float16)
            pipeline = pipeline_class.from_single_file(
                model_path,
                **components,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
    return pipeline.to("cuda")


def build_diffusion_chute(
    username: str,
    name: str,
    model_name_or_url: str,
    node_selector: NodeSelector,
    tagline: str = "",
    readme: str = "",
    concurrency: int = 3,
    version: Optional[str] = None,
    revision: Optional[str] = None,
    image: Optional[Union[str, Image]] = DIFFUSION,
    pipeline_args: Optional[dict] = {},
    max_instances: int = 1,
    scaling_threshold: float = 0.75,
    shutdown_after_seconds: int = 300,
):
    chute = Chute(
        username=username,
        name=name,
        tagline=tagline,
        readme=readme,
        image=image,
        node_selector=node_selector,
        standard_template="diffusion",
        concurrency=concurrency,
        shutdown_after_seconds=shutdown_after_seconds,
        max_instances=max_instances,
        scaling_threshold=scaling_threshold,
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
        single_file = False
        if model_name_or_url.lower().startswith("https://civitai.com/"):
            download_url = (
                "https://civitai.com/api/download/models/{version}?type=Model&format=SafeTensor"
            )
            path_match = re.search(r"^(.+)?/models/([0-9]+)/?.*", urlparse(model_name_or_url).path)
            if not path_match:
                raise Exception("Invalid civitai.com URL!")
            model_id = path_match.group(2)
            if version:
                download_url = download_url.format(version=version)
            else:
                if path_match.group(1) and path_match.group(1).lower().endswith("download"):
                    download_url = download_url.format(version=str(model_id))
                else:
                    # Need to get the actual download link from API.
                    async with aiohttp.ClientSession(
                        raise_for_status=True, trust_env=True
                    ) as session:
                        async with session.get(
                            f"https://civitai.com/api/v1/models/{model_id}"
                        ) as resp:
                            try:
                                data = await resp.json()
                            except Exception:
                                data = json.loads(await resp.text())
                            download_url = download_url.format(
                                version=str(data["modelVersions"][0]["id"])
                            )

            # Now do the actual download.
            model_path = os.path.join(civitai_home, f"{model_id}.safetensors")
            model_identifier = model_path
            single_file = True
            if not os.path.exists(model_path):
                async with aiohttp.ClientSession(raise_for_status=True, trust_env=True) as session:
                    async with session.get(download_url) as resp:
                        with open(model_path, "wb") as outfile:
                            while chunk := await resp.content.read(8192):
                                outfile.write(chunk)

        # Initialize the pipeline.
        if single_file:
            self.pipeline = single_file_pipeline(model_identifier)
        else:
            # Assume SDXL, fallback to SD
            from huggingface_hub import snapshot_download

            download_kwargs = {}
            if revision:
                download_kwargs["revision"] = revision
                pipeline_args["revision"] = revision

            logger.info(f"Downloading {model_identifier} with {download_kwargs=}")
            await asyncio.to_thread(snapshot_download, repo_id=model_identifier, **download_kwargs)
            try:
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_identifier,
                    torch_dtype=torch.float16,
                    **pipeline_args,
                ).to("cuda")
            except Exception:
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    model_identifier,
                    torch_dtype=torch.float16,
                    **pipeline_args,
                ).to("cuda")

    @chute.cord(
        public_api_path="/generate",
        method="POST",
        input_schema=GenerationInput,
        minimal_input_schema=MinifiedGenerationInput,
        output_content_type="image/jpeg",
        pass_chute=True,
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
        return Response(
            content=buffer.getvalue(),
            media_type="image/jpeg",
            headers={"Content-Disposition": f'attachment; filename="{uuid.uuid4()}.jpg"'},
        )

    return DiffusionChute(
        chute=chute,
        generate=generate,
    )
