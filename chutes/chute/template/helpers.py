import os
import re
import time
import random
import aiohttp
import asyncio
from loguru import logger
from huggingface_hub import HfApi


def get_current_hf_commit(model_name: str):
    """
    Helper to load the current main commit for a given repo.
    """
    api = HfApi()
    for ref in api.list_repo_refs(model_name).branches:
        if ref.ref == "refs/heads/main":
            return ref.target_commit
    return None


async def prompt_one(
    model_name: str, base_url: str = "http://127.0.0.1:10101", prompt: str = None
) -> str:
    """
    Send a prompt to the model.
    """
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(0)) as session:
        started_at = time.time()
        if not prompt:
            prompt = (
                "The following is a very long, extraordinarily detailed and verbose story about "
                + random.choice(
                    [
                        "apples",
                        "bananas",
                        "grapes",
                        "raspberries",
                        "dogs",
                        "cats",
                        "goats",
                        "zebras",
                    ]
                )
                + ": "
            )
        async with session.post(
            f"{base_url}/v1/completions",
            json={"model": model_name, "prompt": prompt, "max_tokens": 1000},
        ) as resp:
            if resp.status == 200:
                result = await resp.json()
                delta = time.time() - started_at
                tokens = result["usage"]["completion_tokens"]
                assert tokens <= 1005, "Produced more tokens than asked."
                tps = tokens / delta
                logger.info(f"Generated {tokens=} in {delta=} for {tps=}")
                return result["choices"][0]["text"]
            if resp.status == 400:
                return None
            resp.raise_for_status()


async def warmup_model(chute, base_url: str = "http://127.0.0.1:10101"):
    """
    Warm up a model on startup.
    """
    logger.info(f"Warming up model with max concurrency: {chute.name=} {chute.concurrency=}")

    # Test simple prompts at max concurrency.
    responses = await asyncio.gather(
        *[prompt_one(chute.name, base_url=base_url) for idx in range(chute.concurrency)]
    )
    assert all(responses)
    combined_response = "\n\n".join(responses) + "\n\n"
    logger.info("Now with larger context...")

    # Large-ish context prompts.
    for multiplier in range(1, 4):
        prompt = (
            "Summarize the following stories:\n\n"
            + combined_response * multiplier
            + "\nThe summary is: "
        )
        responses = await asyncio.gather(
            *[
                prompt_one(chute.name, base_url=base_url, prompt=prompt)
                for idx in range(chute.concurrency)
            ]
        )
        if all(responses):
            logger.success(f"Warmed up with {multiplier=}")
        else:
            logger.warning(f"Stopping at {multiplier=}")
            break

    # One final prompt to make sure large context didn't crash it.
    assert await prompt_one(chute.name, base_url=base_url)


def set_default_cache_dirs(download_path):
    for key in [
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "FLASHINFER_WORKSPACE_BASE",
        "XFORMERS_CACHE_DIR",
        "DG_JIT_CACHE_DIR",
        "SGL_DG_CACHE_DIR",
        "SGLANG_DG_CACHE_DIR",
    ]:
        if not os.getenv(key):
            os.environ[key] = os.path.join(download_path, f"_{key.lower()}")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def set_nccl_flags(gpu_count, model_name):
    if gpu_count > 1 and re.search(
        "h[12]0|b[23]00|5090|l40s|6000 ada|a100|h800|pro 6000|sxm", model_name, re.I
    ):
        for key in ["NCCL_P2P_DISABLE", "NCCL_IB_DISABLE", "NCCL_NET_GDR_LEVEL"]:
            if key in os.environ:
                del os.environ[key]
