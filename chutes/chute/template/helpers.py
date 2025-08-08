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


async def prompt_one(model_name: str, base_url: str = "http://127.0.0.1:10101", prompt: str = None):
    """
    Send a prompt to the model.
    """
    async with aiohttp.ClientSession(
        raise_for_status=True, timeout=aiohttp.ClientTimeout(0)
    ) as session:
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
            result = await resp.json()
            delta = time.time() - started_at
            tokens = result["usage"]["completion_tokens"]
            tps = tokens / delta
            logger.info(f"Generated {tokens=} in {delta=} for {tps=}")
            return result["choices"][0]["text"]


async def warmup_model(chute, base_url: str = "http://127.0.0.1:10101"):
    """
    Warm up a model on startup.
    """
    try:
        logger.info(f"Warming up model with max concurrency: {chute.name=} {chute.concurrency=}")
        responses = await asyncio.gather(
            *[prompt_one(chute.name, base_url=base_url) for idx in range(chute.concurrency)]
        )
        combined_response = "\n\n".join([r or "" for r in responses]) + "\n\n"
        logger.info("Now with larger context...")
        multiplier = 1
        while multiplier < 4:
            try:
                prompt = (
                    "Summarize the following stories:\n\n"
                    + combined_response * multiplier
                    + "\nThe summary is: "
                )
                await asyncio.gather(
                    *[
                        prompt_one(chute.name, base_url=base_url, prompt=prompt)
                        for idx in range(chute.concurrency)
                    ]
                )
                logger.success(f"Warmed up with {multiplier=}")
            except Exception:
                logger.warning(f"Stopping at {multiplier=}, failed to get response")
                break
            multiplier += 1

    except Exception as exc:
        logger.warning(f"Failed to warmup with {chute.concurrency=}: {str(exc)}")
