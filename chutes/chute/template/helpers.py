import os
import re
import sys
import time
import uuid
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
    model_name: str,
    base_url: str = "http://127.0.0.1:10101",
    prompt: str = None,
    api_key: str = None,
    require_status: int = None,
) -> str:
    """
    Send a prompt to the model.
    """
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(0)) as session:
        started_at = time.time()
        if not prompt:
            prompt = (
                "They started to tell a long, extraordinarily detailed and verbose story about "
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
            )
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        async with session.post(
            f"{base_url}/v1/completions",
            json={"model": model_name, "prompt": prompt, "max_tokens": 1000},
            headers=headers,
        ) as resp:
            if require_status:
                assert (
                    resp.status == require_status
                ), f"Expected to receive status code {require_status}, received {resp.status}"
                return await resp.json()
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


async def validate_auth(chute, base_url: str = "http://127.0.0.1:10101", api_key: str = None):
    """
    Validate authorization for the engine.
    """
    if not api_key or api_key == "None":
        await prompt_one(chute.name, base_url=base_url, api_key="None")
        return
    await prompt_one(chute.name, base_url=base_url, api_key=api_key)
    await prompt_one(chute.name, base_url=base_url, api_key=None, require_status=401)
    await prompt_one(chute.name, base_url=base_url, api_key=str(uuid.uuid4()), require_status=401)


async def warmup_model(chute, base_url: str = "http://127.0.0.1:10101", api_key: str = None):
    """
    Warm up a model on startup.
    """
    logger.info(f"Warming up model with max concurrency: {chute.name=} {chute.concurrency=}")

    # Test simple prompts at max concurrency.
    responses = await asyncio.gather(
        *[
            prompt_one(chute.name, base_url=base_url, api_key=api_key)
            for idx in range(chute.concurrency)
        ]
    )
    assert all(isinstance(r, str) or r for r in responses)
    combined_response = "\n\n".join(responses) + "\n\n"
    logger.info("Now with larger context...")

    # Large-ish context prompts.
    for multiplier in range(1, 4):
        prompt = (
            "Summarize the following stories:\n\n"
            + combined_response * multiplier
            + "\nThe summary is:"
        )
        responses = await asyncio.gather(
            *[
                prompt_one(chute.name, base_url=base_url, prompt=prompt, api_key=api_key)
                for idx in range(chute.concurrency)
            ]
        )
        if all(isinstance(r, str) or r for r in responses):
            logger.success(f"Warmed up with {multiplier=}")
        else:
            logger.warning(f"Stopping at {multiplier=}")
            break

    # One final prompt to make sure large context didn't crash it.
    assert await prompt_one(chute.name, base_url=base_url, api_key=api_key)


def set_default_cache_dirs(download_path):
    for key in [
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "FLASHINFER_WORKSPACE_BASE",
        "XFORMERS_CACHE_DIR",
        "DG_JIT_CACHE_DIR",
        "SGL_DG_CACHE_DIR",
        "SGLANG_DG_CACHE_DIR",
        "VLLM_CACHE_ROOT",
    ]:
        if not os.getenv(key):
            os.environ[key] = os.path.join(download_path, f"_{key.lower()}")


def set_nccl_flags(gpu_count, model_name):
    if gpu_count > 1 and re.search(
        "h[12]0|b[23]00|5090|l40s|6000 ada|a100|h800|pro 6000|sxm", model_name, re.I
    ):
        for key in ["NCCL_P2P_DISABLE", "NCCL_IB_DISABLE", "NCCL_NET_GDR_LEVEL"]:
            if key in os.environ:
                del os.environ[key]


async def monitor_engine(
    process,
    api_key: str,
    ready_event: asyncio.Event,
    port: int = 10101,
    check_interval: int = 5,
    timeout: float = 3.0,
    failure_threshold: int = 3,
    model_name: str = "Engine",
):
    """
    Monitor the engine process and HTTP endpoint.
    """
    consecutive_failures = 0
    async with aiohttp.ClientSession() as session:
        while True:
            if process.poll() is not None:
                logger.error(f"{model_name} subprocess died with exit code {process.returncode}")
                sys.exit(137)
            if ready_event.is_set():
                try:
                    async with session.get(
                        f"http://127.0.0.1:{port}/v1/models",
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as resp:
                        if resp.status != 200:
                            consecutive_failures += 1
                        else:
                            consecutive_failures = 0
                except Exception:
                    consecutive_failures += 1
                if consecutive_failures >= failure_threshold:
                    logger.error(f"{model_name} server is unresponsive.")
                    sys.exit(137)
            await asyncio.sleep(check_interval)
