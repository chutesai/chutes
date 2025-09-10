"""
Warm up a chute.
"""

import os
import asyncio
import aiohttp
import orjson as json
from loguru import logger
import typer
from chutes.config import get_config
from chutes.entrypoint._shared import load_chute
from chutes.util.auth import sign_request


def warmup_chute(
    chute_id_or_ref_str: str = typer.Argument(
        ...,
        help="The chute file to warm up, format filename:chutevarname",
    ),
    config_path: str = typer.Option(
        None, help="Custom path to the chutes config (credentials, API URL, etc.)"
    ),
    debug: bool = typer.Option(False, help="enable debug logging"),
):
    async def warmup():
        """
        Do the warmup.
        """
        nonlocal chute_id_or_ref_str, config_path, debug
        chute_name = chute_id_or_ref_str
        if ":" in chute_id_or_ref_str and os.path.exists(chute_id_or_ref_str.split(":")[0] + ".py"):
            from chutes.chute.base import Chute

            _, chute = load_chute(chute_id_or_ref_str, config_path=config_path, debug=debug)
            chute_name = chute.name if isinstance(chute, Chute) else chute.chute.name
        config = get_config()
        if config_path:
            os.environ["CHUTES_CONFIG_PATH"] = config_path
        headers, _ = sign_request(purpose="chutes")
        async with aiohttp.ClientSession(base_url=config.generic.api_base_url) as session:
            async with session.get(
                f"/chutes/warmup/{chute_name}",
                headers=headers,
            ) as response:
                if response.status == 200:
                    async for raw_chunk in response.content:
                        if raw_chunk.startswith(b"data:"):
                            chunk = json.loads(raw_chunk[5:])
                            if chunk["status"] == "hot":
                                logger.success(chunk["log"])
                            else:
                                logger.warning(f"Status: {chunk['status']} -- {chunk['log']}")
                else:
                    logger.error(await response.text())

    return asyncio.run(warmup())
