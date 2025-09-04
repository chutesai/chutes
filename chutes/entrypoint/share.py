"""
Share a chute with another user.
"""

import os
import asyncio
import aiohttp
from loguru import logger
import typer
from chutes.config import get_config
from chutes.util.auth import sign_request


def share_chute(
    chute_id: str = typer.Option(..., help="the chute UUID (or name) to share"),
    user_id: str = typer.Option(..., help="the user UUID (or name) to share with"),
    config_path: str = typer.Option(
        None, help="Custom path to the chutes config (credentials, API URL, etc.)"
    ),
    remove: bool = typer.Option(False, help="unshare/delete share"),
):
    async def _share_chute():
        """
        Share (or unshare) the chute.
        """
        nonlocal chute_id, user_id, config_path, remove
        config = get_config()
        if config_path:
            os.environ["CHUTES_CONFIG_PATH"] = config_path
        headers, payload_string = sign_request(
            payload={"chute_id_or_name": chute_id, "user_id_or_name": user_id}
        )
        endpoint = "share" if not remove else "unshare"
        async with aiohttp.ClientSession(base_url=config.generic.api_base_url) as session:
            async with session.post(
                f"/chutes/{endpoint}",
                data=payload_string,
                headers=headers,
            ) as response:
                if response.status == 200:
                    logger.success((await response.json())["status"])
                else:
                    logger.error(await response.json())

    return asyncio.run(_share_chute())
