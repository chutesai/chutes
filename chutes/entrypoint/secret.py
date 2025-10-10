"""
Create a secret for a chute.
"""

import os
import asyncio
import aiohttp
from loguru import logger
import typer
from chutes.config import get_config
from chutes.util.auth import sign_request


def create_secret(
    purpose: str = typer.Option(
        ..., help="the chute UUID or name (other use-cases in the future?)"
    ),
    key: str = typer.Option(..., help="the secret key"),
    value: str = typer.Option(..., help="the secret value"),
    config_path: str = typer.Option(
        None, help="Custom path to the chutes config (credentials, API URL, etc.)"
    ),
):
    async def _create_secret():
        """
        Create a secret for a chute.
        """
        nonlocal purpose, key, value, config_path
        config = get_config()
        if config_path:
            os.environ["CHUTES_CONFIG_PATH"] = config_path
        headers, payload_string = sign_request(
            payload={"purpose": purpose, "key": key, "value": value}
        )
        async with aiohttp.ClientSession(base_url=config.generic.api_base_url) as session:
            async with session.post(
                "/secrets/",
                data=payload_string,
                headers=headers,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.success(
                        f"Successfully created secret_id={data['secret_id']} for {purpose=}"
                    )
                else:
                    logger.error(await response.json())

    return asyncio.run(_create_secret())
