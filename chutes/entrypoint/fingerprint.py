"""
Reset a fingerprint using a coldkey or hotkey already associated with the user.
"""

import os
import asyncio
import aiohttp
import time
from loguru import logger
import typer
import orjson as json
from substrateinterface import Keypair
from chutes.config import get_config
from chutes.constants import (
    HOTKEY_HEADER,
    SIGNATURE_HEADER,
    NONCE_HEADER,
)


def change_fingerprint(
    config_path: str = typer.Option(
        None, help="Custom path to the chutes config (credentials, API URL, etc.)"
    ),
    hotkey_path: str = typer.Option(
        None,
        help="Path to the hotkey file associated with your account (used for generating signature)",
    ),
):
    fingerprint = typer.prompt("Enter new fingerprint", hide_input=True)

    async def _change_fingerprint():
        """
        Change the fingerprint on the account.
        """
        nonlocal config_path, hotkey_path, fingerprint
        config = get_config()
        if config_path:
            os.environ["CHUTES_CONFIG_PATH"] = config_path

        # Create the signature.
        with open(hotkey_path, "r") as infile:
            hotkey_data = json.loads(infile.read())
        keypair = Keypair.create_from_seed(seed_hex=hotkey_data["secretSeed"])
        nonce = str(int(time.time()))
        ss58 = hotkey_data["ss58Address"]
        signature_string = f"{ss58}:{fingerprint}:{nonce}"
        signature = keypair.sign(signature_string.encode()).hex()

        # Send it.
        async with aiohttp.ClientSession(base_url=config.generic.api_base_url) as session:
            async with session.post(
                "/users/change_fingerprint",
                json={"fingerprint": fingerprint},
                headers={
                    HOTKEY_HEADER: ss58,
                    SIGNATURE_HEADER: signature,
                    NONCE_HEADER: nonce,
                },
            ) as response:
                if response.status == 200:
                    logger.success("Fingerprint updated!")
                else:
                    logger.error(f"Error updating fingerprint: {await response.text()}")

    return asyncio.run(_change_fingerprint())
