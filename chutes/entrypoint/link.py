"""
Link a validator or subnet owner hotkey to a user account, providing free access.
"""

import os
import asyncio
import aiohttp
from loguru import logger
import typer
import orjson as json
from enum import Enum
from substrateinterface import Keypair
from chutes.config import get_config
from chutes.util.auth import sign_request


class HotkeyType(str, Enum):
    VALIDATOR = "validator"
    SUBNET_OWNER = "subnet_owner"


def link_hotkey(
    config_path: str = typer.Option(
        None, help="Custom path to the chutes config (credentials, API URL, etc.)"
    ),
    hotkey_path: str = typer.Option(
        ...,
        help="Path to the validator/subnet owner hotkey file (used only for signature)",
    ),
    hotkey_type: HotkeyType = typer.Option(
        ...,
        help="Either 'validator' or 'subnet_owner'",
    ),
):
    async def _link_hotkey():
        """
        Link a hotkey, giving free + developer access.
        """
        nonlocal config_path, hotkey_path, hotkey_type
        config = get_config()
        if config_path:
            os.environ["CHUTES_CONFIG_PATH"] = config_path

        # Create the signature.
        with open(hotkey_path, "r") as infile:
            hotkey_data = json.loads(infile.read())
        keypair = Keypair.create_from_seed(seed_hex=hotkey_data["secretSeed"])
        signature_string = f"{hotkey_data['ss58Address']}:{config.auth.username}"
        signature = keypair.sign(signature_string.encode()).hex()

        # Send it.
        headers, _ = sign_request(purpose="link_account")
        async with aiohttp.ClientSession(base_url=config.generic.api_base_url) as session:
            async with session.get(
                f"/users/link_{hotkey_type.value}",
                params={
                    "hotkey": hotkey_data["ss58Address"],
                    "signature": signature,
                },
                headers=headers,
            ) as response:
                if response.status == 200:
                    logger.success(
                        f"Account is now linked with {hotkey_type.value} {hotkey_data['ss58Address']}"
                    )
                else:
                    logger.error(await response.json())

    return asyncio.run(_link_hotkey())
