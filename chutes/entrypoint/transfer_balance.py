"""
Transfer balance to another user.
"""

import asyncio
import os
import aiohttp
from loguru import logger
from typing import Optional
import typer
from chutes.config import get_config
from chutes.util.auth import sign_request


def transfer_balance(
    user: str = typer.Argument(..., help="Target user_id (UUID) or username"),
    amount: Optional[float] = typer.Option(
        None, help="Amount to transfer (default: entire balance)"
    ),
    config_path: str = typer.Option(
        None, help="Custom path to the chutes config (credentials, API URL, etc.)"
    ),
):
    async def _transfer():
        if config_path:
            os.environ["CHUTES_CONFIG_PATH"] = config_path
        config = get_config()
        payload = {"user_id": user}
        if amount is not None:
            payload["amount"] = amount
        headers, payload_string = sign_request(payload=payload)
        async with aiohttp.ClientSession(base_url=config.generic.api_base_url) as session:
            async with session.post(
                "/users/balance_transfer",
                data=payload_string,
                headers=headers,
            ) as response:
                data = await response.json()
                if response.status == 200:
                    logger.success(
                        f"Transferred ${data['transferred']:.6f} to {user}. "
                        f"Your new balance: ${data['from_balance']:.6f}"
                    )
                else:
                    logger.error(data.get("detail", data))

    asyncio.run(_transfer())
