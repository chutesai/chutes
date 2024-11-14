"""
Generate a report for failed/bad invocation.
"""

import os
import sys
import asyncio
import aiohttp
from loguru import logger
import typer
from chutes.config import get_config
from chutes.util.auth import sign_request


def report_invocation(
    invocation_id: str = typer.Option(..., help="invocation ID to report"),
    config_path: str = typer.Option(
        None, help="Custom path to the chutes config (credentials, API URL, etc.)"
    ),
    reason: str = typer.Option(None, help="explanation/reason for the report"),
):
    async def _report_invocation():
        """
        Report an invocation.
        """
        nonlocal invocation_id, config_path, reason
        config = get_config()
        if config_path:
            os.environ["CHUTES_CONFIG_PATH"] = config_path

        # Ensure we have a reason.
        if not reason:
            reason = input("Please describe the issue with the invocation: ")
            try:
                while True:
                    confirm = input("Submit report? (y/n): ")
                    if confirm.strip().lower() == "y":
                        break
                    reason = input(
                        "Please describe the issue with the invocation (or ctrl+c to quit): "
                    )
            except KeyboardInterrupt:
                sys.exit(0)

        # Send it.
        headers, payload_string = sign_request(payload={"reason": reason})
        async with aiohttp.ClientSession(base_url=config.generic.api_base_url) as session:
            async with session.post(
                f"/invocations/{invocation_id}/report",
                data=payload_string,
                headers=headers,
            ) as response:
                if response.status == 200:
                    logger.success((await response.json())["status"])
                else:
                    logger.error(await response.json())

    return asyncio.run(_report_invocation())
