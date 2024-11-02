"""
Generate a report for failed/bad invocation.
"""

import os
import sys
import aiohttp
from loguru import logger
from chutes.entrypoint._shared import parse_args

CLI_ARGS = {
    "--config-path": {
        "type": str,
        "default": None,
        "help": "custom path to the parachutes config (credentials, API URL, etc.)",
    },
    "--invocation-id": {
        "type": str,
        "required": True,
        "help": "invocation ID to report",
    },
    "--reason": {
        "type": str,
        "default": None,
        "help": "explanation/reason for the report",
    },
}


async def report_invocation(input_args):
    """
    Report an invocation.
    """
    args = parse_args(input_args, CLI_ARGS)
    if args.config_path:
        os.environ["PARACHUTES_CONFIG_PATH"] = args.config_path

    from chutes.util.auth import sign_request
    from chutes.config import API_BASE_URL

    # Ensure we have a reason.
    if not args.reason:
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
    async with aiohttp.ClientSession(base_url=API_BASE_URL) as session:
        async with session.post(
            f"/invocations/{args.invocation_id}/report",
            data=payload_string,
            headers=headers,
        ) as response:
            if response.status == 200:
                logger.success((await response.json())["status"])
            else:
                logger.error(await response.json())
