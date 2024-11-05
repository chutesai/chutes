"""
Create a new API key.
"""

import os
import sys
import json
import aiohttp
from rich import print_json
from loguru import logger
from chutes.util.auth import sign_request
from chutes.entrypoint._shared import parse_args

CLI_ARGS = {
    "--config-path": {
        "type": str,
        "default": None,
        "help": "custom path to the parachutes config (credentials, API URL, etc.)",
    },
    "--name": {
        "type": str,
        "required": True,
        "help": "name to assign to the API key",
    },
    "--admin": {
        "action": "store_true",
        "help": "allow any action for this API key",
    },
    "--images": {
        "action": "store_true",
        "help": "allow full access to images",
    },
    "--chutes": {
        "action": "store_true",
        "help": "allow full access to chutes",
    },
    "--image-ids": {
        "type": str,
        "nargs": "+",
        "help": "allow access to one or more specific images",
    },
    "--chute-ids": {
        "type": str,
        "nargs": "+",
        "help": "allow access to one or more specific chutes",
    },
    "--action": {
        "type": str,
        "choices": ["read", "write", "delete", "invoke"],
        "help": "specify the verb to apply to all scopes",
    },
    "--json": {
        "type": str,
        "help": 'provide a raw scopes document as JSON, for more advanced usage, e.g. {"scopes": [{"object_type": "images", "action": "read"}, {"object_type": "chutes", "object_id": "00...00", "action": "invoke"}]}',
    },
}


async def create_api_key(input_args):
    """
    Create a new API key.
    """
    args = parse_args(input_args, CLI_ARGS)
    if args.config_path:
        os.environ["PARACHUTES_CONFIG_PATH"] = args.config_path

    from chutes.config import API_BASE_URL

    # Build our request payload with nested scopes.
    payload = {
        "name": args.name,
        "admin": args.admin,
    }
    if not args.admin:
        payload["scopes"] = []
        if args.json:
            try:
                payload["scopes"] = json.loads(args.json)["scopes"]
            except Exception:
                logger.error("Invalid scopes JSON provided!")
                sys.exit(1)
        else:
            for object_type, ids in (
                ("images", args.image_ids),
                ("chutes", args.chute_ids),
            ):
                if getattr(args, object_type):
                    payload["scopes"].append(
                        {"object_type": object_type, "action": args.action}
                    )
                elif ids:
                    for _id in ids:
                        payload["scopes"].append(
                            {
                                "object_type": object_type,
                                "object_id": _id,
                                "action": args.action,
                            }
                        )

    # Send it.
    headers, payload_string = sign_request(payload)
    async with aiohttp.ClientSession(base_url=API_BASE_URL) as session:
        async with session.post(
            "/api_keys/",
            data=payload_string,
            headers=headers,
        ) as response:
            if response.status == 200:
                data = await response.json()
                logger.success("API key created successfully")
                print_json(data=data)
                print(
                    f"\nTo use the key, add \"Authorization: Basic {data['secret_key']}\" to your headers!\n"
                )
            else:
                logger.error(await response.json())
