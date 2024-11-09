import os
import sys
import json
import aiohttp
import typer
from rich import print_json
from loguru import logger
from chutes.util.auth import sign_request
from chutes.config import API_BASE_URL

app = typer.Typer()


@app.command()
def create_api_key(
    name: str = typer.Argument(..., help="Name to assign to the API key"),
    config_path: str = typer.Option(
        None, help="Custom path to the parachutes config (credentials, API URL, etc.)"
    ),
    admin: bool = typer.Option(False, help="Allow any action for this API key"),
    images: bool = typer.Option(False, help="Allow full access to images"),
    chutes: bool = typer.Option(False, help="Allow full access to chutes"),
    image_ids: list[str] = typer.Option(
        None, help="Allow access to one or more specific images"
    ),
    chute_ids: list[str] = typer.Option(
        None, help="Allow access to one or more specific chutes"
    ),
    action: str = typer.Option(
        None,
        help="Specify the verb to apply to all scopes",
        prompt=True,
        case_sensitive=False,
        show_choices=True,
        choices=["read", "write", "delete", "invoke"],
    ),
    json_input: str = typer.Option(
        None, help="Provide a raw scopes document as JSON, for more advanced usage"
    ),
):
    """
    Create a new API key as a user
    """
    if config_path:
        os.environ["PARACHUTES_CONFIG_PATH"] = config_path

    # Build our request payload with nested scopes.
    payload = {
        "name": name,
        "admin": admin,
    }
    if not admin:
        payload["scopes"] = []
        if json_input:
            try:
                payload["scopes"] = json.loads(json_input)["scopes"]
            except json.JSONDecodeError:
                logger.error("Invalid scopes JSON provided!")
                sys.exit(1)
        else:
            for object_type, ids in (
                ("images", image_ids),
                ("chutes", chute_ids),
            ):
                if (object_type == "images" and images) or (
                    object_type == "chutes" and chutes
                ):
                    payload["scopes"].append(
                        {"object_type": object_type, "action": action}
                    )
                elif ids:
                    for _id in ids:
                        payload["scopes"].append(
                            {
                                "object_type": object_type,
                                "object_id": _id,
                                "action": action,
                            }
                        )

    # Sign & send request

    headers, payload_string = sign_request(payload)

    async def send_request():
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
                    error_message = await response.text()
                    logger.error(f"Failed to create API key: {error_message}")

    typer.run(send_request)


if __name__ == "__main__":
    app()
