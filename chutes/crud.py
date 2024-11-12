"""
Basic endpoint access stuff.
"""

import asyncio
import aiohttp
import json
from rich import print_json
from loguru import logger

import typer
from chutes.config import get_config
from chutes.util.auth import sign_request


chutes_app = typer.Typer(
    no_args_is_help=True,
    name="chutes",
    help="Manage chutes, e.g. list chutes, delete a chute, etc.",
)
images_app = typer.Typer(no_args_is_help=True, name="images", help="Manage images")
api_keys_app = typer.Typer(no_args_is_help=True, name="keys", help="Manage API keys")


async def _list_objects(
    object_type: str,
    name: str = None,
    limit: int = 25,
    page: int = 0,
):
    """
    List objects of a particular type, paginated.
    """
    config = get_config()
    headers, _ = sign_request(purpose=object_type)
    async with aiohttp.ClientSession(base_url=config.generic.api_base_url) as session:
        params = {
            key: value
            for key, value in {
                "name": name,
                "limit": str(limit),
                "page": str(page),
            }.items()
            if value is not None
        }
        async with session.get(
            f"/{object_type}/",
            headers=headers,
            params=params,
        ) as resp:
            if resp.status != 200:
                logger.error(f"Failed to list {object_type}: {await resp.text()}")
                return
            data = await resp.json()
            logger.info(
                f"Found {data['total']} matching {object_type}, displaying {len(data['items'])}"
            )
            for item in data["items"]:
                singular = object_type.rstrip("s")
                id_field = f"{singular}_id"
                logger.info(f"{singular} {item[id_field]}:")
                print_json(json.dumps(item))


async def _get_object(object_type: str, name_or_id: str):
    """
    Get an object by ID (or name).
    """
    config = get_config()
    headers, _ = sign_request(purpose=object_type)
    async with aiohttp.ClientSession(base_url=config.generic.api_base_url) as session:
        async with session.get(
            f"/{object_type}/{name_or_id}",
            headers=headers,
        ) as resp:
            if resp.status != 200:
                logger.error(f"Failed to get {object_type}/{name_or_id}: {await resp.text()}")
                return
            data = await resp.json()
            singular = object_type.rstrip("s")
            id_field = f"{singular}_id"
            logger.info(f"{singular} {data[id_field]}:")
            print_json(json.dumps(data))


async def _delete_object(object_type: str, name_or_id: str):
    """
    Delete an object by ID (or name).
    """
    config = get_config()
    confirm = input(
        f"Are you sure you want to delete {object_type}/{name_or_id}?  This action is irreversable. (y/n): "
    )
    if confirm.lower() != "y":
        return
    headers, _ = sign_request(purpose=object_type)
    async with aiohttp.ClientSession(base_url=config.generic.api_base_url) as session:
        async with session.delete(
            f"/{object_type}/{name_or_id}",
            headers=headers,
        ) as resp:
            if resp.status != 200:
                logger.error(f"Failed to delete {object_type}/{name_or_id}: {await resp.text()}")
                return
            data = await resp.json()
            singular = object_type.rstrip("s")
            id_field = f"{singular}_id"
            logger.success(f"Successfully deleted {singular} {data[id_field]}")


@chutes_app.command(name="list", help="List chutes")
def list_chutes(
    name: str | None = typer.Option(None, help="Name of chute to filter by"),
    limit: int = typer.Option(25, help="Number of chutes to display per page"),
    page: int = typer.Option(0, help="The page number to display"),
):
    return asyncio.run(_list_objects("chutes", name=name, limit=limit, page=page))


@chutes_app.command(name="get", help="Get a chute by name or ID")
def get_chute(name_or_id: str = typer.Argument(..., help="Name or ID of chute to get")):
    return asyncio.run(_get_object("chutes", name_or_id))


@chutes_app.command(name="delete", help="Delete a chute by name or ID")
def delete_chute(name_or_id: str = typer.Argument(..., help="Name or ID of chute to delete")):
    return asyncio.run(_delete_object("chutes", name_or_id))


@images_app.command(name="list", help="List images")
def list_images(
    name: str | None = typer.Option(None, help="Name of image to filter by"),
    limit: int = typer.Option(25, help="Number of images to display per page"),
    page: int = typer.Option(0, help="The page number to display"),
):
    return asyncio.run(_list_objects("images", name=name, limit=limit, page=page))


@images_app.command(name="get", help="Get an image by name or ID")
def get_image(name_or_id: str = typer.Argument(..., help="Name or ID of image to get")):
    return asyncio.run(_get_object("images", name_or_id))


@images_app.command(name="delete", help="Delete an image by name or ID")
def delete_image(name_or_id: str = typer.Argument(..., help="Name or ID of image to delete")):
    return asyncio.run(_delete_object("images", name_or_id))


@api_keys_app.command(name="list", help="List API keys")
def list_api_keys(
    name: str | None = typer.Option(None, help="Name of API key to filter by"),
    limit: int = typer.Option(25, help="Number of API keys to display per page"),
    page: int = typer.Option(0, help="The page number to display"),
):
    return asyncio.run(_list_objects("api_keys", name=name, limit=limit, page=page))


@api_keys_app.command(name="get", help="Get an API key by name or ID")
def get_api_key(name_or_id: str = typer.Argument(..., help="Name or ID of API key to get")):
    return asyncio.run(_get_object("api_keys", name_or_id))


@api_keys_app.command(name="delete", help="Delete an API key by name or ID")
def delete_api_key(name_or_id: str = typer.Argument(..., help="Name or ID of API key to delete")):
    return asyncio.run(_delete_object("api_keys", name_or_id))
