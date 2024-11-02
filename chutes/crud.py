"""
Basic endpoint access stuff.
"""

import aiohttp
import json
from rich import print_json
from loguru import logger
from functools import partial
from chutes.config import API_BASE_URL
from chutes.util.auth import sign_request


async def list_objects(
    object_type: str, name: str = None, limit: int = 25, page: int = 0
):
    """
    List objects of a particular type, paginated.
    """
    headers, _ = sign_request(purpose=object_type)
    async with aiohttp.ClientSession(base_url=API_BASE_URL) as session:
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


async def get_object(object_type: str, name_or_id: str):
    """
    Get an object by ID (or name).
    """
    headers, _ = sign_request(purpose=object_type)
    async with aiohttp.ClientSession(base_url=API_BASE_URL) as session:
        async with session.get(
            f"/{object_type}/{name_or_id}",
            headers=headers,
        ) as resp:
            if resp.status != 200:
                logger.error(
                    f"Failed to get {object_type}/{name_or_id}: {await resp.text()}"
                )
                return
            data = await resp.json()
            singular = object_type.rstrip("s")
            id_field = f"{singular}_id"
            logger.info(f"{singular} {data[id_field]}:")
            print_json(json.dumps(data))


async def delete_object(object_type: str, name_or_id: str):
    """
    Delete an object by ID (or name).
    """
    confirm = input(
        f"Are you sure you want to delete {object_type}/{name_or_id}?  This action is irreversable. (y/n): "
    )
    if confirm.lower() != "y":
        return
    headers, _ = sign_request(purpose=object_type)
    async with aiohttp.ClientSession(base_url=API_BASE_URL) as session:
        async with session.delete(
            f"/{object_type}/{name_or_id}",
            headers=headers,
        ) as resp:
            if resp.status != 200:
                logger.error(
                    f"Failed to delete {object_type}/{name_or_id}: {await resp.text()}"
                )
                return
            data = await resp.json()
            singular = object_type.rstrip("s")
            id_field = f"{singular}_id"
            logger.success(f"Successfully deleted {singular} {data[id_field]}")


list_images = partial(list_objects, "images")
list_chutes = partial(list_objects, "chutes")
get_image = partial(get_object, "images")
get_chute = partial(get_object, "chutes")
delete_image = partial(delete_object, "images")
delete_chute = partial(delete_object, "chutes")
