import aiohttp
import sys
from copy import deepcopy
from loguru import logger
from chutes.config import API_BASE_URL
from chutes.entrypoint._shared import load_chute
from chutes.util.auth import sign_request


CLI_ARGS = {
    "--config-path": {
        "type": str,
        "default": None,
        "help": "custom path to the parachutes config (credentials, API URL, etc.)",
    },
    "--debug": {
        "action": "store_true",
        "help": "enable debug logging",
    },
    "--public": {
        "action": "store_true",
        "help": "mark an image as public/available to anyone",
    },
}


async def deploy(chute, public=False):
    """
    Perform the actual chute deployment.
    """
    request_body = {
        "name": chute.name,
        "image": chute.image if isinstance(chute.image, str) else chute.image.uid,
        "public": public,
        "standard_template": chute.standard_template,
        "node_selector": chute.node_selector.dict(),
        "cords": [
            {
                "method": cord._method,
                "path": cord.path,
                "stream": cord._stream,
                "function": cord._func.__name__,
            }
            for cord in chute._cords
        ],
    }
    headers, request_string = sign_request(request_body)
    async with aiohttp.ClientSession(base_url=API_BASE_URL) as session:
        async with session.post(
            "/chutes/",
            data=request_string,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=None),
        ) as response:
            if response.status == 409:
                logger.error(f"Chute with name={chute.name} already exists!")
            elif response.status == 401:
                logger.error("Authorization error, please check your credentials.")
            elif response.status != 200:
                logger.error(
                    f"Unexpected error deploying chute: {await response.text()}"
                )
            else:
                logger.success(
                    f"Successfully deployed chute {chute.name}, invocation will be available soon"
                )


async def image_available(image, public):
    """
    Check if an image exists and is built/published in the registry.
    """
    image_id = image if isinstance(image, str) else image.uid
    logger.debug(f"Checking if {image_id=} is available...")
    headers, _ = sign_request(purpose="images")
    async with aiohttp.ClientSession(base_url=API_BASE_URL) as session:
        async with session.get(
            f"/images/{image_id}",
            headers=headers,
        ) as response:
            if response.status == 200:
                data = await response.json()
                if data.get("status") == "built and pushed":
                    if public and not data.get("public"):
                        logger.error(
                            "Unable to create public chutes from non-public images"
                        )
                        return False
                    return True
    return False


async def deploy_chute(input_args):
    """
    Deploy a chute to the platform.
    """
    chute, args = load_chute("chutes deploy", deepcopy(input_args), CLI_ARGS)

    from chutes.chute import ChutePack

    # Get the image reference from the chute.
    chute = chute.chute if isinstance(chute, ChutePack) else chute

    # Ensure the image is ready to be used.
    if not await image_available(chute.image, args.public):
        logger.error(f"Image '{chute.image}' is not available to be used (yet)!")
        sys.exit(1)

    # Deploy!
    return await deploy(chute, args.public)
