import aiohttp
import re
import os
import sys
import shutil
import importlib
import tempfile
import pybase64 as base64
import pickle
import orjson as json
from io import BytesIO
from contextlib import contextmanager
from copy import deepcopy
from loguru import logger
from chutes.config import API_BASE_URL
from chutes.image.directive.add import ADD
from chutes.image.directive.generic_run import RUN
from chutes.entrypoint._shared import load_chute
from chutes.util.auth import sign_request


CLI_ARGS = {
    "--config-path": {
        "type": str,
        "default": None,
        "help": "custom path to the parachutes config (credentials, API URL, etc.)",
    },
    "--local": {
        "action": "store_true",
        "help": "build the image locally, useful for testing/debugging",
    },
    "--debug": {
        "action": "store_true",
        "help": "enable debug logging",
    },
    "--include-cwd": {
        "action": "store_true",
        "help": "include the entire current directory in build context, recursively",
    },
    "--wait": {
        "action": "store_true",
        "help": "wait for image to be built",
    },
    "--public": {
        "action": "store_true",
        "help": "mark an image as public/available to anyone",
    },
}


@contextmanager
def temporary_build_directory(image):
    """
    Helper to copy the build context files to a build directory.
    """
    # Confirm the context files with the user.
    all_input_files = []
    for directive in image._directives:
        all_input_files += directive._build_context

    samples = all_input_files[:10]
    logger.info(
        f"Found {len(all_input_files)} files to include in build context -- \033[1m\033[4mthese will be uploaded for remote builds!\033[0m"
    )
    for path in samples:
        logger.info(f" {path}")
    if len(samples) != len(all_input_files):
        show_all = input(
            f"\033[93mShowing {len(samples)} of {len(all_input_files)}, would you like to see the rest? (y/n) \033[0m"
        )
        if show_all.lower() == "y":
            for path in all_input_files[:10]:
                logger.info(f" {path}")
    confirm = input("\033[1m\033[4mConfirm submitting build context? (y/n) \033[0m")
    if confirm.lower().strip() != "y":
        logger.error("Aborting!")
        sys.exit(1)

    # Copy all of the context files over to a temp dir (to use for local building or a zip file for remote).
    _clean_path = lambda in_: in_[len(os.getcwd()) + 1 :]  # noqa: E731
    with tempfile.TemporaryDirectory() as tempdir:
        for path in all_input_files:
            temp_path = os.path.join(tempdir, _clean_path(path))
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            logger.debug(f"Copying {path} to {temp_path}")
            shutil.copy(path, temp_path)
        yield tempdir


def build_local(image):
    """
    Build an image locally, directly with docker (for testing purposes).
    """
    with temporary_build_directory(image) as build_directory:
        dockerfile_path = os.path.join(build_directory, "Dockerfile")
        with open(dockerfile_path, "w") as outfile:
            outfile.write(str(image))
        logger.info(f"Starting build of {dockerfile_path}...")
        os.chdir(build_directory)
        os.execv(
            "/usr/bin/docker",
            ["/usr/bin/docker", "build", "-t", f"{image.name}:{image.tag}", "."],
        )


async def build_remote(image, wait=None, public=False):
    """
    Build an image remotely, that is, package up the build context and ship it
    off to the parachutes API to have it built.
    """
    with temporary_build_directory(image) as build_directory:
        logger.info(f"Packaging up the build directory to upload: {build_directory}")
        output_path = shutil.make_archive(
            os.path.join(build_directory, "chute"), "zip", build_directory
        )
        logger.info(f"Created the build package: {output_path}, uploading...")

        form_data = aiohttp.FormData()
        form_data.add_field("name", image.name)
        form_data.add_field("tag", image.tag)
        form_data.add_field("dockerfile", str(image))
        form_data.add_field("public", str(public))
        form_data.add_field("wait", str(wait))
        form_data.add_field("image", base64.b64encode(pickle.dumps(image)).decode())
        with open(os.path.join(build_directory, "chute.zip"), "rb") as infile:
            form_data.add_field(
                "build_context",
                BytesIO(infile.read()),
                filename="chute.zip",
                content_type="application/zip",
            )

        class FakeStreamWriter:
            def __init__(self):
                self.output = BytesIO()

            async def write(self, chunk):
                self.output.write(chunk)

            async def drain(self):
                pass

            async def write_eof(self):
                pass

        # Get the payload and write it to the custom writer
        payload = form_data()
        writer = FakeStreamWriter()
        await payload.write(writer)

        # Retrieve the raw bytes of the request body
        raw_data = writer.output.getvalue()

        async with aiohttp.ClientSession(base_url=API_BASE_URL) as session:
            headers, payload_string = sign_request(payload=raw_data)
            headers["Content-Type"] = payload.content_type
            headers["Content-Length"] = str(len(raw_data))
            async with session.post(
                "/images/",
                data=raw_data,  # form_data, #payload_string,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=None),
            ) as response:

                # When the client waits for the image, we just stream the logs.
                if wait:
                    async for data_enc in response.content:
                        data = data_enc.decode()
                        if data and data.strip() and "data: {" in data:
                            data = json.loads(data[6:])
                            log_method = (
                                logger.info
                                if data["log_type"] == "stdout"
                                else logger.warning
                            )
                            log_method(data["log"].strip())
                        elif data.startswith("DONE"):
                            break
                    return
                if response.status == 409:
                    logger.error(
                        f"Image with name={image.name} and tag={image.tag} already exists!"
                    )
                elif response.status == 401:
                    logger.error("Authorization error, please check your credentials.")
                elif response.status != 202:
                    logger.error(
                        f"Unexpected error uploading image data: {await response.text()}"
                    )
                else:
                    data = await response.json()
                    logger.info(
                        f"Uploaded image package: image_id={data['image_id']}, build will run async"
                    )


async def image_exists(image):
    """
    Check if an image already exists.
    """
    logger.debug(f"Checking if image {image.name}:{image.tag} exists...")
    headers, _ = sign_request(purpose="images")
    async with aiohttp.ClientSession(base_url=API_BASE_URL) as session:
        async with session.get(
            f"/images/{image.uid}",
            headers=headers,
        ) as response:
            if response.status == 200:
                return True
            elif response.status == 404:
                return False
            raise Exception(await response.text())
    return False


async def build_image(input_args):
    """
    Build an image for the parachutes platform.
    """
    chute, args = load_chute("chutes build", deepcopy(input_args), CLI_ARGS)

    from chutes.chute import ChutePack

    # Get the image reference from the chute.
    chute = chute.chute if isinstance(chute, ChutePack) else chute
    image = chute.image

    # Pre-built?
    if isinstance(image, str):
        logger.error(
            f"You appear to be using a pre-defined/standard image '{image}', no need to build anything!"
        )
        sys.exit(1)

    # Check if the image is already built.
    if await image_exists(image):
        logger.error(
            f"Image with name={image.name} and tag={image.tag} already exists!"
        )
        sys.exit(1)

    # Always tack on the final directives, which include installing chutes and adding project files.
    image._directives.append(RUN("pip install chutes --upgrade"))
    current_directory = os.getcwd()
    if args.include_cwd:
        image._directives.append(ADD(source=".", dest="/app"))
    else:
        module_name, chute_name = input_args[0].split(":")
        module = importlib.import_module(module_name)
        module_path = os.path.abspath(module.__file__)
        if not module_path.startswith(current_directory):
            logger.error(
                f"You must run the build command from the directory containing your target chute module: {module.__file__} [{current_directory=}]"
            )
            sys.exit(1)
        _clean_path = lambda in_: in_[len(current_directory) + 1 :]  # noqa: E731
        image._directives.append(
            ADD(
                source=_clean_path(module.__file__),
                dest=f"/app/{_clean_path(module.__file__)}",
            )
        )
        imported_files = [
            os.path.abspath(module.__file__)
            for module in sys.modules.values()
            if hasattr(module, "__file__") and module.__file__
        ]
        imported_files = [
            f
            for f in imported_files
            if f.startswith(current_directory)
            and not re.search(r"(site|dist)-packages", f)
            and f != os.path.abspath(module.__file__)
        ]
        for path in imported_files:
            image._directives.append(
                ADD(
                    source=_clean_path(path),
                    dest=f"/app/{_clean_path(path)}",
                )
            )
    logger.debug(f"Generated Dockerfile:\n{str(image)}")

    # Building locally?
    if args.local:
        return build_local(image)

    # Package up the context and ship it off for building.
    return await build_remote(image, wait=args.wait, public=args.public)
