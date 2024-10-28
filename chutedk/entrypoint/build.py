import os
import sys
import importlib
from copy import deepcopy
from loguru import logger
from chutedk.image.directive.add import ADD
from chutedk.image.directive.generic_run import RUN
from chutedk.entrypoint._shared import load_chute


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
}


async def build_image(args):
    """
    Build an image for the parachutes platform.
    """
    chute, args = load_chute("chutedk build", deepcopy(args), CLI_ARGS)

    from chutedk.chute import Chute, ChutePack

    # Get the image reference from the chute.
    chute = chute.chute if isinstance(chute, ChutePack) else chute
    image = chute.image

    # Pre-built?
    if isinstance(image, str):
        logger.error(
            f"You appear to be using a pre-defined/standard image '{image}', no need to build anything!"
        )
        sys.exit(0)

    # XXX check if the image is already built.

    # Always tack on the final directives, which include installing chutedk and adding project files.
    image._directives.append(
        RUN("pip install git+https://github.com/jondurbin/chutedk.git")
    )
    current_directory = os.getcwd()
    if args.include_cwd:
        image._directives.append(ADD(source=".", destination="/app"))
    else:
        module_name, chute_name = args[0].split(":")
        module = importlib.import_module(module_name)
        if not module.__file__.startswith(current_directory):
            logger.error(
                f"You must run the build command from the directory containing your target chute module: {module.__file__}"
            )
            sys.exit(1)
        image._directives.append(
            ADD(
                source=module.__file__[len(current_directory) :],
                destination=f"/app/{module.__file__[len(current_directory):]}",
            )
        )
        imported_files = [
            module.__file__
            for module in sys.modules.values()
            if hasattr(module, "__file__")
            and module.__file__
            and module.__file__.startswith(current_directory)
            and not re.search(r"(site|dist)-packages", module.__file__)
        ]
        for path in imported_files:
            image._directives.append(
                ADD(
                    source=path[len(current_directory) :],
                    destination=f"/app/{path[len(current_directory):]}",
                )
            )
    print(image)
