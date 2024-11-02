import os
import re
import sys
import argparse
import importlib
import importlib.util
from loguru import logger
from typing import List, Dict, Any, Tuple

CHUTE_REF_RE = re.compile(r"^[a-z0-9][a-z0-9_]*:[a-z][a-z0-9_]+$", re.I)


def parse_args(args: List[Any], args_config: Dict[str, Any]):
    """
    Parse the CLI args (or manual dict) to run the chute.
    """
    parser = argparse.ArgumentParser()
    for arg, kwargs in args_config.items():
        parser.add_argument(arg, **kwargs)
    return parser.parse_args(args)


def load_chute(
    log_prefix: str, args: List[Any], args_config: Dict[str, Any]
) -> Tuple[Any, Any]:
    """
    Load a chute from the chute ref string via dynamic imports and such.
    """
    # The first arg is always the module:app variable reference, similar to uvicorn foo:app [other args]
    if not args:
        logger.error(f"usage: {log_prefix} [module_name:chute_name] [args]")
        sys.exit(1)
    chute_ref_str = args.pop(0)
    if not CHUTE_REF_RE.match(chute_ref_str):
        logger.error(
            "Invalid module name '{chute_ref_str}', usage: {log_prefix} {module_name:chute_name} [args]"
        )
        sys.exit(1)
    args = parse_args(args, args_config)

    # Config path updates.
    if args.config_path:
        os.environ["PARACHUTES_CONFIG_PATH"] = args.config_path

    # Debug logging?
    if not args.debug:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    from chutes.chute import Chute, ChutePack

    # Load the module.
    sys.path.append(os.getcwd())
    module_name, chute_name = chute_ref_str.split(":")
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, os.getcwd() + f"/{module_name}.py"
        )
        if spec is None:
            raise ImportError(f"Cannot find module {module_name} in {os.getcwd()}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except ImportError as exc:
        logger.error(f"Unable to import module '{module_name}': {exc}")
        sys.exit(1)

    # Get the Chute reference (FastAPI server).
    try:
        chute = getattr(module, chute_name)
        if not isinstance(chute, (Chute, ChutePack)):
            logger.error(
                f"'{chute_name}' in module '{module_name}' is not of type Chute or ChutePack"
            )
            sys.exit(1)
    except AttributeError:
        logger.error(f"Unable to find chute '{chute_name}' in module '{module_name}'")
        sys.exit(1)

    return chute, args
