import sys
from loguru import logger
from uvicorn import Config, Server
from chutes.entrypoint._shared import load_chute


CLI_ARGS = {
    "--config-path": {
        "type": str,
        "default": None,
        "help": "custom path to the parachutes config (credentials, API URL, etc.)",
    },
    "--port": {
        "type": int,
        "default": None,
        "help": "port to listen on",
    },
    "--host": {
        "type": str,
        "default": None,
        "help": "host to bind to",
    },
    "--uds": {
        "type": str,
        "default": None,
        "help": "unix domain socket path",
    },
    "--debug": {
        "action": "store_true",
        "help": "enable debug logging",
    },
}


async def run_chute(args):
    """
    Run the chute (uvicorn server).
    """
    chute, args = load_chute("chutes run", args, CLI_ARGS)

    from chutes.chute import ChutePack
    from chutes.util.context import is_local

    if is_local():
        logger.error("Cannot run chutes in local context!")
        sys.exit(1)

    # Run the server.
    chute = chute.chute if isinstance(chute, ChutePack) else chute
    await chute.initialize()
    config = Config(app=chute, host=args.host, port=args.port, uds=args.uds)
    server = Server(config)
    await server.serve()
