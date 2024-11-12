import asyncio
import sys
from loguru import logger
import typer
from uvicorn import Config, Server
from chutes.entrypoint._shared import load_chute
from chutes.chute import ChutePack
from chutes.util.context import is_local


# NOTE: Might want to change the name of this to 'start'.
# So `run` means an easy way to perform inference on a chute (pull the cord :P)
def run_chute(
    chute_ref_str: str = typer.Argument(
        ..., help="chute to run, in the form [module]:[app_name], similar to uvicorn"
    ),
    config_path: str = typer.Option(
        None, help="Custom path to the chutes config (credentials, API URL, etc.)"
    ),
    port: int | None = typer.Option(None, help="port to listen on"),
    host: str | None = typer.Option(None, help="host to bind to"),
    uds: str | None = typer.Option(None, help="unix domain socket path"),
    debug: bool = typer.Option(False, help="enable debug logging"),
):
    """
    Run the chute (uvicorn server).
    """

    async def _run_chute():
        # How to get the chute ref string?
        chute = load_chute(chute_ref_str=chute_ref_str, config_path=config_path, debug=debug)

        if is_local():
            logger.error("Cannot run chutes in local context!")
            sys.exit(1)

        # Run the server.
        chute = chute.chute if isinstance(chute, ChutePack) else chute
        await chute.initialize()
        config = Config(app=chute, host=host, port=port, uds=uds)
        server = Server(config)
        await server.serve()

    asyncio.run(_run_chute())
