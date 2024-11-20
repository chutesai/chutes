"""
Run a chute, automatically handling encryption/decryption via GraVal.
"""

import asyncio
import sys
from loguru import logger
import typer
import pybase64 as base64
import orjson as json
from uvicorn import Config, Server
from fastapi import Request, status
from fastapi.responses import ORJSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from graval.miner import Miner
from chutes.entrypoint._shared import load_chute
from chutes.chute import ChutePack
from chutes.util.context import is_local

MINER = Miner()


class GraValMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):
        """
        Transparently handle decryption from validator and encryption back to validator.
        """
        is_encrypted = request.headers.get("X-Chutes-Encrypted", "false").lower() == "true"
        request.state.decrypted = None
        if is_encrypted and request.method in ("POST", "PUT", "PATCH"):
            body_bytes = await request.body()
            encrypted_body = json.loads(body_bytes)
            required_fields = {"ciphertext", "iv", "length", "device_id", "seed"}
            decrypted_body = {}
            for key in encrypted_body:
                if not all(field in encrypted_body[key] for field in required_fields):
                    logger.error(
                        f"Missing encryption fields: {required_fields - set(encrypted_body[key])}"
                    )
                    return ORJSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={
                            "detail": "Missing one or more required fields for encrypted payloads!"
                        },
                    )
                if encrypted_body[key]["seed"] != MINER._seed:
                    logger.error(
                        f"Expecting seed: {MINER._seed}, received {encrypted_body[key]['seed']}"
                    )
                    return ORJSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"detail": "Provided seed does not match initialization seed!"},
                    )

                try:
                    # Decrypt the request body.
                    ciphertext = base64.b64decode(encrypted_body[key]["ciphertext"].encode())
                    iv = bytes.fromhex(encrypted_body[key]["iv"])
                    decrypted = MINER.decrypt(
                        ciphertext,
                        iv,
                        encrypted_body[key]["length"],
                        encrypted_body[key]["device_id"],
                    )
                    assert decrypted, "Decryption failed!"
                    decrypted_body[key] = decrypted
                except Exception as exc:
                    return ORJSONResponse(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        content={"detail": f"Decryption failed: {exc}"},
                    )
            request.state.decrypted = decrypted_body
        elif request.method in ("POST", "PUT", "PATCH"):
            request.state.decrypted = await request.json()

        return await call_next(request)


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
    graval_seed: int | None = typer.Option(None, help="graval seed for encryption/decryption"),
    uds: str | None = typer.Option(None, help="unix domain socket path"),
    debug: bool = typer.Option(False, help="enable debug logging"),
):
    """
    Run the chute (uvicorn server).
    """

    async def _run_chute():
        # How to get the chute ref string?
        _, chute = load_chute(chute_ref_str=chute_ref_str, config_path=config_path, debug=debug)

        if is_local():
            logger.error("Cannot run chutes in local context!")
            sys.exit(1)

        # Run the server.
        chute = chute.chute if isinstance(chute, ChutePack) else chute

        # GraVal enabled?
        if graval_seed is not None:
            logger.info(f"Initializing graval with {graval_seed=}")
            MINER.initialize(graval_seed)
            MINER._seed = graval_seed
            chute.add_middleware(GraValMiddleware)

        await chute.initialize()
        config = Config(app=chute, host=host, port=port, uds=uds)
        server = Server(config)
        await server.serve()

    asyncio.run(_run_chute())
