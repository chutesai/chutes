"""
Run a chute, automatically handling encryption/decryption via GraVal.
"""

import asyncio
import sys
from loguru import logger
import typer
import pybase64 as base64
import orjson as json
from typing import AsyncIterator
from uvicorn import Config, Server
from fastapi import Request, Response, status
from fastapi.responses import ORJSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from graval.miner import Miner
from chutes.entrypoint._shared import load_chute
from chutes.chute import ChutePack
from chutes.util.context import is_local

MINER = Miner()


class GraValMiddleware(BaseHTTPMiddleware):

    async def encrypt_chunk(self, chunk: bytes) -> str:
        """
        Helper method to encrypt a single chunk of data.
        """
        if not chunk or not chunk.decode().strip():
            return None
        ciphertext, iv, length = MINER.encrypt(chunk)
        cipher_payload = {
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "iv": iv.hex(),
            "length": length,
            "device_id": 0,
        }
        return json.dumps(cipher_payload).decode() + "\n\n"

    async def stream_encrypt(self, iterator: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        """
        Generator that encrypts each chunk of a stream.
        """
        async for chunk in iterator:
            if encrypted := await self.encrypt_chunk(chunk):
                yield encrypted

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Transparently handle decryption from validator and encryption back to validator.
        """
        is_encrypted = request.headers.get("X-Chutes-Encrypted", "false").lower() == "true"
        if request.method in ("POST", "PUT", "PATCH") and is_encrypted:
            encrypted_body = await request.json()
            required_fields = {"ciphertext", "iv", "length", "device_id", "seed"}
            if not all(field in encrypted_body for field in required_fields):
                return ORJSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "detail": "Missing one or more required fields for encrypted payloads"
                    },
                )
            if encrypted_body["seed"] != MINER._seed:
                return ORJSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": "Provided seed does not match initialization seed!"},
                )

            try:
                # Decrypt the request body.
                ciphertext = base64.b64decode(encrypted_body["ciphertext"].encode())
                iv = bytes.fromhex(encrypted_body["iv"])
                decrypted = MINER.decrypt(
                    ciphertext,
                    iv,
                    encrypted_body["length"],
                    encrypted_body["device_id"],
                )

                # Create a new request scope with modified body.
                async def receive():
                    if hasattr(request.scope, "_body"):
                        return {"type": "http.request", "body": request.scope._body}
                    request.scope._body = decrypted
                    return {"type": "http.request", "body": decrypted}

                # Override the receive method to return our decrypted data.
                request._receive = receive

            except Exception as exc:
                return ORJSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"detail": f"Decryption failed: {exc}"},
                )

        response = await call_next(request)

        # Encrypt the response(s) if the request was encrypted.
        if is_encrypted:
            if isinstance(response, StreamingResponse):
                return StreamingResponse(
                    self.stream_encrypt(response.body_iterator),
                    status_code=response.status_code,
                )
            else:
                body = b""
                async for chunk in response.body_iterator:
                    body += chunk
                ciphertext, iv, length = MINER.encrypt(body)
                return ORJSONResponse(
                    {
                        "ciphertext": base64.b64encode(ciphertext).decode(),
                        "iv": iv.hex(),
                        "length": length,
                        "device_id": 0,
                    }
                )

        return response


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
