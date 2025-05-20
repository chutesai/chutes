"""
Run a chute, automatically handling encryption/decryption via GraVal.
"""

import os
import asyncio
import sys
import time
import uuid
import hashlib
import inspect
import typer
import psutil
import base64
import aiohttp
import orjson as json
from loguru import logger
from typing import Optional, Any
from datetime import datetime
from functools import lru_cache
from pydantic import BaseModel
from ipaddress import ip_address
from uvicorn import Config, Server
from fastapi import FastAPI, Request, Response, status, HTTPException
from fastapi.responses import ORJSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from substrateinterface import Keypair, KeypairType
from chutes.entrypoint._shared import load_chute
from chutes.chute import ChutePack
from chutes.util.context import is_local
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding


def get_all_process_info():
    """
    Return running process info.
    """
    processes = {}
    for proc in psutil.process_iter(["pid", "name", "cmdline", "open_files", "create_time"]):
        try:
            info = proc.info
            info["open_files"] = [f.path for f in proc.open_files()]
            info["create_time"] = datetime.fromtimestamp(proc.create_time()).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            info["environ"] = dict(proc.environ())
            processes[str(proc.pid)] = info
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return Response(
        content=json.dumps(processes).decode(),
        media_type="application/json",
    )


def get_env_sig(request: Request):
    """
    Environment signature check.
    """
    import chutes.envcheck as envcheck

    return Response(
        content=envcheck.signature(request.state.decrypted["salt"]),
        media_type="text/plain",
    )


def get_env_dump(request: Request):
    """
    Base level environment check, running processes and things.
    """
    import chutes.envcheck as envcheck

    key = bytes.fromhex(request.state.decrypted["key"])
    return Response(
        content=envcheck.dump(key),
        media_type="text/plain",
    )


async def pong(request: Request) -> dict[str, Any]:
    """
    Echo incoming request as a liveness check.
    """
    if hasattr(request.state, "_encrypt"):
        return {"json": request.state._encrypt(json.dumps(request.state.decrypted))}
    return request.state.decrypted


async def get_token(request: Request) -> dict[str, Any]:
    """
    Fetch a token, useful in detecting proxies between the real deployment and API.
    """
    endpoint = request.state.decrypted.get(
        "endpoint", "https://api.chutes.ai/instances/token_check"
    )
    salt = request.state.decrypted.get("salt", 42)
    async with aiohttp.ClientSession(trust_env=True) as session:
        async with session.get(endpoint, params={"salt": salt}) as resp:
            if hasattr(request.state, "_encrypt"):
                return {"json": request.state._encrypt(await resp.text())}
            return await resp.json()


async def is_alive(request: Request):
    """
    Liveness probe endpoint for k8s.
    """
    return {"alive": True}


class Slurp(BaseModel):
    path: str
    start_byte: Optional[int] = 0
    end_byte: Optional[int] = None


@lru_cache(maxsize=1)
def miner():
    from graval import Miner

    return Miner()


class FSChallenge(BaseModel):
    filename: str
    length: int
    offset: int


class DevMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        """
        Dev/dummy dispatch.
        """
        args = await request.json() if request.method in ("POST", "PUT", "PATCH") else None
        request.state.serialized = False
        request.state.decrypted = args
        return await call_next(request)


class GraValMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, concurrency: int = 1):
        """
        Initialize a semaphore for concurrency control/limits.
        """
        super().__init__(app)
        self.concurrency = concurrency
        self.lock = asyncio.Lock()
        self.requests_in_flight = {}
        self.symmetric_key = None
        self.app = app

    async def _dispatch(self, request: Request, call_next):
        """
        Transparently handle decryption and verification.
        """
        if request.client.host == "127.0.0.1":
            return await call_next(request)

        # Internal endpoints.
        path = request.scope.get("path", "")
        if path.endswith(("/_alive", "/_metrics")):
            ip = ip_address(request.client.host)
            is_private = (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
            )
            if not is_private:
                return ORJSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "go away (internal)"},
                )
            else:
                return await call_next(request)

        # Verify the signature.
        miner_hotkey = request.headers.get("X-Chutes-Miner")
        validator_hotkey = request.headers.get("X-Chutes-Validator")
        nonce = request.headers.get("X-Chutes-Nonce")
        signature = request.headers.get("X-Chutes-Signature")
        if (
            any(not v for v in [miner_hotkey, validator_hotkey, nonce, signature])
            or validator_hotkey != miner()._validator_ss58
            or miner_hotkey != miner()._miner_ss58
            or int(time.time()) - int(nonce) >= 30
        ):
            logger.warning(f"Missing auth data: {request.headers}")
            return ORJSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "go away (missing)"},
            )
        body_bytes = await request.body() if request.method in ("POST", "PUT", "PATCH") else None
        payload_string = hashlib.sha256(body_bytes).hexdigest() if body_bytes else "chutes"
        signature_string = ":".join(
            [
                miner_hotkey,
                validator_hotkey,
                nonce,
                payload_string,
            ]
        )
        if not miner()._keypair.verify(signature_string, bytes.fromhex(signature)):
            return ORJSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "go away (sig)"},
            )

        # Decrypt the payload.
        if not self.symmetric_key and path != "/_exchange":
            logger.warning("Received a request but we need the symmetric key first!")
            return ORJSONResponse(
                status_code=status.HTTP_426_UPGRADE_REQUIRED,
                content={"detail": "Exchange a symmetric key via GraVal first."},
            )
        if path == "/_exchange":
            # Initial GraVal payload that contains the symmetric key, encrypted with GraVal.
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
                if encrypted_body[key]["seed"] != miner()._seed:
                    logger.error(
                        f"Expecting seed: {miner()._seed}, received {encrypted_body[key]['seed']}"
                    )
                    return ORJSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"detail": "Provided seed does not match initialization seed!"},
                    )

                try:
                    # Decrypt the request body.
                    ciphertext = base64.b64decode(encrypted_body[key]["ciphertext"].encode())
                    iv = bytes.fromhex(encrypted_body[key]["iv"])
                    decrypted = miner().decrypt(
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

            # Extract our symmetric key.
            secret = decrypted_body.get("symmetric_key")
            if not secret:
                return ORJSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": "Exchange request must contain symmetric key!"},
                )
            self.symmetric_key = bytes.fromhex(secret)
            return ORJSONResponse(
                status_code=status.HTTP_200_OK,
                content={"ok": True},
            )

        # Decrypt using the symmetric key we exchanged via GraVal.
        if request.method in ("POST", "PUT", "PATCH"):
            try:
                iv = bytes.fromhex(body_bytes[:32].decode())
                cipher = Cipher(
                    algorithms.AES(self.symmetric_key),
                    modes.CBC(iv),
                    backend=default_backend(),
                )
                unpadder = padding.PKCS7(128).unpadder()
                decryptor = cipher.decryptor()
                decrypted_data = (
                    decryptor.update(base64.b64decode(body_bytes[32:])) + decryptor.finalize()
                )
                unpadded_data = unpadder.update(decrypted_data) + unpadder.finalize()
                try:
                    request.state.decrypted = json.loads(unpadded_data)
                except Exception:
                    request.state.decrypted = json.loads(unpadded_data.rstrip(bytes(range(1, 17))))
                request.state.iv = iv
            except ValueError as exc:
                return ORJSONResponse(
                    status_code=status.HTTP_451_UNAVAILABLE_FOR_LEGAL_REASONS,
                    content={"detail": f"Decryption failed: {exc}"},
                )

            def _encrypt(plaintext: bytes):
                if isinstance(plaintext, str):
                    plaintext = plaintext.encode()
                padder = padding.PKCS7(128).padder()
                cipher = Cipher(
                    algorithms.AES(self.symmetric_key),
                    modes.CBC(iv),
                    backend=default_backend(),
                )
                padded_data = padder.update(plaintext) + padder.finalize()
                encryptor = cipher.encryptor()
                encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
                return base64.b64encode(encrypted_data).decode()

            request.state._encrypt = _encrypt

        return await call_next(request)

    async def dispatch(self, request: Request, call_next):
        """
        Rate-limiting wrapper around the actual dispatch function.
        """
        request.request_id = str(uuid.uuid4())
        request.state.serialized = request.headers.get("X-Chutes-Serialized") is not None

        # Pass regular, special paths through.
        if (
            request.scope.get("path", "").endswith(
                (
                    "/_fs_challenge",
                    "/_alive",
                    "/_metrics",
                    "/_ping",
                    "/_procs",
                    "/_slurp",
                    "/_device_challenge",
                    "/_devices",
                    "/_env_sig",
                    "/_env_dump",
                    "/_exchange",
                    "/_token",
                )
            )
            or request.client.host == "127.0.0.1"
        ):
            return await self._dispatch(request, call_next)

        # Decrypt encrypted paths, which could be one of the above as well.
        path = request.scope.get("path", "")
        if self.symmetric_key and path != "/_exchange":
            try:
                iv = bytes.fromhex(path[1:33])
                cipher = Cipher(
                    algorithms.AES(self.symmetric_key),
                    modes.CBC(iv),
                    backend=default_backend(),
                )
                unpadder = padding.PKCS7(128).unpadder()
                decryptor = cipher.decryptor()
                decrypted_data = decryptor.update(bytes.fromhex(path[33:])) + decryptor.finalize()
                actual_path = unpadder.update(decrypted_data) + unpadder.finalize()
                actual_path = actual_path.decode().rstrip("?")
                logger.info(f"Decrypted request path: {actual_path} from input path: {path}")
                request.scope["path"] = actual_path
            except ValueError:
                return ORJSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={"detail": f"Bad path: {path}"},
                )

        # Now pass the decrypted special paths through.
        if request.scope.get("path", "").endswith(
            (
                "/_fs_challenge",
                "/_alive",
                "/_metrics",
                "/_ping",
                "/_procs",
                "/_slurp",
                "/_device_challenge",
                "/_devices",
                "/_env_sig",
                "/_env_dump",
                "/_exchange",
                "/_token",
            )
        ):
            return await self._dispatch(request, call_next)

        # Concurrency control with timeouts in case it didn't get cleaned up properly.
        async with self.lock:
            now = time.time()
            if len(self.requests_in_flight) >= self.concurrency:
                purge_keys = []
                for key, val in self.requests_in_flight.items():
                    if now - val >= 600:
                        logger.warning(
                            f"Assuming this request is no longer in flight, killing: {key}"
                        )
                        purge_keys.append(key)
                if purge_keys:
                    for key in purge_keys:
                        self.requests_in_flight.pop(key, None)
                    self.requests_in_flight[request.request_id] = now
                else:
                    return ORJSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={
                            "error": "RateLimitExceeded",
                            "detail": f"Max concurrency exceeded: {self.concurrency}, try again later.",
                        },
                    )
            else:
                self.requests_in_flight[request.request_id] = now

        # Perform the actual request.
        response = None
        try:
            response = await self._dispatch(request, call_next)
            if hasattr(response, "body_iterator"):
                original_iterator = response.body_iterator

                async def wrapped_iterator():
                    try:
                        async for chunk in original_iterator:
                            yield chunk
                    except Exception as exc:
                        logger.warning(f"Unhandled exception in body iterator: {exc}")
                        self.requests_in_flight.pop(request.request_id, None)
                        raise
                    finally:
                        self.requests_in_flight.pop(request.request_id, None)

                response.body_iterator = wrapped_iterator()
                return response
            return response
        finally:
            if not response or not hasattr(response, "body_iterator"):
                self.requests_in_flight.pop(request.request_id, None)


# NOTE: Might want to change the name of this to 'start'.
# So `run` means an easy way to perform inference on a chute (pull the cord :P)
def run_chute(
    chute_ref_str: str = typer.Argument(
        ..., help="chute to run, in the form [module]:[app_name], similar to uvicorn"
    ),
    miner_ss58: str = typer.Option(None, help="miner hotkey ss58 address"),
    validator_ss58: str = typer.Option(None, help="validator hotkey ss58 address"),
    port: int | None = typer.Option(None, help="port to listen on"),
    host: str | None = typer.Option(None, help="host to bind to"),
    graval_seed: int | None = typer.Option(None, help="graval seed for encryption/decryption"),
    debug: bool = typer.Option(False, help="enable debug logging"),
    dev: bool = typer.Option(False, help="dev/local mode"),
):
    """
    Run the chute (uvicorn server).
    """

    async def _run_chute():
        chute_module, chute = load_chute(chute_ref_str=chute_ref_str, config_path=None, debug=debug)
        if is_local():
            logger.error("Cannot run chutes in local context!")
            sys.exit(1)

        # Run the server.
        chute = chute.chute if isinstance(chute, ChutePack) else chute

        # GraVal enabled?
        if dev:
            chute.add_middleware(DevMiddleware)
            if graval_seed is not None:
                logger.info(f"Initializing graval with {graval_seed=}")
                miner().initialize(graval_seed)
                miner()._seed = graval_seed
        else:
            if graval_seed is not None:
                logger.info(f"Initializing graval with {graval_seed=}")
                miner().initialize(graval_seed)
                miner()._seed = graval_seed
            chute.add_middleware(GraValMiddleware, concurrency=chute.concurrency)
            miner()._miner_ss58 = miner_ss58
            miner()._validator_ss58 = validator_ss58
            miner()._keypair = Keypair(ss58_address=validator_ss58, crypto_type=KeypairType.SR25519)

        # Run initialization code.
        await chute.initialize()

        # Metrics endpoint.
        async def _metrics():
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

        chute.add_api_route("/_metrics", _metrics, methods=["GET"])

        # Slurps and processes.
        def handle_slurp(request: Request):
            """
            Read part or all of a file.
            """
            nonlocal chute_module
            slurp = Slurp(**request.state.decrypted)
            if slurp.path == "__file__":
                source_code = inspect.getsource(chute_module)
                return Response(
                    content=base64.b64encode(source_code.encode()).decode(),
                    media_type="text/plain",
                )
            elif slurp.path == "__run__":
                source_code = inspect.getsource(sys.modules[__name__])
                return Response(
                    content=base64.b64encode(source_code.encode()).decode(),
                    media_type="text/plain",
                )
            if not os.path.isfile(slurp.path):
                if os.path.isdir(slurp.path):
                    if hasattr(request.state, "_encrypt"):
                        return {
                            "json": request.state._encrypt(
                                json.dumps({"dir": os.listdir(slurp.path)})
                            )
                        }
                    return {"dir": os.listdir(slurp.path)}
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Path not found: {slurp.path}",
                )
            response_bytes = None
            with open(slurp.path, "rb") as f:
                f.seek(slurp.start_byte)
                if slurp.end_byte is None:
                    response_bytes = f.read()
                else:
                    response_bytes = f.read(slurp.end_byte - slurp.start_byte)
            response_data = {"contents": base64.b64encode(response_bytes).decode()}
            if hasattr(request.state, "_encrypt"):
                return {"json": request.state._encrypt(json.dumps(response_data))}
            return response_data

        chute.add_api_route("/_slurp", handle_slurp, methods=["POST"])
        chute.add_api_route("/_procs", get_all_process_info)

        # Env checks.
        chute.add_api_route("/_env_sig", get_env_sig, methods=["POST"])
        chute.add_api_route("/_env_dump", get_env_dump, methods=["POST"])

        # Add a ping endpoint for validators to use.
        chute.add_api_route("/_ping", pong, methods=["POST"])
        chute.add_api_route("/_token", get_token, methods=["POST"])
        chute.add_api_route("/_alive", is_alive, methods=["GET"])

        async def _devices():
            return [miner().get_device_info(idx) for idx in range(miner()._device_count)]

        chute.add_api_route("/_devices", _devices)

        # Device info challenge endpoint.
        async def _device_challenge(request: Request, challenge: str):
            return Response(
                content=miner().process_device_info_challenge(challenge),
                media_type="text/plain",
            )

        chute.add_api_route("/_device_challenge", _device_challenge, methods=["GET"])

        # Filesystem challenge endpoint.
        async def _fs_challenge(request: Request):
            challenge = FSChallenge(**request.state.decrypted)
            return Response(
                content=miner().process_filesystem_challenge(
                    filename=challenge.filename,
                    offset=challenge.offset,
                    length=challenge.length,
                ),
                media_type="text/plain",
            )

        chute.add_api_route("/_fs_challenge", _fs_challenge, methods=["POST"])

        logger.info("Added validation endpoints.")

        config = Config(app=chute, host=host, port=port, limit_concurrency=1000)
        server = Server(config)
        await server.serve()

    asyncio.run(_run_chute())
