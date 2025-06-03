"""
Run a chute, automatically handling encryption/decryption via GraVal.
"""

import os
import asyncio
import aiohttp
import traceback
import sys
import jwt
import time
import uuid
import hashlib
import inspect
import typer
import psutil
import base64
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
from chutes.job import Job
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


async def get_metrics():
    """
    Get the latest prometheus metrics.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def get_devices():
    """
    Fetch device information.
    """
    return [miner().get_device_info(idx) for idx in range(miner()._device_count)]


async def process_device_challenge(request: Request, challenge: str):
    """
    Process a GraVal device info challenge string.
    """
    return Response(
        content=miner().process_device_info_challenge(challenge),
        media_type="text/plain",
    )


async def process_fs_challenge(request: Request):
    """
    Process a filesystem challenge.
    """
    challenge = FSChallenge(**request.state.decrypted)
    return Response(
        content=miner().process_filesystem_challenge(
            filename=challenge.filename,
            offset=challenge.offset,
            length=challenge.length,
        ),
        media_type="text/plain",
    )


def handle_slurp(request: Request, chute_module):
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
                return {"json": request.state._encrypt(json.dumps({"dir": os.listdir(slurp.path)}))}
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
    def __init__(self, app: FastAPI, concurrency: int = 1, symmetric_key: str = None):
        """
        Initialize a semaphore for concurrency control/limits.
        """
        super().__init__(app)
        self.concurrency = concurrency
        self.lock = asyncio.Lock()
        self.requests_in_flight = {}
        self.symmetric_key = symmetric_key
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
                    "/_exchange",
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
                "/_exchange",
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


async def _gather_devices_and_initialize(token: str) -> dict:
    """
    Gather the GPU info assigned to this pod, submit with our one-time token to get GraVal seed.
    """
    import chutes.envcheck as envcheck

    # Build the GraVal request based on the GPUs that were actually assigned to this pod.
    body = {"gpus": []}
    for idx in range(miner()._device_count):
        body["gpus"].append(miner().get_device_info(idx))
    token_data = jwt.decode(token, options={"verify_signature": False})
    url = token_data.get("url")
    key_salt_hex = token_data.get("env_key"), "a" * 32
    key = bytes.fromhex(key_salt_hex)
    body["env"] = envcheck.dump(key)
    body["sig"] = envcheck.signature(key_salt_hex)

    # Fetch the challenges.
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.post(url, headers={"Authorization": token}, json=body) as resp:
            init_params = await resp.json()
            logger.success(f"Successfully initialized deployment, received {init_params=}")
            miner()._graval_seed = init_params["seed"]
            iterations = init_params.get("iterations", 1)
            miner().initialize(miner()._graval_seed, iterations)

            # Use GraVal to extract the symmetric key from the response.
            sym_key = init_params["symmetric_key"]
            bytes_ = base64.b64decode(sym_key["ciphertext"])
            iv = bytes_[:16]
            cipher = bytes_[16:]
            symmetric_key = bytes.fromhex(
                miner().decrypt(cipher, iv, len(cipher), sym_key["device_index"])
            )

            # Now, we can respond to the URL by encrypting a payload with the symmetric key and sending it back.
            padder = padding.PKCS7(128).padder()
            cipher = Cipher(
                algorithms.AES(symmetric_key),
                modes.CBC(iv),
                backend=default_backend(),
            )
            plaintext = sym_key["response_plaintext"]
            padded_data = padder.update(plaintext) + padder.finalize()
            encryptor = cipher.encryptor()
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            response_cipher = base64.b64encode(encrypted_data).decode()

            # Post the response to the challenge, which returns job data (if any).
            async with session.put(
                init_params["response_url"],
                headers={"Authorization": token},
                json={
                    "response": response_cipher,
                },
            ) as resp:
                logger.success("Successfully negotiated challenge response")
                return symmetric_key, await resp.json()


# Run a chute (which can be an async job or otherwise long-running process).
def run_chute(
    chute_ref_str: str = typer.Argument(
        ..., help="chute to run, in the form [module]:[app_name], similar to uvicorn"
    ),
    miner_ss58: str = typer.Option(None, help="miner hotkey ss58 address"),
    validator_ss58: str = typer.Option(None, help="validator hotkey ss58 address"),
    port: int | None = typer.Option(None, help="port to listen on"),
    host: str | None = typer.Option(None, help="host to bind to"),
    token: str | None = typer.Option(
        None, help="one-time token to fetch graval params from validator"
    ),
    keyfile: str | None = typer.Option(None, help="path to TLS key file"),
    certfile: str | None = typer.Option(None, help="path to TLS certificate file"),
    debug: bool = typer.Option(False, help="enable debug logging"),
    dev: bool = typer.Option(False, help="dev/local mode"),
    job_data_path: str = typer.Option(None, help="dev mode job payload JSON path"),
):
    async def _run_chute():
        """
        Run the chute (or job).
        """
        chute_module, chute = load_chute(chute_ref_str=chute_ref_str, config_path=None, debug=debug)
        if is_local():
            logger.error("Cannot run chutes in local context!")
            sys.exit(1)

        chute = chute.chute if isinstance(chute, ChutePack) else chute

        # GPU verification plus job fetching.
        job_data: dict | None = None
        symmetric_key: str | None = None
        job_id: str | None = None
        job_obj: Job | None = None
        if token:
            symmetric_key, response = await _gather_devices_and_initialize(token)
            job_id = response.get("job_id")
            job_method = response.get("job_method")
            if job_method:
                job_obj = next(j for j in chute._jobs if j.name == job_method)

        elif not dev:
            logger.error("No GraVal token supplied!")
            sys.exit(1)
        if dev and job_data_path:
            with open(job_data_path) as infile:
                job_data = json.load(infile)

        # Run the chute's initialization code.
        await chute.initialize()

        # Encryption/rate-limiting middleware setup.
        if dev:
            chute.add_middleware(DevMiddleware)
        else:
            chute.add_middleware(
                GraValMiddleware, concurrency=chute.concurrency, symmetric_key=symmetric_key
            )
            miner()._miner_ss58 = miner_ss58
            miner()._validator_ss58 = validator_ss58
            miner()._keypair = Keypair(ss58_address=validator_ss58, crypto_type=KeypairType.SR25519)

        # Slurps and processes.
        async def _handle_slurp(request: Request):
            nonlocal chute_module

            return await handle_slurp(request, chute_module)

        # Validation endpoints.
        chute.add_api_route("/_ping", pong, methods=["POST"])
        chute.add_api_route("/_token", get_token, methods=["POST"])
        chute.add_api_route("/_alive", is_alive, methods=["GET"])
        chute.add_api_route("/_metrics", get_metrics, methods=["GET"])
        chute.add_api_route("/_slurp", _handle_slurp, methods=["POST"])
        chute.add_api_route("/_procs", get_all_process_info, methods=["GET"])
        chute.add_api_route("/_env_sig", get_env_sig, methods=["POST"])
        chute.add_api_route("/_env_dump", get_env_dump, methods=["POST"])
        chute.add_api_route("/_devices", get_devices, methods=["GET"])
        chute.add_api_route("/_device_challenge", process_device_challenge, methods=["GET"])
        chute.add_api_route("/_fs_challenge", process_fs_challenge, methods=["POST"])
        chute.add_api_route("/_env_sig", get_env_sig, methods=["POST"])
        chute.add_api_route("/_env_dump", get_env_dump, methods=["POST"])
        logger.success("Added all chutes internal endpoints.")

        # Job related endpoints.
        async def _shutdown():
            nonlocal job_obj, server
            if not job_obj:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Job task not found",
                )
            logger.warning("Shutdown requested.")
            if job_obj and not job_obj.cancel_event.is_set():
                job_obj.cancel_event.set()
            server.should_exit = True
            return {"ok": True}

        async def _launch_job(request: Request):
            nonlocal chute, job_obj
            job_data = request.state.decrypted
            if job_data.get("skip"):
                logger.warning("Instructed to skip job!")
                os._exit(0)
            try:
                final_result = await job_obj.run(**job_data)
                logger.info(f"Job completed with result: {final_result}")
            except asyncio.CancelledError:
                logger.error("Job cancelled.")
            except asyncio.TimeoutError:
                logger.error("Job timed out.")
            except Exception as exc:
                logger.error(f"Job failed unexpectedly: {exc}\n{traceback.format_exc()}")
            server.should_exit = True

        if job_id:
            chute.add_api_route("/_shutdown", _shutdown, methods=["POST"])
            chute.add_api_route("/_launch", _launch_job, methods=["POST"])
            logger.info("Added job endpoints: /_shutdown & /_launch")

        # Start the uvicorn process, whether in job mode or not.
        config = Config(
            app=chute,
            host=host or "0.0.0.0",
            port=port or 8000,
            limit_concurrency=1000,
            ssl_certfile=certfile,
            ssl_keyfile=keyfile,
        )
        server = Server(config)
        await server.serve()

    # Kick everything off
    asyncio.run(_run_chute())
