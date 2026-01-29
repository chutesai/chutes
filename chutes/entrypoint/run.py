"""
Run a chute, automatically handling encryption/decryption via GraVal.
"""

import os
import re
import asyncio
import aiohttp
import sys
import ssl
import site
import ctypes
import time
import uuid
import errno
import inspect
import typer
import psutil
import base64
import socket
import secrets
import threading
import traceback
import orjson as json
from aiohttp import ClientError
from functools import lru_cache
from loguru import logger
from typing import Optional, Any
from datetime import datetime
from pydantic import BaseModel
from ipaddress import ip_address
from uvicorn import Config, Server
from fastapi import Request, Response, status, HTTPException
from fastapi.responses import ORJSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from chutes.entrypoint.verify import GpuVerifier
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from substrateinterface import Keypair, KeypairType
from chutes.entrypoint._shared import (
    get_launch_token,
    get_launch_token_data,
    load_chute,
    miner,
    authenticate_request,
)
from chutes.entrypoint.ssh import setup_ssh_access
from chutes.chute import ChutePack, Job
from chutes.util.context import is_local
from chutes.cfsv_wrapper import get_cfsv
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding


RUNINT_PATH = os.path.join(os.path.dirname(__file__), "..", "chutes-runint.so")


class _ConnStats:
    """Module-level connection stats tracker."""

    def __init__(self):
        self.concurrency = 1
        self.requests_in_flight = {}
        self._lock = None

    @property
    def lock(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def get_stats(self) -> dict:
        now = time.time()
        in_flight = len(self.requests_in_flight)
        available = max(0, self.concurrency - in_flight)
        utilization = in_flight / self.concurrency if self.concurrency > 0 else 0.0
        oldest_age = None
        if self.requests_in_flight:
            oldest_age = max(0.0, now - min(self.requests_in_flight.values()))
        return {
            "concurrency": self.concurrency,
            "in_flight": in_flight,
            "available": available,
            "utilization": round(utilization, 4),
            "oldest_in_flight_age_secs": oldest_age,
        }


_conn_stats = _ConnStats()


@lru_cache(maxsize=1)
def get_netnanny_ref():
    netnanny = ctypes.CDLL(None, ctypes.RTLD_GLOBAL)
    netnanny.generate_challenge_response.argtypes = [ctypes.c_char_p]
    netnanny.generate_challenge_response.restype = ctypes.c_char_p
    netnanny.verify.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint8]
    netnanny.verify.restype = ctypes.c_int
    netnanny.initialize_network_control.argtypes = []
    netnanny.initialize_network_control.restype = ctypes.c_int
    netnanny.unlock_network.argtypes = []
    netnanny.unlock_network.restype = ctypes.c_int
    netnanny.lock_network.argtypes = []
    netnanny.lock_network.restype = ctypes.c_int
    netnanny.set_secure_fs.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    netnanny.set_secure_fs.restype = ctypes.c_int
    netnanny.set_secure_env.argtypes = []
    netnanny.set_secure_env.restype = ctypes.c_int
    return netnanny


class _RunintHandle:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def _load_lib(self):
        if hasattr(self, "_lib") and self._lib is not None:
            return self._lib
        lib_path = RUNINT_PATH
        if not os.path.exists(lib_path):
            return None
        self._lib = ctypes.CDLL(lib_path)
        self._lib._io_pool_init.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._lib._io_pool_init.restype = ctypes.c_void_p
        self._lib._io_pool_sync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._lib._io_pool_sync.restype = ctypes.c_int64
        self._lib._io_pool_pos.argtypes = [ctypes.c_void_p]
        self._lib._io_pool_pos.restype = ctypes.c_int64
        self._lib._io_pool_release.argtypes = [ctypes.c_void_p]
        self._lib._io_pool_release.restype = None
        self._lib._io_pool_get_nonce.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._lib._io_pool_get_nonce.restype = ctypes.c_int

        # Session encryption API
        self._lib._io_pool_derive_session_key.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        self._lib._io_pool_derive_session_key.restype = ctypes.c_int
        self._lib._io_pool_session_ready.argtypes = [ctypes.c_void_p]
        self._lib._io_pool_session_ready.restype = ctypes.c_int
        self._lib._io_pool_encrypt.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._lib._io_pool_encrypt.restype = ctypes.c_int
        self._lib._io_pool_decrypt.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._lib._io_pool_decrypt.restype = ctypes.c_int
        self._lib._io_pool_get_pubkey.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._lib._io_pool_get_pubkey.restype = ctypes.c_int
        self._lib._io_pool_set_session_key.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._lib._io_pool_set_session_key.restype = ctypes.c_int

        return self._lib

    def init(self, validator_nonce: str = None):
        """
        Initialize runtime integrity with validator-provided nonce.

        The commitment format v3 is:
        03 || version (1) || pubkey (64 bytes) || nonce (16 bytes) || lib_fp (16 bytes) || sig (64 bytes)
        = 162 bytes = 324 hex chars

        The validator can verify:
        1. Extract pubkey, nonce, lib_fp, and signature from commitment
        2. Verify signature is valid for hash(version || pubkey || nonce || lib_fp) using pubkey
        3. This proves the keypair holder committed to this specific nonce and library version
        """
        if self._initialized:
            return self._commitment
        with self._lock:
            if self._initialized:
                return self._commitment
            try:
                lib = self._load_lib()
                if lib is None:
                    logger.warning("runint library not found")
                    return None
                # Commitment v3: 03 + ver(2) + pubkey(128) + nonce(32) + lib_fp(32) + sig(128) = 324 chars + null
                commitment_buf = ctypes.create_string_buffer(325)
                nonce_bytes = validator_nonce.encode() if validator_nonce else b""
                self._handle = lib._io_pool_init(commitment_buf, 325, nonce_bytes, len(nonce_bytes))
                if self._handle:
                    self._commitment = commitment_buf.value.decode()
                    # Also get the nonce (stored from validator input)
                    nonce_buf = ctypes.create_string_buffer(33)
                    if lib._io_pool_get_nonce(self._handle, nonce_buf, 33) == 0:
                        self._nonce = nonce_buf.value.decode()
                    else:
                        self._nonce = None
                    self._initialized = True
                    return self._commitment
            except Exception as e:
                logger.warning(f"Failed to initialize runtime integrity: {e}")
            return None

    def get_nonce(self) -> str | None:
        """Get the random nonce generated at init time."""
        return getattr(self, "_nonce", None)

    def prove(self, challenge: str) -> tuple[str, int] | None:
        """Sign a challenge and return (signature, epoch)."""
        if not self._initialized or not self._handle:
            return None
        try:
            sig_buf = ctypes.create_string_buffer(129)
            epoch = self._lib._io_pool_sync(self._handle, challenge.encode(), sig_buf, 129)
            if epoch >= 0:
                return sig_buf.value.decode(), epoch
        except Exception as e:
            logger.warning(f"Failed to generate runtime integrity proof: {e}")
        return None

    def get_pubkey(self) -> str | None:
        """Get our public key in hex format for ECDH."""
        if not self._initialized or not self._handle:
            return None
        try:
            pubkey_buf = ctypes.create_string_buffer(129)
            ret = self._lib._io_pool_get_pubkey(self._handle, pubkey_buf, 129)
            if ret == 0:
                return pubkey_buf.value.decode()
        except Exception as e:
            logger.warning(f"Failed to get runint pubkey: {e}")
        return None

    def derive_session_key(self, validator_pubkey_hex: str) -> bool:
        """Derive session encryption key from validator's public key via ECDH."""
        if not self._initialized or not self._handle:
            return False
        try:
            ret = self._lib._io_pool_derive_session_key(self._handle, validator_pubkey_hex.encode())
            if ret == 0:
                logger.info("Session encryption key derived successfully")
                return True
            logger.warning(f"Failed to derive session key: {ret}")
        except Exception as e:
            logger.warning(f"Failed to derive session key: {e}")
        return False

    def set_session_key(self, key: bytes) -> bool:
        """Set session encryption key directly from raw bytes (for backward compat)."""
        if not self._initialized or not self._handle:
            return False
        try:
            ret = self._lib._io_pool_set_session_key(self._handle, key, len(key))
            if ret == 0:
                logger.info("Session encryption key set successfully")
                return True
            logger.warning(f"Failed to set session key: {ret}")
        except Exception as e:
            logger.warning(f"Failed to set session key: {e}")
        return False

    def session_ready(self) -> bool:
        """Check if session encryption key has been derived."""
        if not self._initialized or not self._handle:
            return False
        try:
            return self._lib._io_pool_session_ready(self._handle) == 1
        except Exception:
            return False

    def encrypt(self, plaintext: bytes) -> bytes | None:
        """Encrypt data using session key (AES-256-GCM)."""
        if not self._initialized or not self._handle:
            return None
        try:
            output_len = len(plaintext) + 16 + 12  # tag + nonce
            output_buf = ctypes.create_string_buffer(output_len)
            ret = self._lib._io_pool_encrypt(
                self._handle, plaintext, len(plaintext), output_buf, output_len
            )
            if ret > 0:
                return output_buf.raw[:ret]
            logger.warning(f"Encryption failed with code {ret}")
        except Exception as e:
            logger.warning(f"Encryption failed: {e}")
        return None

    def decrypt(self, ciphertext: bytes) -> bytes | None:
        """Decrypt data using session key (AES-256-GCM)."""
        if not self._initialized or not self._handle:
            return None
        try:
            output_len = len(ciphertext)
            output_buf = ctypes.create_string_buffer(output_len)
            ret = self._lib._io_pool_decrypt(
                self._handle, ciphertext, len(ciphertext), output_buf, output_len
            )
            if ret >= 0:
                return output_buf.raw[:ret]
            logger.warning(f"Decryption failed with code {ret}")
        except Exception as e:
            logger.warning(f"Decryption failed: {e}")
        return None


@lru_cache(maxsize=1)
def get_runint_handle():
    return _RunintHandle()


def init_runint(validator_nonce: str = None):
    return get_runint_handle().init(validator_nonce)


def runint_get_nonce() -> str | None:
    return get_runint_handle().get_nonce()


def runint_prove(challenge: str) -> tuple[str, int] | None:
    return get_runint_handle().prove(challenge)


def runint_get_pubkey() -> str | None:
    return get_runint_handle().get_pubkey()


def runint_derive_session_key(validator_pubkey_hex: str) -> bool:
    return get_runint_handle().derive_session_key(validator_pubkey_hex)


def runint_set_session_key(key: bytes) -> bool:
    return get_runint_handle().set_session_key(key)


def runint_session_ready() -> bool:
    return get_runint_handle().session_ready()


def runint_encrypt(plaintext: bytes) -> bytes | None:
    return get_runint_handle().encrypt(plaintext)


def runint_decrypt(ciphertext: bytes) -> bytes | None:
    return get_runint_handle().decrypt(ciphertext)


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


def process_netnanny_challenge(chute, request: Request):
    """
    Process a NetNanny challenge.
    """
    challenge = request.state.decrypted.get("challenge", "foo")
    netnanny = get_netnanny_ref()
    return {
        "hash": netnanny.generate_challenge_response(challenge.encode()),
        "allow_external_egress": chute.allow_external_egress,
    }


async def handle_slurp(request: Request, chute_module):
    """
    Read part or all of a file.
    """
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
    async with aiohttp.ClientSession() as session:
        async with session.get(endpoint, params={"salt": salt}) as resp:
            if hasattr(request.state, "_encrypt"):
                return {"json": request.state._encrypt(await resp.text())}
            return await resp.json()


def _conn_err_info(exc: BaseException) -> str:
    """
    Update error info for connectivity tests to be readable.
    """
    if isinstance(exc, OSError):
        name = {
            errno.ENETUNREACH: "ENETUNREACH",
            errno.EHOSTUNREACH: "EHOSTUNREACH",
            errno.ECONNREFUSED: "ECONNREFUSED",
            errno.ETIMEDOUT: "ETIMEDOUT",
        }.get(exc.errno)
        if name:
            return f"{name}: {exc}"
    return str(exc)


async def check_connectivity(request: Request) -> dict[str, Any]:
    """
    Check if network access is allowed.
    """
    endpoint = request.state.decrypted.get(
        "endpoint", "https://api.chutes.ai/instances/token_check"
    )
    timeout = aiohttp.ClientTimeout(total=8, connect=4, sock_connect=4, sock_read=6)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(endpoint) as resp:
                data = await resp.read()
                b64_body = base64.b64encode(data).decode("ascii")
                return {
                    "connection_established": True,
                    "status_code": resp.status,
                    "body": b64_body,
                    "content_type": resp.headers.get("Content-Type"),
                    "error": None,
                }
    except (asyncio.TimeoutError, ssl.SSLError, ClientError, OSError) as e:
        return {
            "connection_established": False,
            "status_code": None,
            "body": None,
            "content_type": None,
            "error": _conn_err_info(e),
        }
    except Exception as e:
        return {
            "connection_established": False,
            "status_code": None,
            "body": None,
            "content_type": None,
            "error": str(e),
        }


async def generate_filesystem_hash(salt: str, exclude_file: str, mode: str = "full"):
    """
    Generate a hash of the filesystem, in either sparse or full mode.
    """
    logger.info(
        f"Running filesystem verification challenge in {mode=} using {salt=} excluding {exclude_file}"
    )
    loop = asyncio.get_event_loop()
    cfsv = get_cfsv()
    fsv_hash = await loop.run_in_executor(
        None,
        cfsv.challenge,
        salt,
        mode,
        "/",
        "/etc/chutesfs.index",
        exclude_file,
    )
    if not fsv_hash:
        logger.warning("Failed to generate filesystem verification hash from cfsv library")
        raise Exception("Failed to generate filesystem challenge response.")
    logger.success(f"Filesystem verification hash: {fsv_hash}")
    return fsv_hash


class Slurp(BaseModel):
    path: str
    start_byte: Optional[int] = 0
    end_byte: Optional[int] = None


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
    def __init__(self, app, concurrency: int = 1):
        """
        Initialize a semaphore for concurrency control/limits.
        """
        super().__init__(app)
        _conn_stats.concurrency = concurrency

    async def _dispatch(self, request: Request, call_next):
        """
        Handle authentication and body decryption.
        """
        if request.client.host == "127.0.0.1":
            return await call_next(request)

        # Authentication...
        body_bytes, failure_response = await authenticate_request(request)
        if failure_response:
            return failure_response

        # Decrypt request body.
        if request.method in ("POST", "PUT", "PATCH"):
            try:
                ciphertext = base64.b64decode(body_bytes)
                decrypted_bytes = runint_decrypt(ciphertext)
                if not decrypted_bytes:
                    return ORJSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"detail": "Decryption failed"},
                    )
                try:
                    request.state.decrypted = json.loads(decrypted_bytes)
                except Exception:
                    request.state.decrypted = json.loads(
                        decrypted_bytes.rstrip(bytes(range(1, 17)))
                    )
            except Exception as exc:
                return ORJSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": f"Decryption failed: {exc}"},
                )

            def _encrypt(plaintext: bytes):
                if isinstance(plaintext, str):
                    plaintext = plaintext.encode()
                encrypted = runint_encrypt(plaintext)
                if not encrypted:
                    raise RuntimeError("Encryption failed")
                return base64.b64encode(encrypted).decode()

            request.state._encrypt = _encrypt

        return await call_next(request)

    async def dispatch(self, request: Request, call_next):
        """
        Rate-limiting wrapper around the actual dispatch function.
        """
        request.request_id = str(uuid.uuid4())
        request.state.serialized = request.headers.get("X-Chutes-Serialized") is not None
        path = request.scope.get("path", "")

        # Verify expected IP if header present.
        expected_ip = request.headers.get("X-Conn-ExpIP")
        if expected_ip:
            client_ip = ip_address(request.client.host)
            if client_ip.is_private or str(client_ip) != expected_ip:
                return ORJSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "IP mismatch"},
                    headers={"X-Conn-ExpIP": expected_ip},
                )
            request.state.exp_ip = expected_ip

        # Localhost bypasses encryption (health checks).
        if request.client.host == "127.0.0.1":
            return await self._dispatch(request, call_next)

        # Metrics/stats from private IPs bypass encryption (prometheus).
        if path.endswith(("/_metrics", "/_conn_stats")):
            ip = ip_address(request.client.host)
            is_private = (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
            )
            if is_private:
                return await call_next(request)
            return ORJSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "unauthorized"},
            )

        # All other paths must be encrypted.
        try:
            ciphertext = bytes.fromhex(path[1:])
            decrypted = runint_decrypt(ciphertext)
            if not decrypted:
                return ORJSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={"detail": f"Bad path: {path}"},
                )
            actual_path = decrypted.decode().rstrip("?")
            logger.info(f"Decrypted request path: {actual_path} from input path: {path}")
            request.scope["path"] = actual_path
        except Exception:
            return ORJSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"detail": f"Bad path: {path}"},
            )

        # Internal paths bypass rate limiting.
        if request.scope.get("path", "").startswith("/_"):
            return await self._dispatch(request, call_next)

        # Concurrency control with timeouts in case it didn't get cleaned up properly.
        async with _conn_stats.lock:
            now = time.time()
            if len(_conn_stats.requests_in_flight) >= _conn_stats.concurrency:
                purge_keys = []
                for key, val in _conn_stats.requests_in_flight.items():
                    if now - val >= 1800:
                        logger.warning(
                            f"Assuming this request is no longer in flight, killing: {key}"
                        )
                        purge_keys.append(key)
                if purge_keys:
                    for key in purge_keys:
                        _conn_stats.requests_in_flight.pop(key, None)
                    _conn_stats.requests_in_flight[request.request_id] = now
                else:
                    return ORJSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={
                            "error": "RateLimitExceeded",
                            "detail": f"Max concurrency exceeded: {_conn_stats.concurrency}, try again later.",
                        },
                    )
            else:
                _conn_stats.requests_in_flight[request.request_id] = now

        # Perform the actual request.
        response = None
        try:
            response = await self._dispatch(request, call_next)

            # Add concurrency headers to the response.
            in_flight = len(_conn_stats.requests_in_flight)
            available = max(0, _conn_stats.concurrency - in_flight)
            utilization = (
                in_flight / _conn_stats.concurrency if _conn_stats.concurrency > 0 else 0.0
            )
            response.headers["X-Chutes-Conn-Used"] = str(in_flight)
            response.headers["X-Chutes-Conn-Available"] = str(available)
            response.headers["X-Chutes-Conn-Utilization"] = f"{utilization:.4f}"
            if hasattr(request.state, "exp_ip"):
                response.headers["X-Conn-ExpIP"] = request.state.exp_ip

            if hasattr(response, "body_iterator"):
                original_iterator = response.body_iterator

                async def wrapped_iterator():
                    try:
                        async for chunk in original_iterator:
                            yield chunk
                    except Exception as exc:
                        logger.warning(f"Unhandled exception in body iterator: {exc}")
                        _conn_stats.requests_in_flight.pop(request.request_id, None)
                        raise
                    finally:
                        _conn_stats.requests_in_flight.pop(request.request_id, None)

                response.body_iterator = wrapped_iterator()
                return response
            return response
        finally:
            if not response or not hasattr(response, "body_iterator"):
                _conn_stats.requests_in_flight.pop(request.request_id, None)


def start_dummy_socket(port_mapping, symmetric_key):
    """
    Start a dummy socket based on the port mapping configuration to validate ports.
    """
    proto = port_mapping["proto"].lower()
    internal_port = port_mapping["internal_port"]
    response_text = f"response from {proto} {internal_port}"
    if proto in ["tcp", "http"]:
        return start_tcp_dummy(internal_port, symmetric_key, response_text)
    return start_udp_dummy(internal_port, symmetric_key, response_text)


def encrypt_response(symmetric_key, plaintext):
    """
    Encrypt the response using AES-CBC with PKCS7 padding.
    """
    padder = padding.PKCS7(128).padder()
    new_iv = secrets.token_bytes(16)
    cipher = Cipher(
        algorithms.AES(symmetric_key),
        modes.CBC(new_iv),
        backend=default_backend(),
    )
    padded_data = padder.update(plaintext.encode()) + padder.finalize()
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
    response_cipher = base64.b64encode(encrypted_data).decode()
    return new_iv, response_cipher


def start_tcp_dummy(port, symmetric_key, response_plaintext):
    """
    TCP port check socket.
    """

    def tcp_handler():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
            sock.listen(1)
            logger.info(f"TCP socket listening on port {port}")
            conn, addr = sock.accept()
            logger.info(f"TCP connection from {addr}")
            data = conn.recv(1024)
            logger.info(f"TCP received: {data.decode('utf-8', errors='ignore')}")
            iv, encrypted_response = encrypt_response(symmetric_key, response_plaintext)
            full_response = f"{iv.hex()}|{encrypted_response}".encode()
            conn.send(full_response)
            logger.info(f"TCP sent encrypted response on port {port}: {full_response=}")
            conn.close()
        except Exception as e:
            logger.info(f"TCP socket error on port {port}: {e}")
            raise
        finally:
            sock.close()
            logger.info(f"TCP socket on port {port} closed")

    thread = threading.Thread(target=tcp_handler, daemon=True)
    thread.start()
    return thread


def start_udp_dummy(port, symmetric_key, response_plaintext):
    """
    UDP port check socket.
    """

    def udp_handler():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
            logger.info(f"UDP socket listening on port {port}")
            data, addr = sock.recvfrom(1024)
            logger.info(f"UDP received from {addr}: {data.decode('utf-8', errors='ignore')}")
            iv, encrypted_response = encrypt_response(symmetric_key, response_plaintext)
            full_response = f"{iv.hex()}|{encrypted_response}".encode()
            sock.sendto(full_response, addr)
            logger.info(f"UDP sent encrypted response on port {port}")
        except Exception as e:
            logger.info(f"UDP socket error on port {port}: {e}")
            raise
        finally:
            sock.close()
            logger.info(f"UDP socket on port {port} closed")

    thread = threading.Thread(target=udp_handler, daemon=True)
    thread.start()
    return thread


async def _gather_devices_and_initialize(
    host: str,
    port_mappings: list[dict[str, Any]],
    chute_abspath: str,
    inspecto_hash: str,
) -> tuple[bool, str, dict[str, Any]]:
    """
    Gather the GPU info assigned to this pod, submit with our one-time token to get GraVal seed.
    """

    # Build the GraVal request based on the GPUs that were actually assigned to this pod.
    logger.info("Collecting GPUs and port mappings...")
    body = {"gpus": [], "port_mappings": port_mappings, "host": host}
    token_data = get_launch_token_data()
    url = token_data.get("url")
    key = token_data.get("env_key", "a" * 32)

    logger.info("Collecting full envdump...")
    import chutes.envdump as envdump

    body["env"] = envdump.DUMPER.dump(key)
    body["run_code"] = envdump.DUMPER.slurp(key, os.path.abspath(__file__), 0, 0)
    body["inspecto"] = inspecto_hash

    body["run_path"] = os.path.abspath(__file__)
    body["py_dirs"] = list(set(site.getsitepackages() + [site.getusersitepackages()]))

    # NetNanny configuration.
    netnanny = get_netnanny_ref()
    egress = token_data.get("egress", False)
    body["egress"] = egress
    body["netnanny_hash"] = netnanny.generate_challenge_response(
        token_data["sub"].encode()
    ).decode()
    body["fsv"] = await generate_filesystem_hash(token_data["sub"], chute_abspath, mode="full")

    # Runtime integrity (already initialized at this point).
    handle = get_runint_handle()
    body["rint_commitment"] = handle._commitment
    body["rint_nonce"] = handle.get_nonce()
    # Include our pubkey for ECDH session key derivation
    rint_pubkey = runint_get_pubkey()
    if rint_pubkey:
        body["rint_pubkey"] = rint_pubkey

    # Disk space.
    disk_gb = token_data.get("disk_gb", 10)
    logger.info(f"Checking disk space availability: {disk_gb}GB required")
    try:
        cfsv = get_cfsv()
        if not cfsv.sizetest("/tmp", disk_gb):
            logger.error("Disk space check failed")
            raise Exception(f"Insufficient disk space: {disk_gb}GB required in /tmp")
        logger.success(f"Disk space check passed: {disk_gb}GB available in /tmp")
    except Exception as e:
        logger.error(f"Error checking disk space: {e}")
        raise Exception(f"Failed to verify disk space availability: {e}")

    # Start up dummy sockets to test port mappings.
    dummy_socket_threads = []
    for port_map in port_mappings:
        if port_map.get("default"):
            continue
        dummy_socket_threads.append(start_dummy_socket(port_map, symmetric_key))

    # Verify GPUs for symmetric key
    verifier = GpuVerifier.create(url, body)
    symmetric_key, response = await verifier.verify_devices()

    # Derive runint session key from validator's pubkey via ECDH if provided
    # Key derivation happens entirely in C - key never touches Python memory
    validator_pubkey = response.get("validator_pubkey")
    if validator_pubkey:
        if runint_derive_session_key(validator_pubkey):
            logger.success("Derived runint session key via ECDH (key never in Python)")
        else:
            logger.warning("Failed to derive runint session key - using legacy encryption")

    return egress, symmetric_key, response


# Run a chute (which can be an async job or otherwise long-running process).
def run_chute(
    chute_ref_str: str = typer.Argument(
        ..., help="chute to run, in the form [module]:[app_name], similar to uvicorn"
    ),
    miner_ss58: str = typer.Option(None, help="miner hotkey ss58 address"),
    validator_ss58: str = typer.Option(None, help="validator hotkey ss58 address"),
    host: str | None = typer.Option("0.0.0.0", help="host to bind to"),
    port: int | None = typer.Option(8000, help="port to listen on"),
    logging_port: int | None = typer.Option(8001, help="logging port"),
    keyfile: str | None = typer.Option(None, help="path to TLS key file"),
    certfile: str | None = typer.Option(None, help="path to TLS certificate file"),
    debug: bool = typer.Option(False, help="enable debug logging"),
    dev: bool = typer.Option(False, help="dev/local mode"),
    dev_job_data_path: str = typer.Option(None, help="dev mode: job payload JSON path"),
    dev_job_method: str = typer.Option(None, help="dev mode: job method"),
    generate_inspecto_hash: bool = typer.Option(False, help="only generate inspecto hash and exit"),
):
    async def _run_chute():
        """
        Run the chute (or job).
        """
        if not (dev or generate_inspecto_hash):
            preload = os.getenv("LD_PRELOAD")
            if preload != "/usr/local/lib/chutes-netnanny.so:/usr/local/lib/chutes-logintercept.so":
                logger.error(f"LD_PRELOAD not set to expected values: {os.getenv('LD_PRELOAD')}")
                sys.exit(137)
            if set(k.lower() for k in os.environ) & {"http_proxy", "https_proxy"}:
                logger.error("HTTP(s) proxy detected, refusing to run.")
                sys.exit(137)

        if generate_inspecto_hash and (miner_ss58 or validator_ss58):
            logger.error("Cannot set --generate-inspecto-hash for real runtime")
            sys.exit(137)

        # Configure net-nanny.
        netnanny = get_netnanny_ref() if not (dev or generate_inspecto_hash) else None

        # If the LD_PRELOAD is already in place, unlock network in dev mode.
        if dev:
            try:
                netnanny = get_netnanny_ref()
                netnanny.initialize_network_control()
                netnanny.unlock_network()
            except AttributeError:
                ...

        if not (dev or generate_inspecto_hash):
            challenge = secrets.token_hex(16).encode("utf-8")
            response = netnanny.generate_challenge_response(challenge)
            if netnanny.set_secure_env() != 0:
                logger.error("NetNanny failed to set secure environment, aborting")
                sys.exit(137)
            try:
                if not response:
                    logger.error("NetNanny validation failed: no response")
                    sys.exit(137)
                if netnanny.verify(challenge, response, 0) != 1:
                    logger.error("NetNanny validation failed: invalid response")
                    sys.exit(137)
                if netnanny.initialize_network_control() != 0:
                    logger.error("Failed to initialize network control")
                    sys.exit(137)

                # Ensure policy is respected.
                netnanny.lock_network()
                request_succeeded = False
                try:
                    async with aiohttp.ClientSession(raise_for_status=True) as session:
                        async with session.get("https://api.chutes.ai/_lbping"):
                            request_succeeded = True
                            logger.error("Should not have been able to ping external https!")
                except Exception:
                    ...
                if request_succeeded:
                    logger.error("Network policy not properly enabled, tampering detected...")
                    sys.exit(137)
                try:
                    async with aiohttp.ClientSession(raise_for_status=True) as session:
                        async with session.get(
                            "https://proxy.chutes.ai/misc/proxy?url=ping"
                        ) as resp:
                            request_succeeded = True
                            logger.success(
                                f"Successfully pinged proxy endpoint: {await resp.text()}"
                            )
                except Exception:
                    ...
                if not request_succeeded:
                    logger.error(
                        "Network policy not properly enabled, failed to connect to proxy URL!"
                    )
                    sys.exit(137)
                # Keep network unlocked for initialization (download models etc.)
                if netnanny.unlock_network() != 0:
                    logger.error("Failed to unlock network")
                    sys.exit(137)
                response = netnanny.generate_challenge_response(challenge)
                if netnanny.verify(challenge, response, 1) != 1:
                    logger.error("NetNanny validation failed: invalid response")
                    sys.exit(137)
                logger.debug("NetNanny initialized and network unlocked")
            except (OSError, AttributeError) as e:
                logger.error(f"NetNanny library not properly loaded: {e}")
                sys.exit(137)
            if not dev and os.getenv("CHUTES_NETNANNY_UNSAFE", "") == "1":
                logger.error("NetNanny library not loaded system wide!")
                sys.exit(137)
            if not dev and os.getpid() != 1:
                logger.error(f"Must be PID 1 (container entrypoint), but got PID {os.getpid()}")
                sys.exit(137)

        # Generate inspecto hash.
        token = get_launch_token()
        token_data = get_launch_token_data()

        # Runtime integrity must be initialized first to get the nonce.
        inspecto_hash = None
        runint_nonce = None
        if not (dev or generate_inspecto_hash):
            # Fetch validator-provided nonce before initializing runint.
            # This nonce is embedded in the commitment (signed by the keypair),
            # proving the keypair was created for this specific session.
            # Attacker cannot pre-compute keypairs because they don't know the nonce.
            validator_nonce = None
            base_url = token_data.get("url", "")
            if base_url:
                nonce_url = base_url.rstrip("/") + "/nonce"
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(nonce_url, headers={"Authorization": token}) as resp:
                            if not resp.ok:
                                logger.error(f"Failed to fetch validator nonce: {resp.status}")
                                sys.exit(137)
                            validator_nonce = await resp.text()
                            logger.info("Fetched validator nonce for key binding")
                except Exception as e:
                    logger.error(f"Failed to fetch validator nonce: {e}")
                    sys.exit(137)
            else:
                logger.error("No URL in token for validator nonce")
                sys.exit(137)

            runint_commitment = init_runint(validator_nonce)
            if not runint_commitment:
                logger.error("Runtime integrity initialization failed")
                sys.exit(137)

            runint_nonce = runint_get_nonce()
            if not runint_nonce:
                logger.error("Runtime integrity nonce retrieval failed")
                sys.exit(137)

            # Generate inspecto hash with seed = nonce + sub
            # This prevents replay attacks because the nonce is fresh per init
            from chutes.inspecto import generate_hash

            inspecto_seed = runint_nonce + token_data["sub"]
            inspecto_hash = await generate_hash(hash_type="base", challenge=inspecto_seed)
            if not inspecto_hash:
                logger.error("Inspecto hash generation failed")
                sys.exit(137)
            logger.info(f"Runtime integrity initialized: commitment={runint_commitment[:16]}...")

        elif generate_inspecto_hash:
            from chutes.inspecto import generate_hash

            inspecto_hash = await generate_hash(hash_type="base")
            print(inspecto_hash)
            return

        if dev:
            os.environ["CHUTES_DEV_MODE"] = "true"
        if is_local():
            logger.error("Cannot run chutes in local context!")
            sys.exit(1)

        # Load token and port mappings from the environment.
        port_mappings = [
            # Main chute pod.
            {
                "proto": "tcp",
                "internal_port": port,
                "external_port": port,
                "default": True,
            },
            # Logging server.
            {
                "proto": "tcp",
                "internal_port": logging_port,
                "external_port": logging_port,
                "default": True,
            },
        ]
        external_host = os.getenv("CHUTES_EXTERNAL_HOST")
        primary_port = os.getenv("CHUTES_PORT_PRIMARY")
        if primary_port and primary_port.isdigit():
            port_mappings[0]["external_port"] = int(primary_port)
        ext_logging_port = os.getenv("CHUTES_PORT_LOGGING")
        if ext_logging_port and ext_logging_port.isdigit():
            port_mappings[1]["external_port"] = int(ext_logging_port)
        for key, value in os.environ.items():
            port_match = re.match(r"^CHUTES_PORT_(TCP|UDP|HTTP)_([0-9]+)", key)
            if port_match and value.isdigit():
                port_mappings.append(
                    {
                        "proto": port_match.group(1),
                        "internal_port": int(port_match.group(2)),
                        "external_port": int(value),
                        "default": False,
                    }
                )

        # GPU verification plus job fetching.
        job_data: dict | None = None
        symmetric_key: str | None = None
        job_id: str | None = None
        job_obj: Job | None = None
        job_method: str | None = None
        job_status_url: str | None = None
        activation_url: str | None = None
        allow_external_egress: bool | None = False

        chute_filename = os.path.basename(chute_ref_str.split(":")[0] + ".py")
        chute_abspath: str = os.path.abspath(os.path.join(os.getcwd(), chute_filename))
        if token:
            (
                allow_external_egress,
                symmetric_key,
                response,
            ) = await _gather_devices_and_initialize(
                external_host,
                port_mappings,
                chute_abspath,
                inspecto_hash,
            )
            job_id = response.get("job_id")
            job_method = response.get("job_method")
            job_status_url = response.get("job_status_url")
            job_data = response.get("job_data")
            activation_url = response.get("activation_url")
            code = response["code"]
            fs_key = response["fs_key"]
            encrypted_cache = response.get("efs") is True
            if (
                fs_key
                and netnanny.set_secure_fs(chute_abspath.encode(), fs_key.encode(), encrypted_cache)
                != 0
            ):
                logger.error("NetNanny failed to set secure FS, aborting!")
                sys.exit(137)
            with open(chute_abspath, "w") as outfile:
                outfile.write(code)

            # Secret environment variables, e.g. HF tokens for private models.
            if response.get("secrets"):
                for secret_key, secret_value in response["secrets"].items():
                    os.environ[secret_key] = secret_value

        elif not dev:
            logger.error("No GraVal token supplied!")
            sys.exit(1)

        # Now we have the chute code available, either because it's dev and the file is plain text here,
        # or it's prod and we've fetched the code from the validator and stored it securely.
        chute_module, chute = load_chute(chute_ref_str=chute_ref_str, config_path=None, debug=debug)
        chute = chute.chute if isinstance(chute, ChutePack) else chute
        if job_method:
            job_obj = next(j for j in chute._jobs if j.name == job_method)

        # Configure dev method job payload/method/etc.
        if dev and dev_job_data_path:
            with open(dev_job_data_path) as infile:
                job_data = json.loads(infile.read())
            job_id = str(uuid.uuid4())
            job_method = dev_job_method
            job_obj = next(j for j in chute._jobs if j.name == dev_job_method)
            logger.info(f"Creating task, dev mode, for {job_method=}")

        # Run the chute's initialization code.
        await chute.initialize()

        # Encryption/rate-limiting middleware setup.
        if dev:
            chute.add_middleware(DevMiddleware)
        else:
            chute.add_middleware(
                GraValMiddleware,
                concurrency=chute.concurrency,
            )

        # Slurps and processes.
        async def _handle_slurp(request: Request):
            nonlocal chute_module

            return await handle_slurp(request, chute_module)

        async def _wait_for_server_ready(timeout: float = 30.0):
            """Wait until the server is accepting connections."""
            import socket

            start = asyncio.get_event_loop().time()
            while (asyncio.get_event_loop().time() - start) < timeout:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex(("127.0.0.1", port))
                    sock.close()
                    if result == 0:
                        return True
                except Exception:
                    pass
                await asyncio.sleep(0.1)
            return False

        async def _do_activation():
            """Activate after server is listening."""
            if not activation_url:
                return

            if not await _wait_for_server_ready():
                logger.error("Server failed to start listening")
                raise Exception("Server not ready for activation")

            activated = False
            for attempt in range(10):
                if attempt > 0:
                    await asyncio.sleep(attempt)
                try:
                    async with aiohttp.ClientSession(raise_for_status=False) as session:
                        async with session.get(
                            activation_url, headers={"Authorization": token}
                        ) as resp:
                            if resp.ok:
                                logger.success(f"Instance activated: {await resp.text()}")
                                activated = True
                                if not dev and not allow_external_egress:
                                    if netnanny.lock_network() != 0:
                                        logger.error("Failed to unlock network")
                                        sys.exit(137)
                                    logger.success("Successfully enabled NetNanny network lock.")
                                break

                            logger.error(
                                f"Instance activation failed: {resp.status=}: {await resp.text()}"
                            )
                            if resp.status == 423:
                                break

                except Exception as e:
                    logger.error(f"Unexpected error attempting to activate instance: {str(e)}")
            if not activated:
                logger.error("Failed to activate instance, aborting...")
                sys.exit(137)

        @chute.on_event("startup")
        async def activate_on_startup():
            asyncio.create_task(_do_activation())

        async def _handle_fs_hash_challenge(request: Request):
            nonlocal chute_abspath
            data = request.state.decrypted
            return {
                "result": await generate_filesystem_hash(
                    data["salt"], chute_abspath, mode=data.get("mode", "sparse")
                )
            }

        async def _handle_conn_stats(request: Request):
            return _conn_stats.get_stats()

        # Validation endpoints.
        chute.add_api_route("/_ping", pong, methods=["POST"])
        chute.add_api_route("/_token", get_token, methods=["POST"])
        chute.add_api_route("/_metrics", get_metrics, methods=["GET"])
        chute.add_api_route("/_conn_stats", _handle_conn_stats, methods=["GET"])
        chute.add_api_route("/_slurp", _handle_slurp, methods=["POST"])
        chute.add_api_route("/_procs", get_all_process_info, methods=["GET"])
        chute.add_api_route("/_env_sig", get_env_sig, methods=["POST"])
        chute.add_api_route("/_env_dump", get_env_dump, methods=["POST"])
        chute.add_api_route("/_devices", get_devices, methods=["GET"])
        chute.add_api_route("/_device_challenge", process_device_challenge, methods=["GET"])
        chute.add_api_route("/_fs_challenge", process_fs_challenge, methods=["POST"])
        chute.add_api_route("/_fs_hash", _handle_fs_hash_challenge, methods=["POST"])
        chute.add_api_route("/_connectivity", check_connectivity, methods=["POST"])

        def _handle_nn(request: Request):
            return process_netnanny_challenge(chute, request)

        chute.add_api_route("/_netnanny_challenge", _handle_nn, methods=["POST"])

        # Runtime integrity challenge endpoint.
        def _handle_rint(request: Request):
            """Handle runtime integrity challenge."""
            challenge = request.state.decrypted.get("challenge")
            if not challenge:
                return {"error": "missing challenge"}
            result = runint_prove(challenge)
            if result is None:
                return {"error": "runtime integrity not initialized or not bound"}
            signature, epoch = result
            return {
                "signature": signature,
                "epoch": epoch,
            }

        chute.add_api_route("/_rint", _handle_rint, methods=["POST"])

        # New envdump endpoints.
        import chutes.envdump as envdump

        chute.add_api_route("/_dump", envdump.handle_dump, methods=["POST"])
        chute.add_api_route("/_sig", envdump.handle_sig, methods=["POST"])
        chute.add_api_route("/_toca", envdump.handle_toca, methods=["POST"])
        chute.add_api_route("/_eslurp", envdump.handle_slurp, methods=["POST"])

        logger.success("Added all chutes internal endpoints.")

        # Job shutdown/kill endpoint.
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

        # Jobs can't be started until the full suite of validation tests run,
        # so we need to provide an endpoint for the validator to use to kick
        # it off.
        if job_id:
            job_task = None

            async def start_job_with_monitoring(**kwargs):
                nonlocal job_task
                ssh_process = None
                job_task = asyncio.create_task(job_obj.run(job_status_url=job_status_url, **kwargs))

                async def monitor_job():
                    try:
                        result = await job_task
                        logger.info(f"Job completed with result: {result}")
                    except Exception as e:
                        logger.error(f"Job failed with error: {e}")
                    finally:
                        logger.info("Job finished, shutting down server...")
                        if ssh_process:
                            try:
                                ssh_process.terminate()
                                await asyncio.sleep(0.5)
                                if ssh_process.poll() is None:
                                    ssh_process.kill()
                                logger.info("SSH server stopped")
                            except Exception as e:
                                logger.error(f"Error stopping SSH server: {e}")
                        server.should_exit = True

                # If the pod defines SSH access, enable it.
                if job_obj.ssh and job_data.get("_ssh_public_key"):
                    ssh_process = await setup_ssh_access(job_data["_ssh_public_key"])

                asyncio.create_task(monitor_job())

            await start_job_with_monitoring(**job_data)
            logger.info("Started job!")

            chute.add_api_route("/_shutdown", _shutdown, methods=["POST"])
            logger.info("Added shutdown endpoint")

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
    async def _logged_run():
        """
        Wrap the actual chute execution with the logging process, which is
        kept alive briefly after the main process terminates.
        """
        from chutes.entrypoint.logger import launch_server

        if not (dev or generate_inspecto_hash):
            miner()._miner_ss58 = miner_ss58
            miner()._validator_ss58 = validator_ss58
            miner()._keypair = Keypair(ss58_address=validator_ss58, crypto_type=KeypairType.SR25519)

        if generate_inspecto_hash:
            await _run_chute()
            return

        def run_logging_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                launch_server(
                    host=host or "0.0.0.0",
                    port=logging_port,
                    dev=dev,
                    certfile=certfile,
                    keyfile=keyfile,
                )
            )

        logging_thread = threading.Thread(target=run_logging_server, daemon=True)
        logging_thread.start()

        await asyncio.sleep(3)
        exception_raised = False
        try:
            await _run_chute()
        except Exception as exc:
            logger.error(
                f"Unexpected error executing _run_chute(): {str(exc)}\n{traceback.format_exc()}"
            )
            exception_raised = True
            await asyncio.sleep(60)
            raise
        finally:
            if not exception_raised:
                await asyncio.sleep(30)

    asyncio.run(_logged_run())
