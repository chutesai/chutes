from abc import abstractmethod
import asyncio
import base64
from contextlib import asynccontextmanager
from functools import lru_cache
import json
import os
import ssl
import socket
import threading
from urllib.parse import urljoin, urlparse

import aiohttp
from loguru import logger
from fastapi import FastAPI, Request, HTTPException, status
from uvicorn import Config, Server

from chutes.entrypoint._shared import encrypt_response, get_launch_token, get_launch_token_data, is_tee_env, miner


# TEE endpoint constants
TEE_VERIFICATION_ENDPOINT = "/verify"  # Validator only: uses nonce from fetch_symmetric_key
TEE_EVIDENCE_RUNTIME_ENDPOINT = "/evidence"  # Third parties: accept nonce from request

# Global nonce storage for TEE verification (validator flow only)
# Set during fetch_symmetric_key; used only by /verify to prove same instance
_evidence_nonce: str | None = None
_evidence_nonce_locked: bool = False



@asynccontextmanager
async def _use_evidence_nonce(validator_url: str):
    """
    Context manager for TEE evidence nonce lifecycle. Fetches nonce from validator,
    makes it available for the duration, and automatically clears it when done.
    
    Args:
        validator_url: The base URL of the validator
    
    Yields:
        The nonce value
        
    Raises:
        RuntimeError: If nonce is already locked (multiple verification processes detected)
    """
    global _evidence_nonce, _evidence_nonce_locked
    
    if _evidence_nonce_locked:
        raise RuntimeError(
            "TEE nonce already locked. Only one verification process should be running."
        )
    
    # Fetch nonce from validator
    url = urljoin(validator_url, "/instances/nonce")
    async with aiohttp.ClientSession(raise_for_status=True) as http_session:
        async with http_session.get(url) as resp:
            logger.success(f"Successfully initiated attestation with validator {validator_url}.")
            nonce = await resp.json()
    
    # Set the nonce and lock
    _evidence_nonce = nonce
    _evidence_nonce_locked = True
    
    try:
        yield nonce
    finally:
        # Clean up nonce state
        _evidence_nonce = None
        _evidence_nonce_locked = False


def _get_evidence_nonce() -> str | None:
    """Get the current evidence nonce (used by evidence endpoint)."""
    return _evidence_nonce


@asynccontextmanager
async def _attestation_session():
    """
    Creates an aiohttp session configured for the attestation service.

    SSL verification is disabled because certificate authenticity is verified
    through TDX quotes, which include a hash of the service's public key.
    """
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    connector = aiohttp.TCPConnector(ssl=ssl_context)

    async with aiohttp.ClientSession(connector=connector, raise_for_status=True) as session:
        yield session


class GpuVerifier:
    def __init__(self, body: dict):
        self._token = get_launch_token()
        token_data = get_launch_token_data()
        self._url = token_data.get("url")
        self._symmetric_key: bytes | None = None
        self._dummy_threads: list[threading.Thread] = []
        self._body = body

    @classmethod
    def create(cls, body: dict) -> "GpuVerifier":
        if is_tee_env():
            return TeeGpuVerifier(body)
        else:
            return GravalGpuVerifier(body)

    def _start_dummy_sockets(self):
        if not self._symmetric_key:
            raise RuntimeError("Cannot start dummy sockets without symmetric key.")
        for port_map in self._body.get("port_mappings", []):
            if port_map.get("default"):
                continue
            self._dummy_threads.append(start_dummy_socket(port_map, self._symmetric_key))

    async def verify(self):
        """
        Execute full verification flow and spin up dummy sockets for port validation.
        """
        await self.fetch_symmetric_key()
        self._start_dummy_sockets()
        response = await self.finalize_verification()
        return response

    @abstractmethod
    async def fetch_symmetric_key(self) -> bytes: ...

    @abstractmethod
    async def finalize_verification(self) -> dict: ...


class GravalGpuVerifier(GpuVerifier):
    def __init__(self, body: dict):
        super().__init__(body)
        self._init_params: dict | None = None
        self._proofs = None
        self._response_plaintext: str | None = None

    async def fetch_symmetric_key(self):
        # Fetch the challenges.
        token = self._token
        url = urljoin(self._url + "/", "graval")

        self._body["gpus"] = self.gather_gpus()
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            logger.info(f"Collected all environment data, submitting to validator: {url}")
            async with session.post(url, headers={"Authorization": token}, json=self._body) as resp:
                self._init_params = await resp.json()
                logger.success(
                    f"Successfully fetched initialization params: {self._init_params=}"
                )

        # First, we initialize graval on all GPUs from the provided seed.
        miner()._graval_seed = self._init_params["seed"]
        iterations = self._init_params.get("iterations", 1)
        logger.info(f"Generating proofs from seed={miner()._graval_seed}")
        self._proofs = miner().prove(miner()._graval_seed, iterations=iterations)

        # Use GraVal to extract the symmetric key from the challenge.
        sym_key = self._init_params["symmetric_key"]
        bytes_ = base64.b64decode(sym_key["ciphertext"])
        iv = bytes_[:16]
        cipher = bytes_[16:]
        logger.info("Decrypting payload via proof challenge matrix...")
        device_index = [
            miner().get_device_info(i)["uuid"] for i in range(miner()._device_count)
        ].index(sym_key["uuid"])
        self._symmetric_key = bytes.fromhex(
            miner().decrypt(
                self._init_params["seed"],
                cipher,
                iv,
                len(cipher),
                device_index,
            )
        )

        # Now, we can respond to the URL by encrypting a payload with the symmetric key and sending it back.
        self._response_plaintext = sym_key["response_plaintext"]

    async def finalize_verification(self):

        token = self._token
        url = urljoin(self._url + "/", "graval")
        plaintext = self._response_plaintext

        async with aiohttp.ClientSession(raise_for_status=True) as session:
            new_iv, response_cipher = encrypt_response(self._symmetric_key, plaintext)
            logger.success(
                f"Completed PoVW challenge, sending back: {plaintext=} "
                f"as {response_cipher=} where iv={new_iv.hex()}"
            )

            # Post the response to the challenge, which returns job data (if any).
            async with session.put(
                url,
                headers={"Authorization": token},
                json={
                    "response": response_cipher,
                    "iv": new_iv.hex(),
                    "proof": self._proofs,
                },
                raise_for_status=False,
            ) as resp:
                if resp.ok:
                    logger.success("Successfully negotiated challenge response!")
                    response = await resp.json()
                    # validator_pubkey is returned in POST response, needed for ECDH session key
                    if "validator_pubkey" in self._init_params:
                        response["validator_pubkey"] = self._init_params["validator_pubkey"]
                    return response
                else:
                    # log down the reason of failure to the challenge
                    detail = await resp.text(encoding="utf-8", errors="replace")
                    logger.error(f"Failed: {resp.reason} ({resp.status}) {detail}")
                    resp.raise_for_status()

    def gather_gpus(self):
        gpus = []
        for idx in range(miner()._device_count):
            gpus.append(miner().get_device_info(idx))

        return gpus


def _parse_evidence_port() -> int:
    """Parse TEE evidence port from env (default 8002)."""
    evidence_port = os.getenv("CHUTES_TEE_EVIDENCE_PORT", "8002")
    if not evidence_port.isdigit():
        raise ValueError(f"CHUTES_TEE_EVIDENCE_PORT must be a valid port number, got: {evidence_port}")
    return int(evidence_port)


class TeeEvidenceService:
    """
    Singleton TEE attestation evidence server. Serves GET /_tee_evidence for the
    validator (during verification) and for third-party runtime verification.
    """

    _instance: "TeeEvidenceService | None" = None

    def __new__(cls) -> "TeeEvidenceService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_port"):
            self._port: int | None = None
            self._server: Server | None = None
            self._task: asyncio.Task | None = None

    async def start(self) -> dict:
        """
        Start the evidence server. Idempotent: if already started, returns the same
        port mapping without starting a second server.

        Returns:
            Port mapping dict to append to port_mappings in the run entrypoint, e.g.:
            {"proto": "tcp", "internal_port": 8002, "external_port": 8002, "default": False}
        """
        if self._task is not None:
            return self._port_mapping()
        self._port = _parse_evidence_port()
        evidence_app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        evidence_app.add_api_route(TEE_VERIFICATION_ENDPOINT, self._get_verification_evidence, methods=["GET"])
        evidence_app.add_api_route(TEE_EVIDENCE_RUNTIME_ENDPOINT, self._get_runtime_evidence, methods=["GET"])
        config = Config(
            app=evidence_app,
            host="0.0.0.0",
            port=self._port,
            limit_concurrency=1000,
            log_level="warning",
        )
        self._server = Server(config)
        self._task = asyncio.create_task(self._server.serve())
        await asyncio.sleep(0.5)
        logger.info(f"Started TEE evidence server on port {self._port}")
        return self._port_mapping()

    def _port_mapping(self) -> dict:
        """Port mapping dict for this service (internal_port == external_port == configured port)."""
        return {
            "proto": "tcp",
            "internal_port": self._port,
            "external_port": self._port,
            "default": False,
        }

    async def stop(self) -> None:
        """Stop the evidence server. No-op if not started."""
        if self._server is None or self._task is None:
            return
        logger.info("Stopping TEE evidence server...")
        self._server.should_exit = True
        try:
            await asyncio.wait_for(self._task, timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning("TEE evidence server did not stop within timeout")
        except Exception as e:
            logger.warning(f"Error stopping TEE evidence server: {e}")
        finally:
            self._server = None
            self._task = None
            self._port = None

    async def _fetch_evidence(self, nonce: str) -> dict:
        """Request evidence from the attestation service for the given nonce."""
        url = "https://attestation-service-internal.attestation-system.svc.cluster.local.:8443/server/attest"
        params = {
            "nonce": nonce,
            "gpu_ids": os.environ.get("CHUTES_NVIDIA_DEVICES"),
        }
        async with _attestation_session() as http_session:
            async with http_session.get(url, params=params) as resp:
                evidence = await resp.json()
                return {"evidence": evidence, "nonce": nonce}

    async def _get_verification_evidence(self, request: Request):
        """
        TEE evidence for initial verification only. Called by the validator during
        fetch_symmetric_key (Phase 2). Uses the nonce retrieved from the validator
        when we started the process so we can prove we are the same instance.
        """
        nonce = _get_evidence_nonce()
        if nonce is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No nonce found. Attestation not initiated. Use /evidence?nonce=... for third-party evidence.",
            )
        try:
            logger.success("Retrieved attestation evidence for validator verification.")
            return await self._fetch_evidence(nonce)
        except Exception as e:
            logger.error(f"Failed to fetch TEE evidence: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch evidence: {str(e)}",
            )

    async def _get_runtime_evidence(self, request: Request):
        """
        TEE evidence for third-party runtime verification. Caller supplies a nonce
        via query param ?nonce=...; we request evidence bound to that nonce and return it.
        """
        nonce = request.query_params.get("nonce")
        if not nonce or not nonce.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query parameter 'nonce' is required for runtime evidence.",
            )
        nonce = nonce.strip()
        try:
            logger.success("Retrieved attestation evidence.")
            return await self._fetch_evidence(nonce)
        except Exception as e:
            logger.error(f"Failed to fetch TEE evidence: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch evidence: {str(e)}",
            )


class TeeGpuVerifier(GpuVerifier):
    @property
    @lru_cache(maxsize=1)
    def validator_url(self) -> str:
        parsed = urlparse(self._url)
        return f"{parsed.scheme}://{parsed.netloc}"
    
    @property
    @lru_cache(maxsize=1)
    def deployment_id(self) -> str:
        hostname = os.environ.get("HOSTNAME")
        # Pod name format: chute-{deployment_id}-{k8s-suffix}
        # Service name format: chute-service-{deployment_id}
        # We need to extract just the deployment_id by removing the prefix and k8s suffix
        if not hostname.startswith("chute-"):
            raise ValueError(f"Unexpected hostname format: {hostname}")
        # Remove 'chute-' prefix
        _deployment_id = hostname[6:]  # len("chute-") = 6
        # Remove k8s-generated pod suffix (everything after the last hyphen)
        _deployment_id = _deployment_id.rsplit("-", 1)[0]
        return _deployment_id

    async def fetch_symmetric_key(self):
        """
        TEE verification flow (3 phases):
        Phase 1: Get nonce from validator (evidence server is already running at chute startup)
        Phase 2: Validator calls our /_tee_evidence endpoint, we fetch evidence from attestation service,
                 validator verifies and returns symmetric key
        Phase 3: Start dummy sockets and finalize verification (handled by base class)
        """
        token = self._token
        
        # Gather GPUs before sending request
        gpus = await self.gather_gpus()
        # Append /tee to instance path; urljoin(base, "/tee") replaces path, urljoin(base, "tee") replaces last segment
        url = urljoin(self._url + "/", "tee")
        
        async with _use_evidence_nonce(self.validator_url) as _nonce:
            async with aiohttp.ClientSession(raise_for_status=True) as session:
                headers = {
                    "Authorization": token,
                    "X-Chutes-Nonce": _nonce
                }
                _body = self._body.copy()
                _body["deployment_id"] = self.deployment_id
                _body["gpus"] = gpus
                logger.info(f"Requesting verification from validator: {url}")
                async with session.post(url, headers=headers, json=_body) as resp:
                    data = await resp.json()
                    self._symmetric_key = bytes.fromhex(data["symmetric_key"])
                    self._validator_pubkey = data["validator_pubkey"] if "validator_pubkey" in data else None
                    logger.success("Successfully received symmetric key from validator")

    async def finalize_verification(self):
        """
        Send final verification with port mappings (same as GraVal flow).
        """
        if not self._symmetric_key:
            raise RuntimeError("Symmetric key must be fetched before finalizing verification.")

        token = self._token
        # Append /tee to instance path; urljoin(base, "/tee") would replace path with /tee (RFC 3986)
        url = urljoin(self._url + "/", "tee")

        async with aiohttp.ClientSession(raise_for_status=False) as session:
            logger.info("Sending final verification request.")
            
            async with session.put(
                url,
                headers={"Authorization": token},
                json={},
                raise_for_status=False,
            ) as resp:
                if resp.ok:
                    logger.success("Successfully completed final verification!")
                    response = await resp.json()
                    if self._validator_pubkey:
                        response["validator_pubkey"] = self._validator_pubkey
                    return response
                else:
                    detail = await resp.text(encoding="utf-8", errors="replace")
                    logger.error(f"Final verification failed: {resp.reason} ({resp.status}) {detail}")
                    resp.raise_for_status()

    async def gather_gpus(self):
        devices = []
        async with _attestation_session() as http_session:
            url = "https://attestation-service-internal.attestation-system.svc.cluster.local.:8443/server/devices"
            params = {"gpu_ids": os.environ.get("CHUTES_NVIDIA_DEVICES")}
            async with http_session.get(url=url, params=params) as resp:
                devices = await resp.json()
                logger.success(f"Retrieved {len(devices)} GPUs.")

        return devices


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


