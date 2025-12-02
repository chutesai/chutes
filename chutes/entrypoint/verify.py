from abc import abstractmethod
import base64
from contextlib import asynccontextmanager
import json
import os
import ssl
import socket
import threading
from urllib.parse import urljoin, urlparse

import aiohttp
from loguru import logger

from chutes.entrypoint._shared import encrypt_response, get_launch_token, is_tee_env, miner


class GpuVerifier:
    def __init__(self, url, body):
        self._token = get_launch_token()
        self._url = url
        self._body = body
        self._symmetric_key: bytes | None = None
        self._dummy_threads: list[threading.Thread] = []

    @classmethod
    def create(cls, url, body) -> "GpuVerifier":
        if is_tee_env():
            return TeeGpuVerifier(url, body)
        else:
            return GravalGpuVerifier(url, body)

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
        symmetric_key = await self.fetch_symmetric_key()
        self._start_dummy_sockets()
        response = await self.finalize_verification()
        return symmetric_key, response

    @abstractmethod
    async def fetch_symmetric_key(self) -> bytes: ...

    @abstractmethod
    async def finalize_verification(self) -> dict: ...


class GravalGpuVerifier(GpuVerifier):
    def __init__(self, url, body):
        super().__init__(url, body)
        self._init_params: dict | None = None
        self._proofs = None
        self._response_plaintext: str | None = None

    async def fetch_symmetric_key(self):
        # Fetch the challenges.
        token = self._token
        url = self._url
        body = self._body

        body["gpus"] = self.gather_gpus()
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            logger.info(f"Collected all environment data, submitting to validator: {url}")
            async with session.post(url, headers={"Authorization": token}, json=body) as resp:
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
        return self._symmetric_key

    async def finalize_verification(self):

        token = self._token
        url = self._url
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


class TeeGpuVerifier(GpuVerifier):

    @asynccontextmanager
    async def _attestation_session(self):
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

    async def _get_nonce(self):
        parsed = urlparse(self._url)

        # Get just the scheme + netloc (host)
        validator_url = f"{parsed.scheme}://{parsed.netloc}"
        url = urljoin(validator_url, "/servers/nonce")
        async with aiohttp.ClientSession(raise_for_status=True) as http_session:
            async with http_session.get(url) as resp:
                logger.success("Successfully retrieved nonce for attestation evidence.")
                data = await resp.json()
                return data["nonce"]

    async def _get_gpu_evidence(self):
        """ """
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        connector = aiohttp.TCPConnector(ssl=ssl_context)

        url = "https://attestation-service-internal.attestation-system.svc.cluster.local.:8443/server/nvtrust/evidence"
        nonce = await self._get_nonce()
        params = {
            "name": os.environ.get("HOSTNAME"),
            "nonce": nonce,
            "gpu_ids": os.environ.get("CHUTES_NVIDIA_DEVICES"),
        }
        async with aiohttp.ClientSession(
            connector=connector, raise_for_status=True
        ) as http_session:
            async with http_session.get(url, params=params) as resp:
                logger.success("Successfully retrieved attestation evidence.")
                evidence = json.loads(await resp.json())
                return nonce, evidence

    async def fetch_symmetric_key(self):
        token = self._token
        url = urljoin(f"{self._url}/", "attest")
        body = self._body

        body["gpus"] = await self.gather_gpus()
        _nonce, evidence = await self._get_gpu_evidence()
        body["gpu_evidence"] = evidence
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            headers = {"Authorization": token, "X-Chutes-Nonce": _nonce}
            logger.info(f"Collected all environment data, submitting to validator for symmetric key: {url}")
            async with session.post(url, headers=headers, json=body) as resp:
                logger.info("Successfully fetched symmetric key and attestation response.")
                data = await resp.json()
                self._symmetric_key = bytes.fromhex(data["symmetric_key"])
                return self._symmetric_key

    async def finalize_verification(self):
        if not self._symmetric_key:
            raise RuntimeError("Symmetric key must be fetched before finalizing verification.")

        token = self._token
        url = urljoin(f"{self._url}/", "attest")
        headers = {"Authorization": token}

        async with aiohttp.ClientSession(raise_for_status=False) as session:
            logger.info("Requesting validator to verify ports with initialized symmetric key.")
            async with session.put(
                url,
                headers=headers,
                json={"port_mappings": self._body.get("port_mappings", [])},
            ) as resp:
                if resp.ok:
                    if resp.content_type == "application/json":
                        return await resp.json()
                    logger.success("Ports verified successfully.")
                    return {}
                if resp.status in (404, 405):
                    logger.warning(
                        f"Port verification endpoint not available ({resp.status}); validator must include final response."
                    )
                    return {}
                detail = await resp.text(encoding="utf-8", errors="replace")
                logger.error(f"Port verification failed: {resp.reason} ({resp.status}) {detail}")
                resp.raise_for_status()

    async def gather_gpus(self):
        devices = []
        async with self._attestation_session() as http_session:
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
