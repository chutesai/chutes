from abc import abstractmethod
import base64
from contextlib import asynccontextmanager
import json
import os
import ssl
from urllib.parse import urljoin, urlparse

import aiohttp
from loguru import logger

from chutes.entrypoint._shared import encrypt_response, get_launch_token, is_tee_env, miner


class GpuVerifier:
    def __init__(self, url, body):
        self._token = get_launch_token()
        self._url = url
        self._body = body

    @classmethod
    def create(cls, url, body) -> "GpuVerifier":
        if is_tee_env():
            return TeeGpuVerifier(url, body)
        else:
            return GravalGpuVerifier(url, body)

    @abstractmethod
    async def verify_devices(self): ...


class GravalGpuVerifier(GpuVerifier):
    async def verify_devices(self):
        # Fetch the challenges.
        token = self._token
        url = self._url
        body = self._body

        body["gpus"] = self.gather_gpus()
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            logger.info(f"Collected all environment data, submitting to validator: {url}")
            async with session.post(url, headers={"Authorization": token}, json=body) as resp:
                init_params = await resp.json()
                logger.success(f"Successfully fetched initialization params: {init_params=}")

                # First, we initialize graval on all GPUs from the provided seed.
                miner()._graval_seed = init_params["seed"]
                iterations = init_params.get("iterations", 1)
                logger.info(f"Generating proofs from seed={miner()._graval_seed}")
                proofs = miner().prove(miner()._graval_seed, iterations=iterations)

                # Use GraVal to extract the symmetric key from the challenge.
                sym_key = init_params["symmetric_key"]
                bytes_ = base64.b64decode(sym_key["ciphertext"])
                iv = bytes_[:16]
                cipher = bytes_[16:]
                logger.info("Decrypting payload via proof challenge matrix...")
                device_index = [
                    miner().get_device_info(i)["uuid"] for i in range(miner()._device_count)
                ].index(sym_key["uuid"])
                symmetric_key = bytes.fromhex(
                    miner().decrypt(
                        init_params["seed"],
                        cipher,
                        iv,
                        len(cipher),
                        device_index,
                    )
                )

                # Now, we can respond to the URL by encrypting a payload with the symmetric key and sending it back.
                plaintext = sym_key["response_plaintext"]
                new_iv, response_cipher = encrypt_response(symmetric_key, plaintext)
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
                        "proof": proofs,
                    },
                    raise_for_status=False,
                ) as resp:
                    if resp.ok:
                        logger.success("Successfully negotiated challenge response!")
                        response = await resp.json()
                        # validator_pubkey is returned in POST response, needed for ECDH session key
                        if "validator_pubkey" in init_params:
                            response["validator_pubkey"] = init_params["validator_pubkey"]
                        return symmetric_key, response
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

    async def verify_devices(self):
        token = self._token
        url = urljoin(f"{self._url}/", "attest")
        body = self._body

        body["gpus"] = await self.gather_gpus()
        nonce, evidence = await self._get_gpu_evidence()
        body["gpu_evidence"] = evidence
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            headers = {"Authorization": token, "X-Chutes-Nonce": nonce}
            logger.info(f"Collected all environment data, submitting to validator: {url}")
            async with session.post(url, headers=headers, json=body) as resp:
                logger.info("Successfully verified instance with validator.")
                data = await resp.json()
                symmetric_key = bytes.fromhex(data["symmetric_key"])
                return symmetric_key, data

    async def gather_gpus(self):
        devices = []
        async with self._attestation_session() as http_session:
            url = "https://attestation-service-internal.attestation-system.svc.cluster.local.:8443/server/devices"
            params = {"gpu_ids": os.environ.get("CHUTES_NVIDIA_DEVICES")}
            async with http_session.get(url=url, params=params) as resp:
                devices = await resp.json()
                logger.success(f"Retrieved {len(devices)} GPUs.")

        return devices
