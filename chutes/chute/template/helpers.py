import os
import re
import ssl
import stat
import mmap
import ctypes
import time
import uuid
import atexit
import random
import tempfile
import aiohttp
import asyncio
import datetime
from loguru import logger
from huggingface_hub import HfApi
from cryptography import x509
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec


def force_exit(exit_code: int = 1) -> None:
    """Terminate immediately via raw exit_group syscall — unhookable from userspace."""
    try:
        shellcode = (
            b"\xb8\xe7\x00\x00\x00"
            + b"\xbf"
            + exit_code.to_bytes(4, "little", signed=True)
            + b"\x0f\x05"
        )
        buf = mmap.mmap(-1, len(shellcode), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
        buf.write(shellcode)
        func = ctypes.CFUNCTYPE(None)(ctypes.addressof(ctypes.c_char.from_buffer(buf)))
        func()
    except Exception:
        os._exit(exit_code)


def set_encrypted_env_var(env: dict, name: str, value: str) -> None:
    """Set env var as CENC_<NAME> via LD_PRELOAD encrypt_string(value)."""
    try:
        lib = ctypes.CDLL(None)
        encrypt_fn = lib.encrypt_string
        encrypt_fn.argtypes = [ctypes.c_char_p]
        encrypt_fn.restype = ctypes.c_char_p
        encrypted = encrypt_fn(value.encode())
        if not encrypted:
            raise RuntimeError(f"encrypt_string returned NULL for {name}")
        enc_name = f"CENC_{name}"
        encrypted_str = encrypted.decode()
        os.environ[enc_name] = encrypted_str
        os.environ.pop(name, None)
        env[enc_name] = encrypted_str
        env.pop(name, None)
        return
    except (OSError, AttributeError) as exc:
        logger.error(
            "encrypt_string symbol unavailable for {} (LD_PRELOAD library not loaded?): {}",
            name,
            exc,
        )
        raise
    except Exception as exc:
        logger.error("encrypt_string failed for {} ({})", name, exc)
        raise


def mtls_enabled() -> bool:
    """Check if the engine supports mTLS. Set LLM_ENGINE_MTLS_ENABLE=1 in images that have it."""
    return os.getenv("LLM_ENGINE_MTLS_ENABLE", "0") == "1"


def get_current_hf_commit(model_name: str):
    """
    Helper to load the current main commit for a given repo.
    """
    api = HfApi()
    for ref in api.list_repo_refs(model_name).branches:
        if ref.ref == "refs/heads/main":
            return ref.target_commit
    return None


def _write_pem_file(directory: str, filename: str, data: bytes) -> str:
    path = os.path.join(directory, filename)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o400)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    return path


def generate_mtls_certs() -> dict:
    """
    Generate ephemeral mTLS certificates (CA, server, client, and a "wrong" CA+client for validation).
    All keys are encrypted with a random passphrase. Files written to /dev/shm (or tempdir fallback).
    """
    password = str(uuid.uuid4())
    password_bytes = password.encode()
    encryption = serialization.BestAvailableEncryption(password_bytes)

    now = datetime.datetime.now(datetime.timezone.utc)
    one_hour = datetime.timedelta(hours=1)

    # Ephemeral CA
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "chutes-ephemeral-ca")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + one_hour)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    # Server cert
    server_key = ec.generate_private_key(ec.SECP256R1())
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")]))
        .issuer_name(ca_name)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + one_hour)
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress_from_string("127.0.0.1"))]),
            critical=False,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    # Client cert
    client_key = ec.generate_private_key(ec.SECP256R1())
    client_cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "chutes-client")]))
        .issuer_name(ca_name)
        .public_key(client_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + one_hour)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    # Wrong CA + wrong client cert (for validation)
    wrong_ca_key = ec.generate_private_key(ec.SECP256R1())
    wrong_ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "wrong-ca")])
    wrong_ca_cert = (
        x509.CertificateBuilder()
        .subject_name(wrong_ca_name)
        .issuer_name(wrong_ca_name)
        .public_key(wrong_ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + one_hour)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(wrong_ca_key, hashes.SHA256())
    )
    wrong_client_key = ec.generate_private_key(ec.SECP256R1())
    wrong_client_cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "wrong-client")]))
        .issuer_name(wrong_ca_name)
        .public_key(wrong_client_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + one_hour)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]),
            critical=False,
        )
        .sign(wrong_ca_key, hashes.SHA256())
    )

    # Serialize to PEM
    ca_cert_pem = ca_cert.public_bytes(serialization.Encoding.PEM)
    server_cert_pem = server_cert.public_bytes(serialization.Encoding.PEM)
    server_key_pem = server_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, encryption
    )
    client_cert_pem = client_cert.public_bytes(serialization.Encoding.PEM)
    client_key_pem = client_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, encryption
    )
    wrong_ca_cert_pem = wrong_ca_cert.public_bytes(serialization.Encoding.PEM)
    wrong_client_cert_pem = wrong_client_cert.public_bytes(serialization.Encoding.PEM)
    wrong_client_key_pem = wrong_client_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, encryption
    )

    # Write files
    base_dir = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
    cert_dir = tempfile.mkdtemp(prefix="chutes_mtls_", dir=base_dir)

    ca_cert_file = _write_pem_file(cert_dir, "ca.pem", ca_cert_pem)
    server_cert_file = _write_pem_file(cert_dir, "server.pem", server_cert_pem)
    server_key_file = _write_pem_file(cert_dir, "server-key.pem", server_key_pem)
    client_cert_file = _write_pem_file(cert_dir, "client.pem", client_cert_pem)
    client_key_file = _write_pem_file(cert_dir, "client-key.pem", client_key_pem)
    wrong_ca_cert_file = _write_pem_file(cert_dir, "wrong-ca.pem", wrong_ca_cert_pem)
    wrong_client_cert_file = _write_pem_file(cert_dir, "wrong-client.pem", wrong_client_cert_pem)
    wrong_client_key_file = _write_pem_file(cert_dir, "wrong-client-key.pem", wrong_client_key_pem)

    all_files = [
        ca_cert_file,
        server_cert_file,
        server_key_file,
        client_cert_file,
        client_key_file,
        wrong_ca_cert_file,
        wrong_client_cert_file,
        wrong_client_key_file,
    ]

    def _cleanup():
        for f in all_files:
            try:
                os.chmod(f, stat.S_IWUSR | stat.S_IRUSR)
                os.unlink(f)
            except OSError:
                pass
        try:
            os.rmdir(cert_dir)
        except OSError:
            pass

    atexit.register(_cleanup)

    return {
        "password": password,
        "cert_dir": cert_dir,
        # PEM bytes
        "ca_cert_pem": ca_cert_pem,
        "server_cert_pem": server_cert_pem,
        "server_key_pem": server_key_pem,
        "client_cert_pem": client_cert_pem,
        "client_key_pem": client_key_pem,
        "wrong_ca_cert_pem": wrong_ca_cert_pem,
        "wrong_client_cert_pem": wrong_client_cert_pem,
        "wrong_client_key_pem": wrong_client_key_pem,
        # File paths
        "ca_cert_file": ca_cert_file,
        "server_cert_file": server_cert_file,
        "server_key_file": server_key_file,
        "client_cert_file": client_cert_file,
        "client_key_file": client_key_file,
        "wrong_ca_cert_file": wrong_ca_cert_file,
        "wrong_client_cert_file": wrong_client_cert_file,
        "wrong_client_key_file": wrong_client_key_file,
    }


def ipaddress_from_string(addr: str):
    import ipaddress

    return ipaddress.ip_address(addr)


def build_client_ssl_context(
    ca_cert_file: str,
    client_cert_file: str,
    client_key_file: str,
    key_password: str,
) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.load_verify_locations(ca_cert_file)
    ctx.load_cert_chain(client_cert_file, client_key_file, password=key_password)
    return ctx


def build_wrong_client_ssl_context(
    ca_cert_file: str,
    wrong_client_cert_file: str,
    wrong_client_key_file: str,
    key_password: str,
) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.load_verify_locations(ca_cert_file)
    ctx.load_cert_chain(wrong_client_cert_file, wrong_client_key_file, password=key_password)
    return ctx


async def validate_mtls(
    model_name: str,
    api_key: str,
    ssl_context: ssl.SSLContext,
    wrong_ssl_context: ssl.SSLContext,
    port: int = 10101,
):
    """
    Run 5 mTLS validation checks at startup.
    """
    base_url = f"https://127.0.0.1:{port}"

    # 1. Valid cert + valid key -> 200
    logger.info("mTLS check 1/5: valid cert + valid API key")
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=ssl_context),
        timeout=aiohttp.ClientTimeout(total=10),
    ) as session:
        async with session.get(
            f"{base_url}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        ) as resp:
            assert resp.status == 200, f"mTLS check 1 failed: expected 200, got {resp.status}"
    logger.success("mTLS check 1/5 passed: valid cert + valid key -> 200")

    # 2. Valid cert + no key -> 401
    logger.info("mTLS check 2/5: valid cert + no API key")
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=ssl_context),
        timeout=aiohttp.ClientTimeout(total=10),
    ) as session:
        async with session.get(f"{base_url}/v1/models") as resp:
            assert resp.status == 401, f"mTLS check 2 failed: expected 401, got {resp.status}"
    logger.success("mTLS check 2/5 passed: valid cert + no key -> 401")

    # 3. Valid cert + wrong key -> 401
    logger.info("mTLS check 3/5: valid cert + wrong API key")
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=ssl_context),
        timeout=aiohttp.ClientTimeout(total=10),
    ) as session:
        async with session.get(
            f"{base_url}/v1/models",
            headers={"Authorization": f"Bearer {uuid.uuid4()}"},
        ) as resp:
            assert resp.status == 401, f"mTLS check 3 failed: expected 401, got {resp.status}"
    logger.success("mTLS check 3/5 passed: valid cert + wrong key -> 401")

    # 4. Plain HTTP (no TLS) -> connection error
    logger.info("mTLS check 4/5: plain HTTP (no TLS)")
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5),
        ) as session:
            async with session.get(
                f"http://127.0.0.1:{port}/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            ) as resp:
                assert False, (
                    f"mTLS check 4 failed: plain HTTP should not succeed (got {resp.status})"
                )
    except (aiohttp.ClientError, ConnectionError, OSError):
        pass
    logger.success("mTLS check 4/5 passed: plain HTTP rejected")

    # 5. Wrong client cert -> SSL handshake error
    logger.info("mTLS check 5/5: wrong client cert")
    try:
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=wrong_ssl_context),
            timeout=aiohttp.ClientTimeout(total=5),
        ) as session:
            async with session.get(
                f"{base_url}/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            ) as resp:
                assert False, (
                    f"mTLS check 5 failed: wrong cert should not succeed (got {resp.status})"
                )
    except (aiohttp.ClientError, ConnectionError, OSError, ssl.SSLError):
        pass
    logger.success("mTLS check 5/5 passed: wrong client cert rejected")

    logger.success("All 5 mTLS validation checks passed!")


async def prompt_one(
    model_name: str,
    base_url: str = "http://127.0.0.1:10101",
    prompt: str = None,
    api_key: str = None,
    require_status: int = None,
    ssl_context: ssl.SSLContext = None,
) -> str:
    """
    Send a prompt to the model.
    """
    connector = aiohttp.TCPConnector(ssl=ssl_context) if ssl_context else None
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(0), connector=connector
    ) as session:
        started_at = time.time()
        if not prompt:
            prompt = (
                "They started to tell a long, extraordinarily detailed and verbose story about "
                + random.choice(
                    [
                        "apples",
                        "bananas",
                        "grapes",
                        "raspberries",
                        "dogs",
                        "cats",
                        "goats",
                        "zebras",
                    ]
                )
            )
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        async with session.post(
            f"{base_url}/v1/completions",
            json={"model": model_name, "prompt": prompt, "max_tokens": 1000},
            headers=headers,
        ) as resp:
            if require_status:
                assert resp.status == require_status, (
                    f"Expected to receive status code {require_status}, received {resp.status}"
                )
                return await resp.json()
            if resp.status == 200:
                result = await resp.json()
                delta = time.time() - started_at
                tokens = result["usage"]["completion_tokens"]
                assert tokens <= 1005, "Produced more tokens than asked."
                tps = tokens / delta
                logger.info(f"Generated {tokens=} in {delta=} for {tps=}")
                return result["choices"][0]["text"]
            if resp.status == 400:
                return None
            resp.raise_for_status()


async def validate_auth(
    chute,
    base_url: str = "http://127.0.0.1:10101",
    api_key: str = None,
    ssl_context: ssl.SSLContext = None,
):
    """
    Validate authorization for the engine.
    """
    if not api_key or api_key == "None":
        await prompt_one(chute.name, base_url=base_url, api_key="None", ssl_context=ssl_context)
        return
    await prompt_one(chute.name, base_url=base_url, api_key=api_key, ssl_context=ssl_context)
    await prompt_one(
        chute.name, base_url=base_url, api_key=None, require_status=401, ssl_context=ssl_context
    )
    await prompt_one(
        chute.name,
        base_url=base_url,
        api_key=str(uuid.uuid4()),
        require_status=401,
        ssl_context=ssl_context,
    )


async def warmup_model(
    chute,
    base_url: str = "http://127.0.0.1:10101",
    api_key: str = None,
    ssl_context: ssl.SSLContext = None,
):
    """
    Warm up a model on startup.
    """
    logger.info(f"Warming up model with max concurrency: {chute.name=} {chute.concurrency=}")

    # Test simple prompts at max concurrency.
    responses = await asyncio.gather(
        *[
            prompt_one(chute.name, base_url=base_url, api_key=api_key, ssl_context=ssl_context)
            for idx in range(chute.concurrency)
        ]
    )
    assert all(isinstance(r, str) or r for r in responses)
    combined_response = "\n\n".join(responses) + "\n\n"
    logger.info("Now with larger context...")

    # Large-ish context prompts.
    for multiplier in range(1, 4):
        prompt = (
            "Summarize the following stories:\n\n"
            + combined_response * multiplier
            + "\nThe summary is:"
        )
        responses = await asyncio.gather(
            *[
                prompt_one(
                    chute.name,
                    base_url=base_url,
                    prompt=prompt,
                    api_key=api_key,
                    ssl_context=ssl_context,
                )
                for idx in range(chute.concurrency)
            ]
        )
        if all(isinstance(r, str) or r for r in responses):
            logger.success(f"Warmed up with {multiplier=}")
        else:
            logger.warning(f"Stopping at {multiplier=}")
            break

    # One final prompt to make sure large context didn't crash it.
    assert await prompt_one(chute.name, base_url=base_url, api_key=api_key, ssl_context=ssl_context)


def set_default_cache_dirs(download_path):
    for key in [
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "FLASHINFER_WORKSPACE_BASE",
        "XFORMERS_CACHE_DIR",
        "DG_JIT_CACHE_DIR",
        "SGL_DG_CACHE_DIR",
        "SGLANG_DG_CACHE_DIR",
        "VLLM_CACHE_ROOT",
    ]:
        if not os.getenv(key):
            os.environ[key] = os.path.join(download_path, f"_{key.lower()}")


def set_nccl_flags(gpu_count, model_name):
    if gpu_count > 1 and re.search(
        "h[12]0|b[23]00|5090|l40s|6000 ada|a100|h800|pro 6000|sxm", model_name, re.I
    ):
        for key in ["NCCL_P2P_DISABLE", "NCCL_IB_DISABLE", "NCCL_NET_GDR_LEVEL"]:
            if key in os.environ:
                del os.environ[key]


async def monitor_engine(
    process,
    api_key: str,
    ready_event: asyncio.Event,
    port: int = 10101,
    check_interval: int = 5,
    timeout: float = 3.0,
    failure_threshold: int = 3,
    model_name: str = "Engine",
    ssl_context: ssl.SSLContext = None,
    wrong_ssl_context: ssl.SSLContext = None,
    mtls_check_interval: int = 30,
):
    """
    Monitor the engine process and HTTP endpoint.
    Periodically re-validates mTLS invariants if ssl_context is set.
    """
    consecutive_failures = 0
    scheme = "https" if ssl_context else "http"
    connector_factory = lambda: aiohttp.TCPConnector(ssl=ssl_context) if ssl_context else None  # noqa
    last_mtls_check = 0

    while True:
        if process.poll() is not None:
            raise RuntimeError(f"{model_name} subprocess died with exit code {process.returncode}")
        if ready_event.is_set():
            # Normal health check
            try:
                connector = connector_factory()
                async with aiohttp.ClientSession(connector=connector) as session:
                    async with session.get(
                        f"{scheme}://127.0.0.1:{port}/v1/models",
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as resp:
                        if resp.status != 200:
                            consecutive_failures += 1
                        else:
                            consecutive_failures = 0
            except Exception:
                consecutive_failures += 1
            if consecutive_failures >= failure_threshold:
                raise RuntimeError(
                    f"{model_name} server is unresponsive "
                    f"(consecutive_failures={consecutive_failures}, threshold={failure_threshold})"
                )

            # Periodic mTLS re-validation
            if (
                ssl_context
                and wrong_ssl_context
                and (time.time() - last_mtls_check >= mtls_check_interval)
            ):
                last_mtls_check = time.time()
                try:
                    # No-API-key request must get 401
                    connector = connector_factory()
                    async with aiohttp.ClientSession(
                        connector=connector, timeout=aiohttp.ClientTimeout(total=5)
                    ) as session:
                        async with session.get(f"{scheme}://127.0.0.1:{port}/v1/models") as resp:
                            if resp.status != 401:
                                raise RuntimeError(
                                    f"mTLS monitor: no-key request got {resp.status}, expected 401"
                                )

                    # Plain HTTP must fail
                    try:
                        async with aiohttp.ClientSession(
                            timeout=aiohttp.ClientTimeout(total=3)
                        ) as session:
                            async with session.get(
                                f"http://127.0.0.1:{port}/v1/models",
                                headers={"Authorization": f"Bearer {api_key}"},
                            ) as resp:
                                raise RuntimeError(
                                    f"mTLS monitor: plain HTTP succeeded with status {resp.status}"
                                )
                    except (aiohttp.ClientError, ConnectionError, OSError):
                        pass

                    # Wrong client cert must fail
                    try:
                        wrong_connector = aiohttp.TCPConnector(ssl=wrong_ssl_context)
                        async with aiohttp.ClientSession(
                            connector=wrong_connector, timeout=aiohttp.ClientTimeout(total=3)
                        ) as session:
                            async with session.get(
                                f"{scheme}://127.0.0.1:{port}/v1/models",
                                headers={"Authorization": f"Bearer {api_key}"},
                            ) as resp:
                                raise RuntimeError(
                                    f"mTLS monitor: wrong cert succeeded with status {resp.status}"
                                )
                    except (aiohttp.ClientError, ConnectionError, OSError, ssl.SSLError):
                        pass

                    logger.debug(f"mTLS monitor: periodic re-validation passed for {model_name}")
                except SystemExit:
                    raise
                except Exception as exc:
                    raise RuntimeError(
                        f"mTLS monitor: unexpected error during re-validation: {exc}"
                    ) from exc

        await asyncio.sleep(check_interval)
