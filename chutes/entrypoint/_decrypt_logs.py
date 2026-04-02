"""Client-side decryption for encrypted startup logs.

Mirrors the ECIES scheme in the API's encrypted_logs/crypto.py:
  ECDH(user_private_scalar, ephemeral_pubkey) → HKDF → AES-256-GCM decrypt.
"""

import pybase64
import rbcl
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

HKDF_INFO = b"chutes-encrypted-logs-v1"


def _derive_symmetric_key(
    shared_secret: bytes,
    ephemeral_pubkey: bytes,
    user_pubkey: bytes,
) -> bytes:
    salt = ephemeral_pubkey + user_pubkey
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=HKDF_INFO,
    )
    return hkdf.derive(shared_secret)


def decrypt_chunk(
    encrypted: bytes,
    private_key_scalar: bytes,
    user_pubkey: bytes,
    ephemeral_pubkey: bytes,
) -> bytes:
    """Decrypt a single encrypted log chunk."""
    shared_secret = rbcl.crypto_scalarmult_ristretto255(private_key_scalar, ephemeral_pubkey)
    if shared_secret is None:
        raise ValueError("ECDH failed: invalid point or identity result")

    aes_key = _derive_symmetric_key(shared_secret, ephemeral_pubkey, user_pubkey)

    nonce = encrypted[:12]
    ciphertext = encrypted[12:]
    aesgcm = AESGCM(aes_key)
    return aesgcm.decrypt(nonce, ciphertext, None)


def decrypt_log_chunks(
    chunks_b64: list[str],
    private_key_scalar: bytes,
    user_pubkey: bytes,
    ephemeral_pubkey: bytes,
) -> str:
    """Decrypt a list of base64-encoded encrypted log chunks and return combined plaintext."""
    lines = []
    for chunk_b64 in chunks_b64:
        encrypted = pybase64.b64decode(chunk_b64)
        plaintext = decrypt_chunk(encrypted, private_key_scalar, user_pubkey, ephemeral_pubkey)
        lines.append(plaintext.decode("utf-8", errors="replace"))
    return "\n".join(lines)
