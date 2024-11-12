import time
import hashlib
import orjson as json
from typing import Dict, Any
from substrateinterface import Keypair
from chutes.constants import (
    USER_ID_HEADER,
    HOTKEY_HEADER,
    NONCE_HEADER,
    SIGNATURE_HEADER,
)
from chutes.config import get_config
from loguru import logger


def get_signing_message(
    hotkey: str,
    nonce: str,
    payload_str: str | bytes | None,
    purpose: str | None = None,
    payload_hash: str | None = None,
) -> str:
    """Get the signing message for a given hotkey, nonce, and payload."""
    if payload_str:
        if isinstance(payload_str, str):
            payload_str = payload_str.encode()
        return f"{hotkey}:{nonce}:{hashlib.sha256(payload_str).hexdigest()}"
    elif purpose:
        return f"{hotkey}:{nonce}:{purpose}"
    elif payload_hash:
        return f"{hotkey}:{nonce}:{payload_hash}"
    else:
        raise ValueError("Either payload_str or purpose must be provided")


def sign_request(payload: Dict[str, Any] | str | None = None, purpose: str = None):
    """
    Generate a signed request.

    # NOTE: Could add the ability to use api keys here too. Important for inference.
    """
    config = get_config()
    nonce = str(int(time.time()))
    headers = {
        USER_ID_HEADER: config.auth.user_id,
        HOTKEY_HEADER: config.auth.hotkey_ss58address,
        NONCE_HEADER: nonce,
    }
    signature_string = None
    payload_string = None
    if payload is not None:
        if isinstance(payload, dict):
            headers["Content-Type"] = "application/json"
            payload_string = json.dumps(payload)
        else:
            payload_string = payload
        signature_string = get_signing_message(
            config.auth.hotkey_ss58address,
            nonce,
            payload_str=payload_string,
            purpose=None,
        )
    else:
        signature_string = get_signing_message(
            config.auth.hotkey_ss58address, nonce, payload_str=None, purpose=purpose
        )
    logger.debug(f"Signing message: {signature_string}")
    keypair = Keypair.create_from_seed(seed_hex=config.auth.hotkey_seed)
    headers[SIGNATURE_HEADER] = keypair.sign(signature_string.encode()).hex()
    return headers, payload_string
