"""
Authentication utilities for the Chutes platform.

This module provides functions for signing requests using Bittensor wallet
signatures to ensure authenticated and tamper-proof API communication.
"""

import time
import hashlib
import orjson as json
from typing import Dict, Any, Tuple, Optional
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
    """
    Generate the message string to be signed for request authentication.
    
    The signing message follows the format: hotkey:nonce:payload_hash
    This ensures that the signature covers the identity (hotkey), timestamp
    (nonce), and content (payload hash) of the request.
    
    Args:
        hotkey: SS58 address of the signing hotkey.
        nonce: Timestamp-based nonce for replay protection.
        payload_str: Request payload as string or bytes (will be hashed).
        purpose: Purpose string for signature (alternative to payload).
        payload_hash: Pre-computed payload hash (alternative to payload_str).
        
    Returns:
        Signing message string in format "hotkey:nonce:hash".
        
    Raises:
        ValueError: If neither payload_str, purpose, nor payload_hash is provided.
        
    Example:
        >>> msg = get_signing_message(
        ...     hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        ...     nonce="1703462400",
        ...     payload_str=b'{"name": "test"}'
        ... )
    """
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


def sign_request(payload: Dict[str, Any] | str | None = None, purpose: str = None) -> Tuple[Dict[str, str], Optional[str]]:
    """
    Generate cryptographically signed request headers and payload.
    
    This function creates authentication headers using the user's Bittensor
    hotkey to sign the request. The signature covers the payload hash to
    ensure request integrity and authenticity.
    
    The signing process:
    1. Creates a timestamp-based nonce for replay protection
    2. Computes SHA256 hash of the payload
    3. Signs "hotkey:nonce:payload_hash" with the Bittensor keypair
    4. Returns headers containing user ID, hotkey, nonce, and signature
    
    Args:
        payload: Request payload (dict or string). If dict, will be JSON serialized.
                If None, purpose parameter must be provided.
        purpose: Purpose string for signature (used when payload is None).
                Typically the API endpoint path.
    
    Returns:
        Tuple of (headers dict, serialized payload string).
        Headers include: X-Chutes-UserID, X-Chutes-Hotkey, X-Chutes-Nonce,
        X-Chutes-Signature, and Content-Type (if payload is dict).
        
    Raises:
        ValueError: If neither payload nor purpose is provided.
        AuthenticationError: If configuration is missing or invalid.
    
    Example:
        >>> from chutes.util.auth import sign_request
        >>> headers, body = sign_request({"name": "test-chute"})
        >>> async with aiohttp.ClientSession() as session:
        ...     await session.post(url, data=body, headers=headers)
        
        >>> # For GET requests without body
        >>> headers, _ = sign_request(purpose="/chutes/")
        >>> async with aiohttp.ClientSession() as session:
        ...     await session.get(url, headers=headers)
    
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
