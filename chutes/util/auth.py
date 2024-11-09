import time
import hashlib
import orjson as json
from typing import Dict, Any
from substrateinterface import Keypair

from chutes.config import get_config



def sign_request(payload: Dict[str, Any] | str = None, purpose: str = None):
    """
    Generate a signed request.

    # NOTE: Could add the ability to use api keys here too. Important for inference.
    """
    config = get_config()
    nonce = str(int(time.time()))
    headers = {
        "X-Parachutes-UserID": config.auth.user_id,
        "X-Parachutes-Hotkey": config.auth.hotkey_ss58address,
        "X-Parachutes-Nonce": nonce,
    }
    signature_string = None
    payload_string = None
    if payload:
        if isinstance(payload, dict):
            headers["Content-Type"] = "application/json"
            payload_string = json.dumps(payload)
        else:
            payload_string = payload
        signature_string = ":".join(
            [config.auth.hotkey_ss58address, nonce, hashlib.sha256(payload_string).hexdigest()]
        )
    else:
        signature_string = ":".join([purpose, nonce, config.auth.hotkey_ss58address])
        headers["X-Parachutes-Auth"] = signature_string
    keypair = Keypair.create_from_seed(seed_hex=config.auth.hotkey_seed)
    headers["X-Parachutes-Signature"] = keypair.sign(signature_string.encode()).hex()
    return headers, payload_string
