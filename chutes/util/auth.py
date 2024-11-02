import time
import hashlib
import orjson as json
from typing import Dict, Any
from substrateinterface import Keypair
from chutes.config import HOTKEY_SEED, HOTKEY_SS58, USER_ID


def sign_request(payload: Dict[str, Any] | str = None, purpose: str = None):
    """
    Generate a signed request.
    """
    nonce = str(int(time.time()))
    headers = {
        "X-Parachutes-UserID": USER_ID,
        "X-Parachutes-Hotkey": HOTKEY_SS58,
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
            [HOTKEY_SS58, nonce, hashlib.sha256(payload_string).hexdigest()]
        )
    else:
        signature_string = ":".join([purpose, nonce, HOTKEY_SS58])
        headers["X-Parachutes-Auth"] = signature_string
    keypair = Keypair.create_from_seed(seed_hex=HOTKEY_SEED)
    headers["X-Parachutes-Signature"] = keypair.sign(signature_string.encode()).hex()
    return headers, payload_string
