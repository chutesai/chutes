import re
from typing import Any


def validate_the_username(value: Any) -> str:
    """
    Simple username validation.
    """
    if not isinstance(value, str):
        raise ValueError("Username must be a string")
    if not re.match(r"^[a-zA-Z0-9_]{3,15}$", value):
        raise ValueError(
            "Username must be 3-15 characters and contain only alphanumeric/underscore characters"
        )
    return value
