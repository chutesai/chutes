"""
Helper to distinguish between local and remote contexts.
"""

import os

def is_remote() -> bool:
    """
    Check if we are in the remote context.
    """
    return os.getenv("CHUTES_EXECUTION_CONTEXT") == "REMOTE"


def is_local() -> bool:
    """
    Check if we are in the local context.
    """
    return not is_remote()
