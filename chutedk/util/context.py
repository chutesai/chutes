"""
Helper to distinguish between local and remote contexts.
"""

import os

IS_REMOTE = os.getenv("PARACHUTES_EXECUTION_CONTEXT") == "REMOTE"


def is_remote() -> bool:
    """
    Check if we are in the remote context.
    """
    return IS_REMOTE


def is_local() -> bool:
    """
    Check if we are in the local context.
    """
    return not IS_REMOTE
