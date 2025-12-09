"""
Pydantic models for warmup status schema v1.
"""

from pydantic import BaseModel
from typing import Literal, Optional


class WarmupStatus(BaseModel):
    """
    Warmup status response schema v1.
    Matches chutes.warmup.status.v1 specification.
    """
    schema: Literal["chutes.warmup.status.v1"] = "chutes.warmup.status.v1"
    state: Literal["idle", "warming", "ready", "error"] = "idle"
    phase: Literal["idle", "pulling", "loading", "tokenizer", "tiny_infer", "ready"] = "idle"
    progress: int = 0  # 0-100
    elapsed_sec: float = 0.0
    error: Optional[str] = None

