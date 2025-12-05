"""
Warmup state manager for canonical warmup cords.
Provides /warmup/kick and /warmup/status endpoints.
"""

import os
import time
import asyncio
import aiohttp
from typing import Optional
from loguru import logger
from chutes.chute.warmup_state import WarmupStatus


class WarmupManager:
    """
    Singleton warmup state manager.
    Manages warmup state, phases, and readiness checking.
    """
    
    _instance: Optional["WarmupManager"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._state = WarmupStatus()
        self._warmup_task: Optional[asyncio.Task] = None
        self._state_lock = asyncio.Lock()
        self._started_at: Optional[float] = None
        self._port: int = 8000  # Default port, can be updated
        self._initialized = True
    
    async def kick(self) -> dict:
        """
        Start or restart warmup (idempotent).
        Returns 202 Accepted response data.
        """
        async with self._state_lock:
            if self._state.state in ("warming", "ready"):
                return {
                    "ok": True,
                    "state": self._state.state,
                    "schema": self._state.schema
                }
            
            # Reset and start warmup (also resets error state)
            self._state.state = "warming"
            self._state.phase = "pulling"
            self._state.progress = 0
            self._state.error = None
            self._started_at = time.time()
            self._state.elapsed_sec = 0.0
            
            # Cancel existing task if running
            if self._warmup_task and not self._warmup_task.done():
                self._warmup_task.cancel()
            
            # Start background warmup task
            self._warmup_task = asyncio.create_task(self._do_warmup())
        
        return {
            "ok": True,
            "state": "warming",
            "schema": self._state.schema
        }
    
    async def get_status(self) -> WarmupStatus:
        """
        Get current warmup status (fast, <500ms).
        """
        async with self._state_lock:
            if self._started_at:
                self._state.elapsed_sec = time.time() - self._started_at
            return self._state
    
    async def update_phase(self, phase: str, progress: int):
        """
        Update warmup phase and progress.
        Called by templates during model initialization.
        
        Args:
            phase: One of "pulling", "loading", "tokenizer", "tiny_infer", "ready"
            progress: Integer between 0-100
        """
        # Validate progress bounds
        progress = max(0, min(100, progress))
        
        async with self._state_lock:
            if self._state.state == "warming":
                # Validate phase (allow any string for flexibility, but log warnings for invalid ones)
                valid_phases = ("pulling", "loading", "tokenizer", "tiny_infer", "ready")
                if phase not in valid_phases:
                    logger.warning(f"Invalid warmup phase: {phase}, expected one of {valid_phases}")
                
                self._state.phase = phase
                self._state.progress = progress
                if self._started_at:
                    self._state.elapsed_sec = time.time() - self._started_at
    
    def set_port(self, port: int):
        """
        Set the port for internal /v1/models checks.
        Should be called during chute initialization.
        """
        self._port = port
    
    async def _check_models_ready(self, base_url: str) -> bool:
        """
        Check if /v1/models endpoint returns 200.
        Used for readiness bridge.
        """
        try:
            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{base_url.rstrip('/')}/v1/models") as response:
                    return response.status == 200
        except Exception as e:
            logger.debug(f"Models readiness check failed: {e}")
            return False
    
    async def _do_warmup(self):
        """
        Background task that monitors warmup progress and checks readiness.
        """
        # Use localhost for internal checks since we're checking the same instance
        base_url = f"http://127.0.0.1:{self._port}"
        
        # Wait for models to become ready (max 3 minutes)
        max_wait = 180  # 3 minutes
        check_interval = 1.0  # Check every second
        
        for attempt in range(max_wait):
            await asyncio.sleep(check_interval)
            
            # Update elapsed time
            async with self._state_lock:
                if self._started_at:
                    self._state.elapsed_sec = time.time() - self._started_at
            
            # Check if models are ready
            if await self._check_models_ready(base_url):
                async with self._state_lock:
                    # Only set ready if we're still in warming state
                    if self._state.state == "warming":
                        self._state.phase = "ready"
                        self._state.progress = 100
                        self._state.state = "ready"
                        if self._started_at:
                            self._state.elapsed_sec = time.time() - self._started_at
                logger.info("Warmup completed: models are ready")
                return
        
        # Timeout
        async with self._state_lock:
            if self._state.state == "warming":
                self._state.state = "error"
                self._state.error = "models_not_ready_within_timeout"
                if self._started_at:
                    self._state.elapsed_sec = time.time() - self._started_at
        logger.warning("Warmup timed out: models not ready within 3 minutes")


def get_warmup_manager() -> WarmupManager:
    """
    Get the singleton warmup manager instance.
    """
    return WarmupManager()

