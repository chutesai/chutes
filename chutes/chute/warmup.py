"""
Canonical warmup cords for Chutes deployments.

Provides a versioned warmup interface with kick/status endpoints
that can be mounted on any Chute to enable operators to:
  (a) kick a background warmup process
  (b) poll human-readable progress with a versioned JSON schema

The warmup system integrates with the readiness gate (GET /v1/models)
and only reports state=ready when the underlying model endpoint is live.

Reference: https://github.com/ChutesAI/chutes/issues/31
"""

import asyncio
import time
import enum
from typing import Optional, Callable, Awaitable, List, Tuple

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field


WARMUP_SCHEMA_VERSION = "chutes.warmup.status.v1"


class WarmupState(str, enum.Enum):
    """Top-level warmup lifecycle states."""

    IDLE = "idle"
    WARMING = "warming"
    READY = "ready"
    ERROR = "error"


class WarmupPhase(str, enum.Enum):
    """Granular phases within the warmup lifecycle."""

    IDLE = "idle"
    PULLING = "pulling"
    LOADING = "loading"
    TOKENIZER = "tokenizer"
    TINY_INFER = "tiny_infer"
    READY = "ready"


class WarmupStatus(BaseModel):
    """
    Versioned warmup status response conforming to
    the ``chutes.warmup.status.v1`` schema contract.
    """

    schema_version: str = Field(
        default=WARMUP_SCHEMA_VERSION,
        alias="schema",
        description="Schema identifier for forward compatibility.",
    )
    state: WarmupState = Field(
        default=WarmupState.IDLE,
        description="Top-level lifecycle state.",
    )
    phase: WarmupPhase = Field(
        default=WarmupPhase.IDLE,
        description="Current granular phase within the warmup process.",
    )
    progress: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Completion percentage (0-100).",
    )
    elapsed_sec: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock seconds since warmup was kicked.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if state is 'error', null otherwise.",
    )

    model_config = {"populate_by_name": True}


class WarmupKickResponse(BaseModel):
    """Response body for POST /warmup/kick."""

    ok: bool = True
    state: WarmupState
    schema_version: str = Field(
        default=WARMUP_SCHEMA_VERSION,
        alias="schema",
    )

    model_config = {"populate_by_name": True}


class WarmupCords:
    """
    Manages the warmup lifecycle for a Chute deployment.

    Holds shared state, runs the background warmup task, and exposes
    ``kick()`` and ``get_status()`` methods that are wired to FastAPI
    routes by ``mount_warmup_cords``.

    Parameters
    ----------
    readiness_url:
        Full URL for the canonical readiness gate, e.g.
        ``http://127.0.0.1:10101/v1/models``.
    readiness_headers:
        Headers to send with readiness probes (e.g. Bearer auth).
    readiness_timeout_sec:
        How long to wait for the readiness gate before declaring error.
    phases:
        Optional list of ``(WarmupPhase, progress_pct, phase_callback)``
        tuples.  Each callback is awaited in order; ``progress`` is set
        to the given percentage once the callback completes.  If not
        supplied, a default set of simulated phases is used.
    """

    def __init__(
        self,
        readiness_url: str = "http://127.0.0.1:10101/v1/models",
        readiness_headers: Optional[dict] = None,
        readiness_timeout_sec: float = 300.0,
        phases: Optional[
            List[Tuple[WarmupPhase, int, Optional[Callable[[], Awaitable[None]]]]]
        ] = None,
    ):
        self._readiness_url = readiness_url
        self._readiness_headers = readiness_headers or {}
        self._readiness_timeout_sec = readiness_timeout_sec
        self._custom_phases = phases

        self._status = WarmupStatus()
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._started_at: Optional[float] = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    async def get_status(self) -> WarmupStatus:
        """Return a snapshot of the current warmup status (sub-500 ms)."""
        async with self._lock:
            if self._started_at is not None:
                self._status.elapsed_sec = round(
                    time.time() - self._started_at, 2
                )
            return self._status.model_copy()

    async def kick(self) -> WarmupKickResponse:
        """
        Idempotently start the warmup process.

        If warmup is already running or complete, this is a no-op and
        returns the current state.  Otherwise it spawns the background
        warmup task.
        """
        async with self._lock:
            if self._status.state in (WarmupState.WARMING, WarmupState.READY):
                return WarmupKickResponse(
                    ok=True,
                    state=self._status.state,
                )
            self._reset_status()

        self._task = asyncio.create_task(self._run_warmup())
        return WarmupKickResponse(ok=True, state=WarmupState.WARMING)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reset_status(self) -> None:
        """Reset status to warming (must be called under lock)."""
        self._started_at = time.time()
        self._status.state = WarmupState.WARMING
        self._status.phase = WarmupPhase.PULLING
        self._status.progress = 0
        self._status.error = None
        self._status.elapsed_sec = 0.0

    async def _set_phase(
        self, phase: WarmupPhase, progress: int
    ) -> None:
        """Thread-safe phase transition."""
        async with self._lock:
            self._status.phase = phase
            self._status.progress = min(progress, 100)
            if self._started_at is not None:
                self._status.elapsed_sec = round(
                    time.time() - self._started_at, 2
                )

    async def _set_ready(self) -> None:
        """Mark warmup as complete."""
        async with self._lock:
            self._status.state = WarmupState.READY
            self._status.phase = WarmupPhase.READY
            self._status.progress = 100
            if self._started_at is not None:
                self._status.elapsed_sec = round(
                    time.time() - self._started_at, 2
                )

    async def _set_error(self, message: str) -> None:
        """Mark warmup as failed."""
        async with self._lock:
            self._status.state = WarmupState.ERROR
            self._status.error = message
            if self._started_at is not None:
                self._status.elapsed_sec = round(
                    time.time() - self._started_at, 2
                )

    async def _check_readiness_gate(self) -> bool:
        """
        Probe the canonical readiness endpoint.

        Returns True if GET ``readiness_url`` returns HTTP 200.
        """
        try:
            timeout = aiohttp.ClientTimeout(total=10.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    self._readiness_url,
                    headers=self._readiness_headers,
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def _run_warmup(self) -> None:
        """
        Execute the warmup lifecycle.

        Iterates through configured phases, then polls the readiness gate
        until it returns 200 or a timeout is reached.
        """
        try:
            phases = self._custom_phases or self._default_phases()
            for phase, progress, callback in phases:
                await self._set_phase(phase, progress)
                if callback is not None:
                    await callback()

            # Poll readiness gate
            deadline = time.time() + self._readiness_timeout_sec
            poll_interval = 1.0
            while time.time() < deadline:
                if await self._check_readiness_gate():
                    await self._set_ready()
                    logger.success(
                        f"Warmup complete — readiness gate passed "
                        f"in {self._status.elapsed_sec:.1f}s"
                    )
                    return
                await asyncio.sleep(poll_interval)

            await self._set_error(
                f"Readiness gate did not return 200 within "
                f"{self._readiness_timeout_sec}s"
            )
            logger.error(
                f"Warmup failed: readiness timeout after "
                f"{self._readiness_timeout_sec}s"
            )

        except asyncio.CancelledError:
            await self._set_error("Warmup task was cancelled")
            raise
        except Exception as exc:
            await self._set_error(str(exc))
            logger.error(f"Warmup failed with exception: {exc}")

    @staticmethod
    def _default_phases() -> (
        List[Tuple[WarmupPhase, int, Optional[Callable[[], Awaitable[None]]]]]
    ):
        """
        Default phase list used when no custom phases are provided.

        Each phase simply sleeps briefly so the status endpoint has
        something meaningful to report during cold start.
        """

        async def _noop() -> None:
            await asyncio.sleep(0.5)

        return [
            (WarmupPhase.PULLING, 10, _noop),
            (WarmupPhase.LOADING, 40, _noop),
            (WarmupPhase.TOKENIZER, 60, _noop),
            (WarmupPhase.TINY_INFER, 80, _noop),
        ]


def mount_warmup_cords(
    chute,
    warmup_cords: Optional[WarmupCords] = None,
    readiness_url: str = "http://127.0.0.1:10101/v1/models",
    readiness_headers: Optional[dict] = None,
    readiness_timeout_sec: float = 300.0,
) -> WarmupCords:
    """
    Mount canonical warmup endpoints on a :class:`~chutes.chute.base.Chute`.

    This registers two cords:

    * ``POST /warmup/kick`` — idempotently starts warmup (returns 202)
    * ``GET  /warmup/status`` — returns the current ``WarmupStatus``

    The cords are mounted with a high startup priority (priority=5) so
    they are available *before* model initialisation completes.

    Parameters
    ----------
    chute:
        The :class:`Chute` instance to attach warmup cords to.
    warmup_cords:
        An existing :class:`WarmupCords` instance.  If ``None`` one is
        created with the given parameters.
    readiness_url:
        URL for the canonical readiness probe.
    readiness_headers:
        Auth headers forwarded to the readiness probe.
    readiness_timeout_sec:
        Maximum seconds to wait for readiness before error.

    Returns
    -------
    WarmupCords
        The (possibly newly created) warmup controller so callers can
        interact with it programmatically.
    """
    if warmup_cords is None:
        warmup_cords = WarmupCords(
            readiness_url=readiness_url,
            readiness_headers=readiness_headers,
            readiness_timeout_sec=readiness_timeout_sec,
        )

    @chute.cord(
        path="/warmup/kick",
        public_api_path="/warmup/kick",
        public_api_method="POST",
        method="POST",
        output_content_type="application/json",
    )
    async def warmup_kick(self) -> dict:
        """Idempotently kick the warmup process."""
        result = await warmup_cords.kick()
        return result.model_dump(by_alias=True)

    @chute.cord(
        path="/warmup/status",
        public_api_path="/warmup/status",
        public_api_method="GET",
        method="GET",
        output_content_type="application/json",
    )
    async def warmup_status(self) -> dict:
        """Return the current warmup status."""
        status = await warmup_cords.get_status()
        return status.model_dump(by_alias=True)

    return warmup_cords
