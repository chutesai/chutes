"""
Tests for the canonical warmup cords module.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from chutes.chute.warmup import (
    WarmupCords,
    WarmupState,
    WarmupPhase,
    WarmupStatus,
    WarmupKickResponse,
    WarmupCords,
    WARMUP_SCHEMA_VERSION,
    mount_warmup_cords,
)


@pytest.fixture
def warmup():
    """Create a WarmupCords instance with a short timeout for testing."""
    return WarmupCords(
        readiness_url="http://127.0.0.1:9999/v1/models",
        readiness_timeout_sec=5.0,
    )


class TestWarmupStatus:
    """Tests for the WarmupStatus pydantic model."""

    def test_default_values(self):
        status = WarmupStatus()
        assert status.schema_version == WARMUP_SCHEMA_VERSION
        assert status.state == WarmupState.IDLE
        assert status.phase == WarmupPhase.IDLE
        assert status.progress == 0
        assert status.elapsed_sec == 0.0
        assert status.error is None

    def test_serialization_uses_alias(self):
        status = WarmupStatus()
        data = status.model_dump(by_alias=True)
        assert "schema" in data
        assert data["schema"] == WARMUP_SCHEMA_VERSION

    def test_progress_bounds(self):
        status = WarmupStatus(progress=0)
        assert status.progress == 0
        status = WarmupStatus(progress=100)
        assert status.progress == 100
        with pytest.raises(Exception):
            WarmupStatus(progress=-1)
        with pytest.raises(Exception):
            WarmupStatus(progress=101)

    def test_all_states_valid(self):
        for state in WarmupState:
            status = WarmupStatus(state=state)
            assert status.state == state

    def test_all_phases_valid(self):
        for phase in WarmupPhase:
            status = WarmupStatus(phase=phase)
            assert status.phase == phase

    def test_error_state(self):
        status = WarmupStatus(
            state=WarmupState.ERROR,
            error="something went wrong",
        )
        assert status.state == WarmupState.ERROR
        assert status.error == "something went wrong"


class TestWarmupKickResponse:
    """Tests for the WarmupKickResponse model."""

    def test_default(self):
        resp = WarmupKickResponse(state=WarmupState.WARMING)
        assert resp.ok is True
        assert resp.state == WarmupState.WARMING

    def test_serialization_alias(self):
        resp = WarmupKickResponse(state=WarmupState.READY)
        data = resp.model_dump(by_alias=True)
        assert data["schema"] == WARMUP_SCHEMA_VERSION
        assert data["state"] == "ready"


class TestWarmupCordsGetStatus:
    """Tests for WarmupCords.get_status()."""

    @pytest.mark.asyncio
    async def test_initial_status_is_idle(self, warmup):
        status = await warmup.get_status()
        assert status.state == WarmupState.IDLE
        assert status.phase == WarmupPhase.IDLE
        assert status.progress == 0

    @pytest.mark.asyncio
    async def test_status_returns_copy(self, warmup):
        s1 = await warmup.get_status()
        s2 = await warmup.get_status()
        assert s1 is not s2
        assert s1 == s2


class TestWarmupCordsKick:
    """Tests for WarmupCords.kick() idempotency and lifecycle."""

    @pytest.mark.asyncio
    async def test_kick_returns_warming(self, warmup):
        with patch.object(warmup, "_run_warmup", new_callable=AsyncMock):
            resp = await warmup.kick()
            assert resp.ok is True
            assert resp.state == WarmupState.WARMING

    @pytest.mark.asyncio
    async def test_kick_is_idempotent(self, warmup):
        with patch.object(warmup, "_run_warmup", new_callable=AsyncMock):
            r1 = await warmup.kick()
            r2 = await warmup.kick()
            assert r1.state == WarmupState.WARMING
            assert r2.state == WarmupState.WARMING

    @pytest.mark.asyncio
    async def test_kick_after_ready_is_noop(self, warmup):
        async with warmup._lock:
            warmup._status.state = WarmupState.READY
        resp = await warmup.kick()
        assert resp.state == WarmupState.READY


class TestWarmupCordsPhaseTransitions:
    """Tests for internal phase transition helpers."""

    @pytest.mark.asyncio
    async def test_set_phase(self, warmup):
        warmup._started_at = time.time()
        await warmup._set_phase(WarmupPhase.LOADING, 40)
        status = await warmup.get_status()
        assert status.phase == WarmupPhase.LOADING
        assert status.progress == 40

    @pytest.mark.asyncio
    async def test_set_phase_clamps_progress(self, warmup):
        warmup._started_at = time.time()
        await warmup._set_phase(WarmupPhase.READY, 150)
        status = await warmup.get_status()
        assert status.progress == 100

    @pytest.mark.asyncio
    async def test_set_ready(self, warmup):
        warmup._started_at = time.time()
        await warmup._set_ready()
        status = await warmup.get_status()
        assert status.state == WarmupState.READY
        assert status.phase == WarmupPhase.READY
        assert status.progress == 100

    @pytest.mark.asyncio
    async def test_set_error(self, warmup):
        warmup._started_at = time.time()
        await warmup._set_error("test error")
        status = await warmup.get_status()
        assert status.state == WarmupState.ERROR
        assert status.error == "test error"


class TestWarmupCordsReadinessGate:
    """Tests for readiness gate probing."""

    @pytest.mark.asyncio
    async def test_check_readiness_returns_true_on_200(self, warmup):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await warmup._check_readiness_gate()
            assert result is True

    @pytest.mark.asyncio
    async def test_check_readiness_returns_false_on_error(self, warmup):
        with patch("aiohttp.ClientSession", side_effect=Exception("connection refused")):
            result = await warmup._check_readiness_gate()
            assert result is False


class TestWarmupCordsRunWarmup:
    """Tests for the full warmup lifecycle."""

    @pytest.mark.asyncio
    async def test_run_warmup_with_immediate_readiness(self, warmup):
        with patch.object(warmup, "_check_readiness_gate", return_value=True):
            warmup._started_at = time.time()
            async with warmup._lock:
                warmup._status.state = WarmupState.WARMING
            await warmup._run_warmup()
            status = await warmup.get_status()
            assert status.state == WarmupState.READY
            assert status.progress == 100

    @pytest.mark.asyncio
    async def test_run_warmup_timeout(self):
        cords = WarmupCords(
            readiness_url="http://127.0.0.1:9999/v1/models",
            readiness_timeout_sec=1.0,
        )
        with patch.object(cords, "_check_readiness_gate", return_value=False):
            cords._started_at = time.time()
            async with cords._lock:
                cords._status.state = WarmupState.WARMING
            await cords._run_warmup()
            status = await cords.get_status()
            assert status.state == WarmupState.ERROR
            assert "did not return 200" in status.error

    @pytest.mark.asyncio
    async def test_run_warmup_custom_phases(self):
        phase_log = []

        async def log_phase():
            phase_log.append(True)

        custom_phases = [
            (WarmupPhase.PULLING, 25, log_phase),
            (WarmupPhase.LOADING, 50, log_phase),
            (WarmupPhase.TOKENIZER, 75, log_phase),
        ]
        cords = WarmupCords(
            readiness_url="http://127.0.0.1:9999/v1/models",
            readiness_timeout_sec=5.0,
            phases=custom_phases,
        )
        with patch.object(cords, "_check_readiness_gate", return_value=True):
            cords._started_at = time.time()
            async with cords._lock:
                cords._status.state = WarmupState.WARMING
            await cords._run_warmup()
            assert len(phase_log) == 3
            status = await cords.get_status()
            assert status.state == WarmupState.READY

    @pytest.mark.asyncio
    async def test_run_warmup_exception_sets_error(self, warmup):
        async def exploding_callback():
            raise RuntimeError("boom")

        warmup._custom_phases = [
            (WarmupPhase.PULLING, 10, exploding_callback),
        ]
        warmup._started_at = time.time()
        async with warmup._lock:
            warmup._status.state = WarmupState.WARMING
        await warmup._run_warmup()
        status = await warmup.get_status()
        assert status.state == WarmupState.ERROR
        assert "boom" in status.error


class TestDefaultPhases:
    """Tests for the default phase configuration."""

    def test_default_phases_structure(self):
        phases = WarmupCords._default_phases()
        assert len(phases) == 4
        for phase, progress, callback in phases:
            assert isinstance(phase, WarmupPhase)
            assert isinstance(progress, int)
            assert 0 <= progress <= 100
            assert callback is not None


class TestMountWarmupCords:
    """Tests for the mount_warmup_cords helper."""

    def test_mount_returns_warmup_cords(self):
        mock_chute = MagicMock()
        mock_chute.cords = []

        # Mock the cord decorator to just return the function
        def mock_cord(**kwargs):
            def decorator(func):
                return func
            return decorator

        mock_chute.cord = mock_cord
        cords = mount_warmup_cords(mock_chute)
        assert isinstance(cords, WarmupCords)

    def test_mount_with_existing_cords(self):
        mock_chute = MagicMock()
        mock_chute.cords = []

        def mock_cord(**kwargs):
            def decorator(func):
                return func
            return decorator

        mock_chute.cord = mock_cord
        existing = WarmupCords(readiness_timeout_sec=42.0)
        result = mount_warmup_cords(mock_chute, warmup_cords=existing)
        assert result is existing
        assert result._readiness_timeout_sec == 42.0
