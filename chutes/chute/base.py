"""
Main application class, along with all of the inference decorators.
"""

import os
import asyncio
import uuid
from loguru import logger
from typing import List, Tuple, Callable
from fastapi import FastAPI, Response, status
from pydantic import BaseModel, ConfigDict, Field
import aiohttp
import time
from chutes.image import Image
from chutes.util.context import is_remote
from chutes.chute.node_selector import NodeSelector

if os.getenv("CHUTES_EXECUTION_CONTEXT") == "REMOTE":
    existing = os.getenv("NO_PROXY")
    os.environ["NO_PROXY"] = ",".join(
        [
            "localhost",
            "127.0.0.1",
            "api",
            "api.chutes.svc",
            "api.chutes.svc.cluster.local",
        ]
    )
    if existing:
        os.environ["NO_PROXY"] += f",{existing}"


class Chute(FastAPI):
    def __init__(
        self,
        username: str,
        name: str,
        image: str | Image,
        tagline: str = "",
        readme: str = "",
        standard_template: str = None,
        revision: str = None,
        node_selector: NodeSelector = None,
        concurrency: int = 1,
        max_instances: int = 1,
        shutdown_after_seconds: int = 300,
        scaling_threshold: float = 0.75,
        allow_external_egress: bool = False,
        encrypted_fs: bool = False,
        passthrough_headers: dict = {},
        tee: bool = False,
        **kwargs,
    ):
        from chutes.chute.cord import Cord
        from chutes.chute.job import Job

        super().__init__(**kwargs)
        self._username = username
        self._name = name
        self._readme = readme
        self._tagline = tagline
        self._uid = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{username}::chute::{name}"))
        self._image = image
        self._standard_template = standard_template
        self._node_selector = node_selector
        # Store hooks as list of tuples: (priority, hook_function)
        self._startup_hooks: List[Tuple[int, Callable]] = []
        self._shutdown_hooks: List[Tuple[int, Callable]] = []
        self._cords: list[Cord] = []
        self._jobs: list[Job] = []
        self.revision = revision
        self.concurrency = concurrency
        self.max_instances = max_instances
        self.scaling_threshold = scaling_threshold
        self.shutdown_after_seconds = shutdown_after_seconds
        self.allow_external_egress = allow_external_egress
        self.encrypted_fs = encrypted_fs
        self.passthrough_headers = passthrough_headers
        self.docs_url = None
        self.redoc_url = None
        self.tee = tee

        # Warmup state
        self._warmup_state = WarmupState()
        self._warmup_lock = asyncio.Lock()
        self._warmup_task: asyncio.Task | None = None

    @property
    def name(self):
        return self._name

    @property
    def readme(self):
        return self._readme

    @property
    def tagline(self):
        return self._tagline

    @property
    def uid(self):
        return self._uid

    @property
    def image(self):
        return self._image

    @property
    def cords(self):
        return self._cords

    @property
    def jobs(self):
        return self._jobs

    @property
    def node_selector(self):
        return self._node_selector

    @property
    def standard_template(self):
        return self._standard_template

    def _on_event(self, hooks: List[Tuple[int, Callable]], priority: int = 50):
        """
        Decorator to register a function for an event type, e.g. startup/shutdown.

        Args:
            hooks: List to store the hook functions
            priority: Execution priority (lower values execute first, default=50)
        """

        def decorator(func):
            if asyncio.iscoroutinefunction(func):

                async def async_wrapper(*args, **kwargs):
                    return await func(self, *args, **kwargs)

                hooks.append((priority, async_wrapper))
                return async_wrapper
            else:

                def sync_wrapper(*args, **kwargs):
                    func(self, *args, **kwargs)

                hooks.append((priority, sync_wrapper))
                return sync_wrapper

        return decorator

    def on_startup(self, priority: int = 50):
        """
        Wrapper around _on_event for startup events.

        Args:
            priority: Execution priority (lower values execute first, default=50).
                     Common values: 0-20 for early initialization,
                     30-70 for normal operations, 80-100 for late initialization.

        Example:
            @app.on_startup(priority=10)  # Runs early
            async def init_database(app):
                await setup_db()

            @app.on_startup(priority=90)  # Runs late
            def log_startup(app):
                logger.info("Application started")
        """
        return self._on_event(self._startup_hooks, priority)

    def on_shutdown(self, priority: int = 50):
        """
        Wrapper around _on_event for shutdown events.

        Args:
            priority: Execution priority (lower values execute first, default=50).
                     Common values: 0-20 for critical cleanup,
                     30-70 for normal cleanup, 80-100 for final cleanup.

        Example:
            @app.on_shutdown(priority=10)  # Runs early
            async def close_connections(app):
                await close_db()

            @app.on_shutdown(priority=90)  # Runs late
            def final_logging(app):
                logger.info("Shutdown complete")
        """
        return self._on_event(self._shutdown_hooks, priority)

    async def initialize(self):
        """
        Initialize the application based on the specified hooks.
        """
        if not is_remote():
            return

        # Sort hooks by priority before execution
        sorted_startup_hooks = sorted(self._startup_hooks, key=lambda x: x[0])

        for priority, hook in sorted_startup_hooks:
            if asyncio.iscoroutinefunction(hook):
                await hook()
            else:
                hook()

        # Add all of the API endpoints.
        dev = os.getenv("CHUTES_DEV_MODE", "false").lower() == "true"
        for cord in self._cords:
            path = cord.path
            method = "POST"
            if dev:
                path = cord._public_api_path
                method = cord._public_api_method
            self.add_api_route(path, cord._request_handler, methods=[method])
            logger.info(f"Added new API route: {path} calling {cord._func.__name__} via {method}")
            logger.debug(f"  {cord.input_schema=}")
            logger.debug(f"  {cord.minimal_input_schema=}")
            logger.debug(f"  {cord.output_content_type=}")
            logger.debug(f"  {cord.output_schema=}")

        # Job methods.
        for job in self._jobs:
            logger.info(f"Found job definition: {job._func.__name__}")

        # Add warmup endpoints
        self.add_api_route(
            "/warmup/kick",
            self._warmup_kick,
            methods=["POST"],
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.add_api_route(
            "/warmup/status",
            self._warmup_status,
            methods=["GET"],
        )

    async def _check_models_ready(self, base_url: str, auth_header: str | None) -> bool:
        """
        Check if /v1/models returns 200.
        """
        headers = {}
        if auth_header:
            headers["Authorization"] = auth_header
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10.0)) as session:
                async with session.get(f"{base_url.rstrip('/')}/v1/models", headers=headers) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def _do_warmup(self, base_url: str, auth_header: str | None):
        """
        Background warmup task.
        """
        t0 = time.time()
        steps = [
            ("pulling", 10),
            ("loading", 40),
            ("tokenizer", 60),
            ("tiny_infer", 80),
        ]
        
        try:
            async with self._warmup_lock:
                self._warmup_state.state = "warming"
                self._warmup_state.phase = "pulling"
                self._warmup_state.progress = 0
                self._warmup_state.error = None
                self._warmup_state.started_at = t0

            # Simulate phases (in a real implementation, we might hook into actual events if possible)
            # For now, we just advance through phases to show progress while waiting for the model.
            # The real gate is /v1/models.
            for phase, pct in steps:
                await asyncio.sleep(1.0)
                async with self._warmup_lock:
                    self._warmup_state.phase = phase
                    self._warmup_state.progress = pct
                    self._warmup_state.elapsed_sec = time.time() - t0
            
            # Wait for canonical gate
            # Try for up to ~3 minutes
            for _ in range(180):
                if await self._check_models_ready(base_url, auth_header):
                    async with self._warmup_lock:
                        self._warmup_state.phase = "ready"
                        self._warmup_state.progress = 100
                        self._warmup_state.state = "ready"
                        self._warmup_state.elapsed_sec = time.time() - t0
                    return
                await asyncio.sleep(1.0)
                async with self._warmup_lock:
                     self._warmup_state.elapsed_sec = time.time() - t0

            # Timeout
            async with self._warmup_lock:
                self._warmup_state.state = "error"
                self._warmup_state.error = "models_not_ready_within_timeout"
                self._warmup_state.elapsed_sec = time.time() - t0

        except Exception as e:
            async with self._warmup_lock:
                self._warmup_state.state = "error"
                self._warmup_state.error = str(e)
                self._warmup_state.elapsed_sec = time.time() - t0

    async def _warmup_kick(self, response: Response, authorization: str | None = None):
        """
        Kick off the warmup process.
        """
        # Auth check is handled by middleware usually, but if we need to pass it to the check loop:
        # We'll assume the request to this endpoint has the same auth as /v1/models needs.
        
        # Determine base URL. Since we are running inside the chute, localhost:8000 is likely where /v1/models is.
        # But we should respect CHUTES_API_BASE if set.
        base = os.environ.get("CHUTES_API_BASE", "http://127.0.0.1:8000")

        async with self._warmup_lock:
            if self._warmup_state.state in ("warming", "ready"):
                 return self._warmup_state.model_dump(by_alias=True)
            
            # Reset and spawn
            self._warmup_state.state = "warming"
            self._warmup_state.phase = "pulling"
            self._warmup_state.progress = 0
            self._warmup_state.error = None
            self._warmup_state.started_at = time.time()
            self._warmup_state.elapsed_sec = 0.0
        
        if self._warmup_task is None or self._warmup_task.done():
            self._warmup_task = asyncio.create_task(self._do_warmup(base, authorization))
            
        return self._warmup_state.model_dump(by_alias=True)

    async def _warmup_status(self):
        """
        Get the current warmup status.
        """
        async with self._warmup_lock:
            return self._warmup_state.model_dump(by_alias=True)

    def cord(self, **kwargs):
        """
        Decorator to define a parachute cord (function).
        """
        from chutes.chute.cord import Cord

        cord = Cord(self, **kwargs)
        self._cords.append(cord)
        return cord

    def job(self, **kwargs):
        """
        Decorator to define a job.
        """
        from chutes.chute.job import Job

        job = Job(self, **kwargs)
        self._jobs.append(job)
        return job


# For returning things from the templates, aside from just a chute.
class ChutePack(BaseModel):
    chute: Chute
    model_config = ConfigDict(arbitrary_types_allowed=True)


# Warmup state model
class WarmupState(BaseModel):
    schema_version: str = Field("chutes.warmup.status.v1", alias="schema")
    state: str = "idle"  # idle|warming|ready|error
    phase: str = "idle"  # pulling|loading|tokenizer|tiny_infer|ready
    progress: int = 0  # 0..100
    elapsed_sec: float = 0.0
    error: str | None = None
    started_at: float | None = None

    model_config = ConfigDict(populate_by_name=True)

