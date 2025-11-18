"""
Main application class, along with all of the inference decorators.
"""

import os
import asyncio
import uuid
from loguru import logger
from typing import List, Tuple, Callable
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
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
        self.docs_url = None
        self.redoc_url = None

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
