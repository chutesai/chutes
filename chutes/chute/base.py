"""
Main application class, along with all of the inference decorators.
"""

import asyncio
import uuid
from loguru import logger
from typing import Any, List, Dict
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from chutes.image import Image
from chutes.config import USER_ID
from chutes.util.context import is_remote
from chutes.chute.node_selector import NodeSelector


async def _pong(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Echo incoming request as a liveness check.
    """
    return request


class Chute(FastAPI):
    def __init__(
        self,
        name: str,
        image: str | Image,
        standard_template: str = None,
        node_selector: NodeSelector = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._name = name
        self._uid = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{USER_ID}::chute::{name}"))
        self._image = image
        self._standard_template = standard_template
        self._node_selector = node_selector
        self._startup_hooks = []
        self._shutdown_hooks = []
        self._cords = []

    @property
    def name(self):
        return self._name

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
    def node_selector(self):
        return self._node_selector

    @property
    def standard_template(self):
        return self._standard_template

    def _on_event(self, hooks: List[Any]):
        """
        Decorator to register a function for an event type, e.g. startup/shutdown.
        """

        def decorator(func):
            if asyncio.iscoroutinefunction(func):

                async def async_wrapper(*args, **kwargs):
                    return await func(self, *args, **kwargs)

                hooks.append(async_wrapper)
                return async_wrapper
            else:

                def sync_wrapper(*args, **kwargs):
                    func(self, *args, **kwargs)

                hooks.append(sync_wrapper)
                return sync_wrapper

        return decorator

    def on_startup(self):
        """
        Wrapper around _on_event for startup events.
        """
        return self._on_event(self._startup_hooks)

    def on_shutdown(self):
        """
        Wrapper around _on_event for shutdown events.
        """
        return self._on_event(self._shutdown_hooks)

    async def initialize(self):
        """
        Initialize the application based on the specified hooks.
        """
        if not is_remote():
            return
        for hook in self._startup_hooks:
            if asyncio.iscoroutinefunction(hook):
                await hook()
            else:
                hook()

        # Add all of the API endpoints.
        for cord in self._cords:
            self.add_api_route(
                f"/{self.uid}{cord.path}", cord._request_handler, methods=["POST"]
            )
            logger.info(
                f"Added new API route: /{self.uid}{cord.path} calling {cord._func.__name__}"
            )

        # Add a liveness check endpoint.
        self.add_api_route(f"/{self.uid}/_ping", _pong, methods=["POST"])
        logger.info(f"Added healthcheck endpoint: /{self.uid}/_ping")

    def cord(self, **kwargs):
        """
        Decorator to define a parachute cord (function).
        """
        from chutes.chute.cord import Cord

        cord = Cord(self, **kwargs)
        self._cords.append(cord)
        return cord


# For returning things from the templates, aside from just a chute.
class ChutePack(BaseModel):
    chute: Chute
    model_config = ConfigDict(arbitrary_types_allowed=True)
