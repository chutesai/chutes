"""
Main application class, along with all of the inference decorators.
"""

import asyncio
import uuid
from typing import Any, Dict, List
from fastapi import FastAPI
from functools import wraps
from chutedk.image import Image
from chutedk.util.context import is_remote
from chutedk.util.config import ACCOUNT_ID


class Application(FastAPI):
    def __init__(self, name: str, image: Image, **kwargs):
        super().__init__(**kwargs)
        self._name = name
        self._uid = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{ACCOUNT_ID}:{name}"))
        self._image = image
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

    def cord(self, backoff_config: Dict[str, Any] = {}, **kwargs):
        """
        Decorator to define a remote function.
        """
        #def decorator(func):
        #    print(f"INSIDE OUTER DECORATOR")

        from chutedk.application.cord import Cord
        cord = Cord(self, backoff_config, **kwargs)
        self._cords.append(cord)
        return cord

        #return decorator
