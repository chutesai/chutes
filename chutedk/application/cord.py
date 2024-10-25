import asyncio
import re
import requests
from typing import Any, Dict, Callable
from chutedk.application.base import Application
from chutedk.application.node_selector import NodeSelector
from chutedk.exception import InvalidPath
from chutedk.util.context import is_local

# Simple regex to check for custom path overrides.
PATH_RE = re.compile(r"^/[a-z0-9]+[a-z0-9-_]*$")


class Cord:
    def __init__(
        self,
        app: Application,
        selector: NodeSelector,
        stream: bool = False,
        path: str = None,
        **kwargs,
    ):
        """
        Constructor.
        """
        self._app = app
        self._path = path
        self._stream = stream

        # GPU selection.

    @property
    def path(self):
        """
        URL getter.
        """
        return self._path

    @path.setter
    def path(self, path: str):
        """
        URL setter with some basic validation.

        :param path: The path to use for the new endpoint.
        :type path: str

        """
        path = "/" + path.lstrip("/").rstrip("/")
        if "//" in path or not PATH_RE.match(path):
            raise InvalidPath(path)
        if any([cord.path == path for cord in self._app.cords]):
            raise DuplicatePath(path)
        self._path = path

    async def _local_call(self, *args, **kwargs):
        """
        Invoke the function from within the local/client side context, meaning
        we're actually just calling the chutes API.
        """
        print(f"GOT HERE _local_call -> {self._func.__name__} with {args=} {kwargs=}")

    async def _local_stream_call(self, *args, **kwargs):
        """
        Invoke the function from within the local/client side context, meaning
        we're actually just calling the chutes API.
        """
        print(
            f"GOT HERE _local_stream_call -> {self._func.__name__} with {args=} {kwargs=}"
        )
        yield "empty"

    async def _remote_call(self, *args, **kwargs):
        """
        Function call from within the remote context, that is, the code that actually
        runs on the miner's deployment.
        """
        print(f"GOT HERE _remote_call -> {self._func.__name__} with {args=} {kwargs=}")
        return await self._func(*args, **kwargs)

    async def _remote_stream_call(self, *args, **kwargs):
        """
        Function call from within the remote context, that is, the code that actually
        runs on the miner's deployment.
        """
        print(
            f"GOT HERE _remote_stream_call -> {self._func.__name__} with {args=} {kwargs=}"
        )
        async for data in self._func(*args, **kwargs):
            print(f"SSE: {data}")
            yield data

    def __call__(self, func):
        self._func = func
        if not self._path:
            self.path = func.__name__
        if is_local():
            return self._local_call if not self._stream else self._local_stream_call
        return self._remote_call if not self._stream else self._remote_stream_call
