import aiohttp
import re
import backoff
import pickle
import gzip
import pybase64 as base64
from typing import Dict
from contextlib import asynccontextmanager
from starlette.responses import StreamingResponse
from chutedk.config import CLIENT_ID, API_BASE_URL
from chutedk.chute.base import Chute
from chutedk.exception import InvalidPath, DuplicatePath, StillProvisioning
from chutedk.util.context import is_local

# Simple regex to check for custom path overrides.
PATH_RE = re.compile(r"^(/[a-z0-9]+[a-z0-9-_]*)+$")


class Cord:
    def __init__(
        self,
        app: Chute,
        stream: bool = False,
        path: str = None,
        passthrough_path: str = None,
        passthrough: bool = False,
        method: str = "GET",
        provision_timeout: int = 180,
        **session_kwargs,
    ):
        """
        Constructor.
        """
        self._app = app
        self._path = None
        if path:
            self.path = path
        self._passthrough_path = None
        if passthrough_path:
            self.passthrough_path = passthrough_path
        self._stream = stream
        self._passthrough = passthrough
        self._method = method
        self._session_kwargs = session_kwargs
        self._provision_timeout = 60

    @property
    def path(self):
        """
        URL path getter.
        """
        return self._path

    @path.setter
    def path(self, path: str):
        """
        URL path setter with some basic validation.

        :param path: The path to use for the new endpoint.
        :type path: str

        """
        path = "/" + path.lstrip("/").rstrip("/")
        if "//" in path or not PATH_RE.match(path):
            raise InvalidPath(path)
        if any([cord.path == path for cord in self._app.cords]):
            raise DuplicatePath(path)
        self._path = path

    @property
    def passthrough_path(self):
        """
        Passthrough/upstream URL path getter.
        """
        return self._passthrough_path

    @passthrough_path.setter
    def passthrough_path(self, path: str):
        """
        Passthrough/usptream path setter with some basic validation.

        :param path: The path to use for the upstream endpoint.
        :type path: str

        """
        path = "/" + path.lstrip("/").rstrip("/")
        if "//" in path or not PATH_RE.match(path):
            raise InvalidPath(path)
        self._passthrough_path = path

    @asynccontextmanager
    async def _local_call_base(self, *args, **kwargs):
        """
        Invoke the function from within the local/client side context, meaning
        we're actually just calling the chutes API.
        """
        print(f"GOT HERE _local_call -> {self._func.__name__} with {args=} {kwargs=}")

        @backoff.on_exception(
            backoff.constant,
            (StillProvisioning,),
            jitter=None,
            interval=1,
        )
        @asynccontextmanager
        async def _call():
            # Pickle is nasty, but... since we're running in ephemeral containers with no
            # security context escalation, no host path access, and limited networking, I think
            # we'll survive, and it allows complex objects as args/return values.
            request_payload = {
                "args": base64.b64encode(gzip.compress(pickle.dumps(args))).decode(),
                "kwargs": base64.b64encode(
                    gzip.compress(pickle.dumps(kwargs))
                ).decode(),
            }
            async with aiohttp.ClientSession(
                base_url=API_BASE_URL, **self._session_kwargs
            ) as session:
                async with session.post(
                    f"/{self._app.uid}{self.path}",
                    json=request_payload,
                    headers={
                        "X-Parachute-ClientID": CLIENT_ID,
                        "X-Parachute-ChuteID": self._app.uid,
                        "X-Parachute-Function": self._func.__name__,
                    },
                ) as response:
                    if response.status == 503:
                        raise StillProvisioning(await response.text())
                    elif response.status != 200:
                        raise Exception(await response.text())
                    yield response

        async with _call() as response:
            yield response

    async def _local_call(self, *args, **kwargs):
        """
        Call the function from the local context, i.e. make an API request.
        """
        async with self._local_call_base(*args, **kwargs) as response:
            return await self._func(response)

    async def _local_stream_call(self, *args, **kwargs):
        """
        Call the function from the local context, i.e. make an API request, but
        instead of just returning the response JSON, we're using a streaming
        response.
        """
        async with self._local_call_base(*args, **kwargs) as response:
            async for content in response.content:
                yield await self._func(content)

    @asynccontextmanager
    async def _passthrough_call(self, **kwargs):
        """
        Call a passthrough endpoint.
        """
        print(f"I AM IN PASSTHROUGH: {self.path=} {self._method=}")
        async with aiohttp.ClientSession(base_url="http://127.0.0.1:8000") as session:
            async with getattr(session, self._method.lower())(
                self.passthrough_path, **kwargs
            ) as response:
                yield response

    async def _remote_call(self, *args, **kwargs):
        """
        Function call from within the remote context, that is, the code that actually
        runs on the miner's deployment.
        """
        print(f"GOT HERE _remote_call -> {self._func.__name__} with {args=} {kwargs=}")
        if self._passthrough:
            async with self._passthrough_call(**kwargs) as response:
                return await response.json()

        return_value = await self._func(*args, **kwargs)
        # Again with the pickle...
        return {"result": base64.b64encode(gzip.compress(pickle.dumps(return_value)))}

    async def _remote_stream_call(self, *args, **kwargs):
        """
        Function call from within the remote context, that is, the code that actually
        runs on the miner's deployment.
        """
        print(
            f"GOT HERE _remote_stream_call -> {self._func.__name__} with {args=} {kwargs=}"
        )
        if self._passthrough:
            async with self._passthrough_call(**kwargs) as response:
                async for content in response.content:
                    yield content
            return

        async for data in self._func(*args, **kwargs):
            yield data

    async def _request_handler(self, request: Dict[str, str]):
        """
        Decode/deserialize incoming request and call the appropriate function.
        """
        args = pickle.loads(gzip.decompress(base64.b64decode(request["args"])))
        kwargs = pickle.loads(gzip.decompress(base64.b64decode(request["kwargs"])))
        if self._stream:
            return StreamingResponse(self._remote_stream_call(*args, **kwargs))
        return await self._remote_call(*args, **kwargs)

    def __call__(self, func):
        self._func = func
        if not self._path:
            self.path = func.__name__
        if not self._passthrough_path:
            self.passthrough_path = func.__name__
        if is_local():
            return self._local_call if not self._stream else self._local_stream_call
        return self._remote_call if not self._stream else self._remote_stream_call
