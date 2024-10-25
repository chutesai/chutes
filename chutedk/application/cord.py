import aiohttp
import re
import backoff
import pickle
import gzip
import pybase64 as base64
from contextlib import asynccontextmanager
from chutedk.config import CLIENT_ID, API_BASE_URL
from chutedk.application.base import Application
from chutedk.application.node_selector import NodeSelector
from chutedk.exception import InvalidPath, DuplicatePath, StillProvisioning
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
        provision_timeout: int = 180,
        **session_kwargs,
    ):
        """
        Constructor.
        """
        self._app = app
        self._path = path
        self._stream = stream
        self._selector = selector
        self._session_kwargs = session_kwargs
        self._provision_timeout = 60

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
                        "X-Parachute-ApplicationID": self._app.uid,
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
            return await response.json()

    async def _local_stream_call(self, *args, **kwargs):
        """
        Call the function from the local context, i.e. make an API request, but
        instead of just returning the response JSON, we're using a streaming
        response.
        """
        async with self._local_call_base(*args, **kwargs) as response:
            async for content in response.content:
                yield content

    async def _remote_call(self, *args, **kwargs):
        """
        Function call from within the remote context, that is, the code that actually
        runs on the miner's deployment.
        """
        print(f"GOT HERE _remote_call -> {self._func.__name__} with {args=} {kwargs=}")
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
        async for data in self._func(*args, **kwargs):
            yield data

    def __call__(self, func):
        self._func = func
        if not self._path:
            self.path = func.__name__
        if is_local():
            return self._local_call if not self._stream else self._local_stream_call
        return self._remote_call if not self._stream else self._remote_stream_call
