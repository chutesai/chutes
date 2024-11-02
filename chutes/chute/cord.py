import aiohttp
import re
import backoff
import gzip
import time
import orjson as json
import fickling
import pickle
import pybase64 as base64
from fastapi import Request, HTTPException, status
from loguru import logger
from contextlib import asynccontextmanager
from starlette.responses import StreamingResponse
from chutes.config import API_BASE_URL
from chutes.chute.base import Chute
from chutes.exception import InvalidPath, DuplicatePath, StillProvisioning
from chutes.util.context import is_local
from chutes.util.auth import sign_request

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
        self._passthrough_port = None
        self._stream = stream
        self._passthrough = passthrough
        self._method = method
        self._session_kwargs = session_kwargs
        self._provision_timeout = provision_timeout

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
        logger.debug(f"Invoking remote function {self._func.__name__} via HTTP...")

        @backoff.on_exception(
            backoff.constant,
            (StillProvisioning,),
            jitter=None,
            interval=1,
            max_time=self._provision_timeout,
        )
        @asynccontextmanager
        async def _call():
            request_payload = {
                "args": base64.b64encode(gzip.compress(pickle.dumps(args))).decode(),
                "kwargs": base64.b64encode(
                    gzip.compress(pickle.dumps(kwargs))
                ).decode(),
            }
            headers, payload_string = sign_request(payload=request_payload)
            headers.update(
                {
                    "X-Parachutes-ChuteID": self._app.uid,
                    "X-Parachutes-Function": self._func.__name__,
                }
            )
            async with aiohttp.ClientSession(
                base_url=API_BASE_URL, **self._session_kwargs
            ) as session:
                async with session.post(
                    f"/chutes/{self._app.uid}{self.path}",
                    data=payload_string,
                    headers=headers,
                ) as response:
                    if response.status == 503:
                        logger.warning(
                            f"Function {self._func.__name__} is still provisioning..."
                        )
                        raise StillProvisioning(await response.text())
                    elif response.status != 200:
                        logger.error(
                            f"Error invoking {self._func.__name__} [status={response.status}]: {await response.text()}"
                        )
                        raise Exception(await response.text())
                    yield response

        started_at = time.time()
        async with _call() as response:
            yield response
        logger.debug(
            f"Completed remote invocation [{self._func.__name__} passthrough={self._passthrough}] in {time.time() - started_at} seconds"
        )

    async def _local_call(self, *args, **kwargs):
        """
        Call the function from the local context, i.e. make an API request.
        """
        result = None
        async for item in self._local_stream_call(*args, **kwargs):
            result = item
        return result

    async def _local_stream_call(self, *args, **kwargs):
        """
        Call the function from the local context, i.e. make an API request, but
        instead of just returning the response JSON, we're using a streaming
        response.
        """
        async with self._local_call_base(*args, **kwargs) as response:
            async for encoded_content in response.content:
                content = encoded_content.decode()
                if not content or not content.strip() or "data: {" not in content:
                    continue
                data = json.loads(content[6:])
                if data.get("trace"):
                    message = "".join(
                        [
                            data["trace"]["timestamp"],
                            " ["
                            + " ".join(
                                [
                                    f"{key}={value}"
                                    for key, value in data["trace"].items()
                                    if key not in ("timestamp", "message")
                                ]
                            ),
                            f"]: {data['trace']['message']}",
                        ]
                    )
                    logger.debug(message)
                elif data.get("error"):
                    logger.error(data["error"])
                    raise Exception(data["error"])
                elif data.get("result"):
                    if self._passthrough:
                        yield await self._func(data["result"])
                    else:
                        yield data["result"]

    @asynccontextmanager
    async def _passthrough_call(self, **kwargs):
        """
        Call a passthrough endpoint.
        """
        logger.debug(
            f"Received passthrough call, passing along to {self.passthrough_path} via {self._method}"
        )
        async with aiohttp.ClientSession(
            base_url=f"http://127.0.0.1:{self._passthrough_port or 8000}"
        ) as session:
            async with getattr(session, self._method.lower())(
                self.passthrough_path, **kwargs
            ) as response:
                yield response

    async def _remote_call(self, *args, **kwargs):
        """
        Function call from within the remote context, that is, the code that actually
        runs on the miner's deployment.
        """
        logger.info(
            f"Received invocation request [{self._func.__name__} passthrough={self._passthrough}]"
        )
        started_at = time.time()
        if self._passthrough:
            async with self._passthrough_call(**kwargs) as response:
                logger.success(
                    f"Completed request [{self._func.__name__} passthrough={self._passthrough}] in {time.time() - started_at} seconds"
                )
                return await response.json()

        return_value = await self._func(*args, **kwargs)
        logger.success(
            f"Completed request [{self._func.__name__} passthrough={self._passthrough}] in {time.time() - started_at} seconds"
        )
        return return_value

    async def _remote_stream_call(self, *args, **kwargs):
        """
        Function call from within the remote context, that is, the code that actually
        runs on the miner's deployment.
        """
        logger.info(f"Received streaming invocation request [{self._func.__name__}]")
        started_at = time.time()
        if self._passthrough:
            async with self._passthrough_call(**kwargs) as response:
                async for content in response.content:
                    yield content
            logger.success(
                f"Completed request [{self._func.__name__} (passthrough)] in {time.time() - started_at} seconds"
            )
            return

        async for data in self._func(*args, **kwargs):
            yield data
        logger.success(
            f"Completed request [{self._func.__name__}] in {time.time() - started_at} seconds"
        )

    async def _request_handler(self, request: Request):
        """
        Decode/deserialize incoming request and call the appropriate function.
        """
        if self._passthrough_port is None:
            self._passthrough_port = request.url.port
        request = await request.json()
        try:
            args = fickling.load(gzip.decompress(base64.b64decode(request["args"])))
            kwargs = fickling.load(gzip.decompress(base64.b64decode(request["kwargs"])))
        except fickling.exception.UnsafeFileError as exc:
            message = f"Detected potentially hazardous call arguments, blocking: {exc}"
            logger.error(message)
            raise HTTPException(
                status_code=status.HTTP_401_FORBIDDEN,
                detail=message,
            )
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
