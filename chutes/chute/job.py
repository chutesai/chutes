"""
Jobs (or server rentals, or anything really that isn't an API).
"""

import os
import glob
import time
import asyncio
import traceback
import tempfile
import backoff
import aiohttp
from pydantic import BaseModel, Field, constr
from typing import Optional, Any, Literal
from loguru import logger
import chutes.metrics as metrics
from chutes.chute.base import Chute


class Port(BaseModel):
    name: str = constr(pattern=r"^[a-z]+[0-9]*$")
    port: int = Field(gt=8001, le=65535, description="Numeric port number")
    proto: str = Literal["tcp", "udp", "http"]


class Job:
    def __init__(
        self,
        app: Chute,
        ports: list[Port] = [],
        timeout: Optional[int] = None,
        upload: Optional[bool] = True,
    ):
        self._app = app
        self._timeout = None
        self._ports = None
        self._name = None
        self.timeout = timeout
        self.ports = ports
        self._upload = upload
        self.cancel_event = asyncio.Event()

    @property
    def name(self):
        return self._name

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value: int):
        """
        Allow the timeout to be None (e.g. for server rental), or up to one day.
        """
        assert value is None or (isinstance(value, int) and 30 <= value <= 24 * 60 * 60)
        self._timeout = value

    @property
    def ports(self):
        return self._ports

    @ports.setter
    def ports(self, ports: list[Port] = []):
        """
        Validate port ranges - no dupes, 8001-65535, TCP, UDP, HTTPS.
        Where of course, HTTPS is just TCP but wrapped in a chutes API TLS proxy.
        """
        self._ports = []
        assert isinstance(ports, (list, None))
        if not ports:
            return
        validated = []
        for port in ports:
            assert isinstance(port, (Port, dict))
            validated.append(Port(**port) if isinstance(port, dict) else port)
        self._ports = validated

    @property
    def upload(self):
        return bool(self._upload)

    @backoff.on_exception(
        backoff.constant,
        Exception,
        jitter=None,
        interval=10,
        max_tries=7,
    )
    async def _upload_job_file(self, path: str, signed_url: str) -> None:
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            with open(path, "rb") as f:
                async with session.post(signed_url, data=f) as resp:
                    logger.success(f"Uploaded job output file: {path}: {await resp.text()}")

    @backoff.on_exception(
        backoff.constant,
        Exception,
        jitter=None,
        interval=10,
        max_tries=7,
    )
    async def _update_job_status(self, job_data: dict, final_result: Any) -> dict:
        """
        Notify an external endpoint that the job completed (or failed).
        """
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            async with session.post(job_data["job_status_url"], json=final_result) as resp:
                return await resp.json()

    async def run(self, dev: bool = False, **job_data):
        """
        Run a job, uploading output files and handling cancellation, etc.
        """
        tempdir = tempfile.mkdtemp()
        job_data["output_dir"] = tempdir
        os.environ["TMPDIR"] = tempdir

        # Start metrics
        started_at = time.time()
        metrics.last_request_timestamp.labels(
            chute_id=self._app.uid, function=self._func.__name__
        ).set_to_current_time()

        # Wrap the user job in a task, so we can wait on both the task and a cancel event
        job_task = asyncio.create_task(self._func(self._app, **job_data))
        done, pending = await asyncio.wait(
            [job_task, self.cancel_event.wait()],
            timeout=self._timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # If the cancel_event was triggered, we want to cancel the actual job
        if self.cancel_event.is_set():
            logger.warning("Job cancelled via external signal.")
            for task in pending:
                task.cancel()
            for task in done:
                task.cancel()
            raise asyncio.CancelledError("Job was externally cancelled.")

        # If the job didn't finish within our timeout, we cancel it
        if job_task not in done:
            logger.error(f"Job timed out after {self._timeout} seconds.")
            job_task.cancel()
            raise asyncio.TimeoutError(
                f"Job [{self._func.__name__}] timed out after {self._timeout} seconds."
            )

        # If we got here, the job actually completed
        upload_required = False
        final_result = None
        try:
            final_result = {
                "status": "complete",
                "result": job_task.result(),
            }
            logger.success(f"Job [{self._func.__name__}] finished successfully.")
            upload_required = True
        except asyncio.CancelledError:
            logger.error("Job was cancelled.")
            final_result = {
                "status": "cancelled",
                "detail": "Job was cancelled.",
            }
        except Exception as exc:
            final_result = {
                "status": "error",
                "detail": f"Job failed: {exc}\n{traceback.format_exc()}",
            }
            logger.error(final_result["detail"])

        # Update the job object.
        if not dev:
            upload_cfg = await self._update_job_status(job_data, final_result)
            if self.upload and upload_required:
                output_files = [
                    p
                    for p in glob.glob(os.path.join(tempdir, "**"), recursive=True)
                    if os.path.isfile(p)
                ]
                if output_files and "output_storage_urls" in upload_cfg:
                    sem = asyncio.Semaphore(8)

                    async def _wrapped_upload(idx: int):
                        async with sem:
                            await self._upload_job_file(
                                output_files[idx],
                                upload_cfg["output_storage_urls"][idx],
                            )

                    await asyncio.gather(
                        *[
                            _wrapped_upload(i)
                            for i in range(
                                min(len(output_files), len(upload_cfg["output_storage_urls"]))
                            )
                        ]
                    )
                    logger.success("Uploaded all output files, job complete!")

            # Record job completion.
            elapsed = time.time() - started_at
            status = 700 if final_result["status"] == "complete" else 750
            metrics.total_requests.labels(
                chute_id=self._app.uid, function=self._func.__name__, status=status
            ).inc()
            metrics.request_duration.labels(
                chute_id=self._app.uid, function=self._func.__name__, status=status
            ).observe(elapsed)

        return final_result

    def __call__(self, func):
        """
        Decorator.
        """
        self._func = func
        self._name = func.__name__
        return self
