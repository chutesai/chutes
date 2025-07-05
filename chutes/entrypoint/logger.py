import json
import asyncio
import os
from pathlib import Path
from typing import Optional, List, AsyncGenerator
from datetime import datetime
import aiofiles
from fastapi import Query, HTTPException, APIRouter
from fastapi.responses import StreamingResponse

LOG_BASE = os.getenv("LOG_BASE", "/tmp/_chute.log")
POLL_INTERVAL = 0.25

router = APIRouter()


def get_available_logs() -> List[str]:
    """
    Get list of available log files.
    """
    logs = []
    if Path(LOG_BASE).exists():
        logs.append("current")
    for i in range(1, 5):
        if Path(f"{LOG_BASE}.{i}").exists():
            logs.append(str(i))
    return logs


def get_log_path(filename: str) -> Path:
    """
    Convert filename parameter to actual path.
    """
    if filename == "current":
        return Path(LOG_BASE)
    elif filename in ["1", "2", "3", "4"]:
        return Path(f"{LOG_BASE}.{filename}")
    else:
        raise ValueError(f"Invalid filename: {filename}")


async def read_last_n_lines(filepath: Path, n: Optional[int] = None) -> List[str]:
    """
    Read last n lines from a file asynchronously.
    """
    if not filepath.exists():
        return []
    async with aiofiles.open(filepath, "r") as f:
        if n is None:
            content = await f.read()
            return content.splitlines()
        else:
            async with aiofiles.open(filepath, "rb") as fb:
                await fb.seek(0, 2)
                file_length = await fb.tell()
                buffer = bytearray()
                lines_found = 0
                position = file_length
                while lines_found < n and position > 0:
                    chunk_size = min(4096, position)
                    position -= chunk_size
                    await fb.seek(position)
                    chunk = await fb.read(chunk_size)
                    buffer = chunk + buffer
                    lines_found = buffer.count(b"\n")

                text = buffer.decode("utf-8", errors="ignore")
                all_lines = text.splitlines()
                return all_lines[-n:] if len(all_lines) > n else all_lines


@router.get("")
async def list_logs():
    """
    List available log files.
    """
    logs = get_available_logs()
    log_info = []
    for log in logs:
        path = get_log_path(log)
        if path.exists():
            stat = path.stat()
            log_info.append(
                {
                    "name": log,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "path": str(path),
                }
            )
    return {"logs": log_info}


@router.get("/read/{filename}")
async def read_log(
    filename: str,
    lines: Optional[int] = Query(None, description="Number of lines to read from end (None = all)"),
):
    """
    Read contents from a log file.
    """
    try:
        path = get_log_path(filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Log file '{filename}' not found")
    log_lines = await read_last_n_lines(path, lines)
    return {
        "filename": filename,
        "path": str(path),
        "lines_requested": lines,
        "lines_returned": len(log_lines),
        "content": log_lines,
    }


async def log_streamer(filename: str, backfill: Optional[int] = None) -> AsyncGenerator[str, None]:
    """
    Stream log updates via SSE.
    """
    try:
        path = get_log_path(filename)
    except ValueError:
        yield f'data: {json.dumps({"error": f"Invalid filename: {filename}"})}\n\n'
        return
    if backfill is not None:
        lines = await read_last_n_lines(path, backfill)
        for line in lines:
            yield f'data: {json.dumps({"log": line.strip()})}\n\n'
    last_position = 0
    if path.exists():
        last_position = path.stat().st_size

    while True:
        try:
            if path.exists():
                current_size = path.stat().st_size
                if current_size > last_position:
                    async with aiofiles.open(path, "r") as f:
                        await f.seek(last_position)
                        new_content = await f.read()
                        for line in new_content.splitlines():
                            if line.strip():
                                yield f'data: {json.dumps({"log": line.strip()})}\n\n'

                        last_position = current_size
                elif current_size < last_position:
                    yield f'data: {json.dumps({"event": "file_rotated", "filename": filename})}\n\n'
                    last_position = 0
            await asyncio.sleep(POLL_INTERVAL)
        except asyncio.CancelledError:
            break
        except Exception as e:
            yield f'data: {json.dumps({"error": str(e)})}\n\n'
            await asyncio.sleep(POLL_INTERVAL)


@router.get("/stream")
async def stream_log(
    backfill: Optional[int] = Query(
        None, description="Number of recent lines to send before streaming (None = all)"
    ),
):
    """
    Stream log updates via SSE, current file only.
    """
    return StreamingResponse(
        log_streamer("current", backfill),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
