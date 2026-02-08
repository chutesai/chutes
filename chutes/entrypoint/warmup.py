"""
Warm up a chute.
"""

import os
import asyncio
import aiohttp
import orjson as json
import sys
import time
from loguru import logger
import typer
from chutes.config import get_config
from chutes.entrypoint._shared import load_chute
from chutes.util.auth import sign_request


async def poll_for_instances(chute_name: str, config, headers, poll_interval: float = 2.0, max_wait: float = 600.0):
    """
    Poll for instances of a chute. Returns a list of instance info dicts (with instance_id and status).
    
    Args:
        chute_name: Name or ID of the chute
        config: Config object
        headers: Request headers
        poll_interval: Seconds between polls
        max_wait: Maximum seconds to wait for an instance (default 10 minutes)
    
    Returns:
        List of dicts with 'instance_id' and status information
    """
    start_time = time.time()
    async with aiohttp.ClientSession(base_url=config.generic.api_base_url) as session:
        while time.time() - start_time < max_wait:
            try:
                async with session.get(
                    f"/chutes/{chute_name}",
                    headers=headers,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        instances = data.get("instances", [])
                        if instances:
                            # Return all instances with their status info
                            instance_infos = [
                                {
                                    "instance_id": inst.get("instance_id"),
                                    "active": inst.get("active", False),
                                    "verified": inst.get("verified", False),
                                    "region": inst.get("region", "n/a"),
                                    "last_verified_at": inst.get("last_verified_at"),
                                }
                                for inst in instances
                                if inst.get("instance_id")
                            ]
                            if instance_infos:
                                return instance_infos
                    elif response.status == 404:
                        # Chute doesn't exist - this is an error, not a polling condition
                        error_text = await response.text()
                        raise ValueError(f"Chute '{chute_name}' not found: {error_text}")
                    else:
                        error_text = await response.text()
                        logger.debug(f"Failed to get chute (status {response.status}): {error_text}")
            except ValueError:
                # Re-raise ValueError (chute not found) immediately
                raise
            except Exception as e:
                logger.debug(f"Error polling for instances: {e}")
            
            await asyncio.sleep(poll_interval)
        
        # Timeout reached
        raise TimeoutError(f"No instances found for chute {chute_name} within {max_wait} seconds")


async def stream_instance_logs(instance_id: str, config, backfill: int = 100):
    """
    Stream logs from an instance.
    
    Raises:
        aiohttp.ClientResponseError: If the request fails
        Exception: For other streaming errors
    """
    # Sign the request with purpose for instances endpoint
    headers, _ = sign_request(purpose="logs")
    async with aiohttp.ClientSession(base_url=config.generic.api_base_url) as session:
        async with session.get(
            f"/instances/{instance_id}/logs",
            headers=headers,
            params={"backfill": str(backfill)},
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=f"Failed to stream logs from instance {instance_id}: {error_text}",
                )
            
            logger.info(f"Streaming logs from instance {instance_id}...")
            # Parse SSE format and extract log content
            buffer = b""
            skipped_lines_count = 0
            try:
                async for chunk in response.content.iter_any():
                    if chunk:
                        buffer += chunk
                        # Process complete lines
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            # Skip empty lines completely
                            if not line.strip():
                                skipped_lines_count += 1
                                continue
                            # Skip SSE comment lines (lines starting with :)
                            if line.startswith(b":"):
                                skipped_lines_count += 1
                                continue
                            # Only process SSE data lines - ignore everything else
                            if line.startswith(b"data: "):
                                # Check for empty data: lines (just "data: " or "data:")
                                data_content = line[6:].strip() if len(line) > 6 else b""
                                if not data_content:
                                    skipped_lines_count += 1
                                    continue
                                try:
                                    # Parse JSON from SSE data line
                                    data = json.loads(data_content)
                                    log_message = data.get("log", "")
                                    # Only output if we have actual non-empty log content
                                    # Also check that it's not just a single character like "."
                                    if log_message and log_message.strip() and len(log_message.strip()) > 1:
                                        # Write just the log message with a newline
                                        sys.stdout.buffer.write(log_message.encode("utf-8") + b"\n")
                                        sys.stdout.buffer.flush()
                                    elif log_message and log_message.strip():
                                        # Single character - likely a keepalive, skip it
                                        skipped_lines_count += 1
                                        logger.debug(f"Skipping single-character log message (likely keepalive): {repr(log_message)}")
                                    else:
                                        skipped_lines_count += 1
                                except (json.JSONDecodeError, KeyError) as e:
                                    # Log what we're skipping for debugging
                                    logger.debug(f"Skipping unparseable SSE line: {line[:100]}, error: {e}")
                                    skipped_lines_count += 1
                            else:
                                # Skip non-SSE lines silently (likely keepalive messages)
                                skipped_lines_count += 1
            except asyncio.CancelledError:
                raise
            except (aiohttp.ClientError, ConnectionError, OSError) as e:
                # Connection errors - instance might have been deleted
                raise Exception(f"Stream ended for instance {instance_id} (instance may have been deleted): {e}") from e
            except Exception as e:
                # Re-raise to allow caller to handle (e.g., try another instance)
                raise Exception(f"Error streaming logs from instance {instance_id}: {e}") from e


async def monitor_warmup(chute_name: str, config, headers):
    """
    Monitor the warmup stream and log status updates.
    """
    async with aiohttp.ClientSession(base_url=config.generic.api_base_url) as session:
        async with session.get(
            f"/chutes/warmup/{chute_name}",
            headers=headers,
        ) as response:
            if response.status == 200:
                async for raw_chunk in response.content:
                    if raw_chunk.startswith(b"data:"):
                        chunk = json.loads(raw_chunk[5:])
                        if chunk["status"] == "hot":
                            logger.success(chunk["log"])
                        else:
                            logger.warning(f"Status: {chunk['status']} -- {chunk['log']}")
            else:
                logger.error(await response.text())


async def poll_and_stream_logs(chute_name: str, config, headers):
    """
    Poll for instances and stream logs when found. Tries multiple instances if one fails.
    """
    try:
        instance_infos = await poll_for_instances(chute_name, config, headers)
        logger.info(f"Found {len(instance_infos)} instance(s), attempting to stream logs...")
        
        
        # Try each instance until one works
        # Keep trying instances even if they get deleted during streaming
        tried_instance_ids = set()
        max_retries = 10  # Limit retries to avoid infinite loops
        
        for attempt in range(max_retries):
            # Refresh instance list in case instances were deleted
            try:
                current_instance_infos = await poll_for_instances(chute_name, config, headers, poll_interval=1.0, max_wait=5.0)
                # Filter out instances we've already tried
                available_instances = [
                    inst for inst in current_instance_infos
                    if inst["instance_id"] not in tried_instance_ids
                ]
                
                if not available_instances:
                    # No new instances available
                    if tried_instance_ids:
                        if attempt < max_retries - 1:
                            logger.info("All instances have been tried or deleted. Waiting for new instances...")
                            await asyncio.sleep(2.0)
                            continue
                        else:
                            raise Exception("No new instances available after retries")
                    else:
                        raise Exception("No instances available for log streaming")
                
                # Try the first available instance
                inst_info = available_instances[0]
                instance_id = inst_info["instance_id"]
                tried_instance_ids.add(instance_id)
                
                logger.info(f"Attempting to stream logs from instance {instance_id}...")
                
                try:
                    # Stream logs - this will continue until interrupted or stream ends
                    await stream_instance_logs(instance_id, config)
                    # If we get here, streaming completed successfully (unlikely for a stream)
                    return
                except asyncio.CancelledError:
                    # User interrupted, don't try other instances
                    raise
                except Exception as e:
                    # Stream ended (instance deleted, connection error, etc.)
                    logger.warning(f"Stream ended for instance {instance_id}: {e}")
                    logger.info("Looking for another instance to stream from...")
                    # Continue the loop to try another instance
                    continue
            except TimeoutError:
                # No instances found when refreshing
                if tried_instance_ids and attempt < max_retries - 1:
                    logger.info("No new instances found. Waiting...")
                    await asyncio.sleep(2.0)
                    continue
                else:
                    raise
        
        # Exhausted retries
        raise Exception(f"Failed to maintain stream after {max_retries} attempts")
            
    except asyncio.CancelledError:
        pass
    except TimeoutError as e:
        logger.warning(str(e))
    except Exception as e:
        logger.error(f"Error in poll_and_stream_logs: {e}")
        raise


def warmup_chute(
    chute_id_or_ref_str: str = typer.Argument(
        ...,
        help="The chute file to warm up, format filename:chutevarname",
    ),
    config_path: str = typer.Option(
        None, help="Custom path to the chutes config (credentials, API URL, etc.)"
    ),
    debug: bool = typer.Option(False, help="enable debug logging"),
    stream_logs: bool = typer.Option(False, help="automatically stream logs from the first instance that appears"),
):
    async def warmup():
        """
        Do the warmup.
        """
        nonlocal chute_id_or_ref_str, config_path, debug, stream_logs
        chute_name = chute_id_or_ref_str
        if ":" in chute_id_or_ref_str and os.path.exists(chute_id_or_ref_str.split(":")[0] + ".py"):
            from chutes.chute.base import Chute

            _, chute = load_chute(chute_id_or_ref_str, config_path=config_path, debug=debug)
            chute_name = chute.name if isinstance(chute, Chute) else chute.chute.name
        config = get_config()
        if config_path:
            os.environ["CHUTES_CONFIG_PATH"] = config_path
        headers, _ = sign_request(purpose="chutes")
        
        if stream_logs:
            # Run warmup monitoring and log streaming in parallel
            warmup_task = asyncio.create_task(monitor_warmup(chute_name, config, headers))
            poll_task = asyncio.create_task(poll_and_stream_logs(chute_name, config, headers))
            
            # Wait for both tasks, but log streaming should continue even after warmup completes
            try:
                # Wait for warmup to complete (or be cancelled)
                try:
                    await warmup_task
                except Exception as e:
                    logger.debug(f"Warmup task ended: {e}")
                
                # Log streaming continues independently - wait for it or user interrupt
                try:
                    await poll_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.debug(f"Poll task ended: {e}")
            except KeyboardInterrupt:
                warmup_task.cancel()
                poll_task.cancel()
                try:
                    await asyncio.gather(warmup_task, poll_task, return_exceptions=True)
                except Exception:
                    pass
                raise
        else:
            # Just monitor warmup
            await monitor_warmup(chute_name, config, headers)

    return asyncio.run(warmup())
