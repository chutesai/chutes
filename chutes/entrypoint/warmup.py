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


async def poll_for_instance(chute_name: str, config, headers, poll_interval: float = 2.0, max_wait: float = 600.0):
    """
    Poll for instances of a chute. Returns the first instance_id found (regardless of active status).
    
    Args:
        chute_name: Name or ID of the chute
        config: Config object
        headers: Request headers
        poll_interval: Seconds between polls
        max_wait: Maximum seconds to wait for an instance (default 10 minutes)
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
                            # Return the first instance_id found (not just active ones)
                            instance_id = instances[0].get("instance_id")
                            if instance_id:
                                return instance_id
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


async def stream_instance_logs(instance_id: str, config, headers, backfill: int = 100):
    """
    Stream logs from an instance.
    """
    async with aiohttp.ClientSession(base_url=config.generic.api_base_url) as session:
        try:
            async with session.get(
                f"/instances/{instance_id}/logs",
                headers=headers,
                params={"backfill": str(backfill)},
            ) as response:
                if response.status == 200:
                    logger.info(f"Streaming logs from instance {instance_id}...")
                    # Stream the response content directly to stdout
                    async for chunk in response.content.iter_any():
                        if chunk:
                            sys.stdout.buffer.write(chunk)
                            sys.stdout.buffer.flush()
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to stream logs: {error_text}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error streaming logs: {e}")


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
    Poll for instances and stream logs when found.
    """
    try:
        instance_id = await poll_for_instance(chute_name, config, headers)
        logger.info(f"Found instance {instance_id}, starting log stream...")
        # Stream logs - this will continue until interrupted or stream ends
        await stream_instance_logs(instance_id, config, headers)
    except asyncio.CancelledError:
        pass
    except TimeoutError as e:
        logger.warning(str(e))
    except Exception as e:
        logger.error(f"Error in poll_and_stream_logs: {e}")


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
