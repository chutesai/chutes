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
from chutes.entrypoint.log_read import SessionClient, prefix_instance_logs, read_instance_logs
from chutes.util.auth import sign_request


async def poll_for_instances(
    chute_name: str, config, headers, poll_interval: float = 2.0, max_wait: float = 600.0
):
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
                        logger.debug(
                            f"Failed to get chute (status {response.status}): {error_text}"
                        )
            except ValueError:
                # Re-raise ValueError (chute not found) immediately
                raise
            except Exception as e:
                logger.debug(f"Error polling for instances: {e}")

            await asyncio.sleep(poll_interval)

        # Timeout reached
        raise TimeoutError(f"No instances found for chute {chute_name} within {max_wait} seconds")


async def stream_instance_logs(instance_id: str, config):
    """Read stored logs for one instance and write them to stdout."""
    headers, _ = sign_request(purpose="logs")
    async with aiohttp.ClientSession(base_url=config.generic.api_base_url) as session:
        result = await read_instance_logs(SessionClient(session), instance_id, headers)
    if result.error:
        raise Exception(f"Failed to read logs from instance {instance_id}: {result.error}")
    logger.info(f"Read logs from instance {instance_id}...")
    if result.text:
        sys.stdout.buffer.write(result.text.encode("utf-8"))
        sys.stdout.buffer.flush()


async def plain_stream_all_instances(chute_name: str, config, headers):
    """Stream logs from all instances to stdout with instance ID prefixes."""
    active_tasks: dict[str, asyncio.Task] = {}
    seen_ids: set[str] = set()

    async def _stream_one(instance_id: str):
        short_id = instance_id[:8]
        log_headers, _ = sign_request(purpose="logs")
        try:
            async with aiohttp.ClientSession(base_url=config.generic.api_base_url) as session:
                result = await read_instance_logs(
                    SessionClient(session), instance_id, log_headers
                )
            if result.error:
                logger.warning(f"[{short_id}] {result.error}")
                return
            logger.info(f"[{short_id}] Read logs...")
            text = prefix_instance_logs(instance_id, result.text)
            if text:
                sys.stdout.buffer.write(text.encode("utf-8"))
                sys.stdout.buffer.flush()
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.warning(f"[{short_id}] Log read failed: {e}")
        finally:
            active_tasks.pop(instance_id, None)

    async def _discover_instances():
        """Get instance IDs from the chute info endpoint."""
        try:
            h, _ = sign_request(purpose="chutes")
            async with aiohttp.ClientSession(base_url=config.generic.api_base_url) as session:
                # Try quick warmup first (has gpu info too)
                async with session.get(
                    f"/chutes/warmup/{chute_name}",
                    headers=h,
                    params={"quick": "true"},
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        instances = result.get("instances", [])
                        if instances:
                            return [i.get("instance_id") for i in instances if i.get("instance_id")]
                # Fallback: chute detail endpoint
                async with session.get(f"/chutes/{chute_name}", headers=h) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [
                            i.get("instance_id")
                            for i in data.get("instances", [])
                            if i.get("instance_id")
                        ]
        except Exception as e:
            logger.warning(f"Failed to discover instances: {e}")
        return []

    logger.info(f"Watching for instances of {chute_name}...")
    try:
        while True:
            instance_ids = await _discover_instances()
            for iid in instance_ids:
                if iid not in seen_ids:
                    seen_ids.add(iid)
                    active_tasks[iid] = asyncio.create_task(_stream_one(iid))
            if not seen_ids:
                logger.info("Waiting for instances...")
            await asyncio.sleep(10.0)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        for task in active_tasks.values():
            task.cancel()


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
                current_instance_infos = await poll_for_instances(
                    chute_name, config, headers, poll_interval=1.0, max_wait=5.0
                )
                # Filter out instances we've already tried
                available_instances = [
                    inst
                    for inst in current_instance_infos
                    if inst["instance_id"] not in tried_instance_ids
                ]

                if not available_instances:
                    # No new instances available
                    if tried_instance_ids:
                        if attempt < max_retries - 1:
                            logger.info(
                                "All instances have been tried or deleted. Waiting for new instances..."
                            )
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
    stream_logs: bool = typer.Option(
        False, help="stream plain-text logs from all instances to stdout"
    ),
    dashboard: bool = typer.Option(
        False, help="launch TUI dashboard with real-time events and instance logs"
    ),
):
    async def _resolve_chute(chute_name, config, headers):
        """Fetch chute info from the API to get chute_id, node_selector, and existing instances."""
        async with aiohttp.ClientSession(base_url=config.generic.api_base_url) as session:
            async with session.get(f"/chutes/{chute_name}", headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Failed to resolve chute '{chute_name}': {error_text}")
                data = await response.json()
                return {
                    "chute_id": data.get("chute_id"),
                    "node_selector": data.get("node_selector", {}),
                    "instances": data.get("instances", []),
                }

    async def warmup():
        """
        Do the warmup.
        """
        nonlocal chute_id_or_ref_str, config_path, debug, stream_logs, dashboard
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
            await plain_stream_all_instances(chute_name, config, headers)
        else:
            # Dashboard is the default (also explicit with --dashboard)
            chute_info = await _resolve_chute(chute_name, config, headers)
            chute_id = chute_info["chute_id"]
            if not chute_id:
                logger.error("Could not resolve chute_id — falling back to simple warmup")
                await monitor_warmup(chute_name, config, headers)
                return

            from chutes.entrypoint.warmup_dashboard import WarmupDashboard

            app = WarmupDashboard(
                chute_name=chute_name,
                chute_id=chute_id,
                node_selector=chute_info["node_selector"],
                existing_instances=chute_info["instances"],
                config=config,
            )
            await app.run_async()

    return asyncio.run(warmup())
