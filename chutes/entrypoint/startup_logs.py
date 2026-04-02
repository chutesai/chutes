"""Fetch and decrypt encrypted startup logs for private chutes."""

import asyncio
import os
from datetime import datetime

import aiohttp
import pybase64
import typer
from rich.console import Console
from rich.prompt import Prompt
from substrateinterface import Keypair

from chutes.config import get_config
from chutes.util.auth import sign_request

console = Console()


def _format_session(idx: int, s: dict) -> str:
    """Format a session for display in the picker."""
    instance_id = s["instance_id"][:12]
    chunks = s.get("chunk_count", "?")
    config = s.get("config_id", "?")[:12]
    started = s.get("started_at")
    time_str = ""
    if started:
        dt = datetime.fromtimestamp(started)
        time_str = f"  {dt.strftime('%H:%M:%S')}"
    return f"  [bold]{idx}[/]) [cyan]{instance_id}[/]{time_str}  chunks={chunks}  config={config}"


def fetch_startup_logs(
    chute_id: str = typer.Argument(..., help="Chute ID to fetch startup logs for"),
    instance_id: str = typer.Option(None, "--instance", "-i", help="Specific instance ID"),
    latest: bool = typer.Option(
        False, "--latest", "-l", help="Automatically select the most recent session"
    ),
    config_path: str = typer.Option(None, help="Custom config path"),
):
    """Fetch and decrypt encrypted startup logs for a private chute."""

    async def _fetch():
        nonlocal chute_id, instance_id, config_path

        if config_path:
            os.environ["CHUTES_CONFIG_PATH"] = config_path

        config = get_config()
        keypair = Keypair.create_from_seed(seed_hex=config.auth.hotkey_seed)
        private_key_scalar = keypair.private_key[:32]
        user_pubkey = bytes.fromhex(
            keypair.public_key[2:] if keypair.public_key.startswith("0x") else keypair.public_key
        )

        headers, _ = sign_request(purpose="chutes")

        async with aiohttp.ClientSession(base_url=config.generic.api_base_url) as session:
            # List sessions
            async with session.get(
                f"/encrypted_logs/{chute_id}/sessions",
                headers=headers,
            ) as response:
                if response.status == 404:
                    console.print("[yellow]No encrypted startup logs found for this chute.[/]")
                    return
                if response.status != 200:
                    error = await response.text()
                    console.print(f"[red]Error fetching log sessions: {response.status} {error}[/]")
                    return
                sessions = await response.json()

            if not sessions:
                console.print(
                    "[yellow]No encrypted startup logs available (logs expire after 4 hours).[/]"
                )
                return

            # Sort newest first
            sessions.sort(key=lambda x: x.get("started_at", 0), reverse=True)

            # Pick session
            if instance_id:
                target = next((s for s in sessions if s["instance_id"] == instance_id), None)
                if not target:
                    console.print(f"[red]No logs found for instance {instance_id}[/]")
                    console.print("\nAvailable sessions:")
                    for i, s in enumerate(sessions, 1):
                        console.print(_format_session(i, s))
                    return
            elif latest or len(sessions) == 1:
                target = sessions[0]
            else:
                # Interactive picker
                console.print(f"\n[bold]Found {len(sessions)} startup log sessions:[/]\n")
                for i, s in enumerate(sessions, 1):
                    console.print(_format_session(i, s))
                console.print()

                choice = Prompt.ask(
                    "Select session",
                    choices=[str(i) for i in range(1, len(sessions) + 1)],
                    default="1",
                )
                target = sessions[int(choice) - 1]

            target_instance_id = target["instance_id"]
            ephemeral_pubkey = pybase64.b64decode(target["ephemeral_pubkey"])

            console.print(
                f"\n[dim]Fetching logs for instance {target_instance_id[:12]}... "
                f"({target.get('chunk_count', '?')} chunks)[/]\n"
            )

            # Fetch chunks
            async with session.get(
                f"/encrypted_logs/{chute_id}/sessions/{target_instance_id}/chunks",
                headers=headers,
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    console.print(f"[red]Error fetching log chunks: {response.status} {error}[/]")
                    return
                data = await response.json()

        # Decrypt
        from chutes.entrypoint._decrypt_logs import decrypt_log_chunks

        plaintext = decrypt_log_chunks(
            chunks_b64=data["chunks"],
            private_key_scalar=private_key_scalar,
            user_pubkey=user_pubkey,
            ephemeral_pubkey=ephemeral_pubkey,
        )

        console.print(f"[bold]Startup logs for instance {target_instance_id[:12]}:[/]\n")
        console.print(plaintext)

    asyncio.run(_fetch())
