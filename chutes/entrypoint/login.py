"""
Login to the chutes website via hotkey signature.
"""

import asyncio
import glob
import json
import os
import sys
import webbrowser
from pathlib import Path
from urllib.parse import urlencode

import aiohttp
import typer
from loguru import logger
from rich import print as rprint
from substrateinterface import Keypair

from chutes.config import get_config, get_generic_config
from chutes.exception import NotConfigured, AuthenticationRequired


def login(
    wallets_path: str = typer.Option(
        os.path.join(Path.home(), ".bittensor", "wallets"),
        help="path to the bittensor wallets directory (only used with --wallet)",
    ),
    wallet: str | None = typer.Option(
        None, help="name of the wallet to use (overrides chutes config)"
    ),
    hotkey: str | None = typer.Option(
        None, help="hotkey to use for signing (overrides chutes config)"
    ),
    browser: bool = typer.Option(
        True, help="automatically open the login URL in your default browser"
    ),
):
    """
    Login to the chutes platform using your hotkey for authentication.

    By default, uses the hotkey from your chutes config. Use --wallet and --hotkey
    to override with a different bittensor wallet.
    """

    async def _login():
        nonlocal wallet, hotkey
        ss58 = None
        keypair = None
        base_url = None

        # Try to use existing chutes config first (unless wallet is explicitly specified)
        if wallet is None:
            try:
                config = get_config()
                base_url = config.generic.api_base_url.rstrip("/")
                if config.auth.hotkey_seed and config.auth.hotkey_ss58address:
                    ss58 = config.auth.hotkey_ss58address
                    keypair = Keypair.create_from_seed(seed_hex=config.auth.hotkey_seed)
                    rprint(f"[dim]Using hotkey from chutes config: {ss58}[/dim]")
            except (NotConfigured, AuthenticationRequired):
                pass

        # Fall back to generic config if no base_url from config file
        if base_url is None:
            base_url = get_generic_config().api_base_url.rstrip("/")

        # Fall back to wallet selection if no config or wallet explicitly specified
        if keypair is None:
            # Interactive mode for wallet selection.
            if not wallet:
                available_wallets = sorted(
                    [
                        os.path.basename(item)
                        for item in glob.glob(os.path.join(wallets_path, "*"))
                        if os.path.isdir(item)
                    ]
                )
                if len(available_wallets) == 0:
                    logger.error("No wallets found in the wallets path!")
                    sys.exit(1)
                rprint("Wallets available:")
                for idx in range(len(available_wallets)):
                    rprint(f"[{idx:2d}] {available_wallets[idx]}")
                choice = input("Enter your choice (number, not name): ")
                if not choice.isdigit() or not 0 <= int(choice) < len(available_wallets):
                    logger.error("Bad choice!")
                    sys.exit(1)
                wallet = available_wallets[int(choice)]
            else:
                if not os.path.isdir(wallet_path := os.path.join(wallets_path, wallet)):
                    logger.error(f"No wallet found: {wallet_path}")
                    sys.exit(1)

            # Interactive mode for hotkey selection.
            if not hotkey:
                available_hotkeys = sorted(
                    [
                        os.path.basename(item)
                        for item in glob.glob(os.path.join(wallets_path, wallet, "hotkeys", "*"))
                        if os.path.isfile(item)
                    ]
                )
                rprint(f"Hotkeys available for {wallet}:")
                for idx in range(len(available_hotkeys)):
                    rprint(f"[{idx:2d}] {available_hotkeys[idx]}")
                choice = input("Enter your choice (number, not name): ")
                if not choice.isdigit() or not 0 <= int(choice) < len(available_hotkeys):
                    logger.error("Bad choice!")
                    sys.exit(1)
                hotkey = available_hotkeys[int(choice)]

            hotkey_path = os.path.join(wallets_path, wallet, "hotkeys", hotkey)
            if not os.path.isfile(hotkey_path):
                logger.error(f"No hotkey found: {hotkey_path}")
                sys.exit(1)

            # Load the hotkey
            with open(hotkey_path) as infile:
                hotkey_data = json.load(infile)
            ss58 = hotkey_data["ss58Address"]
            secret_seed = hotkey_data["secretSeed"].replace("0x", "")
            keypair = Keypair.create_from_seed(seed_hex=secret_seed)

        # Get nonce from IDP
        rprint("\n[bold]Fetching login nonce...[/bold]")
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(f"{base_url}/idp/cli_login/nonce") as response:
                    response.raise_for_status()
                    data = await response.json()
                    nonce = data["nonce"]
        except aiohttp.ClientError as e:
            logger.error(f"Failed to get nonce from IDP: {e}")
            sys.exit(1)

        # Sign the nonce
        signature = keypair.sign(nonce.encode()).hex()

        # Build the login URL
        params = urlencode(
            {
                "hotkey": ss58,
                "signature": signature,
                "nonce": nonce,
            }
        )
        login_url = f"{base_url}/idp/cli_login?{params}"

        if browser:
            rprint("\n[bold green]Opening browser for login...[/bold green]")
            webbrowser.open(login_url)
            rprint("[dim]If the browser didn't open, use the URL below:[/dim]")

        rprint("\n[bold]Login URL:[/bold]")
        rprint(f"[cyan]{login_url}[/cyan]\n")

    asyncio.run(_login())
