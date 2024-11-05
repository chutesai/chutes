"""
Register on the chutes run platform.
"""

import os
import sys
import json
import glob
import time
import aiohttp
import hashlib
from loguru import logger
from pathlib import Path
from substrateinterface import Keypair
from chutes.entrypoint._shared import parse_args

CLI_ARGS = {
    "--config-path": {
        "type": str,
        "default": None,
        "help": "custom path to the parachutes config (credentials, API URL, etc.)",
    },
    "--username": {
        "type": str,
        "help": "username",
    },
    "--wallets-path": {
        "type": str,
        "default": os.path.join(Path.home(), ".bittensor", "wallets"),
        "help": "path to the bittensor wallets directory",
    },
    "--wallet": {
        "type": str,
        "help": "name of the wallet to use",
    },
    "--hotkey": {
        "type": str,
        "help": "hotkey to register with",
    },
}


async def register(input_args):
    """
    Register a user!
    """
    args = parse_args(input_args, CLI_ARGS)
    if args.config_path:
        os.environ["PARACHUTES_CONFIG_PATH"] = args.config_path
    os.environ["PARACHUTES_ALLOW_MISSING"] = "true"

    from chutes.config import API_BASE_URL, CONFIG_PATH

    # Interactive mode for username.
    if not args.username:
        args.username = input("Enter desired username: ").strip()
        if not args.username:
            logger.error("Bad choice!")
            sys.exit(1)

    # Interactive mode for wallet selection.
    if not args.wallet:
        available_wallets = sorted(
            [
                os.path.basename(item)
                for item in glob.glob(os.path.join(args.wallets_path, "*"))
                if os.path.isdir(item)
            ]
        )
        print("Wallets available (commissions soon\u2122 for image/chute use):")
        for idx in range(len(available_wallets)):
            print(f"[{idx:2d}] {available_wallets[idx]}")
        choice = input("Enter your choice (number, not name): ")
        if not choice.isdigit() or not 0 <= int(choice) < len(available_wallets):
            logger.error("Bad choice!")
            sys.exit(1)
        args.wallet = available_wallets[int(choice)]
    else:
        if not os.path.isdir(
            wallet_path := os.path.join(args.wallets_path, args.wallet)
        ):
            logger.error(f"No wallet found: {wallet_path}")
            sys.exit(1)

    # Interactive model for hotkey selection.
    if not args.hotkey:
        available_hotkeys = sorted(
            [
                os.path.basename(item)
                for item in glob.glob(
                    os.path.join(args.wallets_path, args.wallet, "hotkeys", "*")
                )
                if os.path.isfile(item)
            ]
        )
        print(f"Hotkeys available for {args.wallet}:")
        for idx in range(len(available_hotkeys)):
            print(f"[{idx:2d}] {available_hotkeys[idx]}")
        choice = input("Enter your choice (number, not name): ")
        if not choice.isdigit() or not 0 <= int(choice) < len(available_hotkeys):
            logger.error("Bad choice!")
            sys.exit(1)
        args.hotkey = available_hotkeys[int(choice)]
    if not os.path.isfile(
        hotkey_path := os.path.join(
            args.wallets_path, args.wallet, "hotkeys", args.hotkey
        )
    ):
        logger.error(f"No hotkey found: {hotkey_path}")
        sys.exit(1)

    # Send it.
    with open(hotkey_path) as infile:
        hotkey_data = json.load(infile)
    with open(os.path.join(args.wallets_path, args.wallet, "coldkeypub.txt")) as infile:
        coldkey_pub_data = json.load(infile)
    ss58 = hotkey_data["ss58Address"]
    secret_seed = hotkey_data["secretSeed"].replace("0x", "")
    coldkey_ss58 = coldkey_pub_data["ss58Address"]
    payload = json.dumps(
        {
            "username": args.username,
            "coldkey": coldkey_ss58,
        }
    )
    keypair = Keypair.create_from_seed(seed_hex=secret_seed)
    headers = {
        "Content-Type": "application/json",
        "X-Parachutes-Hotkey": ss58,
        "X-Parachutes-Nonce": str(int(time.time())),
    }
    sig_str = ":".join(
        [
            ss58,
            headers["X-Parachutes-Nonce"],
            hashlib.sha256(payload.encode()).hexdigest(),
        ]
    )
    headers["X-Parachutes-Signature"] = keypair.sign(sig_str.encode()).hex()
    async with aiohttp.ClientSession(base_url=API_BASE_URL) as session:
        async with session.post(
            "/users/register",
            data=payload,
            headers=headers,
        ) as response:
            if response.status == 200:
                data = await response.json()
                logger.success(
                    f"User created successfully: user_id={data['user_id']}, updated config.ini:"
                )
                updated_config = "\n".join(
                    [
                        "[api]",
                        f"base_url = {API_BASE_URL}",
                        "",
                        "[auth]",
                        f"user_id = {data['user_id']}",
                        f"hotkey_seed = {secret_seed}",
                        f"hotkey_name = {args.hotkey}",
                        f"hotkey_ss58address = {ss58}",
                        "",
                        "[payment]",
                        f"address = {data['payment_address']}",
                    ]
                )
                print(updated_config + "\n\n")
                save = input(f"Save to {CONFIG_PATH} (y/n): ")
                if save.strip().lower() == "y":
                    with open(CONFIG_PATH, "w") as outfile:
                        outfile.write(updated_config + "\n")
                logger.success(
                    f"Successfully registered username={data['username']}! Send tao funds to {data['payment_address']} when you are ready."
                )
            else:
                logger.error(await response.json())
