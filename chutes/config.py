import os
from loguru import logger
from pathlib import Path
from configparser import ConfigParser, NoSectionError
from chutes.constants import CHUTES_DIR
from chutes.exception import AuthenticationRequired, NotConfigured
from dataclasses import dataclass

os.makedirs(os.path.join(Path.home(), CHUTES_DIR), exist_ok=True)
CONFIG_PATH = os.getenv("PARACHUTES_CONFIG_PATH") or os.path.join(
    Path.home(), CHUTES_DIR, "config.ini"
)
ALLOW_MISSING = os.getenv("PARACHUTES_ALLOW_MISSING", "false").lower() == "true"


# Have a class for config to prevent errors at import time.
@dataclass
class AuthConfig:
    user_id: str | None
    hotkey_seed: str | None
    hotkey_name: str | None
    hotkey_ss58address: str | None


@dataclass
class Config:
    auth: AuthConfig
    api_base_url: str


_config = None


def get_config(without_config_file: bool = False) -> Config:
    global _config
    if _config is None:
        # def load_config(self):
        api_base_url = None
        if not os.path.exists(CONFIG_PATH):
            if not (ALLOW_MISSING or without_config_file):
                raise NotConfigured(
                    f"Please set either populate {CONFIG_PATH} or set PARACHUTES_CONFIG_PATH to alternative/valid config path!"
                )
        elif not without_config_file:
            logger.debug(f"Loading parachutes config from {CONFIG_PATH}...")
            raw_config = ConfigParser()
            raw_config.read(CONFIG_PATH)

            try:
                auth_config = AuthConfig(
                    user_id=raw_config.get("auth", "user_id"),
                    hotkey_seed=raw_config.get("auth", "hotkey_seed"),
                    hotkey_name=raw_config.get("auth", "hotkey_name"),
                    hotkey_ss58address=raw_config.get("auth", "hotkey_ss58address"),
                )

            except NoSectionError:
                if not ALLOW_MISSING:
                    raise AuthenticationRequired(
                        f"Please ensure you have an [auth] section defined in {CONFIG_PATH} with 'hotkey_seed', 'hotkey_name', and 'hotkey_ss58address' values"
                    )
            api_base_url = raw_config.get("api", "base_url")
        if not api_base_url:
            api_base_url = os.getenv("PARACHUTES_API_URL", "https://api.parachutes.ai")
        logger.debug(f"Configured parachutes: with api_base_url={api_base_url}")
        _config = Config(auth=auth_config, api_base_url=api_base_url)
    return _config
