from functools import lru_cache
import os
from loguru import logger
from pathlib import Path
from configparser import ConfigParser, NoSectionError
from chutes.constants import CHUTES_DIR
from chutes.exception import AuthenticationRequired, NotConfigured
from dataclasses import dataclass

CONFIG_PATH = os.getenv("CHUTES_CONFIG_PATH") or os.path.join(Path.home(), CHUTES_DIR, "config.ini")
ALLOW_MISSING = os.getenv("CHUTES_ALLOW_MISSING", "false").lower() == "true"


# Have a class for config to prevent errors at import time.
@dataclass
class AuthConfig:
    user_id: str | None
    username: str | None
    hotkey_seed: str | None
    hotkey_name: str | None
    hotkey_ss58address: str | None


@dataclass
class GenericConfig:
    api_base_url: str


@dataclass
class Config:
    auth: AuthConfig
    generic: GenericConfig


_config = None


@lru_cache
def get_generic_config() -> GenericConfig:
    api_base_url = os.getenv("CHUTES_API_URL", "https://api.chutes.ai")
    return GenericConfig(api_base_url=api_base_url)


def get_config() -> Config:
    global _config
    if _config is None:
        # def load_config(self):
        if not os.path.exists(CONFIG_PATH):
            os.makedirs(os.path.dirname(os.path.abspath(CONFIG_PATH)), exist_ok=True)
            if not ALLOW_MISSING:
                raise NotConfigured(
                    f"Please set either populate {CONFIG_PATH} or set CHUTES_CONFIG_PATH to alternative/valid config path!"
                )
        else:
            logger.debug(f"Loading chutes config from {CONFIG_PATH}...")
            raw_config = ConfigParser()
            raw_config.read(CONFIG_PATH)

            try:
                auth_config = AuthConfig(
                    user_id=raw_config.get("auth", "user_id"),
                    username=raw_config.get("auth", "username"),
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
            api_base_url = os.getenv("CHUTES_API_URL", "https://api.chutes.ai")
        generic_config = GenericConfig(api_base_url=api_base_url)
        logger.debug(f"Configured chutes: with api_base_url={api_base_url}")
        _config = Config(auth=auth_config, generic=generic_config)
    return _config
