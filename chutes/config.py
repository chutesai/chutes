import os
from loguru import logger
from pathlib import Path
from configparser import ConfigParser, NoSectionError
from chutes.exception import AuthenticationRequired, NotConfigured

CONFIG_PATH = os.getenv("PARACHUTES_CONFIG_PATH") or os.path.join(
    Path.home(), ".parachutes", "config.ini"
)
CONFIG = None
HOTKEY_SEED = None
HOTKEY_NAME = None
HOTKEY_SS58 = None
API_BASE_URL = None
USER_ID = None
ALLOW_MISSING = os.getenv("PARACHUTES_ALLOW_MISSING", "false").lower() == "true"
if not os.path.exists(CONFIG_PATH):
    if not ALLOW_MISSING:
        raise NotConfigured(
            f"Please set either populate {CONFIG_PATH} or set PARACHUTES_CONFIG_PATH to alternative/valid config path!"
        )
else:
    logger.debug(f"Loading parachutes config from {CONFIG_PATH}...")
    CONFIG = ConfigParser()
    CONFIG.read(CONFIG_PATH)
    try:
        USER_ID = CONFIG.get("auth", "user_id")
        HOTKEY_SEED = CONFIG.get("auth", "hotkey_seed")
        HOTKEY_NAME = CONFIG.get("auth", "hotkey_name")
        HOTKEY_SS58 = CONFIG.get("auth", "hotkey_ss58address")
    except NoSectionError:
        if not ALLOW_MISSING:
            raise AuthenticationRequired(
                f"Please ensure you have an [auth] section defined in {CONFIG_PATH} with 'hotkey_seed', 'hotkey_name', and 'hotkey_ss58address' values"
            )
API_BASE_URL = None
if CONFIG:
    try:
        API_BASE_URL = CONFIG.get("api", "base_url")
    except NoSectionError:
        ...
if not API_BASE_URL:
    API_BASE_URL = os.getenv("PARACHUTES_API_URL", "https://api.parachutes.ai")
logger.debug(f"Configured parachutes: with api_base_url={API_BASE_URL}")
