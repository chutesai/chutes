import os
from pathlib import Path
from configparser import ConfigParser, NoSectionError
from chutedk.exception import AuthenticationRequired, NotConfigured

CONFIG_PATH = os.getenv("PARACHUTES_CONFIG_PATH") or os.path.join(
    Path.home(), ".parachute", "config"
)
if not os.path.exists(CONFIG_PATH):
    raise NotConfigured(
        f"Please set either populate {CONFIG_PATH} or set PARACHUTES_CONFIG_PATH to alternative/valid config path!"
    )
CONFIG = ConfigParser()
CONFIG.read(CONFIG_PATH)
if not (CLIENT_ID := CONFIG.get("auth", "client_id")):
    raise AuthenticationRequired(
        f"Please ensure you have an [auth] section defined in {CONFIG_PATH} with 'client_id' value"
    )
if not (API_KEY := CONFIG.get("auth", "api_key")):
    raise AuthenticationRequired(
        f"Please ensure you have an [auth] section defined in {CONFIG_PATH} with 'api_key' value"
    )
API_BASE_URL = None
try:
    API_BASE_URL = CONFIG.get("api", "base_url")
except NoSectionError:
    ...
if not API_BASE_URL:
    API_BASE_URL = (
        "https://api.parachutes.ai"
        if not os.getenv("PARACHUTES_DEV_MODE")
        else "http://127.0.0.1:8000"
    )
