from functools import lru_cache
import os
import threading
from typing import Optional
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


class ConfigManager:
    """
    Thread-safe singleton configuration manager.
    
    This class ensures that configuration is loaded only once and safely
    accessible from multiple threads without race conditions.
    """
    _instance: Optional['ConfigManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._config: Optional[Config] = None
            self._initialized = True
    
    def get_config(self) -> Config:
        """Get or load the configuration."""
        if self._config is None:
            self._config = self._load_config()
        return self._config
    
    def reload_config(self) -> Config:
        """Force reload the configuration from disk."""
        self._config = self._load_config()
        return self._config
    
    def _load_config(self) -> Config:
        """Load configuration from file."""
        auth_config = None
        if not os.path.exists(CONFIG_PATH):
            os.makedirs(os.path.dirname(os.path.abspath(CONFIG_PATH)), exist_ok=True)
            if not ALLOW_MISSING:
                raise NotConfigured(
                    f"Please set either populate {CONFIG_PATH} or set CHUTES_CONFIG_PATH to alternative/valid config path!"
                )
        else:
            logger.info(f"Loading chutes config from {CONFIG_PATH}...")
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

            api_base_url = raw_config.get("api", "base_url", fallback=None)
            if not api_base_url:
                api_base_url = os.getenv("CHUTES_API_URL", "https://api.chutes.ai")
            generic_config = GenericConfig(api_base_url=api_base_url)
            logger.debug(f"Configured chutes: with api_base_url={api_base_url}")
            return Config(auth=auth_config, generic=generic_config)
        
        # Handle case where auth_config is None
        api_base_url = os.getenv("CHUTES_API_URL", "https://api.chutes.ai")
        generic_config = GenericConfig(api_base_url=api_base_url)
        if auth_config is None:
            auth_config = AuthConfig(None, None, None, None, None)
        return Config(auth=auth_config, generic=generic_config)


@lru_cache(maxsize=1)
def get_generic_config() -> GenericConfig:
    """
    Get generic configuration (API URL, etc.).
    
    This function is cached to avoid repeated environment variable lookups.
    
    Returns:
        GenericConfig: Generic configuration object.
    """
    api_base_url = os.getenv("CHUTES_API_URL", "https://api.chutes.ai")
    return GenericConfig(api_base_url=api_base_url)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """
    Get the global Chutes configuration.
    
    This function uses a thread-safe singleton pattern to ensure configuration
    is loaded only once and safely accessible from multiple threads.
    
    Returns:
        Config: Configuration object containing auth and generic settings.
        
    Raises:
        NotConfigured: If config file doesn't exist and ALLOW_MISSING is False.
        AuthenticationRequired: If auth section is missing and ALLOW_MISSING is False.
    """
    return ConfigManager().get_config()


def reload_config() -> Config:
    """
    Force reload the configuration from disk.
    
    Useful for testing or when configuration has changed.
    
    Returns:
        Config: Freshly loaded configuration object.
    """
    get_config.cache_clear()
    return ConfigManager().reload_config()
