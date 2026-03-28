"""
Tests for chutes/config.py — get_config() error handling.

Covers all edge cases surfaced in PR #65 review and follow-up analysis:
  - Config file missing entirely
  - Config file is empty
  - Config file has no [auth] section
  - Config file has [auth] section but is missing individual options (NoOptionError)
  - Config file is malformed / unparseable (MissingSectionHeaderError / ParsingError)
  - Config file is fully valid
  - Config file has [auth] but no [api] section → falls back to env/default
  - Config file has [api] section but no base_url key → falls back to env/default
  - CHUTES_ALLOW_MISSING=true suppresses all errors
  - CHUTES_API_URL env var overrides default api_base_url
"""

import os
import tempfile
import textwrap

import pytest

from chutes.exception import AuthenticationRequired, NotConfigured


# ---------------------------------------------------------------------------
# Helper: patch the module-level CONFIG_PATH and reset cached _config between
# tests so each test gets a fresh run.
# ---------------------------------------------------------------------------

def _reset_config(monkeypatch, path: str):
    """Point get_config() at a custom path and clear the module-level cache."""
    import chutes.config as cfg_module
    monkeypatch.setenv("CHUTES_CONFIG_PATH", path)
    monkeypatch.setattr(cfg_module, "_config", None)
    # Re-evaluate CONFIG_PATH (it's a module-level expression, re-read from env)
    monkeypatch.setattr(cfg_module, "CONFIG_PATH", path)


def _reset_allow_missing(monkeypatch, value: str = "false"):
    import chutes.config as cfg_module
    monkeypatch.setenv("CHUTES_ALLOW_MISSING", value)
    monkeypatch.setattr(cfg_module, "ALLOW_MISSING", value.lower() == "true")


# ---------------------------------------------------------------------------
# 1. Config file does not exist → NotConfigured
# ---------------------------------------------------------------------------

def test_missing_config_raises_not_configured(monkeypatch, tmp_path):
    nonexistent = str(tmp_path / "does_not_exist.ini")
    _reset_config(monkeypatch, nonexistent)
    _reset_allow_missing(monkeypatch, "false")

    from chutes.config import get_config
    with pytest.raises(NotConfigured):
        get_config()


# ---------------------------------------------------------------------------
# 2. Config file does not exist + ALLOW_MISSING=true → returns defaults
# ---------------------------------------------------------------------------

def test_missing_config_allow_missing(monkeypatch, tmp_path):
    nonexistent = str(tmp_path / "does_not_exist.ini")
    _reset_config(monkeypatch, nonexistent)
    _reset_allow_missing(monkeypatch, "true")

    from chutes.config import get_config
    cfg = get_config()
    assert cfg.auth.user_id is None
    assert cfg.auth.hotkey_seed is None
    assert cfg.generic.api_base_url == "https://api.chutes.ai"


# ---------------------------------------------------------------------------
# 3. Config file is empty → no [auth] section → AuthenticationRequired
# ---------------------------------------------------------------------------

def test_empty_config_raises_auth_required(monkeypatch, tmp_path):
    cfg_file = tmp_path / "config.ini"
    cfg_file.write_text("")
    _reset_config(monkeypatch, str(cfg_file))
    _reset_allow_missing(monkeypatch, "false")

    from chutes.config import get_config
    with pytest.raises(AuthenticationRequired):
        get_config()


# ---------------------------------------------------------------------------
# 4. Config has [auth] section but some options are missing (NoOptionError)
# ---------------------------------------------------------------------------

def test_partial_auth_section_raises_auth_required(monkeypatch, tmp_path):
    cfg_file = tmp_path / "config.ini"
    cfg_file.write_text(textwrap.dedent("""\
        [auth]
        user_id = abc123
        username = testuser
        hotkey_name = default
        # hotkey_seed and hotkey_ss58address intentionally omitted
    """))
    _reset_config(monkeypatch, str(cfg_file))
    _reset_allow_missing(monkeypatch, "false")

    from chutes.config import get_config
    with pytest.raises(AuthenticationRequired):
        get_config()


# ---------------------------------------------------------------------------
# 5. Config has [auth] section header only, no keys at all (NoOptionError)
# ---------------------------------------------------------------------------

def test_empty_auth_section_raises_auth_required(monkeypatch, tmp_path):
    cfg_file = tmp_path / "config.ini"
    cfg_file.write_text("[auth]\n")
    _reset_config(monkeypatch, str(cfg_file))
    _reset_allow_missing(monkeypatch, "false")

    from chutes.config import get_config
    with pytest.raises(AuthenticationRequired):
        get_config()


# ---------------------------------------------------------------------------
# 6. Malformed config (no section headers) → NotConfigured
# ---------------------------------------------------------------------------

def test_malformed_config_raises_not_configured(monkeypatch, tmp_path):
    cfg_file = tmp_path / "config.ini"
    cfg_file.write_text("this_is_not_a_section = value\n")
    _reset_config(monkeypatch, str(cfg_file))
    _reset_allow_missing(monkeypatch, "false")

    from chutes.config import get_config
    with pytest.raises(NotConfigured):
        get_config()


# ---------------------------------------------------------------------------
# 7. Malformed config + ALLOW_MISSING=true → returns defaults (no raise)
# ---------------------------------------------------------------------------

def test_malformed_config_allow_missing(monkeypatch, tmp_path):
    cfg_file = tmp_path / "config.ini"
    cfg_file.write_text("not_ini_format\n")
    _reset_config(monkeypatch, str(cfg_file))
    _reset_allow_missing(monkeypatch, "true")

    from chutes.config import get_config
    cfg = get_config()
    assert cfg.auth.user_id is None
    assert cfg.generic.api_base_url == "https://api.chutes.ai"


# ---------------------------------------------------------------------------
# 8. Valid [auth] but no [api] section → falls back to default URL
# ---------------------------------------------------------------------------

def test_valid_auth_no_api_section_uses_default_url(monkeypatch, tmp_path):
    cfg_file = tmp_path / "config.ini"
    cfg_file.write_text(textwrap.dedent("""\
        [auth]
        user_id = u1
        username = alice
        hotkey_seed = deadbeef
        hotkey_name = default
        hotkey_ss58address = 5Abcdef
    """))
    _reset_config(monkeypatch, str(cfg_file))
    _reset_allow_missing(monkeypatch, "false")
    monkeypatch.delenv("CHUTES_API_URL", raising=False)

    from chutes.config import get_config
    cfg = get_config()
    assert cfg.auth.user_id == "u1"
    assert cfg.auth.username == "alice"
    assert cfg.auth.hotkey_seed == "deadbeef"
    assert cfg.generic.api_base_url == "https://api.chutes.ai"


# ---------------------------------------------------------------------------
# 9. Valid [auth] + [api] section present → uses the configured URL
# ---------------------------------------------------------------------------

def test_valid_full_config(monkeypatch, tmp_path):
    cfg_file = tmp_path / "config.ini"
    cfg_file.write_text(textwrap.dedent("""\
        [auth]
        user_id = u2
        username = bob
        hotkey_seed = cafebabe
        hotkey_name = myhotkey
        hotkey_ss58address = 5Xyzabc

        [api]
        base_url = https://api.custom.example.com
    """))
    _reset_config(monkeypatch, str(cfg_file))
    _reset_allow_missing(monkeypatch, "false")

    from chutes.config import get_config
    cfg = get_config()
    assert cfg.auth.user_id == "u2"
    assert cfg.auth.hotkey_ss58address == "5Xyzabc"
    assert cfg.generic.api_base_url == "https://api.custom.example.com"


# ---------------------------------------------------------------------------
# 10. [api] section present but base_url key missing (NoOptionError) →
#     falls back to env/default
# ---------------------------------------------------------------------------

def test_api_section_missing_base_url_uses_default(monkeypatch, tmp_path):
    cfg_file = tmp_path / "config.ini"
    cfg_file.write_text(textwrap.dedent("""\
        [auth]
        user_id = u3
        username = carol
        hotkey_seed = 0011aabb
        hotkey_name = default
        hotkey_ss58address = 5Zzz

        [api]
        some_other_key = irrelevant
    """))
    _reset_config(monkeypatch, str(cfg_file))
    _reset_allow_missing(monkeypatch, "false")
    monkeypatch.delenv("CHUTES_API_URL", raising=False)

    from chutes.config import get_config
    cfg = get_config()
    assert cfg.generic.api_base_url == "https://api.chutes.ai"


# ---------------------------------------------------------------------------
# 11. CHUTES_API_URL env var overrides the hardcoded default
# ---------------------------------------------------------------------------

def test_env_var_overrides_default_api_url(monkeypatch, tmp_path):
    cfg_file = tmp_path / "config.ini"
    cfg_file.write_text(textwrap.dedent("""\
        [auth]
        user_id = u4
        username = dave
        hotkey_seed = aabbccdd
        hotkey_name = default
        hotkey_ss58address = 5Dave
    """))
    _reset_config(monkeypatch, str(cfg_file))
    _reset_allow_missing(monkeypatch, "false")
    monkeypatch.setenv("CHUTES_API_URL", "https://api.staging.example.com")

    from chutes.config import get_config
    cfg = get_config()
    assert cfg.generic.api_base_url == "https://api.staging.example.com"


# ---------------------------------------------------------------------------
# 12. ALLOW_MISSING=true + partial auth → returns None fields, no raise
# ---------------------------------------------------------------------------

def test_allow_missing_with_partial_auth(monkeypatch, tmp_path):
    cfg_file = tmp_path / "config.ini"
    cfg_file.write_text("[auth]\nuser_id = partial\n")
    _reset_config(monkeypatch, str(cfg_file))
    _reset_allow_missing(monkeypatch, "true")

    from chutes.config import get_config
    cfg = get_config()
    # Should not raise; auth fields that were missing remain None
    assert cfg.auth.user_id is None   # reset to default since exception was swallowed
    assert cfg.auth.hotkey_seed is None
