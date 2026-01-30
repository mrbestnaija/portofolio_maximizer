import pytest

from execution.ctrader_client import CTraderClientConfig


def _set_common(monkeypatch):
    monkeypatch.setenv("USERNAME_CTRADER", "generic_user")
    monkeypatch.setenv("PASSWORD_CTRADER", "generic_pass")
    monkeypatch.setenv("APPLICATION_NAME_CTRADER", "generic_app")
    monkeypatch.setenv("CTRADER_ACCOUNT_ID", "999")


def test_demo_env_keys_take_precedence(monkeypatch):
    _set_common(monkeypatch)
    monkeypatch.setenv("CTRADER_DEMO_USERNAME", "demo_user")
    monkeypatch.setenv("CTRADER_DEMO_PASSWORD", "demo_pass")
    monkeypatch.setenv("CTRADER_DEMO_APPLICATION_ID", "demo_app")
    monkeypatch.setenv("CTRADER_DEMO_ACCOUNT_ID", "123")

    cfg = CTraderClientConfig.from_env(environment="demo")

    assert cfg.username == "demo_user"
    assert cfg.password == "demo_pass"
    assert cfg.application_id == "demo_app"
    assert cfg.account_id == 123


def test_live_env_keys_take_precedence(monkeypatch):
    _set_common(monkeypatch)
    monkeypatch.setenv("CTRADER_LIVE_USERNAME", "live_user")
    monkeypatch.setenv("CTRADER_LIVE_PASSWORD", "live_pass")
    monkeypatch.setenv("CTRADER_LIVE_APPLICATION_ID", "live_app")
    monkeypatch.setenv("CTRADER_LIVE_ACCOUNT_ID", "321")

    cfg = CTraderClientConfig.from_env(environment="live")

    assert cfg.username == "live_user"
    assert cfg.password == "live_pass"
    assert cfg.application_id == "live_app"
    assert cfg.account_id == 321


def test_fallback_to_generic_keys_when_env_specific_missing(monkeypatch):
    _set_common(monkeypatch)
    monkeypatch.delenv("CTRADER_LIVE_USERNAME", raising=False)
    monkeypatch.delenv("CTRADER_LIVE_PASSWORD", raising=False)
    monkeypatch.delenv("CTRADER_LIVE_APPLICATION_ID", raising=False)
    monkeypatch.delenv("CTRADER_LIVE_ACCOUNT_ID", raising=False)

    cfg = CTraderClientConfig.from_env(environment="live")

    assert cfg.username == "generic_user"
    assert cfg.password == "generic_pass"
    assert cfg.application_id == "generic_app"
    assert cfg.account_id == 999
