from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_auto_trader_core_sanitizes_forecast_audits_before_running() -> None:
    text = (_repo_root() / "bash" / "production_cron.sh").read_text(encoding="utf-8")

    assert "sanitize_forecast_audits()" in text
    sanitize_idx = text.index("sanitize_production_forecast_audits.py")
    trader_idx = text.index('scripts/run_auto_trader.py --tickers "${CORE_TICKERS}"')
    assert sanitize_idx < trader_idx
    assert "--apply" in text
