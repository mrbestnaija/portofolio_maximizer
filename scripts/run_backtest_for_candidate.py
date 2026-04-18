#!/usr/bin/env python3
"""
run_backtest_for_candidate.py
-----------------------------

Offline backtest harness for a single strategy candidate.

Design goals:
- Accept candidate parameters + a regime evaluation window.
- Replay historical performance from the database under guardrails
  (min_expected_return, max_risk_score) without hardcoding any strategy.
- Produce candidate-specific PnL metrics that can be consumed by
  StrategyOptimizer for non-convex, stochastic optimization.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import click
import yaml

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT_PATH))

from backtesting.candidate_backtester import backtest_candidate
from etl.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _load_quant_guardrails(path: Optional[Path]) -> Dict[str, Any]:
    """
    Load quant success criteria (e.g., min_expected_return, max_risk_score)
    from the existing configuration. This function does not enforce or change
    those guardrails; it only reads them for information.
    """
    if not path or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Unable to read guardrail config %s: %s", path, exc)
        return {}

    if isinstance(payload, dict) and "quant_success" in payload:
        return payload["quant_success"]
    return payload if isinstance(payload, dict) else {}


def _parse_candidate_params(candidate_json: Optional[str]) -> Dict[str, Any]:
    if not candidate_json:
        return {}
    try:
        payload = json.loads(candidate_json)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"candidate_json is not valid JSON: {exc}") from exc
    return payload if isinstance(payload, dict) else {}


@click.command()
@click.option(
    "--db-path",
    default="data/portfolio_maximizer.db",
    show_default=True,
    help="SQLite database path to read historical trades from.",
)
@click.option(
    "--regime",
    default="default",
    show_default=True,
    help="Regime label (used for evaluation window selection).",
)
@click.option(
    "--lookback-days",
    default=365,
    show_default=True,
    help="Evaluation window size in days (end = today, start = today - lookback).",
)
@click.option(
    "--candidate-json",
    default=None,
    help="Optional JSON string with candidate params for logging/audit.",
)
@click.option(
    "--quant-config-path",
    default="config/quant_success_config.yml",
    show_default=True,
    help="Path to quant success / guardrail configuration.",
)
@click.option(
    "--forecasting-config-path",
    default="config/forecasting_config.yml",
    show_default=True,
    help="Path to the canonical forecasting config used by the candidate backtest.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable debug logging.",
)
@click.option(
    "--prefer-gpu/--no-prefer-gpu",
    default=True,
    show_default=True,
    help="Prefer GPU when available; sets PIPELINE_DEVICE for downstream components.",
)
def main(
    db_path: str,
    regime: str,
    lookback_days: int,
    candidate_json: Optional[str],
    quant_config_path: str,
    forecasting_config_path: str,
    verbose: bool,
    prefer_gpu: bool,
) -> None:
    """Run an offline evaluation for a single candidate over a regime window."""
    _configure_logging(verbose)

    guardrails = _load_quant_guardrails(ROOT_PATH / quant_config_path)
    min_expected_return = guardrails.get("min_expected_return")
    max_risk_score = guardrails.get("max_risk_score")

    logger.info("Regime=%s lookback_days=%s", regime, lookback_days)
    if min_expected_return is not None or max_risk_score is not None:
        logger.info(
            "Guardrails (informational): min_expected_return=%s, max_risk_score=%s",
            min_expected_return,
            max_risk_score,
        )

    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=int(lookback_days))
    forecasting_cfg_path = Path(forecasting_config_path)
    if not forecasting_cfg_path.is_absolute():
        forecasting_cfg_path = ROOT_PATH / forecasting_cfg_path

    db_manager = DatabaseManager(db_path=db_path)
    try:
        tickers = db_manager.get_distinct_tickers()
        if not tickers:
            raise click.ClickException(
                "Candidate backtest requires ticker-backed OHLCV data; no distinct tickers were available."
            )

        candidate_params = _parse_candidate_params(candidate_json)
        result = backtest_candidate(
            db_manager=db_manager,
            tickers=tickers,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            candidate_params=candidate_params,
            guardrails=guardrails,
            forecasting_config_path=str(forecasting_cfg_path),
        )
    finally:
        db_manager.close()

    # Set device hint for downstream components/backtests.
    try:
        from scripts.run_etl_pipeline import _detect_device

        device = _detect_device(prefer_gpu=prefer_gpu)
    except Exception:
        device = "cpu"
    __import__("os").environ["PIPELINE_DEVICE"] = device

    logger.info(
        "Backtest window: %s -> %s | total_trades=%s win_rate=%.3f profit_factor=%.3f total_profit=%.2f alpha=%.4f ir=%.4f",
        start_date,
        end_date,
        result.total_trades,
        float(result.win_rate),
        float(result.profit_factor),
        float(result.total_profit),
        float(result.alpha),
        float(result.information_ratio),
    )
    logger.info("Candidate backtest device: %s (prefer_gpu=%s)", device, prefer_gpu)

    result_payload = dict(vars(result))
    strategy_returns = result_payload.pop("strategy_returns", None)
    strategy_returns_count = 0
    if strategy_returns is not None:
        try:
            strategy_returns_count = int(len(strategy_returns))
        except Exception:
            strategy_returns_count = 0
    result_payload["strategy_returns_count"] = strategy_returns_count

    report = {
        "regime": regime,
        "window": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "lookback_days": int(lookback_days),
        },
        "tickers": tickers,
        "candidate_params": candidate_params,
        "guardrails": guardrails,
        "device": device,
        "benchmark_proxy": result.benchmark_proxy,
        "benchmark_metrics_status": result.benchmark_metrics_status,
        "metrics": result_payload,
    }

    click.echo(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
