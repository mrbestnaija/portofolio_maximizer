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

CURRENT SCOPE (SCAFFOLD):
- Uses realized performance summary over the specified regime window as
  a proxy for candidate performance. It does NOT yet re-simulate trades
  per candidate; that requires wiring candidate.params into a dedicated
  backtest runner and execution stack.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timedelta

import click
import yaml

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT_PATH))

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
    verbose: bool,
    prefer_gpu: bool,
) -> None:
    """Run an offline evaluation for a single candidate over a regime window.

    This is intentionally lightweight and does not participate in the
    higher-order hyper-parameter search; that orchestration is handled by
    shell helpers (e.g., bash/run_post_eval.sh) which may invoke this script.
    """
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

    db_manager = DatabaseManager(db_path=db_path)
    summary = db_manager.get_performance_summary(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    logger.info(
        "Backtest window: %s -> %s | total_trades=%s win_rate=%.3f profit_factor=%.3f total_profit=%.2f",
        start_date,
        end_date,
        summary.get("total_trades", 0),
        float(summary.get("win_rate", 0.0)),
        float(summary.get("profit_factor", 0.0)),
        float(summary.get("total_profit", 0.0)),
    )

    if candidate_json:
        logger.info("Candidate params (opaque JSON): %s", candidate_json)

    # Set device hint for downstream components/backtests
    try:
        from scripts.run_etl_pipeline import _detect_device

        device = _detect_device(prefer_gpu=prefer_gpu)
    except Exception:
        device = "cpu"
    __import__("os").environ["PIPELINE_DEVICE"] = device
    logger.info("Candidate backtest device: %s (prefer_gpu=%s)", device, prefer_gpu)


if __name__ == "__main__":
    main()
