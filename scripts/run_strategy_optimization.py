#!/usr/bin/env python3
"""
run_strategy_optimization.py
----------------------------

Stochastic, configuration-driven strategy optimization.

This script:
- Loads a strategy optimization config (search space, objectives, constraints).
- Samples candidate configurations via StrategyOptimizer.
- Evaluates each candidate using realized performance metrics from the database.

IMPORTANT:
- This is infrastructure only. It does not hardcode any "best" strategy.
- The evaluation function currently uses aggregate performance summary as a
  placeholder. Future work should plug in per-candidate backtests that respect
  min_expected_return, max_risk_score, and other guardrails.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

import click
import yaml

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from etl.database_manager import DatabaseManager
from etl.portfolio_math import portfolio_metrics_ngn
from etl.strategy_optimizer import StrategyOptimizer, StrategyCandidate
from backtesting.candidate_backtester import backtest_candidate


logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Strategy optimization config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload.get("strategy_optimization", {})


def _load_signal_guardrails(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    signal_routing = payload.get("signal_routing") if isinstance(payload, dict) else {}
    return (signal_routing or {}).get("time_series") or {}


@click.command()
@click.option(
    "--config-path",
    default="config/strategy_optimization_config.yml",
    show_default=True,
    help="Path to strategy optimization YAML config.",
)
@click.option(
    "--db-path",
    default="data/portfolio_maximizer.db",
    show_default=True,
    help="SQLite database path to read realized metrics from.",
)
@click.option(
    "--n-candidates",
    default=20,
    show_default=True,
    help="Number of candidate configurations to sample and evaluate.",
)
@click.option(
    "--regime",
    default="default",
    show_default=True,
    help="Regime label to associate with this optimization batch.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable debug logging.",
)
def main(
    config_path: str,
    db_path: str,
    n_candidates: int,
    regime: str,
    verbose: bool,
) -> None:
    """Entry point for stochastic strategy optimization."""
    _configure_logging(verbose)

    cfg = _load_config(ROOT_PATH / config_path)
    search_space = cfg.get("search_space", {})
    objectives = cfg.get("objectives", {})
    constraints = cfg.get("constraints", {})
    evaluation_cfg = cfg.get("evaluation", {}) or {}
    regimes_cfg = cfg.get("regimes", {}) or {}
    if not search_space:
        raise click.UsageError("Empty search_space in strategy optimization config.")

    optimizer = StrategyOptimizer(
        search_space=search_space,
        objectives=objectives,
        constraints=constraints,
    )

    db_manager = DatabaseManager(db_path=db_path)
    signal_guardrails = _load_signal_guardrails(
        ROOT_PATH / str(evaluation_cfg.get("signal_routing_config_path") or "config/signal_routing_config.yml")
    )
    raw_ticker_limit = evaluation_cfg.get("ticker_limit")
    try:
        ticker_limit = int(raw_ticker_limit) if raw_ticker_limit is not None else None
    except (TypeError, ValueError):
        ticker_limit = None
    tickers = db_manager.get_distinct_tickers(limit=ticker_limit)
    if not tickers:
        raise click.ClickException(
            "Strategy optimization requires ticker-backed OHLCV data; no distinct tickers were available."
        )

    def evaluation_fn(candidate: StrategyCandidate) -> Dict[str, float]:
        """
        Evaluate a candidate using realized performance metrics for a regime.

        Notes
        -----
        - Regime-aware via the evaluation window and uses ONLY realized trades
          stored in the database; it does not synthesize new signals/trades.
        - Consults time series forecaster regression metrics
          (RMSE/sMAPE/tracking_error) stored in time_series_forecasts so that
          candidates are only rewarded when the TS ensemble is performing
          above configured monitoring thresholds.
        - Barbell/tail-risk aware evaluation (Sortino, Omega, CVaR, antifragility
          scenarios) is intentionally deferred to the portfolio math layer and
          `BARBELL_INTEGRATION_TODO.md` so that this function remains a thin,
          config-driven hook into realized PnL/health metrics instead of
          hardcoding any specific risk model.
        """
        from datetime import datetime, timedelta, timezone as _tz; UTC = _tz.utc  # noqa: E702

        regime_cfg = regimes_cfg.get(candidate.regime or regime, {})
        lookback_days = int(
            regime_cfg.get(
                "lookback_days",
                evaluation_cfg.get("default_lookback_days", 365),
            )
        )
        end_date = datetime.now(UTC).date()
        start_date = end_date - timedelta(days=lookback_days)
        start_iso = start_date.isoformat()
        end_iso = end_date.isoformat()

        backtest = backtest_candidate(
            db_manager=db_manager,
            tickers=tickers,
            start=start_iso,
            end=end_iso,
            candidate_params=candidate.params,
            guardrails=signal_guardrails,
        )
        total_profit = float(backtest.total_profit)
        total_return = float(backtest.total_return)
        profit_factor = float(backtest.profit_factor)
        win_rate = float(backtest.win_rate)
        total_trades = int(backtest.total_trades)
        max_dd = float(backtest.max_drawdown)
        strategy_returns = backtest.strategy_returns

        barbell_metrics: Dict[str, float] = {}
        if strategy_returns is not None and not strategy_returns.empty:
            try:
                barbell_metrics = portfolio_metrics_ngn(strategy_returns)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Unable to compute barbell metrics for candidate %s: %s", candidate.params, exc)

        # Incorporate forecaster monitoring thresholds (RMSE-aware).
        # Missing or malformed regression summaries are fatal because the
        # optimizer must not score candidates against mismatched evidence.
        monitoring_cfg_path = ROOT_PATH / "config" / "forecaster_monitoring.yml"
        if not monitoring_cfg_path.exists():
            raise click.ClickException(f"Missing monitoring config: {monitoring_cfg_path}")

        cfg_raw = yaml.safe_load(monitoring_cfg_path.read_text()) or {}
        fm_cfg = cfg_raw.get("forecaster_monitoring") or {}
        rm_cfg = fm_cfg.get("regression_metrics") or {}
        max_ratio = float(rm_cfg.get("max_rmse_ratio_vs_baseline", 1.10))
        regression_summary = db_manager.get_forecast_regression_summary(
            start_date=start_iso,
            end_date=end_iso,
        ) or {}
        ens = regression_summary.get("ensemble") or {}
        baseline_summary = db_manager.get_forecast_regression_summary(
            start_date=start_iso,
            end_date=end_iso,
            model_type="SAMOSSA",
        ) or {}
        base = baseline_summary.get("samossa") or {}
        ensemble_rmse = ens.get("rmse")
        baseline_rmse = base.get("rmse")
        if not (
            isinstance(ensemble_rmse, (int, float))
            and isinstance(baseline_rmse, (int, float))
            and float(baseline_rmse) > 0.0
        ):
            raise click.ClickException(
                "Strategy optimization requires ensemble and baseline RMSE for the evaluation window."
            )
        rmse_ratio_vs_baseline = float(ensemble_rmse) / float(baseline_rmse)
        rmse_within_threshold = rmse_ratio_vs_baseline <= max_ratio

        return {
            "total_return": float(total_return),
            "total_profit": float(total_profit),
            "profit_factor": float(profit_factor),
            "win_rate": float(win_rate),
            "total_trades": int(total_trades),
            "max_drawdown": float(max_dd),
            "omega_ratio": barbell_metrics.get("omega_ratio"),
            "expected_shortfall": barbell_metrics.get("expected_shortfall"),
            "cvar_95": barbell_metrics.get("cvar_95"),
            "fractional_kelly_fat_tail": barbell_metrics.get("fractional_kelly_fat_tail"),
            "rmse_ratio_vs_baseline": float(rmse_ratio_vs_baseline),
            "rmse_within_threshold": float(1.0 if rmse_within_threshold else 0.0),
        }

    evaluations = optimizer.run(
        n_candidates=n_candidates,
        evaluation_fn=evaluation_fn,
        regime=regime,
    )

    if not evaluations:
        logger.warning("No candidates satisfied constraints; nothing to record.")
        return

    best = evaluations[0]
    for ev in evaluations:
        db_manager.save_strategy_config(
            regime=regime,
            params=ev.candidate.params,
            metrics=ev.metrics,
            score=ev.score,
        )

    logger.info("Best candidate for regime=%s score=%.4f", regime, best.score)
    logger.info("Parameters: %s", best.candidate.params)
    logger.info("Metrics: %s", best.metrics)


if __name__ == "__main__":
    main()
