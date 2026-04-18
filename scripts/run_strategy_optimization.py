#!/usr/bin/env python3
"""
run_strategy_optimization.py
----------------------------

Stochastic, configuration-driven strategy optimization.

This script:
- Loads a strategy optimization config (search space, objectives, constraints).
- Samples candidate configurations via StrategyOptimizer.
- Evaluates each candidate using the causal walk-forward simulator and the
  barbell-aware portfolio metric bundle.

IMPORTANT:
- This is infrastructure only. It does not hardcode any "best" strategy.
- The evaluation function must stay causal: no same-bar look-ahead and no
  fallback to heuristic placeholder backtests.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any, Dict

import click
import yaml

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from etl.database_manager import DatabaseManager
from etl.strategy_optimizer import StrategyOptimizer, StrategyCandidate
from backtesting.candidate_simulator import simulate_candidate


logger = logging.getLogger(__name__)

_OBJECTIVE_THRESHOLD_POLARITY = {
    "total_return": "min",
    "alpha": "min",
    "information_ratio": "min",
    "omega_ratio": "min",
    "payoff_asymmetry_effective": "min",
    "profit_factor": "min",
    "expected_shortfall": "min",
    "max_drawdown": "max",
}


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


def _merge_hard_constraints(
    base_constraints: Dict[str, Any],
    objective_thresholds: Dict[str, Any],
    *,
    max_rmse_ratio_vs_baseline: float,
) -> Dict[str, Dict[str, float]]:
    """Merge soft config thresholds into fail-closed min/max constraints."""
    merged: Dict[str, Dict[str, float]] = {
        "min": dict((base_constraints or {}).get("min", {}) or {}),
        "max": dict((base_constraints or {}).get("max", {}) or {}),
    }

    def _merge(bucket: str, key: str, value: Any) -> None:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise click.ClickException(f"Invalid threshold for {key!r}: {value!r}") from exc
        existing = merged[bucket].get(key)
        if existing is None:
            merged[bucket][key] = parsed
            return
        try:
            existing_value = float(existing)
        except (TypeError, ValueError) as exc:
            raise click.ClickException(f"Invalid existing constraint for {key!r}: {existing!r}") from exc
        merged[bucket][key] = max(existing_value, parsed) if bucket == "min" else min(existing_value, parsed)

    for key, value in (objective_thresholds or {}).items():
        polarity = _OBJECTIVE_THRESHOLD_POLARITY.get(str(key))
        if polarity is None:
            raise click.ClickException(
                f"Unsupported objective_thresholds metric {key!r}; add an explicit polarity mapping before using it."
            )
        _merge(polarity, str(key), value)

    _merge("max", "rmse_ratio_vs_baseline", max_rmse_ratio_vs_baseline)
    return merged


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
    objective_thresholds = cfg.get("objective_thresholds", {}) or {}
    evaluation_cfg = cfg.get("evaluation", {}) or {}
    regimes_cfg = cfg.get("regimes", {}) or {}
    if not search_space:
        raise click.UsageError("Empty search_space in strategy optimization config.")

    monitoring_cfg_path = ROOT_PATH / "config" / "forecaster_monitoring.yml"
    if not monitoring_cfg_path.exists():
        raise click.ClickException(f"Missing monitoring config: {monitoring_cfg_path}")
    cfg_raw = yaml.safe_load(monitoring_cfg_path.read_text(encoding="utf-8")) or {}
    fm_cfg = cfg_raw.get("forecaster_monitoring") or {}
    rm_cfg = fm_cfg.get("regression_metrics") or {}
    max_ratio = float(rm_cfg.get("max_rmse_ratio_vs_baseline", 1.10))

    hard_constraints = _merge_hard_constraints(
        constraints,
        objective_thresholds,
        max_rmse_ratio_vs_baseline=max_ratio,
    )

    optimizer = StrategyOptimizer(
        search_space=search_space,
        objectives=objectives,
        constraints=hard_constraints,
    )

    db_manager = DatabaseManager(db_path=db_path)
    signal_guardrails = _load_signal_guardrails(
        ROOT_PATH / str(evaluation_cfg.get("signal_routing_config_path") or "config/signal_routing_config.yml")
    )
    forecasting_config_path = Path(
        evaluation_cfg.get("forecasting_config_path") or "config/forecasting_config.yml"
    )
    if not forecasting_config_path.is_absolute():
        forecasting_config_path = ROOT_PATH / forecasting_config_path
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

    def evaluation_fn(candidate: StrategyCandidate) -> Dict[str, Any]:
        """
        Evaluate a candidate using causal walk-forward simulation and realized
        regression health for the configured regime.

        Notes
        -----
        - Uses the walk-forward simulator so the evaluation source matches the
          live causal execution path.
        - Keeps raw upside metrics visible while scoring on support-aware and
          tail-aware fields.
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

        simulation = simulate_candidate(
            source_db=db_manager,
            tickers=tickers,
            start_date=start_iso,
            end_date=end_iso,
            candidate_params=candidate.params,
            guardrails=signal_guardrails,
            forecasting_config_path=str(forecasting_config_path),
        )
        metrics = dict(simulation or {})
        metrics.pop("strategy_returns", None)

        # Incorporate forecaster monitoring thresholds (RMSE-aware).
        # Missing or malformed regression summaries are fatal because the
        # optimizer must not score candidates against mismatched evidence.
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

        metrics.update(
            {
                "rmse_ratio_vs_baseline": float(rmse_ratio_vs_baseline),
                "rmse_within_threshold": float(1.0 if rmse_within_threshold else 0.0),
            }
        )

        return metrics

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
