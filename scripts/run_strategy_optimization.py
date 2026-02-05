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
from etl.strategy_optimizer import StrategyOptimizer, StrategyCandidate


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
        from datetime import datetime, timedelta, UTC

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

        summary = db_manager.get_performance_summary(
            start_date=start_iso,
            end_date=end_iso,
        )
        total_profit = summary.get("total_profit") or 0.0
        profit_factor = summary.get("profit_factor") or 0.0
        win_rate = summary.get("win_rate") or 0.0
        total_trades = summary.get("total_trades") or 0

        equity_curve = db_manager.get_equity_curve(
            start_date=start_iso,
            end_date=end_iso,
            initial_capital=0.0,
        )
        peak = -float("inf")
        max_dd = 0.0
        for pt in equity_curve:
            val = float(pt.get("equity", 0.0))
            peak = max(peak, val)
            if peak > 0:
                dd = (peak - val) / peak
                max_dd = max(max_dd, dd)

        # Incorporate forecaster monitoring thresholds (RMSE-aware).
        # This is intentionally coarse-grained: we aggregate ensemble
        # regression metrics over the evaluation window and, if the
        # ensemble underperforms configured thresholds, we penalise the
        # candidate by driving total_return toward zero. In addition, we
        # apply soft gating based on realised profit_factor / win_rate so
        # that hyperopt only rewards regimes where the TS ensemble is both
        # statistically healthy (RMSE) and economically sensible (PF / WR).
        monitoring_cfg_path = ROOT_PATH / "config" / "forecaster_monitoring.yml"
        penalty = 1.0
        if monitoring_cfg_path.exists():
            try:
                cfg_raw = yaml.safe_load(monitoring_cfg_path.read_text()) or {}
                fm_cfg = cfg_raw.get("forecaster_monitoring") or {}
                # 1) RMSE / regression health
                rm_cfg = fm_cfg.get("regression_metrics") or {}
                max_ratio = float(rm_cfg.get("max_rmse_ratio_vs_baseline", 1.10))
                regression_summary = db_manager.get_forecast_regression_summary(
                    start_date=start_iso,
                    end_date=end_iso,
                )
                ens = regression_summary.get("ensemble") or {}
                ensemble_rmse = ens.get("rmse")
                baseline_summary = db_manager.get_forecast_regression_summary(
                    model_type="SAMOSSA"
                )
                base = (baseline_summary.get("samossa") or {}) if baseline_summary else {}
                baseline_rmse = base.get("rmse")
                if (
                    isinstance(ensemble_rmse, (int, float))
                    and isinstance(baseline_rmse, (int, float))
                    and baseline_rmse > 0
                ):
                    ratio = float(ensemble_rmse) / float(baseline_rmse)
                    if ratio > max_ratio:
                        logger.info(
                            "Forecaster RMSE ratio %.3f exceeds max %.3f for regime %s; "
                            "penalising candidate total_return.",
                            ratio,
                            max_ratio,
                            candidate.regime or regime,
                        )
                        penalty = 0.0

                # 2) Quant validation-style PF / WR health at the portfolio level.
                qv_cfg = fm_cfg.get("quant_validation") or {}
                min_pf = qv_cfg.get("min_profit_factor")
                min_wr = qv_cfg.get("min_win_rate")

                pf_bad = (
                    isinstance(min_pf, (int, float))
                    and profit_factor is not None
                    and float(profit_factor) < float(min_pf)
                )
                wr_bad = (
                    isinstance(min_wr, (int, float))
                    and win_rate is not None
                    and float(win_rate) < float(min_wr)
                )
                if pf_bad or wr_bad:
                    logger.info(
                        "Forecaster economic health check failed for regime %s "
                        "(PF=%.3f, WR=%.3f, min_pf=%s, min_wr=%s); penalising candidate.",
                        candidate.regime or regime,
                        float(profit_factor),
                        float(win_rate),
                        min_pf,
                        min_wr,
                    )
                    penalty = 0.0
            except Exception:  # pragma: no cover - monitoring is advisory
                penalty = 1.0

        return {
            "total_return": float(total_profit) * float(penalty),
            "profit_factor": float(profit_factor),
            "win_rate": float(win_rate),
            "total_trades": int(total_trades),
            "max_drawdown": float(max_dd),
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
