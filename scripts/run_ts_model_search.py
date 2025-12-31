#!/usr/bin/env python3
"""
run_ts_model_search.py
----------------------

Time-series model hyper-parameter search helper (institution-grade scaffold).

- Loads OHLCV data for a set of tickers.
- Runs rolling-window CV for a small SARIMAX / SAMOSSA grid.
- Computes per-candidate aggregate metrics, fold-level RMSE stability, and a
  simple Diebold–Mariano-style comparison against a baseline candidate.
- Records per-ticker candidate metrics into the ts_model_candidates table.

This script is read-only w.r.t. configs: it produces evidence for model /
hyper-parameter choices but does not mutate any production YAML.
"""

from __future__ import annotations

import copy
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import pandas as pd

from etl.database_manager import DatabaseManager
from etl.time_series_forecaster import (
    TimeSeriesForecasterConfig,
    RollingWindowValidator,
    RollingWindowCVConfig,
)
from etl.model_profiles import (
    ModelProfile,
    TSModelOverride,
    select_profile_for_sleeve_and_returns,
    select_profile_with_overrides,
)
from etl.statistical_tests import diebold_mariano
from risk.barbell_policy import BarbellConfig

ROOT_PATH = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _load_price_series(
    db: DatabaseManager,
    ticker: str,
    lookback_days: int,
) -> Optional[pd.Series]:
    """Load a simple close-price series for one ticker."""
    end_date = date.today()
    start_date = end_date - timedelta(days=max(lookback_days, 365))
    frame = db.load_ohlcv(
        [ticker],
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )
    if frame.empty:
        logger.warning("No OHLCV data found for %s in requested window.", ticker)
        return None
    sub = frame[frame["ticker"] == ticker].copy()
    if sub.empty or "close" not in sub.columns:
        logger.warning("Close column missing for %s; skipping.", ticker)
        return None
    sub = sub.sort_index()
    return sub["close"].astype(float)


def _build_candidate_configs(
    profile: Optional[ModelProfile] = None,
) -> List[Tuple[str, TimeSeriesForecasterConfig]]:
    """
    Define a small grid of candidate TS configs.

    These are deliberately compact to keep runtime manageable and aligned with
    the institutionalisation roadmap in OPTIMIZATION_IMPLEMENTATION_PLAN.md.
    """
    configs: List[Tuple[str, TimeSeriesForecasterConfig]] = []

    # Baseline SARIMAX-only ensemble (no SAMOSSA/MSSA, ensemble disabled)
    cfg_sarimax = TimeSeriesForecasterConfig()
    cfg_sarimax.sarimax_enabled = True
    cfg_sarimax.garch_enabled = False
    cfg_sarimax.samossa_enabled = False
    cfg_sarimax.mssa_rl_enabled = False
    cfg_sarimax.ensemble_enabled = False
    cfg_sarimax.sarimax_kwargs = {
        "max_p": 2,
        "max_d": 1,
        "max_q": 2,
    }
    sarimax_profile = (profile.payload.get("sarimax") or {}) if profile else {}
    sarimax_enabled = bool(sarimax_profile.get("enabled", True)) if profile else True
    if sarimax_profile:
        for k, v in sarimax_profile.items():
            if k == "enabled":
                continue
            cfg_sarimax.sarimax_kwargs[k] = v
    if sarimax_enabled:
        configs.append(("sarimax_only", cfg_sarimax))

    # SAMOSSA-only profile (no SARIMAX, ensemble disabled)
    cfg_samossa = TimeSeriesForecasterConfig()
    cfg_samossa.sarimax_enabled = False
    cfg_samossa.garch_enabled = False
    cfg_samossa.samossa_enabled = True
    cfg_samossa.mssa_rl_enabled = False
    cfg_samossa.ensemble_enabled = False
    cfg_samossa.samossa_kwargs = {
        "window_length": 40,
        "n_components": 6,
        "min_series_length": 120,
    }
    samossa_profile = (profile.payload.get("samossa") or {}) if profile else {}
    samossa_enabled = bool(samossa_profile.get("enabled", True)) if profile else True
    if samossa_profile:
        for k, v in samossa_profile.items():
            if k == "enabled":
                continue
            cfg_samossa.samossa_kwargs[k] = v
    if samossa_enabled:
        configs.append(("samossa_only", cfg_samossa))

    # SARIMAX + SAMOSSA hybrid (ensemble on)
    cfg_hybrid = copy.deepcopy(cfg_sarimax)
    cfg_hybrid.samossa_enabled = True
    cfg_hybrid.ensemble_enabled = True
    cfg_hybrid.samossa_kwargs = {
        "window_length": 40,
        "n_components": 6,
        "min_series_length": 120,
    }
    if sarimax_enabled and samossa_enabled:
        configs.append(("sarimax_samossa", cfg_hybrid))

    return configs


def _select_primary_model(aggregate_metrics: Dict[str, Dict[str, float]]) -> str:
    """Choose the primary model name for scoring from aggregate metrics."""
    if not aggregate_metrics:
        return ""
    for key in aggregate_metrics.keys():
        if key.lower() in {"combined", "ensemble"}:
            return key
    # Fallback: first model in sorted order.
    return sorted(aggregate_metrics.keys())[0]


def _score_candidate(aggregate_metrics: Dict[str, Dict[str, float]]) -> float:
    """
    Derive a simple scalar score from aggregate metrics.

    Currently: negative RMSE for the primary model (lower RMSE => higher score).
    """
    if not aggregate_metrics:
        return 0.0
    primary = _select_primary_model(aggregate_metrics)
    metrics = aggregate_metrics.get(primary) or {}
    rmse_val = metrics.get("rmse")
    if rmse_val is None:
        # Fallback to any metric; keep score monotone decreasing in error.
        vals = [float(v) for v in metrics.values() if v is not None]
        rmse_val = vals[0] if vals else 0.0
    return -float(rmse_val)


def _serialize_config(config: TimeSeriesForecasterConfig) -> Dict[str, Any]:
    """Serialize a forecaster config into a JSON-friendly dict."""
    return {
        "sarimax_enabled": bool(config.sarimax_enabled),
        "garch_enabled": bool(config.garch_enabled),
        "samossa_enabled": bool(config.samossa_enabled),
        "mssa_rl_enabled": bool(config.mssa_rl_enabled),
        "ensemble_enabled": bool(config.ensemble_enabled),
        "forecast_horizon": int(config.forecast_horizon),
        "sarimax_kwargs": config.sarimax_kwargs,
        "garch_kwargs": config.garch_kwargs,
        "samossa_kwargs": config.samossa_kwargs,
        "mssa_rl_kwargs": config.mssa_rl_kwargs,
        "ensemble_kwargs": config.ensemble_kwargs,
    }


@click.command()
@click.option(
    "--tickers",
    default="AAPL,MSFT",
    show_default=True,
    help="Comma-separated list of tickers to search over.",
)
@click.option(
    "--db-path",
    default="data/portfolio_maximizer.db",
    show_default=True,
    help="SQLite DB with OHLCV data.",
)
@click.option(
    "--lookback-days",
    default=730,
    show_default=True,
    help="Lookback window (days) for price history used in CV.",
)
@click.option(
    "--regime",
    default="default",
    show_default=True,
    help="Regime label to tag candidates with.",
)
@click.option(
    "--min-train-size",
    default=180,
    show_default=True,
    help="Minimum training size for rolling-window CV.",
)
@click.option(
    "--horizon",
    default=20,
    show_default=True,
    help="Forecast horizon (days) for CV folds.",
)
@click.option(
    "--step-size",
    default=20,
    show_default=True,
    help="Step size between CV folds.",
)
@click.option(
    "--max-folds",
    default=5,
    show_default=True,
    help="Maximum number of CV folds per ticker.",
)
@click.option(
    "--use-profiles",
    is_flag=True,
    help="Use config/model_profiles.yml + config/barbell.yml to select model profiles per (ticker, regime) before building candidates.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable debug logging.",
)
def main(
    tickers: str,
    db_path: str,
    lookback_days: int,
    regime: str,
    min_train_size: int,
    horizon: int,
    step_size: int,
    max_folds: int,
    use_profiles: bool,
    verbose: bool,
) -> None:
    """Run a TS model search and persist CV metrics per candidate."""
    _configure_logging(verbose)
    ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        raise click.UsageError("At least one ticker is required.")

    db = DatabaseManager(db_path=db_path)

    barbell_cfg: Optional[BarbellConfig]
    if use_profiles:
        try:
            barbell_cfg = BarbellConfig.from_yaml()
        except Exception:
            barbell_cfg = None
            logger.warning(
                "Unable to load config/barbell.yml; proceeding without sleeve-aware profiles."
            )
    else:
        barbell_cfg = None

    cv_cfg = RollingWindowCVConfig(
        min_train_size=min_train_size,
        horizon=horizon,
        step_size=step_size,
        max_folds=max_folds,
    )

    logger.info(
        "Running TS model search for tickers=%s | use_profiles=%s",
        ",".join(ticker_list),
        use_profiles,
    )

    # Optional ticker -> sleeve map from barbell config for profile selection.
    ticker_to_sleeve: Dict[str, str] = {}
    if barbell_cfg is not None:
        safe = set(barbell_cfg.safe_symbols)
        core = set(barbell_cfg.core_symbols)
        spec = set(barbell_cfg.speculative_symbols)
        for t in ticker_list:
            sym = t.upper()
            if sym in safe:
                ticker_to_sleeve[t] = "safe"
            elif sym in core:
                ticker_to_sleeve[t] = "core"
            elif sym in spec:
                ticker_to_sleeve[t] = "speculative"
            else:
                ticker_to_sleeve[t] = "other"

    for ticker in ticker_list:
        price_series = _load_price_series(db, ticker, lookback_days=lookback_days)
        if price_series is None:
            continue

        returns_series = price_series.pct_change().dropna()

        # Optionally select a model profile based on sleeve + volatility regime,
        # and consult ts_model_overrides.yml for any explicit (ticker, regime)
        # candidate/profile hints.
        profile: Optional[ModelProfile] = None
        regime_label: Optional[str] = None
        override: Optional[TSModelOverride] = None
        if use_profiles:
            sleeve = ticker_to_sleeve.get(ticker, "other")
            try:
                profile, regime_state, override = select_profile_with_overrides(
                    ticker=ticker,
                    sleeve=sleeve,
                    returns=returns_series.values,
                )
            except Exception:
                # Fallback to sleeve/regime-based selection only.
                profile, regime_state = select_profile_for_sleeve_and_returns(
                    sleeve=sleeve,
                    returns=returns_series.values,
                )
                override = None

            regime_label = regime_state.regime_type
            if override is not None:
                logger.info(
                    "Ticker %s mapped to sleeve=%s regime=%s -> override candidate=%s profile_hint=%s (profile=%s)",
                    ticker,
                    sleeve,
                    regime_label,
                    override.candidate_name,
                    override.profile_hint,
                    profile.name if profile is not None else None,
                )
            elif profile is not None:
                logger.info(
                    "Ticker %s mapped to sleeve=%s regime=%s -> profile=%s",
                    ticker,
                    sleeve,
                    regime_label,
                    profile.name,
                )
            else:
                logger.info(
                    "Ticker %s mapped to sleeve=%s regime=%s -> no profile; using default candidates",
                    ticker,
                    sleeve,
                    regime_label,
                )

        candidates = _build_candidate_configs(profile)

        # First collect CV results for all candidates so we can compute
        # per-ticker stability and pairwise comparisons.
        cv_results: Dict[str, Dict[str, Any]] = {}
        for candidate_name, base_config in candidates:
            cfg = copy.deepcopy(base_config)
            validator = RollingWindowValidator(
                forecaster_config=cfg,
                cv_config=cv_cfg,
            )
            try:
                result = validator.run(
                    price_series=price_series,
                    returns_series=returns_series,
                )
            except Exception as exc:
                logger.warning(
                    "CV failed for %s (%s): %s", ticker, candidate_name, exc
                )
                continue
            cv_results[candidate_name] = {
                "config": cfg,
                "result": result,
            }

        if not cv_results:
            logger.warning("No successful CV results for %s; skipping.", ticker)
            continue

        # Compute per-candidate fold-level RMSE series for the primary model.
        rmse_by_candidate: Dict[str, List[float]] = {}
        primary_by_candidate: Dict[str, str] = {}
        for candidate_name, payload in cv_results.items():
            result = payload["result"]
            aggregate_metrics = result.get("aggregate_metrics") or {}
            primary = _select_primary_model(aggregate_metrics)
            primary_by_candidate[candidate_name] = primary
            folds = result.get("folds") or []
            fold_rmses: List[float] = []
            for fold in folds:
                metrics_map = fold.get("metrics") or {}
                fold_metrics = metrics_map.get(primary) or {}
                rmse_val = fold_metrics.get("rmse")
                if rmse_val is not None:
                    fold_rmses.append(float(rmse_val))
            rmse_by_candidate[candidate_name] = fold_rmses

        # Baseline candidate for DM comparisons.
        if "sarimax_only" in cv_results:
            baseline = "sarimax_only"
        else:
            baseline = sorted(cv_results.keys())[0]

        for candidate_name, payload in cv_results.items():
            cfg = payload["config"]
            result = payload["result"]
            aggregate_metrics = result.get("aggregate_metrics") or {}
            folds = result.get("folds") or []

            base_score = _score_candidate(aggregate_metrics)

            # Stability: coefficient-of-variation based on fold RMSE.
            fold_rmses = rmse_by_candidate.get(candidate_name) or []
            if len(fold_rmses) >= 2:
                arr = pd.Series(fold_rmses, dtype="float64")
                mean_rmse = float(arr.mean())
                std_rmse = float(arr.std(ddof=1))
                if mean_rmse > 0 and std_rmse >= 0:
                    cv = std_rmse / mean_rmse
                    stability = max(0.0, min(1.0, 1.0 / (1.0 + cv)))
                else:
                    stability = None
            else:
                stability = None

            # Effective score: penalise unstable candidates.
            if stability is None:
                effective_score = base_score
            else:
                effective_score = base_score * (0.5 + 0.5 * stability)

            # Diebold–Mariano-style comparison vs baseline (per-fold RMSE proxy).
            dm_payload: Optional[Dict[str, Any]] = None
            if candidate_name != baseline:
                baseline_rmses = rmse_by_candidate.get(baseline) or []
                m = min(len(baseline_rmses), len(fold_rmses))
                if m >= 3:
                    dm_res = diebold_mariano(
                        baseline_rmses[:m],
                        fold_rmses[:m],
                        loss="absolute",
                        alternative="two-sided",
                    )
                    dm_payload = {
                        "statistic": dm_res.statistic,
                        "p_value": dm_res.p_value,
                        "better_model": dm_res.better_model,
                        "baseline": baseline,
                    }

            metrics_payload: Dict[str, Any] = {
                "aggregate_metrics": aggregate_metrics,
                "fold_count": result.get("fold_count"),
                "horizon": result.get("horizon"),
                "primary_model": primary_by_candidate.get(candidate_name),
                "fold_rmse": fold_rmses,
                "baseline": baseline,
                "dm_vs_baseline": dm_payload,
            }
            params_payload = _serialize_config(cfg)

            row_id = db.save_ts_model_candidate(
                ticker=ticker,
                regime=regime,
                candidate_name=candidate_name,
                params=params_payload,
                metrics=metrics_payload,
                stability=stability,
                score=effective_score,
            )
            logger.info(
                "Saved TS model candidate id=%s for %s (%s) with score=%.4f (stability=%s)",
                row_id,
                ticker,
                candidate_name,
                effective_score,
                "None" if stability is None else f"{stability:.3f}",
            )

    logger.info("TS model search complete.")


if __name__ == "__main__":
    main()
