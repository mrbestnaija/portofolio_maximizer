#!/usr/bin/env python3
"""
Ensemble weight optimization utilities.

Modes:
- rolling_cv: run rolling-window CV on historical OHLCV in SQLite and optimize per-regime weights.
- files: optimize from forecast parquet files that include actuals.
- database: deprecated (the DB doesn't store realised values needed for optimisation).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover - optional dependency for optimisation
    minimize = None  # type: ignore[assignment]

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _connect_sqlite_readonly(db_path: str):
    """
    Open a read-only SQLite connection in a way that's robust under WSL/DrvFS.

    We prefer immutable URI mode to avoid file-locking edge-cases when the DB
    lives on `/mnt/c` (Windows filesystem mount).
    """
    import sqlite3

    path = Path(db_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    uri = f"file:{path.as_posix()}?mode=ro&immutable=1"
    try:
        return sqlite3.connect(uri, uri=True)
    except Exception:
        return sqlite3.connect(str(path))


class EnsembleWeightOptimizer:
    """
    Optimize ensemble weights to minimize validation RMSE.
    Uses scipy.optimize.minimize with constraints.
    """

    def __init__(
        self,
        min_weight: float = 0.05,
        max_weight: float = 0.95,
        method: str = 'SLSQP'
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.method = method

    def optimize_weights(
        self,
        forecasts: Dict[str, np.ndarray],
        actuals: np.ndarray,
        initial_weights: Dict[str, float] = None
    ) -> Tuple[Dict[str, float], float, Dict]:
        """
        Find optimal ensemble weights.

        Args:
            forecasts: {model_name: forecast_array}
            actuals: actual values array
            initial_weights: starting weights (optional)

        Returns:
            (optimal_weights, final_rmse, optimization_info)
        """
        if minimize is None:
            raise ImportError("scipy is required for weight optimisation (pip install scipy)")
        models = list(forecasts.keys())
        n_models = len(models)

        if n_models == 0:
            raise ValueError("No forecasts provided")
        if self.min_weight > self.max_weight:
            raise ValueError("min_weight cannot exceed max_weight")
        if self.min_weight * n_models > 1.0 + 1e-9:
            raise ValueError(
                f"Infeasible constraints: min_weight={self.min_weight} across {n_models} models (sum(min)>1)"
            )
        if self.max_weight * n_models < 1.0 - 1e-9:
            raise ValueError(
                f"Infeasible constraints: max_weight={self.max_weight} across {n_models} models (sum(max)<1)"
            )

        # Stack forecasts into matrix (n_samples x n_models)
        forecast_matrix = np.column_stack([forecasts[m] for m in models])

        # Handle NaN values
        mask = ~np.isnan(forecast_matrix).any(axis=1) & ~np.isnan(actuals)
        forecast_matrix = forecast_matrix[mask]
        actuals = actuals[mask]

        if len(actuals) == 0:
            raise ValueError("No valid samples after removing NaN")

        if n_models == 1:
            rmse_val = float(np.sqrt(np.mean((forecast_matrix[:, 0] - actuals) ** 2)))
            info = {
                "initial_rmse": rmse_val,
                "final_rmse": rmse_val,
                "improvement_pct": 0.0,
                "n_iterations": 0,
                "success": True,
                "message": "single-model (no optimisation)",
                "n_samples": int(len(actuals)),
            }
            return {models[0]: 1.0}, rmse_val, info

        # Initial weights (uniform if not provided)
        if initial_weights:
            x0 = np.array([initial_weights.get(m, 1/n_models) for m in models])
        else:
            x0 = np.ones(n_models) / n_models

        # Objective function: RMSE
        def objective(weights):
            ensemble_forecast = forecast_matrix @ weights
            rmse = np.sqrt(np.mean((ensemble_forecast - actuals) ** 2))
            return rmse

        # Constraints: weights sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]

        # Bounds: min_weight to max_weight for each model
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_models)]

        # Optimize
        logger.info(f"Optimizing weights for {n_models} models ({len(actuals)} samples)")
        result = minimize(
            objective,
            x0=x0,
            method=self.method,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")

        # Extract optimal weights
        optimal_weights = {model: float(w) for model, w in zip(models, result.x)}

        # Calculate performance metrics
        initial_rmse = objective(x0)
        final_rmse = result.fun
        improvement = ((initial_rmse - final_rmse) / initial_rmse) * 100

        info = {
            'initial_rmse': float(initial_rmse),
            'final_rmse': float(final_rmse),
            'improvement_pct': float(improvement),
            'n_iterations': result.nit,
            'success': result.success,
            'message': result.message,
            'n_samples': len(actuals),
        }

        logger.info(f"Optimization complete: RMSE {initial_rmse:.4f} -> {final_rmse:.4f} "
                   f"({improvement:+.2f}%)")

        return optimal_weights, final_rmse, info

    def optimize_from_database(
        self,
        db_path: str,
        ticker: str,
        validation_start: str = None,
        validation_end: str = None
    ) -> Tuple[Dict[str, float], float, Dict]:
        """
        Optimize weights using forecasts from database.

        Args:
            db_path: Path to SQLite database
            ticker: Ticker symbol
            validation_start: Start date for validation period
            validation_end: End date for validation period

        Returns:
            (optimal_weights, final_rmse, info)
        """
        conn = _connect_sqlite_readonly(db_path)
        cols = [row[1] for row in conn.execute("PRAGMA table_info(time_series_forecasts)")]
        if "actual_value" not in cols:
            conn.close()
            raise ValueError(
                "Database source requires `time_series_forecasts.actual_value`, which is not present. "
                "Use --source rolling_cv instead."
            )

        # Build query
        query = """
        SELECT
            forecast_date,
            model_type,
            forecast_value,
            actual_value
        FROM time_series_forecasts
        WHERE ticker = ?
        """
        params = [ticker]

        if validation_start:
            query += " AND forecast_date >= ?"
            params.append(validation_start)

        if validation_end:
            query += " AND forecast_date <= ?"
            params.append(validation_end)

        query += " ORDER BY forecast_date, model_type"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df.empty:
            raise ValueError(f"No forecast data found for {ticker}")

        # Pivot to get forecasts by model
        pivot = df.pivot(index='forecast_date', columns='model_type', values='forecast_value')

        # Get actuals (use first non-null actual_value per date)
        actuals = df.groupby('forecast_date')['actual_value'].first()

        # Align forecasts and actuals
        common_dates = pivot.index.intersection(actuals.index)
        forecasts = {col: pivot.loc[common_dates, col].values for col in pivot.columns}
        actuals_array = actuals.loc[common_dates].values

        return self.optimize_weights(forecasts, actuals_array)

    def optimize_from_files(
        self,
        forecast_dir: str,
        ticker: str,
        models: List[str] = None
    ) -> Tuple[Dict[str, float], float, Dict]:
        """
        Optimize weights using forecast files (parquet format).

        Args:
            forecast_dir: Directory containing forecast files
            ticker: Ticker symbol
            models: List of model names to include (default: all)

        Returns:
            (optimal_weights, final_rmse, info)
        """
        forecast_dir = Path(forecast_dir)

        # Find forecast files
        pattern = f"{ticker}_*_forecast.parquet"
        files = list(forecast_dir.glob(pattern))

        if not files:
            raise ValueError(f"No forecast files found for {ticker} in {forecast_dir}")

        # Load forecasts
        forecasts = {}
        actuals = None

        for file in files:
            # Extract model name from filename
            parts = file.stem.split('_')
            if len(parts) >= 2:
                model = parts[1]  # ticker_MODEL_forecast
            else:
                continue

            if models and model not in models:
                continue

            df = pd.read_parquet(file)

            # Assume columns: date, forecast, actual
            if 'forecast' in df.columns:
                forecasts[model] = df['forecast'].values

            if 'actual' in df.columns and actuals is None:
                actuals = df['actual'].values

        if not forecasts:
            raise ValueError(f"No valid forecast data loaded")

        if actuals is None:
            raise ValueError("No actual values found in forecast files")

        return self.optimize_weights(forecasts, actuals)

    def optimize_per_regime_from_rolling_cv(
        self,
        *,
        db_path: str,
        tickers: List[str],
        config_path: str = "config/forecasting_config.yml",
        models: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        ohlcv_source: Optional[str] = None,
        price_field: str = "close",
        min_train_size: int = 180,
        horizon: int = 5,
        step_size: Optional[int] = None,
        max_folds: Optional[int] = None,
        min_samples_per_regime: int = 25,
        per_regime: bool = True,
    ) -> Dict[str, Any]:
        """
        Optimise ensemble weights from rolling-window CV runs over historical OHLCV data.

        Returns:
            Mapping:
                {
                    "LIQUID_RANGEBOUND": {
                        "weights": {...},
                        "rmse": float,
                        "info": {...},
                        "n_samples": int,
                        "n_folds": int,
                    },
                    ...
                }
        """
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - config dependency
            raise ImportError("PyYAML is required for --source rolling_cv") from exc

        from forcester_ts.ensemble import canonical_model_key
        from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig

        tickers = [t.strip().upper() for t in tickers if str(t).strip()]
        if not tickers:
            raise ValueError("No tickers provided")

        model_list = [canonical_model_key(m) for m in (models or ["sarimax", "samossa", "mssa_rl"])]
        model_list = [m for m in model_list if m]
        supported_models = {"sarimax", "samossa", "mssa_rl"}
        dropped = sorted(set(model_list) - supported_models)
        if dropped:
            logger.warning("Ignoring unsupported rolling_cv models: %s", dropped)
        model_list = [m for m in model_list if m in supported_models]
        if not model_list:
            raise ValueError("No models selected")

        if step_size is None:
            step_size = int(horizon)

        # Load forecasting config (either forecasting_config.yml or pipeline_config.yml)
        config_raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        forecasting_cfg = config_raw.get("forecasting")
        if forecasting_cfg is None:
            forecasting_cfg = (config_raw.get("pipeline") or {}).get("forecasting")
        if not isinstance(forecasting_cfg, dict):
            raise ValueError(f"Unable to locate forecasting config in {config_path}")

        sarimax_cfg = forecasting_cfg.get("sarimax", {}) or {}
        garch_cfg = forecasting_cfg.get("garch", {}) or {}
        samossa_cfg = forecasting_cfg.get("samossa", {}) or {}
        mssa_rl_cfg = forecasting_cfg.get("mssa_rl", {}) or {}
        regime_cfg = forecasting_cfg.get("regime_detection", {}) or {}

        # Always force regime detection ON for optimisation runs; the feature flag can stay OFF in prod.
        regime_kwargs = {k: v for k, v in regime_cfg.items() if k != "enabled"}

        def _build_forecaster_config(target_horizon: int) -> TimeSeriesForecasterConfig:
            return TimeSeriesForecasterConfig(
                forecast_horizon=int(target_horizon),
                sarimax_enabled=bool(sarimax_cfg.get("enabled", True)) and "sarimax" in model_list,
                garch_enabled=False,  # GARCH does not emit a price forecast series for the ensemble today
                samossa_enabled=bool(samossa_cfg.get("enabled", True)) and "samossa" in model_list,
                mssa_rl_enabled=bool(mssa_rl_cfg.get("enabled", True)) and "mssa_rl" in model_list,
                ensemble_enabled=False,  # We optimise blending offline from component forecasts
                sarimax_kwargs={k: v for k, v in sarimax_cfg.items() if k != "enabled"},
                garch_kwargs={k: v for k, v in garch_cfg.items() if k != "enabled"},
                samossa_kwargs={k: v for k, v in samossa_cfg.items() if k != "enabled"},
                mssa_rl_kwargs={k: v for k, v in mssa_rl_cfg.items() if k != "enabled"},
                ensemble_kwargs={"audit_log_dir": None},
                regime_detection_enabled=True,
                regime_detection_kwargs=regime_kwargs,
            )

        def _load_price_series(conn, ticker: str) -> pd.Series:
            if price_field not in {"close", "adj_close", "open", "high", "low"}:
                raise ValueError(f"Unsupported price_field '{price_field}'")

            query = f"SELECT date, {price_field} AS price FROM ohlcv_data WHERE ticker = ?"
            params: list[Any] = [ticker]
            if ohlcv_source:
                query += " AND source = ?"
                params.append(ohlcv_source)
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            query += " ORDER BY date"

            df = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])
            if df.empty:
                raise ValueError(f"No OHLCV rows found for {ticker}")
            series = pd.Series(df["price"].astype(float).to_numpy(), index=pd.DatetimeIndex(df["date"]), name="Close")
            return series

        def _iter_folds(series: pd.Series) -> List[slice]:
            total_points = len(series)
            min_train = int(min_train_size)
            h = int(horizon)
            step = max(1, int(step_size))
            if total_points < min_train + h:
                raise ValueError(
                    f"Insufficient data for rolling CV (need >= {min_train + h}, received {total_points})"
                )
            folds: List[slice] = []
            fold_index = min_train
            while fold_index + h <= total_points:
                folds.append(slice(fold_index, fold_index + h))
                if max_folds is not None and len(folds) >= int(max_folds):
                    break
                fold_index += step
            return folds

        buckets: Dict[str, Dict[str, Any]] = {}

        conn = _connect_sqlite_readonly(db_path)
        try:
            for ticker in tickers:
                price_series = _load_price_series(conn, ticker)
                price_series = price_series.sort_index()
                folds = _iter_folds(price_series)

                for fold_slice in folds:
                    train = price_series.iloc[: fold_slice.start]
                    test = price_series.iloc[fold_slice]
                    if len(test) < 1:
                        continue

                    forecaster = TimeSeriesForecaster(config=_build_forecaster_config(len(test)))
                    train_returns = train.pct_change().dropna()
                    forecaster.fit(price_series=train, returns_series=train_returns)
                    result = forecaster.forecast(steps=len(test))

                    regime = result.get("regime") or "UNKNOWN"
                    if not per_regime:
                        regime = "ALL"

                    fold_df = pd.DataFrame({"actual": test})
                    for model in model_list:
                        payload = result.get(f"{model}_forecast")
                        if not isinstance(payload, dict):
                            fold_df = pd.DataFrame()
                            break
                        forecast_series = payload.get("forecast")
                        if not isinstance(forecast_series, pd.Series):
                            fold_df = pd.DataFrame()
                            break
                        aligned = forecast_series.reindex(test.index)
                        # Some forecasters may emit a calendar-day index while the
                        # realised series is business-day (or vice-versa). When
                        # this happens, fall back to position-based alignment so
                        # we still have usable samples for optimisation.
                        non_na = int(aligned.notna().sum())
                        if non_na < len(test):
                            values = forecast_series.to_numpy(dtype=float, copy=False)
                            aligned = pd.Series(values[: len(test)], index=test.index)
                        fold_df[model] = aligned
                    if fold_df.empty:
                        continue

                    fold_df = fold_df.dropna()
                    if fold_df.empty:
                        continue

                    bucket = buckets.setdefault(
                        regime,
                        {
                            "forecasts": {m: [] for m in model_list},
                            "actuals": [],
                            "n_folds": 0,
                        },
                    )
                    bucket["n_folds"] += 1
                    bucket["actuals"].extend(fold_df["actual"].tolist())
                    for model in model_list:
                        bucket["forecasts"][model].extend(fold_df[model].tolist())
        finally:
            conn.close()

        results: Dict[str, Any] = {}
        for regime, bucket in buckets.items():
            actuals = np.asarray(bucket["actuals"], dtype=float)
            if actuals.size < int(min_samples_per_regime):
                continue

            forecasts = {
                model: np.asarray(bucket["forecasts"][model], dtype=float)
                for model in model_list
            }
            if any(arr.size != actuals.size for arr in forecasts.values()):
                raise RuntimeError(f"Internal alignment error for regime '{regime}'")

            weights, rmse, info = self.optimize_weights(forecasts, actuals)
            results[regime] = {
                "weights": weights,
                "rmse": float(rmse),
                "info": info,
                "n_samples": int(actuals.size),
                "n_folds": int(bucket["n_folds"]),
                "models": list(model_list),
            }

        if not results:
            raise ValueError(
                "No regimes had enough samples to optimise. "
                "Try increasing history, decreasing min_train_size/horizon, or lowering --min-samples-per-regime."
            )

        return results


def main():
    parser = argparse.ArgumentParser(description="Optimize ensemble weights")
    parser.add_argument(
        "--source",
        choices=["rolling_cv", "files", "database"],
        default="rolling_cv",
        help="Data source: rolling_cv (recommended), files, or database (deprecated)",
    )

    parser.add_argument("--ticker", help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--tickers", nargs="+", help="Tickers list (e.g., AAPL MSFT NVDA)")

    parser.add_argument(
        "--db",
        default="data/portfolio_maximizer.db",
        help="Database path (rolling_cv/database sources)",
    )
    parser.add_argument(
        "--forecast-dir",
        default="data/forecasts",
        help="Forecast directory (source=files)",
    )
    parser.add_argument(
        "--config",
        default="config/forecasting_config.yml",
        help="Forecasting YAML config (source=rolling_cv)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to include (rolling_cv default: sarimax samossa mssa_rl)",
    )

    # rolling_cv options
    parser.add_argument("--start-date", help="OHLCV start date filter (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="OHLCV end date filter (YYYY-MM-DD)")
    parser.add_argument("--ohlcv-source", help="OHLCV source filter (e.g., yfinance)")
    parser.add_argument(
        "--price-field",
        default="close",
        help="OHLCV field to optimise on (close, adj_close, open, high, low)",
    )
    parser.add_argument("--min-train-size", type=int, default=180, help="Rolling CV minimum train size")
    parser.add_argument("--horizon", type=int, default=5, help="Rolling CV horizon (test window size)")
    parser.add_argument("--step-size", type=int, help="Rolling CV step size (default: horizon)")
    parser.add_argument("--max-folds", type=int, help="Max rolling CV folds per ticker")
    parser.add_argument(
        "--min-samples-per-regime",
        type=int,
        default=25,
        help="Minimum samples required to optimise a regime bucket",
    )
    parser.add_argument(
        "--no-per-regime",
        action="store_true",
        help="Optimise a single ALL bucket (ignore regimes)",
    )

    # optimiser options
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.05,
        help="Minimum weight per model (default: 0.05)",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.95,
        help="Maximum weight per model (default: 0.95)",
    )
    parser.add_argument(
        "--method",
        default="SLSQP",
        choices=["SLSQP", "trust-constr", "COBYLA"],
        help="Optimization method",
    )

    parser.add_argument("--output", help="Output JSON file for optimal weights")
    parser.add_argument("--update-config", action="store_true", help="Print YAML snippet for config updates")

    args = parser.parse_args()

    # Create optimizer
    optimizer = EnsembleWeightOptimizer(
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        method=args.method
    )

    # Run optimization
    try:
        if args.source == "rolling_cv":
            tickers: List[str] = []
            if args.tickers:
                tickers = list(args.tickers)
            elif args.ticker:
                tickers = [t for t in str(args.ticker).split(",") if t.strip()]
            else:
                parser.error("Provide --ticker or --tickers for --source rolling_cv")

            results = optimizer.optimize_per_regime_from_rolling_cv(
                db_path=args.db,
                tickers=tickers,
                config_path=args.config,
                models=args.models,
                start_date=args.start_date,
                end_date=args.end_date,
                ohlcv_source=args.ohlcv_source,
                price_field=args.price_field,
                min_train_size=args.min_train_size,
                horizon=args.horizon,
                step_size=args.step_size,
                max_folds=args.max_folds,
                min_samples_per_regime=args.min_samples_per_regime,
                per_regime=not args.no_per_regime,
            )

            print("\n" + "=" * 80)
            print("PER-REGIME ENSEMBLE WEIGHT OPTIMIZATION (rolling_cv)")
            print("=" * 80)
            for regime in sorted(results.keys()):
                payload = results[regime]
                weights = payload["weights"]
                info = payload["info"]
                print(f"\n## {regime} (samples={payload['n_samples']}, folds={payload['n_folds']})")
                for model, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {model:<12} {weight:>6.1%}")
                print(f"  RMSE: {info['initial_rmse']:.4f} -> {info['final_rmse']:.4f} ({info['improvement_pct']:+.2f}%)")

            print("\n" + "=" * 80)

            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "source": "rolling_cv",
                            "db": args.db,
                            "tickers": tickers,
                            "config": args.config,
                            "models": args.models,
                            "cv": {
                                "min_train_size": args.min_train_size,
                                "horizon": args.horizon,
                                "step_size": args.step_size or args.horizon,
                                "max_folds": args.max_folds,
                                "min_samples_per_regime": args.min_samples_per_regime,
                                "per_regime": not args.no_per_regime,
                            },
                            "results": results,
                        },
                        f,
                        indent=2,
                    )
                print(f"\nResults saved to: {output_path}")

            if args.update_config:
                print("\n## YAML Snippet (paste under forecasting.regime_detection)")
                print("regime_candidate_weights:")
                for regime in sorted(results.keys()):
                    weights = results[regime]["weights"]
                    weights_str = ", ".join(f"{m}: {w:.2f}" for m, w in weights.items())
                    print(f"  {regime}:")
                    print(f"    - {{{weights_str}}}")

            return 0

        if args.source == "database":
            if not args.ticker:
                parser.error("Provide --ticker for --source database")
            weights, rmse, info = optimizer.optimize_from_database(
                db_path=args.db,
                ticker=args.ticker,
                validation_start=None,
                validation_end=None
            )
        else:
            if not args.ticker:
                parser.error("Provide --ticker for --source files")
            weights, rmse, info = optimizer.optimize_from_files(
                forecast_dir=args.forecast_dir,
                ticker=args.ticker,
                models=args.models
            )

        # Print results
        print("\n" + "=" * 80)
        print(f"ENSEMBLE WEIGHT OPTIMIZATION - {args.ticker}")
        print("=" * 80)

        print("\n## Optimal Weights")
        for model, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model:<12} {weight:>6.1%}")

        print("\n## Performance")
        print(f"  Initial RMSE:   {info['initial_rmse']:.4f}")
        print(f"  Optimized RMSE: {info['final_rmse']:.4f}")
        print(f"  Improvement:    {info['improvement_pct']:+.2f}%")
        print(f"  Samples:        {info['n_samples']}")

        print("\n## Optimization Info")
        print(f"  Method:      {args.method}")
        print(f"  Iterations:  {info['n_iterations']}")
        print(f"  Status:      {'SUCCESS' if info['success'] else 'FAILED'}")
        print(f"  Message:     {info['message']}")

        print("\n" + "=" * 80)

        # Save to file if requested
        if args.output:
            output_data = {
                'ticker': args.ticker,
                'optimal_weights': weights,
                'rmse': float(rmse),
                'optimization_info': info,
                'config': {
                    'min_weight': args.min_weight,
                    'max_weight': args.max_weight,
                    'method': args.method,
                }
            }

            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"\nOptimal weights saved to: {output_path}")

        # Update config if requested
        if args.update_config:
            print("\n## Update Config")
            print("Add this to forecasting.ensemble.candidate_weights:")
            print(f"  # Optimized for {args.ticker} (RMSE: {rmse:.4f})")
            weights_str = ", ".join(f"{m}: {w:.2f}" for m, w in weights.items())
            print(f"  - {{{weights_str}}}")

        return 0

    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
