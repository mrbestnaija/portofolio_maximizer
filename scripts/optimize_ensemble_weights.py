#!/usr/bin/env python3
"""
Ensemble Weight Optimization using scipy.optimize.
Finds optimal weights that minimize validation RMSE for each ticker.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        models = list(forecasts.keys())
        n_models = len(models)

        if n_models == 0:
            raise ValueError("No forecasts provided")

        # Stack forecasts into matrix (n_samples x n_models)
        forecast_matrix = np.column_stack([forecasts[m] for m in models])

        # Handle NaN values
        mask = ~np.isnan(forecast_matrix).any(axis=1) & ~np.isnan(actuals)
        forecast_matrix = forecast_matrix[mask]
        actuals = actuals[mask]

        if len(actuals) == 0:
            raise ValueError("No valid samples after removing NaN")

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

        logger.info(f"Optimization complete: RMSE {initial_rmse:.4f} → {final_rmse:.4f} "
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
        import sqlite3

        conn = sqlite3.connect(db_path)

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
        import glob

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


def main():
    parser = argparse.ArgumentParser(description='Optimize ensemble weights')
    parser.add_argument('--ticker', required=True, help='Ticker symbol (e.g., AAPL)')
    parser.add_argument('--source', choices=['database', 'files'], default='database',
                       help='Data source: database or parquet files')
    parser.add_argument('--db', default='data/portfolio_maximizer.db',
                       help='Database path (if source=database)')
    parser.add_argument('--forecast-dir', default='data/forecasts',
                       help='Forecast directory (if source=files)')
    parser.add_argument('--models', nargs='+',
                       help='Models to include (default: all)')
    parser.add_argument('--validation-start', help='Validation start date (YYYY-MM-DD)')
    parser.add_argument('--validation-end', help='Validation end date (YYYY-MM-DD)')
    parser.add_argument('--min-weight', type=float, default=0.05,
                       help='Minimum weight per model (default: 0.05)')
    parser.add_argument('--max-weight', type=float, default=0.95,
                       help='Maximum weight per model (default: 0.95)')
    parser.add_argument('--method', default='SLSQP',
                       choices=['SLSQP', 'trust-constr', 'COBYLA'],
                       help='Optimization method')
    parser.add_argument('--output', help='Output JSON file for optimal weights')
    parser.add_argument('--update-config', action='store_true',
                       help='Update pipeline_config.yml with optimal weights')

    args = parser.parse_args()

    # Create optimizer
    optimizer = EnsembleWeightOptimizer(
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        method=args.method
    )

    # Run optimization
    try:
        if args.source == 'database':
            weights, rmse, info = optimizer.optimize_from_database(
                db_path=args.db,
                ticker=args.ticker,
                validation_start=args.validation_start,
                validation_end=args.validation_end
            )
        else:
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
                    'validation_start': args.validation_start,
                    'validation_end': args.validation_end,
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
            print("Add this to config/pipeline_config.yml:")
            print(f"  # Optimized for {args.ticker} (RMSE: {rmse:.4f})")
            weights_str = ", ".join(f"{m}: {w:.2f}" for m, w in weights.items())
            print(f"  - {{{weights_str}}}")

        return 0

    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
