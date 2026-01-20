#!/usr/bin/env python3
"""
Quick script to check ensemble weights from database after pipeline run.

Usage:
    python scripts/check_ensemble_weights.py --ticker AAPL
"""

import argparse
import sqlite3
import json
from pathlib import Path

def check_ensemble_weights(ticker: str):
    """Check ensemble weights and model RMSE from database."""

    db_path = Path("data/portfolio_maximizer.db")
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    print("=" * 80)
    print(f"ENSEMBLE WEIGHTS VERIFICATION - {ticker}")
    print("=" * 80)

    # Check if ensemble forecasts exist
    cursor.execute("""
        SELECT COUNT(*)
        FROM time_series_forecasts
        WHERE ticker = ? AND model_type = 'ENSEMBLE'
    """, (ticker,))
    ensemble_count = cursor.fetchone()[0]

    print(f"\n1. Ensemble Forecasts in Database: {ensemble_count}")
    if ensemble_count == 0:
        print("   WARNING: No ensemble forecasts found. Check if ensemble is enabled.")

    # Get ensemble diagnostics (weights stored in diagnostics JSON)
    cursor.execute("""
        SELECT diagnostics, forecast_value, lower_ci, upper_ci
        FROM time_series_forecasts
        WHERE ticker = ? AND model_type = 'ENSEMBLE'
        ORDER BY forecast_date DESC, forecast_horizon ASC
        LIMIT 1
    """, (ticker,))

    ensemble_row = cursor.fetchone()
    if ensemble_row:
        diagnostics_json = ensemble_row[0]
        if diagnostics_json:
            diagnostics = json.loads(diagnostics_json)
            weights = diagnostics.get('weights', {})
            confidence = diagnostics.get('confidence', {})
            selection_score = diagnostics.get('selection_score')

            print(f"\n2. Latest Ensemble Weights:")
            for model, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                print(f"   {model:12s}: {weight:6.2%}")

            print(f"\n3. Model Confidence Scores:")
            for model, conf in sorted(confidence.items(), key=lambda x: x[1], reverse=True):
                print(f"   {model:12s}: {conf:6.4f}")

            print(f"\n4. Selection Score: {selection_score:.4f}")

            # Check if GARCH is in the weights
            if 'garch' in weights:
                garch_weight = weights['garch']
                print(f"\n5. GARCH Weight: {garch_weight:.2%}")
                if garch_weight >= 0.60:
                    print("   ✓ GARCH-dominant ensemble (>= 60%)")
                elif garch_weight >= 0.40:
                    print("   ~ GARCH-balanced ensemble (40-60%)")
                else:
                    print("   ✗ GARCH under-weighted (< 40%)")
            else:
                print("\n5. GARCH Weight: 0.00%")
                print("   ✗ GARCH NOT included in ensemble (config issue?)")
    else:
        print("\n2. No ensemble diagnostics found in database")

    # Get individual model RMSE from regression_metrics
    print(f"\n6. Individual Model Performance (RMSE):")
    cursor.execute("""
        SELECT model_type, regression_metrics, COUNT(*)
        FROM time_series_forecasts
        WHERE ticker = ? AND model_type IN ('SARIMAX', 'GARCH', 'SAMoSSA', 'MSSA-RL', 'ENSEMBLE')
        GROUP BY model_type
        ORDER BY model_type
    """, (ticker,))

    model_metrics = {}
    for row in cursor.fetchall():
        model_type = row[0]
        metrics_json = row[1]
        count = row[2]
        if metrics_json:
            try:
                metrics = json.loads(metrics_json)
                rmse = metrics.get('rmse')
                if rmse:
                    model_metrics[model_type] = rmse
                    print(f"   {model_type:12s}: RMSE={rmse:8.4f}  (n={count} forecasts)")
            except:
                print(f"   {model_type:12s}: (no metrics available)")
        else:
            print(f"   {model_type:12s}: (no metrics available, n={count})")

    # Calculate RMSE ratio
    if model_metrics:
        best_rmse = min(rmse for model, rmse in model_metrics.items() if model != 'ENSEMBLE')
        best_model = min(
            ((model, rmse) for model, rmse in model_metrics.items() if model != 'ENSEMBLE'),
            key=lambda x: x[1]
        )[0]

        if 'ENSEMBLE' in model_metrics:
            ensemble_rmse = model_metrics['ENSEMBLE']
            rmse_ratio = ensemble_rmse / best_rmse

            print(f"\n7. RMSE Analysis:")
            print(f"   Best Single Model: {best_model} (RMSE={best_rmse:.4f})")
            print(f"   Ensemble RMSE:     {ensemble_rmse:.4f}")
            print(f"   RMSE Ratio:        {rmse_ratio:.3f}x")

            if rmse_ratio < 1.1:
                print(f"   ✓✓ EXCELLENT: Ensemble within 10% of best model (TARGET MET)")
            elif rmse_ratio < 1.2:
                print(f"   ✓ GOOD: Ensemble within 20% of best model")
            elif rmse_ratio < 1.5:
                print(f"   ~ ACCEPTABLE: Ensemble within 50% of best model")
            else:
                print(f"   ✗ POOR: Ensemble significantly worse than best model")
        else:
            print(f"\n7. RMSE Analysis:")
            print(f"   Best Single Model: {best_model} (RMSE={best_rmse:.4f})")
            print(f"   Ensemble RMSE:     N/A (no ensemble forecasts)")

    conn.close()
    print("\n" + "=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check ensemble weights from database")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker to check")
    args = parser.parse_args()

    check_ensemble_weights(args.ticker)
