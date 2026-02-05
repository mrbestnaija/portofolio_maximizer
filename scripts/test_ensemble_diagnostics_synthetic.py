#!/usr/bin/env python3
"""Test ensemble diagnostics with synthetic forecast data.

This creates realistic synthetic forecast data to test and demonstrate
the ensemble diagnostics visualization system.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from forcester_ts.ensemble_diagnostics import (
    EnsembleDiagnostics,
    ModelPerformance,
    EnsemblePerformance
)

def generate_synthetic_forecasts(n_steps: int = 100, seed: int = 42) -> dict:
    """Generate synthetic forecast data simulating ensemble underperformance.

    Creates scenario where:
    - SARIMAX is best single model (RMSE ≈ 0.8)
    - SAMoSSA is slightly worse (RMSE ≈ 0.9)
    - MSSA-RL is worst (RMSE ≈ 1.2)
    - Ensemble has suboptimal weights, performing worse than SARIMAX
    """
    np.random.seed(seed)

    # Generate true values (sine wave + noise)
    t = np.linspace(0, 4*np.pi, n_steps)
    actuals = 100 + 10*np.sin(t) + np.random.randn(n_steps) * 0.5

    # SARIMAX: Best model (low bias, low variance)
    sarimax_preds = actuals + np.random.randn(n_steps) * 0.8
    sarimax_conf = 0.85 + 0.1 * np.random.rand(n_steps)

    # SAMoSSA: Slightly worse (slightly more variance)
    samossa_preds = actuals + np.random.randn(n_steps) * 0.9
    samossa_conf = 0.75 + 0.15 * np.random.rand(n_steps)

    # MSSA-RL: Worst model (high variance, some bias)
    mssa_preds = actuals + 0.3 + np.random.randn(n_steps) * 1.2  # +0.3 bias
    mssa_conf = 0.60 + 0.2 * np.random.rand(n_steps)

    # Ensemble with SUBOPTIMAL weights (gives too much weight to worse models)
    # Optimal would be ~90% SARIMAX, 10% SAMoSSA, 0% MSSA-RL
    # But system uses suboptimal: 40% SARIMAX, 30% SAMoSSA, 30% MSSA-RL
    suboptimal_weights = {
        'SARIMAX': 0.4,
        'SAMoSSA': 0.3,
        'MSSA-RL': 0.3
    }

    ensemble_preds = (
        suboptimal_weights['SARIMAX'] * sarimax_preds +
        suboptimal_weights['SAMoSSA'] * samossa_preds +
        suboptimal_weights['MSSA-RL'] * mssa_preds
    )

    return {
        'actuals': actuals,
        'models': {
            'SARIMAX': {'predictions': sarimax_preds, 'confidence': sarimax_conf},
            'SAMoSSA': {'predictions': samossa_preds, 'confidence': samossa_conf},
            'MSSA-RL': {'predictions': mssa_preds, 'confidence': mssa_conf}
        },
        'ensemble': {
            'predictions': ensemble_preds,
            'weights': suboptimal_weights
        }
    }

def compute_metrics(predictions, actuals):
    """Compute performance metrics."""
    errors = predictions - actuals
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(errors / actuals)) * 100

    # Directional accuracy
    pred_dir = np.diff(predictions)
    actual_dir = np.diff(actuals)
    dir_acc = ((pred_dir * actual_dir) > 0).sum() / len(pred_dir)

    return rmse, mae, mape, dir_acc

def main():
    print("=" * 80)
    print("ENSEMBLE DIAGNOSTICS TEST WITH SYNTHETIC DATA")
    print("=" * 80)
    print("\nScenario: Ensemble with suboptimal weights underperforms best single model")
    print("  - SARIMAX: Best single model (~0.8 RMSE)")
    print("  - SAMoSSA: Slightly worse (~0.9 RMSE)")
    print("  - MSSA-RL: Worst model (~1.2 RMSE)")
    print("  - Ensemble: Suboptimal weights -> worse than SARIMAX")
    print("\n" + "=" * 80)

    # Generate data
    data = generate_synthetic_forecasts(n_steps=100)
    actuals = data['actuals']

    # Create diagnostics instance
    diagnostics = EnsembleDiagnostics(
        output_dir="visualizations/ensemble_diagnostics/test_synthetic"
    )

    # Add model performances
    print("\n1. Adding individual model performances...")
    for model_name, model_data in data['models'].items():
        preds = model_data['predictions']
        conf = model_data['confidence']

        rmse, mae, mape, dir_acc = compute_metrics(preds, actuals)

        perf = ModelPerformance(
            name=model_name,
            predictions=preds,
            actuals=actuals,
            confidence=conf,
            rmse=rmse,
            mae=mae,
            mape=mape,
            directional_accuracy=dir_acc
        )

        diagnostics.add_model_performance(perf)
        print(f"  OK {model_name}: RMSE={rmse:.4f}, DA={dir_acc:.2%}")

    # Add ensemble performance
    print("\n2. Adding ensemble performance...")
    ens_preds = data['ensemble']['predictions']
    ens_weights = data['ensemble']['weights']

    errors = ens_preds - actuals
    ens_rmse = np.sqrt(np.mean(errors**2))
    ens_bias = np.mean(errors)
    ens_var = np.var(errors)

    ens_perf = EnsemblePerformance(
        predictions=ens_preds,
        actuals=actuals,
        weights=ens_weights,
        rmse=ens_rmse,
        bias=ens_bias,
        variance=ens_var
    )

    diagnostics.add_ensemble_performance(ens_perf)
    print(f"  OK Ensemble: RMSE={ens_rmse:.4f}")
    print(f"    Weights: {ens_weights}")

    # Run diagnostics
    print("\n3. Running full diagnostics...")
    print("   This will generate:")
    print("   - Error decomposition visualization")
    print("   - Confidence calibration analysis")
    print("   - Weight optimization recommendations")
    print("   - Comprehensive diagnostic report")
    print()

    diagnostics.run_full_diagnostics()

    print("\n" + "=" * 80)
    print("OK TEST COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: visualizations/ensemble_diagnostics/test_synthetic/")
    print("\nGenerated files:")
    print("  - error_decomposition.png: RMSE comparison and error analysis")
    print("  - confidence_calibration.png: Model confidence vs actual accuracy")
    print("  - weight_optimization.png: Current vs optimal ensemble weights")
    print("  - ensemble_diagnostics_report.txt: Comprehensive text report")
    print("\nReview these visualizations to understand:")
    print("  1. Why ensemble RMSE > best single model RMSE")
    print("  2. Which models have miscalibrated confidence scores")
    print("  3. What optimal weights would minimize RMSE")
    print()

if __name__ == '__main__':
    main()
