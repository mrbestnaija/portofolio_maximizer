#!/usr/bin/env python3
"""Ensemble Forecasting Error Tracking and Diagnostics.

This module provides comprehensive error analysis and visualization tools
for diagnosing ensemble forecasting performance issues, particularly:
- Why ensemble RMSE > best single model RMSE
- Confidence calibration errors
- Weight optimization opportunities

Mathematical Framework:
======================

1. Ensemble Error Decomposition:
   RMSE_ens² = RMSE_best² + Bias² + Variance_extra

   where:
   - RMSE_best: Best single model error
   - Bias²: Systematic ensemble bias
   - Variance_extra: Additional variance from poor weighting

2. Confidence Calibration:
   For well-calibrated model: P(|error| < σ) ≈ 68.3%

   Calibration error = |empirical_coverage - theoretical_coverage|

3. Weight Optimality:
   Optimal weights minimize: E[(y - Σ w_i f_i)²]

   Subject to: Σ w_i = 1, w_i ≥ 0

References:
- Timmermann (2006): Forecast Combinations
- Armstrong (2001): Principles of Forecasting
- Makridakis et al. (2020): M4 Competition
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Single model performance metrics."""
    name: str
    predictions: np.ndarray
    actuals: np.ndarray
    confidence: np.ndarray  # Model's confidence scores
    rmse: float
    mae: float
    mape: float
    directional_accuracy: float

    @property
    def errors(self) -> np.ndarray:
        """Prediction errors."""
        return self.predictions - self.actuals

    @property
    def squared_errors(self) -> np.ndarray:
        """Squared errors for variance analysis."""
        return self.errors ** 2


@dataclass
class EnsemblePerformance:
    """Ensemble performance with decomposition."""
    predictions: np.ndarray
    actuals: np.ndarray
    weights: Dict[str, float]
    rmse: float
    bias: float
    variance: float

    @property
    def errors(self) -> np.ndarray:
        return self.predictions - self.actuals


class EnsembleDiagnostics:
    """Comprehensive ensemble error tracking and visualization."""

    def __init__(self, output_dir: str = "visualizations/ensemble_diagnostics"):
        """Initialize diagnostics.

        Args:
            output_dir: Directory to save diagnostic visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_performances: Dict[str, ModelPerformance] = {}
        self.ensemble_performance: Optional[EnsemblePerformance] = None

    def add_model_performance(self, perf: ModelPerformance) -> None:
        """Add individual model performance data.

        Args:
            perf: Model performance metrics
        """
        self.model_performances[perf.name] = perf
        logger.info(f"Added {perf.name}: RMSE={perf.rmse:.4f}, DA={perf.directional_accuracy:.2%}")

    def add_ensemble_performance(self, perf: EnsemblePerformance) -> None:
        """Add ensemble performance data.

        Args:
            perf: Ensemble performance metrics
        """
        self.ensemble_performance = perf
        logger.info(f"Added ensemble: RMSE={perf.rmse:.4f}, Bias={perf.bias:.4f}")

    def compute_error_decomposition(self) -> Dict[str, float]:
        """Decompose ensemble error vs best single model.

        Returns:
            Dictionary with error components:
            - best_model_rmse: Best single model RMSE
            - ensemble_rmse: Ensemble RMSE
            - rmse_ratio: ensemble_rmse / best_model_rmse
            - excess_error: ensemble_rmse - best_model_rmse
            - bias_component: Systematic bias²
            - variance_component: Extra variance from weighting
        """
        if not self.ensemble_performance:
            raise ValueError("No ensemble performance data available")

        # Find best single model
        best_model = min(self.model_performances.values(), key=lambda m: m.rmse)

        # Error decomposition
        ens_rmse = self.ensemble_performance.rmse
        best_rmse = best_model.rmse
        ratio = ens_rmse / best_rmse if best_rmse > 0 else float('inf')

        # Bias and variance components
        ens_errors = self.ensemble_performance.errors
        bias_sq = np.mean(ens_errors) ** 2
        variance = np.var(ens_errors)

        # Excess variance vs best model
        best_variance = np.var(best_model.errors)
        excess_variance = variance - best_variance

        decomposition = {
            'best_model': best_model.name,
            'best_model_rmse': best_rmse,
            'ensemble_rmse': ens_rmse,
            'rmse_ratio': ratio,
            'excess_error': ens_rmse - best_rmse,
            'bias_squared': bias_sq,
            'ensemble_variance': variance,
            'best_model_variance': best_variance,
            'excess_variance': excess_variance,
        }

        logger.info(f"Error decomposition: ratio={ratio:.3f}, excess_error={decomposition['excess_error']:.4f}")
        return decomposition

    def compute_confidence_calibration(self, model_name: str) -> Dict[str, Any]:
        """Analyze confidence calibration for a model.

        For well-calibrated forecasts:
        - High confidence predictions should have low errors
        - Confidence should correlate with actual accuracy

        Args:
            model_name: Name of model to analyze

        Returns:
            Calibration metrics including correlation and coverage stats
        """
        if model_name not in self.model_performances:
            raise ValueError(f"Model {model_name} not found")

        perf = self.model_performances[model_name]

        # Correlation between confidence and accuracy
        # High confidence should mean low error
        abs_errors = np.abs(perf.errors)
        correlation = stats.spearmanr(perf.confidence, -abs_errors)  # Negative: high conf = low error

        # Binned calibration
        n_bins = 5
        conf_bins = pd.qcut(perf.confidence, q=n_bins, duplicates='drop')

        calibration_data = []
        for bin_label in conf_bins.unique():
            mask = conf_bins == bin_label
            avg_conf = perf.confidence[mask].mean()
            avg_error = abs_errors[mask].mean()
            count = mask.sum()

            calibration_data.append({
                'bin': str(bin_label),
                'avg_confidence': avg_conf,
                'avg_error': avg_error,
                'count': count
            })

        return {
            'model': model_name,
            'correlation': correlation.correlation,
            'correlation_pvalue': correlation.pvalue,
            'calibration_bins': calibration_data,
            'is_well_calibrated': correlation.correlation < -0.3 and correlation.pvalue < 0.05
        }

    def optimize_weights(self, method: str = 'rmse') -> Dict[str, float]:
        """Find optimal ensemble weights to minimize error.

        Args:
            method: Optimization objective ('rmse', 'mae', or 'sharpe')

        Returns:
            Dictionary of optimal weights per model
        """
        if len(self.model_performances) < 2:
            raise ValueError("Need at least 2 models for ensemble")

        model_names = list(self.model_performances.keys())
        n_models = len(model_names)

        # Stack predictions (samples x models)
        predictions_matrix = np.column_stack([
            self.model_performances[name].predictions
            for name in model_names
        ])

        # Get actuals (use first model's actuals, all should be same)
        actuals = self.model_performances[model_names[0]].actuals

        def objective(weights):
            """RMSE objective function."""
            ensemble_pred = predictions_matrix @ weights
            errors = ensemble_pred - actuals
            return np.sqrt(np.mean(errors ** 2))

        # Constraints: weights sum to 1, all non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]

        # Initial guess: uniform weights
        x0 = np.ones(n_models) / n_models

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )

        if not result.success:
            logger.warning(f"Weight optimization did not converge: {result.message}")

        optimal_weights = dict(zip(model_names, result.x))
        optimal_rmse = result.fun

        # Compare to current ensemble
        current_rmse = self.ensemble_performance.rmse if self.ensemble_performance else float('inf')
        improvement = ((current_rmse - optimal_rmse) / current_rmse * 100) if current_rmse < float('inf') else 0

        logger.info(f"Optimal weights: {optimal_weights}")
        logger.info(f"Optimal RMSE: {optimal_rmse:.4f} (current: {current_rmse:.4f}, improvement: {improvement:.1f}%)")

        return {
            **optimal_weights,
            '_optimal_rmse': optimal_rmse,
            '_current_rmse': current_rmse,
            '_improvement_pct': improvement
        }

    def plot_error_decomposition(self, save_path: Optional[str] = None) -> None:
        """Visualize error decomposition: ensemble vs individual models.

        Creates a comprehensive plot showing:
        - RMSE comparison (bar chart)
        - Error distribution (violin plots)
        - Cumulative errors over time
        - Error contribution by model
        """
        if not self.ensemble_performance or not self.model_performances:
            raise ValueError("Need both ensemble and model performance data")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Ensemble Error Decomposition Analysis', fontsize=16, fontweight='bold')

        # 1. RMSE Comparison
        ax1 = axes[0, 0]
        model_names = list(self.model_performances.keys()) + ['Ensemble']
        rmses = [m.rmse for m in self.model_performances.values()] + [self.ensemble_performance.rmse]
        colors = ['#3498db' if r == min(rmses[:-1]) else '#95a5a6' for r in rmses[:-1]] + ['#e74c3c']

        bars = ax1.bar(range(len(model_names)), rmses, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.set_ylabel('RMSE', fontsize=12)
        ax1.set_title('RMSE Comparison (Red=Ensemble, Blue=Best)', fontsize=12)
        ax1.axhline(y=min(rmses[:-1]), color='green', linestyle='--', alpha=0.5, label='Best Single Model')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Add ratio text
        best_rmse = min(rmses[:-1])
        ratio = self.ensemble_performance.rmse / best_rmse
        ax1.text(0.02, 0.98, f'RMSE Ratio: {ratio:.3f}x',
                transform=ax1.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 2. Error Distribution Violin Plot
        ax2 = axes[0, 1]
        all_errors = [m.errors for m in self.model_performances.values()] + [self.ensemble_performance.errors]
        parts = ax2.violinplot(all_errors, positions=range(len(model_names)),
                               showmeans=True, showmedians=True)
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.set_ylabel('Prediction Error', fontsize=12)
        ax2.set_title('Error Distribution (Violin Plots)', fontsize=12)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.grid(axis='y', alpha=0.3)

        # 3. Cumulative Squared Error Over Time
        ax3 = axes[1, 0]
        for name, perf in self.model_performances.items():
            cumsum_sq_errors = np.cumsum(perf.squared_errors)
            ax3.plot(cumsum_sq_errors, label=name, alpha=0.7)

        ens_cumsum = np.cumsum(self.ensemble_performance.errors ** 2)
        ax3.plot(ens_cumsum, label='Ensemble', linewidth=2, linestyle='--', color='red')

        ax3.set_xlabel('Forecast Step', fontsize=12)
        ax3.set_ylabel('Cumulative Squared Error', fontsize=12)
        ax3.set_title('Cumulative Error Accumulation', fontsize=12)
        ax3.legend(loc='upper left')
        ax3.grid(alpha=0.3)

        # 4. Error Decomposition Components
        ax4 = axes[1, 1]
        decomp = self.compute_error_decomposition()

        components = [
            ('Best Model\nRMSE', decomp['best_model_rmse']),
            ('Ensemble\nBias²', decomp['bias_squared']),
            ('Excess\nVariance', max(0, decomp['excess_variance'])),
        ]

        comp_names, comp_values = zip(*components)
        colors_comp = ['#2ecc71', '#e67e22', '#e74c3c']

        ax4.bar(range(len(comp_names)), comp_values, color=colors_comp, alpha=0.7, edgecolor='black')
        ax4.set_xticks(range(len(comp_names)))
        ax4.set_xticklabels(comp_names, fontsize=10)
        ax4.set_ylabel('Error Component', fontsize=12)
        ax4.set_title('Error Decomposition: RMSE² ≈ Best² + Bias² + Excess Var', fontsize=11)
        ax4.grid(axis='y', alpha=0.3)

        # Add decomposition formula
        formula = f"RMSE²_ens ({self.ensemble_performance.rmse**2:.4f}) ≈\nBest² ({decomp['best_model_rmse']**2:.4f}) + Bias² ({decomp['bias_squared']:.4f}) + ExVar ({max(0, decomp['excess_variance']):.4f})"
        ax4.text(0.02, 0.98, formula, transform=ax4.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / 'error_decomposition.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved error decomposition plot to {save_path}")
        plt.close()

    def plot_confidence_calibration(self, save_path: Optional[str] = None) -> None:
        """Visualize confidence calibration for all models.

        Well-calibrated models show:
        - Strong negative correlation between confidence and error
        - Linear relationship in calibration plot
        """
        if not self.model_performances:
            raise ValueError("No model performance data")

        n_models = len(self.model_performances)
        fig, axes = plt.subplots(n_models, 2, figsize=(14, 5*n_models))
        if n_models == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle('Model Confidence Calibration Analysis', fontsize=16, fontweight='bold')

        for idx, (name, perf) in enumerate(self.model_performances.items()):
            calib = self.compute_confidence_calibration(name)

            # Left: Scatter plot of confidence vs error
            ax_left = axes[idx, 0]
            abs_errors = np.abs(perf.errors)
            ax_left.scatter(perf.confidence, abs_errors, alpha=0.5, s=20)

            # Add trend line
            z = np.polyfit(perf.confidence, abs_errors, 1)
            p = np.poly1d(z)
            x_line = np.linspace(perf.confidence.min(), perf.confidence.max(), 100)
            ax_left.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Trend (corr={calib["correlation"]:.3f})')

            ax_left.set_xlabel('Model Confidence', fontsize=11)
            ax_left.set_ylabel('Absolute Error', fontsize=11)
            ax_left.set_title(f'{name} - Confidence vs Error', fontsize=12)
            ax_left.legend()
            ax_left.grid(alpha=0.3)

            # Add calibration status
            status = "✓ Well-Calibrated" if calib['is_well_calibrated'] else "✗ Poor Calibration"
            color = 'green' if calib['is_well_calibrated'] else 'red'
            ax_left.text(0.02, 0.98, status, transform=ax_left.transAxes, va='top',
                        color=color, fontweight='bold', fontsize=11,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Right: Binned calibration plot
            ax_right = axes[idx, 1]
            bins_df = pd.DataFrame(calib['calibration_bins'])

            x_pos = range(len(bins_df))
            ax_right.bar(x_pos, bins_df['avg_error'], alpha=0.7, color='coral', label='Avg Error')
            ax_right.set_xlabel('Confidence Bin', fontsize=11)
            ax_right.set_ylabel('Average Error', fontsize=11)
            ax_right.set_title(f'{name} - Error by Confidence Bin', fontsize=12)
            ax_right.set_xticks(x_pos)
            ax_right.set_xticklabels([f"{row['avg_confidence']:.2f}" for _, row in bins_df.iterrows()], rotation=45)
            ax_right.grid(axis='y', alpha=0.3)
            ax_right.legend()

            # Ideal: error should decrease as confidence increases
            # Add expected trend line
            ideal_errors = np.linspace(bins_df['avg_error'].max(), bins_df['avg_error'].min(), len(bins_df))
            ax_right.plot(x_pos, ideal_errors, 'g--', alpha=0.6, linewidth=2, label='Ideal Trend')
            ax_right.legend()

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / 'confidence_calibration.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved confidence calibration plot to {save_path}")
        plt.close()

    def plot_weight_optimization(self, save_path: Optional[str] = None) -> None:
        """Visualize current vs optimal ensemble weights.

        Shows:
        - Current weights used by ensemble
        - Optimal weights that minimize RMSE
        - Expected improvement from reweighting
        """
        if not self.ensemble_performance or not self.model_performances:
            raise ValueError("Need ensemble performance data")

        optimal_weights = self.optimize_weights()

        # Extract current and optimal weights
        model_names = list(self.model_performances.keys())
        current_weights = [self.ensemble_performance.weights.get(name, 0) for name in model_names]
        optimal_weight_values = [optimal_weights.get(name, 0) for name in model_names]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Ensemble Weight Optimization Analysis', fontsize=16, fontweight='bold')

        # Left: Current vs Optimal Weights
        ax1 = axes[0]
        x = np.arange(len(model_names))
        width = 0.35

        bars1 = ax1.bar(x - width/2, current_weights, width, label='Current Weights',
                       alpha=0.7, color='#3498db', edgecolor='black')
        bars2 = ax1.bar(x + width/2, optimal_weight_values, width, label='Optimal Weights',
                       alpha=0.7, color='#2ecc71', edgecolor='black')

        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('Weight', fontsize=12)
        ax1.set_title('Current vs Optimal Ensemble Weights', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 1.0])

        # Add weight change annotations
        for i, (curr, opt) in enumerate(zip(current_weights, optimal_weight_values)):
            change = opt - curr
            if abs(change) > 0.05:  # Only annotate significant changes
                ax1.annotate(f'{change:+.2f}',
                           xy=(i, max(curr, opt) + 0.05),
                           ha='center', fontsize=9,
                           color='red' if change < 0 else 'green')

        # Right: RMSE Improvement
        ax2 = axes[1]

        # Individual model RMSEs
        model_rmses = [perf.rmse for perf in self.model_performances.values()]
        current_rmse = self.ensemble_performance.rmse
        optimal_rmse = optimal_weights.get('_optimal_rmse', current_rmse)
        improvement_pct = optimal_weights.get('_improvement_pct', 0)

        # Bar chart
        categories = model_names + ['Current\nEnsemble', 'Optimal\nEnsemble']
        rmse_values = model_rmses + [current_rmse, optimal_rmse]
        colors = ['#95a5a6'] * len(model_names) + ['#e74c3c', '#2ecc71']

        bars = ax2.bar(range(len(categories)), rmse_values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(len(categories)))
        ax2.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
        ax2.set_ylabel('RMSE', fontsize=12)
        ax2.set_title('RMSE: Current vs Optimized Ensemble', fontsize=12)
        ax2.grid(axis='y', alpha=0.3)

        # Highlight best single model
        best_single_rmse = min(model_rmses)
        ax2.axhline(y=best_single_rmse, color='blue', linestyle='--', alpha=0.5,
                   label='Best Single Model', linewidth=2)
        ax2.legend()

        # Add improvement text
        improvement_text = f"Potential Improvement:\n{improvement_pct:.1f}%\n\nCurrent RMSE: {current_rmse:.4f}\nOptimal RMSE: {optimal_rmse:.4f}"
        ax2.text(0.98, 0.98, improvement_text, transform=ax2.transAxes,
                va='top', ha='right', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / 'weight_optimization.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved weight optimization plot to {save_path}")
        plt.close()

    def generate_diagnostic_report(self) -> str:
        """Generate comprehensive text report of ensemble diagnostics.

        Returns:
            Formatted diagnostic report string
        """
        if not self.ensemble_performance or not self.model_performances:
            return "Insufficient data for diagnostic report"

        report = []
        report.append("=" * 80)
        report.append("ENSEMBLE FORECASTING DIAGNOSTICS REPORT")
        report.append("=" * 80)
        report.append("")

        # 1. Error Decomposition
        decomp = self.compute_error_decomposition()
        report.append("1. ERROR DECOMPOSITION")
        report.append("-" * 80)
        report.append(f"Best Single Model: {decomp['best_model']} (RMSE: {decomp['best_model_rmse']:.4f})")
        report.append(f"Ensemble RMSE: {decomp['ensemble_rmse']:.4f}")
        report.append(f"RMSE Ratio: {decomp['rmse_ratio']:.3f}x {'✓' if decomp['rmse_ratio'] < 1.1 else '✗ FAILED (>1.1x)'}")
        report.append(f"Excess Error: {decomp['excess_error']:.4f}")
        report.append(f"  - Bias²: {decomp['bias_squared']:.6f}")
        report.append(f"  - Excess Variance: {decomp['excess_variance']:.6f}")
        report.append("")

        # 2. Individual Model Performance
        report.append("2. INDIVIDUAL MODEL PERFORMANCE")
        report.append("-" * 80)
        for name, perf in sorted(self.model_performances.items(), key=lambda x: x[1].rmse):
            report.append(f"{name}:")
            report.append(f"  RMSE: {perf.rmse:.4f}")
            report.append(f"  MAE: {perf.mae:.4f}")
            report.append(f"  Directional Accuracy: {perf.directional_accuracy:.2%}")
            report.append(f"  Avg Confidence: {perf.confidence.mean():.3f}")
        report.append("")

        # 3. Confidence Calibration
        report.append("3. CONFIDENCE CALIBRATION")
        report.append("-" * 80)
        for name in self.model_performances.keys():
            calib = self.compute_confidence_calibration(name)
            status = "✓ GOOD" if calib['is_well_calibrated'] else "✗ POOR"
            report.append(f"{name}: {status}")
            report.append(f"  Correlation (conf vs -error): {calib['correlation']:.3f} (p={calib['correlation_pvalue']:.4f})")
        report.append("")

        # 4. Weight Optimization
        report.append("4. WEIGHT OPTIMIZATION")
        report.append("-" * 80)
        optimal = self.optimize_weights()
        report.append("Current Weights:")
        for name in self.model_performances.keys():
            curr_w = self.ensemble_performance.weights.get(name, 0)
            opt_w = optimal.get(name, 0)
            change = opt_w - curr_w
            report.append(f"  {name}: {curr_w:.3f} → {opt_w:.3f} ({change:+.3f})")

        report.append("")
        report.append(f"Potential RMSE Improvement: {optimal.get('_improvement_pct', 0):.1f}%")
        report.append(f"  Current: {optimal.get('_current_rmse', 0):.4f}")
        report.append(f"  Optimal: {optimal.get('_optimal_rmse', 0):.4f}")
        report.append("")

        # 5. Recommendations
        report.append("5. RECOMMENDATIONS")
        report.append("-" * 80)

        if decomp['rmse_ratio'] >= 1.1:
            report.append("❌ CRITICAL: Ensemble underperforms best single model")
            report.append("   → Review weight assignment logic")
            report.append("   → Consider using optimal weights from optimization")

            if decomp['bias_squared'] > 0.01:
                report.append("   → Significant bias detected - check for systematic errors")

            if decomp['excess_variance'] > 0.01:
                report.append("   → High excess variance - weights may be suboptimal")

        # Check calibration issues
        poorly_calibrated = []
        for name in self.model_performances.keys():
            calib = self.compute_confidence_calibration(name)
            if not calib['is_well_calibrated']:
                poorly_calibrated.append(name)

        if poorly_calibrated:
            report.append(f"⚠️ WARNING: {len(poorly_calibrated)} model(s) have poor confidence calibration:")
            for name in poorly_calibrated:
                report.append(f"   → {name}: Recalibrate confidence scoring")

        # Optimal weight improvement
        improvement_pct = optimal.get('_improvement_pct', 0)
        if improvement_pct > 5:
            report.append(f"✓ OPPORTUNITY: {improvement_pct:.1f}% RMSE improvement possible with optimal weights")
            report.append("   → Update ensemble weight logic in production")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def save_diagnostic_report(self, filename: Optional[str] = None) -> None:
        """Save diagnostic report to file.

        Args:
            filename: Output filename (default: ensemble_diagnostics_report.txt)
        """
        if filename is None:
            filename = self.output_dir / 'ensemble_diagnostics_report.txt'

        report = self.generate_diagnostic_report()

        with open(filename, 'w') as f:
            f.write(report)

        logger.info(f"Saved diagnostic report to {filename}")
        print(report)  # Also print to console

    def run_full_diagnostics(self) -> None:
        """Run all diagnostic analyses and generate visualizations."""
        logger.info("Starting full ensemble diagnostics...")

        try:
            self.plot_error_decomposition()
            logger.info("✓ Generated error decomposition plot")
        except Exception as e:
            logger.error(f"Failed to generate error decomposition: {e}")

        try:
            self.plot_confidence_calibration()
            logger.info("✓ Generated confidence calibration plot")
        except Exception as e:
            logger.error(f"Failed to generate confidence calibration: {e}")

        try:
            self.plot_weight_optimization()
            logger.info("✓ Generated weight optimization plot")
        except Exception as e:
            logger.error(f"Failed to generate weight optimization: {e}")

        try:
            self.save_diagnostic_report()
            logger.info("✓ Generated diagnostic report")
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")

        logger.info(f"Full diagnostics complete. Results saved to {self.output_dir}")
