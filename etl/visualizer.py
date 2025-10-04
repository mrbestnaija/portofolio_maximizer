"""Robust Time Series Visualization Module with Statistical Rigor.

Mathematical Foundations:
========================

1. Trend Decomposition (Additive Model):
   y_t = T_t + S_t + R_t
   where T_t = trend, S_t = seasonal, R_t = residual

2. QQ-Plot (Quantile-Quantile):
   Theoretical quantiles vs sample quantiles
   Tests normality assumption: X ~ N(μ, σ²)

3. Spectral Density (Power Spectrum):
   S(f) = |Σ x_t e^(-i2πft)|²
   Identifies dominant frequencies in time series

4. Rolling Statistics:
   μ_t(w) = (1/w) Σ_{k=0}^{w-1} x_{t-k}
   σ_t²(w) = (1/w) Σ_{k=0}^{w-1} (x_{t-k} - μ_t(w))²

5. Confidence Intervals (ACF):
   CI = ±z_{α/2} / √n, where z_{0.025} = 1.96 for 95% CI

References:
- Tufte: The Visual Display of Quantitative Information
- Cleveland: Visualizing Data
- Wickham: ggplot2: Elegant Graphics for Data Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict
from pathlib import Path
from scipy import signal, stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import logging

logger = logging.getLogger(__name__)


class TimeSeriesVisualizer:
    """Comprehensive visualization for time series analysis with statistical rigor."""

    def __init__(self, figsize: Tuple[int, int] = (16, 12), style: str = 'seaborn-v0_8-darkgrid'):
        """Initialize visualizer with plotting defaults.

        Args:
            figsize: Default figure size (width, height)
            style: Matplotlib style
        """
        self.figsize = figsize
        self.style = style
        plt.style.use('default')  # Reset to default first
        sns.set_palette("husl")
        self.colors = sns.color_palette("husl", 10)

    def plot_time_series_overview(self, data: pd.DataFrame, columns: Optional[List[str]] = None,
                                   title: str = "Time Series Overview") -> plt.Figure:
        """Create comprehensive time series overview with multiple panels.

        Mathematical components:
        - Raw time series: x_t vs t
        - Rolling mean: μ_t(w) with window w
        - Rolling std: σ_t(w) for volatility
        - Returns: r_t = ln(x_t / x_{t-1})

        Args:
            data: DataFrame with time series data (DatetimeIndex)
            columns: Columns to plot (default: all numeric)
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()[:4]

        n_cols = min(len(columns), 4)
        fig, axes = plt.subplots(n_cols, 1, figsize=(self.figsize[0], n_cols * 3))
        if n_cols == 1:
            axes = [axes]

        for idx, col in enumerate(columns[:n_cols]):
            ax = axes[idx]

            # Plot raw series
            ax.plot(data.index, data[col], label=f'{col}',
                   color=self.colors[idx], linewidth=1.5, alpha=0.8)

            # Add rolling mean (30-day window)
            window = min(30, len(data) // 10)
            if window > 1:
                rolling_mean = data[col].rolling(window=window).mean()
                ax.plot(data.index, rolling_mean, label=f'MA({window})',
                       color='red', linewidth=2, linestyle='--', alpha=0.7)

            ax.set_ylabel(col, fontsize=10, fontweight='bold')
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)

            # Add descriptive statistics
            mean_val = data[col].mean()
            std_val = data[col].std()
            ax.axhline(mean_val, color='green', linestyle=':', alpha=0.5,
                      label=f'μ={mean_val:.2f}')

        axes[-1].set_xlabel('Time', fontsize=11, fontweight='bold')
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        logger.info(f"Created time series overview for {len(columns)} columns")
        return fig

    def plot_distribution_analysis(self, data: pd.DataFrame, column: str) -> plt.Figure:
        """Comprehensive distribution analysis with statistical tests.

        Components:
        - Histogram with KDE overlay
        - QQ-plot for normality assessment
        - Box plot for outlier detection
        - Statistical summary

        Args:
            data: DataFrame with time series
            column: Column to analyze

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        series = data[column].dropna()

        # 1. Histogram with KDE
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax1.hist(series, bins=50, density=True, alpha=0.6, color=self.colors[0],
                edgecolor='black')

        # Fit normal distribution
        mu, sigma = series.mean(), series.std()
        x = np.linspace(series.min(), series.max(), 100)
        ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                label=f'N(μ={mu:.2f}, σ={sigma:.2f})')

        # KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(series)
        ax1.plot(x, kde(x), 'g--', linewidth=2, label='KDE')

        ax1.set_xlabel(column, fontsize=10, fontweight='bold')
        ax1.set_ylabel('Density', fontsize=10, fontweight='bold')
        ax1.set_title('Distribution with Normal Fit', fontsize=11, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. QQ-Plot (Quantile-Quantile)
        ax2 = fig.add_subplot(gs[0:2, 2])
        stats.probplot(series, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normality Test)', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Box Plot (Outlier Detection)
        ax3 = fig.add_subplot(gs[2, 0])
        bp = ax3.boxplot(series, vert=False, patch_artist=True,
                        boxprops=dict(facecolor=self.colors[1], alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        ax3.set_xlabel(column, fontsize=10)
        ax3.set_title('Box Plot (Outliers)', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. Statistical Summary
        ax4 = fig.add_subplot(gs[2, 1:])
        ax4.axis('off')

        # Compute statistics
        stats_summary = {
            'Count': len(series),
            'Mean (μ)': series.mean(),
            'Std (σ)': series.std(),
            'Min': series.min(),
            'Q1': series.quantile(0.25),
            'Median': series.median(),
            'Q3': series.quantile(0.75),
            'Max': series.max(),
            'Skewness (γ₁)': series.skew(),
            'Kurtosis (γ₂)': series.kurtosis(),
            'CV': series.std() / series.mean() if series.mean() != 0 else np.nan
        }

        # Jarque-Bera test
        jb_stat, jb_pvalue = stats.jarque_bera(series)

        summary_text = "Statistical Summary:\n" + "="*40 + "\n"
        for key, val in stats_summary.items():
            summary_text += f"{key:20s}: {val:12.6f}\n"
        summary_text += "="*40 + "\n"
        summary_text += f"Jarque-Bera Test:\n"
        summary_text += f"  Statistic: {jb_stat:.6f}\n"
        summary_text += f"  p-value: {jb_pvalue:.6f}\n"
        summary_text += f"  Normal: {'Yes' if jb_pvalue > 0.05 else 'No'} (α=0.05)"

        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle(f'Distribution Analysis: {column}', fontsize=14, fontweight='bold')

        logger.info(f"Created distribution analysis for {column}")
        return fig

    def plot_autocorrelation(self, data: pd.DataFrame, column: str,
                           nlags: int = 40) -> plt.Figure:
        """Plot ACF and PACF with confidence intervals.

        Mathematical basis:
        - ACF: ρ(k) = Cov(y_t, y_{t-k}) / Var(y_t)
        - PACF: Partial correlation after removing intermediate lags
        - CI: ±1.96/√n for 95% confidence

        Args:
            data: DataFrame with time series
            column: Column to analyze
            nlags: Number of lags

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(self.figsize[0], 8))

        series = data[column].dropna()

        # ACF
        plot_acf(series, lags=nlags, ax=axes[0], alpha=0.05)
        axes[0].set_title(f'Autocorrelation Function (ACF): {column}',
                         fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Lag', fontsize=10)
        axes[0].set_ylabel('ACF', fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Add mathematical annotation
        ci_value = 1.96 / np.sqrt(len(series))
        axes[0].text(0.02, 0.98, f'95% CI: ±{ci_value:.4f}\nFormula: ±1.96/√n',
                    transform=axes[0].transAxes, fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        # PACF
        plot_pacf(series, lags=nlags, ax=axes[1], alpha=0.05)
        axes[1].set_title(f'Partial Autocorrelation Function (PACF): {column}',
                         fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Lag', fontsize=10)
        axes[1].set_ylabel('PACF', fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        logger.info(f"Created ACF/PACF plots for {column}")
        return fig

    def plot_decomposition(self, data: pd.DataFrame, column: str,
                          period: Optional[int] = None) -> plt.Figure:
        """Trend-Seasonal decomposition plot.

        Additive model: y_t = T_t + S_t + R_t
        where:
        - T_t: Trend component
        - S_t: Seasonal component
        - R_t: Residual component

        Args:
            data: DataFrame with time series
            column: Column to decompose
            period: Seasonal period (auto-detected if None)

        Returns:
            Matplotlib figure
        """
        series = data[column].dropna()

        # Auto-detect period if not provided
        if period is None:
            # Assume daily data, use 30 days as default
            period = min(30, len(series) // 2)

        if len(series) < 2 * period:
            logger.warning(f"Series too short for decomposition (need >= {2*period})")
            period = len(series) // 2

        # Perform decomposition
        decomposition = seasonal_decompose(series, model='additive', period=period)

        fig, axes = plt.subplots(4, 1, figsize=(self.figsize[0], 10))

        # Original
        axes[0].plot(series.index, series, color=self.colors[0], linewidth=1.5)
        axes[0].set_ylabel('Original', fontsize=10, fontweight='bold')
        axes[0].set_title(f'Time Series Decomposition: {column} (Period={period})',
                         fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Trend
        axes[1].plot(decomposition.trend.index, decomposition.trend,
                    color=self.colors[1], linewidth=2)
        axes[1].set_ylabel('Trend (T_t)', fontsize=10, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # Seasonal
        axes[2].plot(decomposition.seasonal.index, decomposition.seasonal,
                    color=self.colors[2], linewidth=1.5)
        axes[2].set_ylabel('Seasonal (S_t)', fontsize=10, fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        # Residual
        axes[3].plot(decomposition.resid.index, decomposition.resid,
                    color=self.colors[3], linewidth=1, alpha=0.7)
        axes[3].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[3].set_ylabel('Residual (R_t)', fontsize=10, fontweight='bold')
        axes[3].set_xlabel('Time', fontsize=10, fontweight='bold')
        axes[3].grid(True, alpha=0.3)

        # Add model equation
        fig.text(0.99, 0.01, 'Model: y_t = T_t + S_t + R_t (Additive)',
                ha='right', fontsize=9, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        plt.tight_layout()

        logger.info(f"Created decomposition plot for {column}")
        return fig

    def plot_rolling_statistics(self, data: pd.DataFrame, column: str,
                                window: int = 30) -> plt.Figure:
        """Rolling statistics for stationarity assessment.

        Plots:
        - Original series
        - Rolling mean: μ_t(w)
        - Rolling std: σ_t(w)

        Args:
            data: DataFrame with time series
            column: Column to analyze
            window: Rolling window size

        Returns:
            Matplotlib figure
        """
        series = data[column].dropna()

        # Compute rolling statistics
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()

        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], 9))

        # Original series with rolling mean
        axes[0].plot(series.index, series, label='Original',
                    color=self.colors[0], linewidth=1, alpha=0.7)
        axes[0].plot(rolling_mean.index, rolling_mean, label=f'Rolling Mean (w={window})',
                    color='red', linewidth=2)
        axes[0].set_ylabel(column, fontsize=10, fontweight='bold')
        axes[0].set_title(f'Rolling Statistics Analysis: {column}',
                         fontsize=12, fontweight='bold')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)

        # Rolling mean
        axes[1].plot(rolling_mean.index, rolling_mean, color=self.colors[1], linewidth=2)
        axes[1].set_ylabel(f'Rolling Mean\nμ_t({window})', fontsize=10, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # Rolling std (volatility)
        axes[2].plot(rolling_std.index, rolling_std, color=self.colors[2], linewidth=2)
        axes[2].set_ylabel(f'Rolling Std\nσ_t({window})', fontsize=10, fontweight='bold')
        axes[2].set_xlabel('Time', fontsize=10, fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        # Add interpretation note
        fig.text(0.99, 0.01,
                'Stationary series: constant μ_t and σ_t over time',
                ha='right', fontsize=9, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        plt.tight_layout()

        logger.info(f"Created rolling statistics plot for {column}")
        return fig

    def plot_spectral_density(self, data: pd.DataFrame, column: str) -> plt.Figure:
        """Power spectral density plot (frequency domain analysis).

        S(f) = |FFT(x_t)|²
        Identifies dominant frequencies/periodicities

        Args:
            data: DataFrame with time series
            column: Column to analyze

        Returns:
            Matplotlib figure
        """
        series = data[column].dropna()

        # Compute power spectral density using Welch's method
        # Convert to numpy array to avoid pandas indexing issues
        series_values = series.values
        frequencies, psd = signal.welch(series_values, fs=1.0, nperseg=min(256, len(series_values)//2))

        fig, axes = plt.subplots(2, 1, figsize=(self.figsize[0], 8))

        # PSD (linear scale)
        axes[0].semilogy(frequencies, psd, color=self.colors[0], linewidth=2)
        axes[0].set_ylabel('Power Spectral Density', fontsize=10, fontweight='bold')
        axes[0].set_title(f'Spectral Density Analysis: {column}',
                         fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, which='both')

        # PSD (log-log scale)
        axes[1].loglog(frequencies[1:], psd[1:], color=self.colors[1], linewidth=2)
        axes[1].set_xlabel('Frequency (cycles/period)', fontsize=10, fontweight='bold')
        axes[1].set_ylabel('PSD (log scale)', fontsize=10, fontweight='bold')
        axes[1].grid(True, alpha=0.3, which='both')

        # Find dominant frequency
        if len(frequencies) > 1:
            dominant_idx = np.argmax(psd[1:]) + 1
            dominant_freq = frequencies[dominant_idx]
            axes[0].axvline(dominant_freq, color='red', linestyle='--', alpha=0.7,
                          label=f'Dominant: {dominant_freq:.4f}')
            axes[0].legend()

        # Add mathematical note
        fig.text(0.99, 0.01, 'S(f) = |FFT(x_t)|² (Welch\'s method)',
                ha='right', fontsize=9, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        plt.tight_layout()

        logger.info(f"Created spectral density plot for {column}")
        return fig

    def plot_comprehensive_dashboard(self, data: pd.DataFrame, column: str,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive dashboard with all analyses.

        Includes:
        - Time series with rolling statistics
        - Distribution with QQ-plot
        - ACF/PACF
        - Spectral density
        - Decomposition

        Args:
            data: DataFrame with time series
            column: Column to analyze
            save_path: Path to save figure (optional)

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        series = data[column].dropna()

        # 1. Time series with rolling mean
        ax1 = fig.add_subplot(gs[0, :])
        window = min(30, len(series) // 10)
        rolling_mean = series.rolling(window=window).mean()
        ax1.plot(series.index, series, label='Original', alpha=0.7, linewidth=1)
        ax1.plot(rolling_mean.index, rolling_mean, label=f'MA({window})',
                linewidth=2, color='red')
        ax1.set_title(f'Comprehensive Analysis Dashboard: {column}',
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel(column, fontsize=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(series, bins=50, density=True, alpha=0.6, edgecolor='black')
        mu, sigma = series.mean(), series.std()
        x = np.linspace(series.min(), series.max(), 100)
        ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2)
        ax2.set_title('Distribution', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. QQ-Plot
        ax3 = fig.add_subplot(gs[1, 1])
        stats.probplot(series, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. Box plot
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.boxplot(series, vert=True, patch_artist=True)
        ax4.set_title('Box Plot', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. ACF
        ax5 = fig.add_subplot(gs[2, :2])
        plot_acf(series, lags=min(40, len(series)//2), ax=ax5, alpha=0.05)
        ax5.set_title('ACF', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # 6. PACF
        ax6 = fig.add_subplot(gs[2, 2])
        plot_pacf(series, lags=min(20, len(series)//2), ax=ax6, alpha=0.05)
        ax6.set_title('PACF', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3)

        # 7. Rolling std
        ax7 = fig.add_subplot(gs[3, 0])
        rolling_std = series.rolling(window=window).std()
        ax7.plot(rolling_std.index, rolling_std, linewidth=2)
        ax7.set_title('Rolling Std (Volatility)', fontsize=11, fontweight='bold')
        ax7.set_xlabel('Time', fontsize=9)
        ax7.grid(True, alpha=0.3)

        # 8. Spectral density
        ax8 = fig.add_subplot(gs[3, 1:])
        series_values = series.values
        frequencies, psd = signal.welch(series_values, fs=1.0,
                                       nperseg=min(256, len(series_values)//2))
        ax8.semilogy(frequencies, psd, linewidth=2)
        ax8.set_title('Power Spectral Density', fontsize=11, fontweight='bold')
        ax8.set_xlabel('Frequency', fontsize=9)
        ax8.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Dashboard saved to {save_path}")

        logger.info(f"Created comprehensive dashboard for {column}")
        return fig

    def save_all_plots(self, data: pd.DataFrame, column: str,
                      output_dir: str = "visualizations") -> Dict[str, str]:
        """Generate and save all visualization types.

        Args:
            data: DataFrame with time series
            column: Column to analyze
            output_dir: Directory to save plots

        Returns:
            Dictionary mapping plot type to file path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # 1. Time series overview
        fig = self.plot_time_series_overview(data, columns=[column])
        path = output_path / f"{column}_overview.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files['overview'] = str(path)

        # 2. Distribution analysis
        fig = self.plot_distribution_analysis(data, column)
        path = output_path / f"{column}_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files['distribution'] = str(path)

        # 3. Autocorrelation
        fig = self.plot_autocorrelation(data, column)
        path = output_path / f"{column}_acf_pacf.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files['autocorrelation'] = str(path)

        # 4. Decomposition
        fig = self.plot_decomposition(data, column)
        path = output_path / f"{column}_decomposition.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files['decomposition'] = str(path)

        # 5. Rolling statistics
        fig = self.plot_rolling_statistics(data, column)
        path = output_path / f"{column}_rolling_stats.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files['rolling_stats'] = str(path)

        # 6. Spectral density
        fig = self.plot_spectral_density(data, column)
        path = output_path / f"{column}_spectral.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files['spectral'] = str(path)

        # 7. Comprehensive dashboard
        fig = self.plot_comprehensive_dashboard(data, column)
        path = output_path / f"{column}_dashboard.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files['dashboard'] = str(path)

        logger.info(f"Saved {len(saved_files)} visualizations to {output_dir}")
        return saved_files
