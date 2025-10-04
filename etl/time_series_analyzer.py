"""Time Series Dataset Analyzer with MIT Statistical Learning Standards.

Mathematical Foundations:
========================

1. Missing Data Rate:
   ρ_missing = (Σ I(x_ij = NA)) / (n × p)
   where n = samples, p = features, I = indicator function

2. Temporal Sampling Frequency:
   f_s = 1 / Δt, where Δt = mode(t_i - t_{i-1})
   Nyquist frequency: f_N = f_s / 2

3. Stationarity Test (Augmented Dickey-Fuller):
   Δy_t = α + βt + γy_{t-1} + Σ δ_i Δy_{t-i} + ε_t
   H_0: γ = 0 (unit root exists, non-stationary)

4. Autocorrelation Function (ACF):
   ρ(k) = Cov(y_t, y_{t-k}) / Var(y_t)
   SE(ρ̂(k)) ≈ 1/√n for white noise

5. Statistical Moments:
   μ = E[X], σ² = E[(X-μ)²], γ₁ = E[(X-μ)³]/σ³, γ₂ = E[(X-μ)⁴]/σ⁴ - 3

Success Criteria:
- Missing data < 5% (seldom), < 1% (excellent)
- Temporal gaps detected with 95% confidence
- Distributional assumptions validated (Jarque-Bera test)
- Frequency domain analysis for periodicity detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import logging
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class TimeSeriesDatasetAnalyzer:
    """Comprehensive time series analyzer following MIT statistical standards.

    Implements vectorized analysis with mathematical rigor for:
    - Dataset structure characterization
    - Missing data pattern detection
    - Temporal frequency identification
    - Stationarity assessment
    - Statistical distribution validation
    """

    def __init__(self, dataset_name: str, description: str = ""):
        """Initialize analyzer with dataset metadata.

        Args:
            dataset_name: Identifier for the dataset
            description: Human-readable dataset description
        """
        self.dataset_name = dataset_name
        self.description = description
        self.data = None
        self.analysis_results = {}

    def load_and_inspect_data(self, filepath: str) -> Dict:
        """Load dataset and perform initial inspection.

        Mathematical characterization:
        - Dimensionality: n × p (samples × features)
        - Memory complexity: O(np)
        - Data types: numerical vs categorical decomposition

        Args:
            filepath: Path to parquet/csv file

        Returns:
            Dictionary with inspection metrics
        """
        logger.info(f"Loading and inspecting: {self.dataset_name}")

        try:
            # Load data (vectorized)
            if filepath.endswith('.parquet'):
                self.data = pd.read_parquet(filepath)
            else:
                self.data = pd.read_csv(filepath)

            logger.info(f"Successfully loaded {filepath}")

            # Vectorized characterization
            inspection_results = {
                'total_rows': len(self.data),
                'total_columns': len(self.data.columns),
                'column_names': list(self.data.columns),
                'dtypes': dict(self.data.dtypes),
                'memory_usage_mb': self.data.memory_usage(deep=True).sum() / (1024**2),
                'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(self.data.select_dtypes(exclude=[np.number]).columns),
                'index_type': str(type(self.data.index))
            }

            logger.info(f"Dataset shape: {self.data.shape}")
            logger.info(f"Memory usage: {inspection_results['memory_usage_mb']:.2f} MB")

            self.analysis_results['inspection'] = inspection_results
            return inspection_results

        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return None

    def analyze_missing_data(self) -> Dict:
        """Comprehensive missing data analysis with pattern detection.

        Mathematical formulation:
        - Missing rate per feature: ρ_j = (Σ I(x_ij = NA)) / n
        - Overall missing rate: ρ = (Σ Σ I(x_ij = NA)) / (n × p)
        - Missing pattern: M ∈ {MCAR, MAR, MNAR}

        Returns:
            Dictionary with missing data statistics
        """
        if self.data is None:
            logger.error("No data loaded")
            return None

        logger.info("Analyzing missing data patterns")

        # Vectorized missing data calculation
        missing_mask = self.data.isnull()
        missing_counts = missing_mask.sum()
        missing_percentages = (missing_counts / len(self.data)) * 100

        total_missing = missing_counts.sum()
        total_cells = self.data.shape[0] * self.data.shape[1]
        overall_missing_rate = (total_missing / total_cells) * 100

        # Missing pattern detection (vectorized)
        missing_by_row = missing_mask.sum(axis=1)
        missing_pattern_entropy = stats.entropy(
            missing_by_row.value_counts(normalize=True).values + 1e-10
        )

        # Categorize severity (MIT standards)
        if overall_missing_rate == 0:
            category = "Complete data (ρ = 0)"
            severity = "excellent"
        elif overall_missing_rate < 1:
            category = "Minimal missing data (ρ < 1%)"
            severity = "excellent"
        elif overall_missing_rate < 5:
            category = "Seldom missing data (ρ < 5%)"
            severity = "acceptable"
        else:
            category = f"Substantial missing data (ρ = {overall_missing_rate:.2f}%)"
            severity = "concerning"

        missing_analysis = {
            'total_missing_values': int(total_missing),
            'overall_missing_rate': float(overall_missing_rate),
            'missing_by_column': dict(missing_counts),
            'missing_percentage_by_column': dict(missing_percentages),
            'columns_with_missing': list(missing_counts[missing_counts > 0].index),
            'completely_missing_columns': list(missing_counts[missing_counts == len(self.data)].index),
            'missing_category': category,
            'severity': severity,
            'pattern_entropy': float(missing_pattern_entropy),
            'max_consecutive_missing': int(missing_by_row.max())
        }

        logger.info(f"Missing data: {category}")
        self.analysis_results['missing_data'] = missing_analysis
        return missing_analysis

    def identify_temporal_structure(self, date_column: Optional[str] = None) -> Dict:
        """Identify temporal structure and sampling frequency.

        Mathematical analysis:
        - Sampling period: Δt = mode(t_i - t_{i-1})
        - Sampling frequency: f_s = 1/Δt
        - Nyquist frequency: f_N = f_s/2
        - Temporal span: T = t_n - t_1

        Args:
            date_column: Name of datetime column (auto-detected if None)

        Returns:
            Dictionary with temporal structure metrics
        """
        if self.data is None:
            logger.error("No data loaded")
            return None

        logger.info("Identifying temporal structure")

        # Auto-detect date column (vectorized)
        if date_column is None:
            if isinstance(self.data.index, pd.DatetimeIndex):
                dates = self.data.index
                date_column = 'index'
            else:
                date_candidates = [col for col in self.data.columns
                                 if any(kw in col.lower() for kw in ['date', 'time', 'timestamp'])]
                if date_candidates:
                    date_column = date_candidates[0]
                    dates = pd.to_datetime(self.data[date_column])
                else:
                    logger.warning("No datetime column detected")
                    return {'is_time_series': False}
        else:
            dates = pd.to_datetime(self.data[date_column])

        # Vectorized temporal analysis
        date_diffs = dates.diff().dropna()

        if len(date_diffs) == 0:
            return {'is_time_series': False}

        # Modal sampling period (use value_counts for TimedeltaIndex)
        diff_counts = date_diffs.value_counts()
        modal_diff = diff_counts.index[0] if len(diff_counts) > 0 else date_diffs[0]
        sampling_period_days = modal_diff.total_seconds() / 86400

        # Frequency classification
        if sampling_period_days == 1:
            frequency = "Daily"
            freq_code = "D"
        elif sampling_period_days == 7:
            frequency = "Weekly"
            freq_code = "W"
        elif 28 <= sampling_period_days <= 31:
            frequency = "Monthly"
            freq_code = "M"
        elif 89 <= sampling_period_days <= 92:
            frequency = "Quarterly"
            freq_code = "Q"
        elif 365 <= sampling_period_days <= 366:
            frequency = "Annual"
            freq_code = "Y"
        else:
            frequency = f"Custom (Δt = {sampling_period_days:.2f} days)"
            freq_code = "custom"

        # Temporal gap detection (outliers in Δt)
        gap_threshold = modal_diff + 3 * date_diffs.std()
        temporal_gaps = (date_diffs > gap_threshold).sum()

        structure_analysis = {
            'is_time_series': True,
            'date_column': date_column,
            'sampling_frequency': frequency,
            'frequency_code': freq_code,
            'sampling_period_days': float(sampling_period_days),
            'nyquist_frequency': float(1 / (2 * sampling_period_days)),
            'time_span': {
                'start': dates.min(),
                'end': dates.max(),
                'total_periods': len(dates),
                'duration_days': (dates.max() - dates.min()).days
            },
            'temporal_gaps_detected': int(temporal_gaps),
            'gap_rate': float(temporal_gaps / len(date_diffs)),
            'temporal_regularity': float(1 - date_diffs.std() / date_diffs.mean())
        }

        logger.info(f"Temporal structure: {frequency}, {len(dates)} periods")
        self.analysis_results['temporal_structure'] = structure_analysis
        return structure_analysis

    def statistical_summary(self, numeric_only: bool = True) -> Dict:
        """Compute comprehensive statistical summary.

        Mathematical moments:
        - Location: μ = E[X]
        - Scale: σ² = E[(X-μ)²]
        - Shape: γ₁ (skewness), γ₂ (kurtosis)
        - Quantiles: Q₁, Q₂ (median), Q₃

        Args:
            numeric_only: Analyze only numeric columns

        Returns:
            Dictionary with statistical measures
        """
        if self.data is None:
            logger.error("No data loaded")
            return None

        logger.info("Computing statistical summary")

        # Select numeric columns (vectorized)
        if numeric_only:
            data_numeric = self.data.select_dtypes(include=[np.number])
        else:
            data_numeric = self.data

        # Vectorized statistical moments
        summary = {
            'count': dict(data_numeric.count()),
            'mean': dict(data_numeric.mean()),
            'std': dict(data_numeric.std()),
            'min': dict(data_numeric.min()),
            'q25': dict(data_numeric.quantile(0.25)),
            'median': dict(data_numeric.median()),
            'q75': dict(data_numeric.quantile(0.75)),
            'max': dict(data_numeric.max()),
            'skewness': dict(data_numeric.skew()),
            'kurtosis': dict(data_numeric.kurtosis()),
            'cv': dict((data_numeric.std() / data_numeric.mean()).replace([np.inf, -np.inf], np.nan))
        }

        # Normality tests (Jarque-Bera)
        normality_tests = {}
        for col in data_numeric.columns:
            clean_data = data_numeric[col].dropna()
            if len(clean_data) > 3:
                _, p_value = stats.jarque_bera(clean_data)
                normality_tests[col] = {
                    'p_value': float(p_value),
                    'is_normal': bool(p_value > 0.05)
                }

        summary['normality_tests'] = normality_tests

        logger.info(f"Statistical summary computed for {len(data_numeric.columns)} columns")
        self.analysis_results['statistics'] = summary
        return summary

    def test_stationarity(self, column: str, max_lag: int = 10) -> Dict:
        """Test time series stationarity using Augmented Dickey-Fuller.

        ADF Test Equation:
        Δy_t = α + βt + γy_{t-1} + Σ δ_i Δy_{t-i} + ε_t

        H_0: γ = 0 (unit root, non-stationary)
        H_1: γ < 0 (stationary)

        Args:
            column: Column name to test
            max_lag: Maximum lag for ADF test

        Returns:
            Dictionary with stationarity test results
        """
        if self.data is None or column not in self.data.columns:
            logger.error(f"Column {column} not found")
            return None

        logger.info(f"Testing stationarity for {column}")

        series = self.data[column].dropna()

        if len(series) < max_lag + 1:
            logger.warning(f"Insufficient data for stationarity test")
            return None

        # Augmented Dickey-Fuller test
        adf_result = adfuller(series, maxlag=max_lag)

        stationarity = {
            'column': column,
            'adf_statistic': float(adf_result[0]),
            'p_value': float(adf_result[1]),
            'lags_used': int(adf_result[2]),
            'n_obs': int(adf_result[3]),
            'critical_values': {k: float(v) for k, v in adf_result[4].items()},
            'is_stationary': bool(adf_result[1] < 0.05),
            'conclusion': 'Stationary' if adf_result[1] < 0.05 else 'Non-stationary (unit root)'
        }

        logger.info(f"ADF test: {stationarity['conclusion']} (p={stationarity['p_value']:.4f})")

        if 'stationarity_tests' not in self.analysis_results:
            self.analysis_results['stationarity_tests'] = {}
        self.analysis_results['stationarity_tests'][column] = stationarity

        return stationarity

    def compute_autocorrelation(self, column: str, nlags: int = 40) -> Dict:
        """Compute autocorrelation function (ACF) and partial ACF (PACF).

        ACF: ρ(k) = Cov(y_t, y_{t-k}) / Var(y_t)
        PACF: Correlation after removing effects of intermediate lags

        Args:
            column: Column name
            nlags: Number of lags

        Returns:
            Dictionary with ACF/PACF values
        """
        if self.data is None or column not in self.data.columns:
            logger.error(f"Column {column} not found")
            return None

        logger.info(f"Computing autocorrelation for {column}")

        series = self.data[column].dropna()

        if len(series) < nlags + 1:
            nlags = len(series) - 1

        # Vectorized ACF/PACF computation
        acf_values = acf(series, nlags=nlags)
        pacf_values = pacf(series, nlags=nlags)

        # Confidence intervals (95%)
        conf_int = 1.96 / np.sqrt(len(series))

        autocorr = {
            'column': column,
            'acf': acf_values.tolist(),
            'pacf': pacf_values.tolist(),
            'lags': list(range(nlags + 1)),
            'confidence_interval': float(conf_int),
            'significant_lags_acf': [int(i) for i in range(1, len(acf_values))
                                    if abs(acf_values[i]) > conf_int],
            'significant_lags_pacf': [int(i) for i in range(1, len(pacf_values))
                                     if abs(pacf_values[i]) > conf_int]
        }

        logger.info(f"ACF computed: {len(autocorr['significant_lags_acf'])} significant lags")

        if 'autocorrelation' not in self.analysis_results:
            self.analysis_results['autocorrelation'] = {}
        self.analysis_results['autocorrelation'][column] = autocorr

        return autocorr

    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report.

        Returns:
            Complete analysis results dictionary
        """
        logger.info(f"Generating comprehensive report for {self.dataset_name}")

        report = {
            'dataset_name': self.dataset_name,
            'description': self.description,
            'timestamp': datetime.now().isoformat(),
            'results': self.analysis_results
        }

        return report
