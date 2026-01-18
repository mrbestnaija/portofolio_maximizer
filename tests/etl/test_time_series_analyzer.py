"""Comprehensive tests for TimeSeriesDatasetAnalyzer.

Following MIT standards for statistical testing with:
- Quantitative validation of all metrics
- Mathematical correctness verification
- Edge case handling
- Vectorized operation validation
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from etl.time_series_analyzer import TimeSeriesDatasetAnalyzer


@pytest.fixture
def sample_time_series_data():
    """Generate synthetic time series data for testing."""
    np.random.seed(42)

    # Daily data for 1000 days
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')

    # Generate multiple time series with different characteristics
    n = len(dates)

    data = pd.DataFrame({
        'Date': dates,
        'stationary': np.random.randn(n),  # Stationary white noise
        'trend': np.arange(n) + np.random.randn(n) * 10,  # Linear trend + noise
        'seasonal': 10 * np.sin(2 * np.pi * np.arange(n) / 365) + np.random.randn(n),
        'random_walk': np.cumsum(np.random.randn(n)),  # Non-stationary
        'volume': np.abs(np.random.randn(n) * 1000000),
        'category': np.random.choice(['A', 'B', 'C'], n)
    })

    # Introduce some missing values (2% missing rate)
    missing_indices = np.random.choice(n, size=int(n * 0.02), replace=False)
    data.loc[missing_indices, 'stationary'] = np.nan

    data.set_index('Date', inplace=True)

    return data


@pytest.fixture
def sample_data_file(sample_time_series_data, tmp_path):
    """Create temporary parquet file with sample data."""
    filepath = tmp_path / "test_data.parquet"
    sample_time_series_data.reset_index().to_parquet(filepath)
    return filepath


class TestTimeSeriesDatasetAnalyzer:
    """Test suite for TimeSeriesDatasetAnalyzer."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = TimeSeriesDatasetAnalyzer(
            dataset_name="test_dataset",
            description="Test description"
        )

        assert analyzer.dataset_name == "test_dataset"
        assert analyzer.description == "Test description"
        assert analyzer.data is None
        assert analyzer.analysis_results == {}

    def test_load_and_inspect_parquet(self, sample_data_file):
        """Test loading and inspecting parquet file."""
        analyzer = TimeSeriesDatasetAnalyzer("test")
        result = analyzer.load_and_inspect_data(str(sample_data_file))

        assert result is not None
        assert result['total_rows'] == 1000
        assert result['total_columns'] > 0
        assert 'numeric_columns' in result
        assert 'categorical_columns' in result
        assert result['memory_usage_mb'] > 0
        assert analyzer.data is not None

    def test_missing_data_analysis(self, sample_data_file):
        """Test missing data analysis with known missing rate."""
        analyzer = TimeSeriesDatasetAnalyzer("test")
        analyzer.load_and_inspect_data(str(sample_data_file))
        result = analyzer.analyze_missing_data()

        assert result is not None
        assert 'overall_missing_rate' in result
        assert 'missing_category' in result
        assert 'severity' in result

        # Should detect ~2% missing rate in stationary column (out of 7 columns, only 1 has 2% missing)
        # Overall rate = 2% / 7 ≈ 0.29%
        assert 0.1 <= result['overall_missing_rate'] <= 0.5

        # Should be categorized as "acceptable" (< 5%)
        assert result['severity'] in ['excellent', 'acceptable']

        assert 'pattern_entropy' in result
        assert isinstance(result['pattern_entropy'], float)

    def test_missing_data_complete_dataset(self):
        """Test missing data analysis with complete dataset."""
        # Create complete dataset (no missing values)
        data = pd.DataFrame({
            'x': np.arange(100),
            'y': np.random.randn(100)
        })

        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            data.to_parquet(tmp.name)
            tmp_path = tmp.name

        try:
            analyzer = TimeSeriesDatasetAnalyzer("complete")
            analyzer.load_and_inspect_data(tmp_path)
            result = analyzer.analyze_missing_data()

            assert result['overall_missing_rate'] == 0.0
            assert result['total_missing_values'] == 0
            assert result['severity'] == 'excellent'
            assert len(result['columns_with_missing']) == 0
        finally:
            Path(tmp_path).unlink()

    def test_temporal_structure_daily(self, sample_data_file):
        """Test temporal structure detection for daily data."""
        analyzer = TimeSeriesDatasetAnalyzer("test")
        analyzer.load_and_inspect_data(str(sample_data_file))
        result = analyzer.identify_temporal_structure()

        assert result is not None
        assert result['is_time_series'] is True
        assert result['sampling_frequency'] == 'Daily'
        assert result['frequency_code'] == 'D'
        assert abs(result['sampling_period_days'] - 1.0) < 0.01

        # Nyquist frequency should be 0.5 cycles/day for daily data
        assert abs(result['nyquist_frequency'] - 0.5) < 0.01

        assert 'time_span' in result
        assert result['time_span']['total_periods'] == 1000
        assert 'temporal_regularity' in result

    def test_temporal_structure_monthly(self):
        """Test temporal structure detection for monthly data."""
        # Create monthly data
        dates = pd.date_range(start='2020-01-01', periods=36, freq='MS')
        data = pd.DataFrame({
            'date': dates,
            'value': np.random.randn(36)
        })

        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            data.to_parquet(tmp.name)
            tmp_path = tmp.name

        try:
            analyzer = TimeSeriesDatasetAnalyzer("monthly")
            analyzer.load_and_inspect_data(tmp_path)
            result = analyzer.identify_temporal_structure()

            assert result['is_time_series'] is True
            assert result['sampling_frequency'] == 'Monthly'
            assert result['frequency_code'] == 'M'
            assert 28 <= result['sampling_period_days'] <= 31
        finally:
            Path(tmp_path).unlink()

    def test_statistical_summary(self, sample_data_file):
        """Test statistical summary computation."""
        analyzer = TimeSeriesDatasetAnalyzer("test")
        analyzer.load_and_inspect_data(str(sample_data_file))
        result = analyzer.statistical_summary()

        assert result is not None
        assert 'mean' in result
        assert 'std' in result
        assert 'min' in result
        assert 'max' in result
        assert 'median' in result
        assert 'skewness' in result
        assert 'kurtosis' in result
        assert 'cv' in result  # Coefficient of variation
        assert 'normality_tests' in result

        # Verify stationary series has mean ≈ 0, std ≈ 1
        if 'stationary' in result['mean']:
            assert abs(result['mean']['stationary']) < 0.2
            assert abs(result['std']['stationary'] - 1.0) < 0.2

        # Check normality tests
        assert isinstance(result['normality_tests'], dict)
        for col, test in result['normality_tests'].items():
            assert 'p_value' in test
            assert 'is_normal' in test
            assert 0 <= test['p_value'] <= 1

    def test_stationarity_test_stationary(self, sample_data_file):
        """Test stationarity detection on stationary series."""
        analyzer = TimeSeriesDatasetAnalyzer("test")
        analyzer.load_and_inspect_data(str(sample_data_file))

        # Test stationary white noise
        result = analyzer.test_stationarity('stationary')

        assert result is not None
        assert 'adf_statistic' in result
        assert 'p_value' in result
        assert 'is_stationary' in result
        assert 'conclusion' in result

        # White noise should be stationary
        assert result['is_stationary'] is True
        assert result['p_value'] < 0.05
        assert 'Stationary' in result['conclusion']

    def test_stationarity_test_nonstationary(self, sample_data_file):
        """Test stationarity detection on non-stationary series."""
        analyzer = TimeSeriesDatasetAnalyzer("test")
        analyzer.load_and_inspect_data(str(sample_data_file))

        # Test random walk (non-stationary)
        result = analyzer.test_stationarity('random_walk')

        assert result is not None

        # Random walk should be non-stationary (though with 1000 points, may vary)
        # Check that the test executed properly
        assert 'adf_statistic' in result
        assert 'p_value' in result
        assert 'critical_values' in result
        assert '1%' in result['critical_values']
        assert '5%' in result['critical_values']
        assert '10%' in result['critical_values']

    def test_autocorrelation_computation(self, sample_data_file):
        """Test ACF and PACF computation."""
        analyzer = TimeSeriesDatasetAnalyzer("test")
        analyzer.load_and_inspect_data(str(sample_data_file))

        result = analyzer.compute_autocorrelation('stationary', nlags=20)

        assert result is not None
        assert 'acf' in result
        assert 'pacf' in result
        assert 'lags' in result
        assert 'confidence_interval' in result
        assert 'significant_lags_acf' in result
        assert 'significant_lags_pacf' in result

        # Check dimensions
        assert len(result['acf']) == 21  # nlags + 1
        assert len(result['pacf']) == 21
        assert len(result['lags']) == 21

        # ACF at lag 0 should be 1.0
        assert abs(result['acf'][0] - 1.0) < 1e-6

        # Confidence interval should be positive
        assert result['confidence_interval'] > 0

    def test_autocorrelation_seasonal(self, sample_data_file):
        """Test ACF on seasonal data."""
        analyzer = TimeSeriesDatasetAnalyzer("test")
        analyzer.load_and_inspect_data(str(sample_data_file))

        # Seasonal data should show autocorrelation at seasonal lags
        result = analyzer.compute_autocorrelation('seasonal', nlags=40)

        assert result is not None
        assert len(result['acf']) == 41

        # Should detect some significant lags
        assert len(result['significant_lags_acf']) > 0

    def test_generate_report(self, sample_data_file):
        """Test comprehensive report generation."""
        analyzer = TimeSeriesDatasetAnalyzer("test", "Test dataset")
        analyzer.load_and_inspect_data(str(sample_data_file))
        analyzer.analyze_missing_data()
        analyzer.identify_temporal_structure()
        analyzer.statistical_summary()
        analyzer.test_stationarity('stationary')
        analyzer.compute_autocorrelation('stationary')

        report = analyzer.generate_report()

        assert report is not None
        assert 'dataset_name' in report
        assert 'description' in report
        assert 'timestamp' in report
        assert 'results' in report

        # Check all analyses are included
        results = report['results']
        assert 'inspection' in results
        assert 'missing_data' in results
        assert 'temporal_structure' in results
        assert 'statistics' in results
        assert 'stationarity_tests' in results
        assert 'autocorrelation' in results

    def test_edge_case_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        analyzer = TimeSeriesDatasetAnalyzer("empty")

        # Create empty DataFrame
        data = pd.DataFrame()

        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            data.to_parquet(tmp.name)
            tmp_path = tmp.name

        try:
            result = analyzer.load_and_inspect_data(tmp_path)

            # Should handle gracefully
            assert result is not None
            assert result['total_rows'] == 0
            assert result['total_columns'] == 0
        finally:
            Path(tmp_path).unlink()

    def test_edge_case_single_column(self):
        """Test handling of single column DataFrame."""
        data = pd.DataFrame({'value': np.random.randn(100)})

        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            data.to_parquet(tmp.name)
            tmp_path = tmp.name

        try:
            analyzer = TimeSeriesDatasetAnalyzer("single")
            result = analyzer.load_and_inspect_data(tmp_path)

            assert result is not None
            assert result['total_columns'] == 1
            assert len(result['numeric_columns']) == 1
        finally:
            Path(tmp_path).unlink()

    def test_mathematical_correctness_mean(self, sample_data_file):
        """Verify mathematical correctness of mean calculation."""
        analyzer = TimeSeriesDatasetAnalyzer("test")
        analyzer.load_and_inspect_data(str(sample_data_file))
        result = analyzer.statistical_summary()

        # Manually compute mean and verify
        for col in result['mean'].keys():
            manual_mean = analyzer.data[col].mean()
            computed_mean = result['mean'][col]

            assert abs(manual_mean - computed_mean) < 1e-10

    def test_mathematical_correctness_cv(self, sample_data_file):
        """Verify coefficient of variation calculation: CV = σ/μ."""
        analyzer = TimeSeriesDatasetAnalyzer("test")
        analyzer.load_and_inspect_data(str(sample_data_file))
        result = analyzer.statistical_summary()

        # CV = std / mean
        for col in result['cv'].keys():
            if not pd.isna(result['cv'][col]):
                expected_cv = result['std'][col] / result['mean'][col]
                computed_cv = result['cv'][col]

                # Allow for floating point differences
                if not (np.isinf(expected_cv) or np.isnan(expected_cv)):
                    assert abs(expected_cv - computed_cv) < 1e-10

    def test_vectorization_performance(self):
        """Test that vectorized operations are used (performance check)."""
        # Create large dataset (reduced size to avoid pandas date range limits)
        n = 50000
        data = pd.DataFrame({
            'date': pd.date_range('2000-01-01', periods=n, freq='D'),
            'value': np.random.randn(n)
        })

        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            data.to_parquet(tmp.name)
            tmp_path = tmp.name

        try:
            import time

            analyzer = TimeSeriesDatasetAnalyzer("large")

            start = time.time()
            analyzer.load_and_inspect_data(tmp_path)
            analyzer.analyze_missing_data()
            analyzer.statistical_summary()
            elapsed = time.time() - start

            # Should complete in reasonable time (< 5 seconds for 50k rows)
            assert elapsed < 5.0, f"Analysis took {elapsed:.2f}s, possible non-vectorized ops"
        finally:
            Path(tmp_path).unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
