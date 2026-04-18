"""Tests for data preprocessing with vectorized transformations."""
import pytest
import numpy as np
import pandas as pd
from etl.preprocessor import Preprocessor

@pytest.fixture
def preprocessor():
    return Preprocessor()

@pytest.fixture
def sample_data():
    """Sample OHLCV data for testing."""
    dates = pd.date_range('2014-01-01', periods=100)
    return pd.DataFrame({
        'Close': np.random.lognormal(0, 0.02, 100).cumprod() * 100,
        'Volume': np.random.randint(1e6, 1e7, 100),
        'ticker': ['AAPL'] * 100  # Non-numeric column
    }, index=dates)

@pytest.fixture
def sample_data_with_missing():
    """Sample data with missing values."""
    dates = pd.date_range('2014-01-01', periods=50)
    return pd.DataFrame({
        'Close': [100, np.nan, 102, 103, np.nan, 105, 106, np.nan, 108, 109] + list(range(110, 150)),
        'Volume': [1e6, 2e6, np.nan, 4e6, 5e6, np.nan, 7e6, 8e6, 9e6, 1e7] + list(range(int(1.1e7), int(5.1e7), int(1e6)))
    }, index=dates)


def test_handle_missing_forward_fill(preprocessor, sample_data_with_missing):
    """Test forward-fill missing value handling."""
    filled = preprocessor.handle_missing(sample_data_with_missing, method='forward')
    assert not filled.isna().any().any(), "Missing values should be filled"
    assert len(filled) == len(sample_data_with_missing)


def test_handle_missing_basic(preprocessor):
    """Test basic missing value handling."""
    data = pd.DataFrame({
        'A': [1.0, np.nan, 3.0, np.nan, 5.0],
        'B': [10.0, 20.0, np.nan, 40.0, 50.0]
    })
    filled = preprocessor.handle_missing(data)
    numeric_cols = filled.select_dtypes(include=[np.number]).columns
    assert not filled[numeric_cols].isna().any().any()


def test_normalize_zscore(preprocessor, sample_data):
    """Test z-score normalization (μ=0, σ²=1)."""
    normalized, stats = preprocessor.normalize(sample_data)
    numeric_cols = sample_data.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        assert np.abs(normalized[col].mean()) < 1e-10, f"{col} should have mean ≈ 0"
        assert np.abs(normalized[col].std() - 1.0) < 1e-10, f"{col} should have std ≈ 1"

    assert isinstance(stats, dict)
    for col in numeric_cols:
        assert 'mean' in stats[col]
        assert 'std' in stats[col]


def test_normalize_preserves_non_numeric(preprocessor, sample_data):
    """Test that normalization preserves non-numeric columns."""
    normalized, stats = preprocessor.normalize(sample_data)
    assert 'ticker' in normalized.columns
    assert (normalized['ticker'] == sample_data['ticker']).all()


def test_handle_missing_empty_dataframe(preprocessor):
    """Test handling of empty DataFrame."""
    empty_df = pd.DataFrame()
    filled = preprocessor.handle_missing(empty_df)
    assert len(filled) == 0
    assert filled.empty


def test_normalize_single_column(preprocessor):
    """Test normalization with single numeric column."""
    data = pd.DataFrame({'price': [100, 110, 120, 130, 140]})
    normalized, stats = preprocessor.normalize(data)
    assert np.abs(normalized['price'].mean()) < 1e-10
    assert np.abs(normalized['price'].std() - 1.0) < 1e-10
    assert 'price' in stats


def test_preprocessing_pipeline_integration(preprocessor, sample_data_with_missing):
    """Test full preprocessing pipeline: handle_missing -> normalize."""
    filled = preprocessor.handle_missing(sample_data_with_missing, method='forward')
    normalized, stats = preprocessor.normalize(filled)

    numeric_cols = normalized.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        assert not normalized[col].isna().any(), f"{col} should have no NaN after pipeline"
        assert np.abs(normalized[col].mean()) < 1e-10, f"{col} should be normalized"
        assert np.abs(normalized[col].std() - 1.0) < 1e-10, f"{col} should have unit variance"


def test_validate_post_preprocess_marks_clean_frame_production_ready(preprocessor, sample_data):
    health = preprocessor.validate_post_preprocess(
        sample_data,
        sample_data.copy(),
        min_usable_bars=50,
        max_imputed_fraction=0.30,
        max_padding_fraction=0.20,
    )

    assert health["status"] == "PASS"
    assert health["production_ok"] is True
    assert health["research_ok"] is True
    assert health["quality_tag"] == "CLEAN"
    assert health["imputed_fraction"] == pytest.approx(0.0)
    assert health["padding_fraction"] == pytest.approx(0.0)


def test_validate_post_preprocess_marks_sparse_frames_research_only(preprocessor, sample_data_with_missing):
    raw = sample_data_with_missing.iloc[:10].copy()
    filled = preprocessor.handle_missing(raw, method="forward")

    health = preprocessor.validate_post_preprocess(
        raw,
        filled,
        min_usable_bars=5,
        max_imputed_fraction=0.10,
        max_padding_fraction=0.0,
    )

    assert health["status"] == "WARN"
    assert health["production_ok"] is False
    assert health["research_ok"] is True
    assert health["quality_tag"] in {"SPARSE_DATA", "HIGH_IMPUTE"}
    assert health["imputed_fraction"] > 0.10


def test_validate_post_preprocess_fails_on_structural_defects(preprocessor):
    raw = pd.DataFrame(
        {
            "Close": [100.0, 101.0, 102.0],
            "Volume": [1_000_000, 1_100_000, 1_200_000],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    processed = raw.copy()
    processed.index = pd.to_datetime([
        "2024-01-01",
        "2024-01-01",
        "2024-01-03",
    ])
    processed.loc[processed.index[1], "Close"] = np.inf

    health = preprocessor.validate_post_preprocess(
        raw,
        processed,
        min_usable_bars=3,
    )

    assert health["status"] == "FAIL"
    assert health["production_ok"] is False
    assert health["research_ok"] is False
    assert "DUPLICATE_ROWS" in health["reason"] or "INFINITE_VALUES" in health["reason"]
