"""Tests for data storage with time series persistence."""
import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from etl.data_storage import DataStorage

@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    temp_dir = tempfile.mkdtemp()
    storage = DataStorage(base_path=temp_dir)
    yield storage
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_data():
    """Create sample time series data."""
    dates = pd.date_range('2024-01-01', periods=100)
    return pd.DataFrame({
        'Close': np.random.lognormal(0, 0.02, 100).cumprod() * 100,
        'Volume': np.random.randint(1e6, 1e7, 100)
    }, index=dates)

def test_save_and_load(temp_storage, sample_data):
    """Test basic save and load operations."""
    # Save data
    filepath = temp_storage.save(sample_data, stage='raw', symbol='TEST')

    assert filepath.exists()
    assert filepath.suffix == '.parquet'

    # Load data
    loaded = temp_storage.load(stage='raw', symbol='TEST')

    assert len(loaded) == len(sample_data)
    pd.testing.assert_frame_equal(loaded, sample_data)

def test_timestamp_index_validation(temp_storage):
    """Test that non-timestamp index raises error."""
    data = pd.DataFrame({'A': [1, 2, 3]})

    with pytest.raises(ValueError, match="DatetimeIndex"):
        temp_storage.save(data, stage='raw', symbol='TEST')

def test_date_filtering(temp_storage, sample_data):
    """Test date range filtering on load."""
    temp_storage.save(sample_data, stage='raw', symbol='TEST')

    # Filter by date range
    loaded = temp_storage.load(
        stage='raw',
        symbol='TEST',
        start_date='2024-01-15',
        end_date='2024-01-25'
    )

    assert len(loaded) <= len(sample_data)
    assert loaded.index.min() >= pd.Timestamp('2024-01-15')
    assert loaded.index.max() <= pd.Timestamp('2024-01-25')

def test_directory_structure(temp_storage):
    """Test storage directory structure creation."""
    stages = ['raw', 'processed', 'training', 'validation', 'testing']

    for stage in stages:
        stage_dir = temp_storage.base_path / stage
        assert stage_dir.exists()
        assert stage_dir.is_dir()

def test_missing_data_error(temp_storage):
    """Test error when loading non-existent data."""
    with pytest.raises(FileNotFoundError):
        temp_storage.load(stage='raw', symbol='NONEXISTENT')

def test_sorted_index(temp_storage):
    """Test that saved data maintains sorted timestamp index."""
    # Create unsorted data
    dates = pd.date_range('2024-01-01', periods=10)
    shuffled_dates = np.random.permutation(dates)
    data = pd.DataFrame({'Close': range(10)}, index=shuffled_dates)

    temp_storage.save(data, stage='raw', symbol='TEST')
    loaded = temp_storage.load(stage='raw', symbol='TEST')

    # Verify sorted
    assert loaded.index.is_monotonic_increasing

def test_cleanup_old_files(temp_storage, sample_data):
    """Test auto-cleanup of old files (vectorized)."""
    # Save file with current timestamp
    temp_storage.save(sample_data, stage='raw', symbol='RECENT')

    # Create old file by saving and then modifying its mtime
    old_path = temp_storage.save(sample_data, stage='raw', symbol='OLD')
    old_time = (datetime.now() - timedelta(days=10)).timestamp()
    import os
    os.utime(old_path, (old_time, old_time))

    # Cleanup files older than 7 days
    temp_storage.cleanup_old_files('raw', retention_days=7)

    # Verify: OLD file deleted, RECENT file remains
    assert not old_path.exists()
    recent_files = list((temp_storage.base_path / 'raw').glob('RECENT_*.parquet'))
    assert len(recent_files) == 1