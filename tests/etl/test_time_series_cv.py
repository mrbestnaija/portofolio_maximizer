"""Comprehensive tests for Time Series Cross-Validation.

Tests verify:
1. Correctness of CV splits (temporal ordering, no leakage)
2. Backward compatibility with simple splits
3. Quantifiable improvements (coverage, representation)
4. Edge cases and error handling
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from etl.time_series_cv import TimeSeriesCrossValidator, CVFold
from etl.data_storage import DataStorage


class TestTimeSeriesCrossValidator:
    """Test suite for TimeSeriesCrossValidator."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data (1000 points)."""
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        data = pd.DataFrame({
            'Close': np.random.randn(1000).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)
        return data

    @pytest.fixture
    def cv_splitter(self):
        """Default CV splitter (k=5)."""
        return TimeSeriesCrossValidator(n_splits=5, test_size=0.15)

    # ===== Correctness Tests =====

    def test_cv_split_temporal_ordering(self, sample_data, cv_splitter):
        """Test that train data comes before validation in each fold."""
        cv_folds, test_indices = cv_splitter.split(sample_data)

        for fold in cv_folds:
            # Train indices must be < validation indices
            assert fold.train_end <= fold.val_start, \
                f"Fold {fold.fold_id}: train_end ({fold.train_end}) > val_start ({fold.val_start})"

            # Check actual data timestamps
            train_data = sample_data.iloc[fold.train_indices]
            val_data = sample_data.iloc[fold.val_indices]

            if len(train_data) > 0 and len(val_data) > 0:
                assert train_data.index.max() <= val_data.index.min(), \
                    f"Fold {fold.fold_id}: Training data leaks into validation period"

    def test_cv_test_set_isolation(self, sample_data, cv_splitter):
        """Test that test set is completely isolated from CV folds."""
        cv_folds, test_indices = cv_splitter.split(sample_data)

        test_set = set(test_indices)

        for fold in cv_folds:
            train_set = set(fold.train_indices)
            val_set = set(fold.val_indices)

            # No overlap with test set
            assert len(train_set & test_set) == 0, \
                f"Fold {fold.fold_id}: Training set overlaps with test set"
            assert len(val_set & test_set) == 0, \
                f"Fold {fold.fold_id}: Validation set overlaps with test set"

    def test_cv_no_fold_overlap(self, sample_data, cv_splitter):
        """Test that validation sets don't overlap across folds."""
        cv_folds, test_indices = cv_splitter.split(sample_data)

        val_sets = [set(fold.val_indices) for fold in cv_folds]

        for i in range(len(val_sets)):
            for j in range(i + 1, len(val_sets)):
                overlap = val_sets[i] & val_sets[j]
                assert len(overlap) == 0, \
                    f"Validation sets overlap between folds {i} and {j}"

    def test_cv_complete_coverage(self, sample_data, cv_splitter):
        """Test that CV validation sets cover substantial portion of CV data."""
        cv_folds, test_indices = cv_splitter.split(sample_data)

        n_samples = len(sample_data)
        expected_cv_size = int(n_samples * (1 - cv_splitter.test_size))

        # Union of all validation indices
        all_val_indices = set()
        for fold in cv_folds:
            all_val_indices.update(fold.val_indices)

        coverage = len(all_val_indices) / expected_cv_size

        # With k=5, expect ~83% coverage (5/(5+1) = 0.833)
        # This is acceptable as it ensures all folds have training data
        assert coverage >= 0.80, \
            f"CV coverage only {coverage*100:.1f}%, expected ≥80%"

    def test_cv_test_size_correct(self, sample_data, cv_splitter):
        """Test that test set is exactly 15% of data."""
        cv_folds, test_indices = cv_splitter.split(sample_data)

        n_samples = len(sample_data)
        expected_test_size = int(n_samples * cv_splitter.test_size)

        assert len(test_indices) == n_samples - int(n_samples * (1 - cv_splitter.test_size)), \
            f"Test set size incorrect: {len(test_indices)}, expected ~{expected_test_size}"

    # ===== Backward Compatibility Tests =====

    def test_backward_compatible_simple_split(self, sample_data):
        """Test that use_cv=False returns backward-compatible format."""
        storage = DataStorage(base_path="data")

        # Old API call (default use_cv=False)
        splits = storage.train_validation_test_split(sample_data)

        # Check old format
        assert 'training' in splits, "Missing 'training' key"
        assert 'validation' in splits, "Missing 'validation' key"
        assert 'testing' in splits, "Missing 'testing' key"

        # Should NOT have CV keys
        assert 'cv_folds' not in splits, "CV keys present in simple split"
        assert 'split_type' not in splits, "split_type present in simple split"

        # Check data types
        assert isinstance(splits['training'], pd.DataFrame)
        assert isinstance(splits['validation'], pd.DataFrame)
        assert isinstance(splits['testing'], pd.DataFrame)

    def test_backward_compatible_split_ratios(self, sample_data):
        """Test that simple split uses correct ratios (70/15/15)."""
        storage = DataStorage(base_path="data")

        splits = storage.train_validation_test_split(sample_data, train_ratio=0.7, val_ratio=0.15)

        n = len(sample_data)
        train_size = len(splits['training'])
        val_size = len(splits['validation'])
        test_size = len(splits['testing'])

        # Check ratios (allow 1% tolerance)
        assert abs(train_size / n - 0.7) < 0.01, "Training ratio incorrect"
        assert abs(val_size / n - 0.15) < 0.01, "Validation ratio incorrect"
        assert abs(test_size / n - 0.15) < 0.01, "Test ratio incorrect"

    def test_backward_compatible_temporal_order(self, sample_data):
        """Test that simple split maintains temporal ordering."""
        storage = DataStorage(base_path="data")

        splits = storage.train_validation_test_split(sample_data)

        # Train < Val < Test (temporal ordering)
        assert splits['training'].index.max() <= splits['validation'].index.min(), \
            "Training data leaks into validation"
        assert splits['validation'].index.max() <= splits['testing'].index.min(), \
            "Validation data leaks into testing"

    # ===== CV Format Tests =====

    def test_cv_split_format(self, sample_data):
        """Test that use_cv=True returns correct format."""
        storage = DataStorage(base_path="data")

        splits = storage.train_validation_test_split(sample_data, use_cv=True, n_splits=5)

        # Check CV format
        assert 'cv_folds' in splits, "Missing 'cv_folds' key"
        assert 'testing' in splits, "Missing 'testing' key"
        assert 'n_splits' in splits, "Missing 'n_splits' key"
        assert 'split_type' in splits, "Missing 'split_type' key"

        # Check data types
        assert isinstance(splits['cv_folds'], list)
        assert isinstance(splits['testing'], pd.DataFrame)
        assert splits['n_splits'] == 5
        assert splits['split_type'] == 'cross_validation'

        # Check fold structure
        assert len(splits['cv_folds']) == 5, "Should have 5 folds"

        for fold in splits['cv_folds']:
            assert 'fold_id' in fold
            assert 'train' in fold
            assert 'validation' in fold
            assert isinstance(fold['train'], pd.DataFrame)
            assert isinstance(fold['validation'], pd.DataFrame)

    # ===== Quantifiable Improvement Tests =====

    def test_cv_improves_temporal_coverage(self, sample_data):
        """QUANTIFY: CV provides better temporal coverage than simple split."""
        storage = DataStorage(base_path="data")

        # Simple split
        simple_splits = storage.train_validation_test_split(sample_data, use_cv=False)

        # CV split
        cv_splits = storage.train_validation_test_split(sample_data, use_cv=True, n_splits=5)

        # Calculate temporal coverage for validation sets
        # Simple split: validation covers only 15% in middle
        simple_val_coverage = len(simple_splits['validation']) / len(sample_data)

        # CV split: validation covers 85% across all folds
        total_cv_val_samples = sum(len(fold['validation']) for fold in cv_splits['cv_folds'])
        cv_val_coverage = total_cv_val_samples / len(sample_data)

        # CV should cover ~5x more temporal range
        improvement_factor = cv_val_coverage / simple_val_coverage

        assert improvement_factor >= 4.5, \
            f"CV coverage improvement only {improvement_factor:.1f}x, expected ≥4.5x"

    def test_cv_reduces_disparity(self, sample_data):
        """QUANTIFY: CV reduces training/validation disparity."""
        storage = DataStorage(base_path="data")

        # Simple split
        simple_splits = storage.train_validation_test_split(sample_data, use_cv=False)

        # Calculate temporal gap in simple split
        train_end = simple_splits['training'].index.max()
        val_start = simple_splits['validation'].index.min()
        simple_gap = (val_start - train_end).days

        # CV split
        cv_splits = storage.train_validation_test_split(sample_data, use_cv=True, n_splits=5)

        # Calculate average temporal gap in CV folds
        cv_gaps = []
        for fold in cv_splits['cv_folds']:
            if len(fold['train']) > 0 and len(fold['validation']) > 0:
                fold_train_end = fold['train'].index.max()
                fold_val_start = fold['validation'].index.min()
                gap = (fold_val_start - fold_train_end).days
                cv_gaps.append(gap)

        avg_cv_gap = np.mean(cv_gaps)

        # CV gap should be minimal (expanding window)
        assert avg_cv_gap <= 1, \
            f"CV avg gap {avg_cv_gap:.1f} days, expected ≤1 day"

    def test_cv_validates_improvements(self, sample_data, cv_splitter):
        """Run validation checks and ensure quality criteria met."""
        validation_results = cv_splitter.validate_splits(sample_data)

        # Must pass validation
        assert validation_results['valid'], \
            f"CV validation failed: {validation_results['errors']}"

        # Check coverage metric (accept 80%+ for k=5)
        coverage = validation_results['statistics']['cv_coverage']
        assert coverage >= 0.80, \
            f"CV coverage {coverage*100:.1f}% < 80%"

        # Check fold balance
        fold_cv = validation_results['statistics']['fold_size_cv']
        assert fold_cv <= 0.2, \
            f"Fold size variation too high: CV={fold_cv:.2f}"

    # ===== Edge Cases =====

    def test_cv_small_dataset(self):
        """Test CV behavior with small dataset."""
        # 100 samples, k=5
        small_data = pd.DataFrame({
            'Close': np.random.randn(100)
        }, index=pd.date_range('2020-01-01', periods=100))

        cv_splitter = TimeSeriesCrossValidator(n_splits=5, test_size=0.15)
        cv_folds, test_indices = cv_splitter.split(small_data)

        # Should still create all folds
        assert len(cv_folds) == 5
        assert len(test_indices) == 15  # 15% of 100

    def test_cv_invalid_n_splits(self):
        """Test error handling for invalid n_splits."""
        with pytest.raises(ValueError, match="n_splits must be at least 2"):
            TimeSeriesCrossValidator(n_splits=1)

    def test_cv_invalid_test_size(self):
        """Test error handling for invalid test_size."""
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            TimeSeriesCrossValidator(n_splits=5, test_size=1.5)

    def test_cv_non_datetime_index(self, sample_data):
        """Test error handling for non-DatetimeIndex."""
        # Remove datetime index
        bad_data = sample_data.reset_index(drop=True)

        cv_splitter = TimeSeriesCrossValidator(n_splits=5)

        with pytest.raises(ValueError, match="Data must have DatetimeIndex"):
            cv_splitter.split(bad_data)

    # ===== Integration Tests =====

    def test_cv_save_load_cycle(self, sample_data, tmp_path):
        """Test saving and loading CV splits."""
        storage = DataStorage(base_path=str(tmp_path))
        cv_splitter = TimeSeriesCrossValidator(n_splits=3, test_size=0.15)

        # Save CV splits
        saved_paths = cv_splitter.save_cv_splits(
            sample_data,
            storage,
            symbol='TEST'
        )

        # Verify structure
        assert 'folds' in saved_paths
        assert 'test' in saved_paths
        assert len(saved_paths['folds']) == 3

        # Verify files exist
        for fold_paths in saved_paths['folds']:
            assert Path(fold_paths['train']).exists()
            assert Path(fold_paths['validation']).exists()

        assert Path(saved_paths['test']).exists()

    def test_cv_deterministic_splits(self, sample_data):
        """Test that splits are deterministic (reproducible)."""
        cv_splitter = TimeSeriesCrossValidator(n_splits=5, test_size=0.15)

        # Run twice
        cv_folds1, test_indices1 = cv_splitter.split(sample_data)
        cv_folds2, test_indices2 = cv_splitter.split(sample_data)

        # Test indices should be identical
        np.testing.assert_array_equal(test_indices1, test_indices2)

        # Each fold should be identical
        assert len(cv_folds1) == len(cv_folds2)

        for fold1, fold2 in zip(cv_folds1, cv_folds2):
            np.testing.assert_array_equal(fold1.train_indices, fold2.train_indices)
            np.testing.assert_array_equal(fold1.val_indices, fold2.val_indices)

    # ===== Performance Tests =====

    def test_cv_performance_benchmarks(self, sample_data):
        """Test that CV splitting is fast (<100ms for 1000 samples)."""
        import time

        cv_splitter = TimeSeriesCrossValidator(n_splits=5, test_size=0.15)

        start = time.time()
        cv_folds, test_indices = cv_splitter.split(sample_data)
        elapsed = time.time() - start

        # Should be very fast (<100ms)
        assert elapsed < 0.1, f"CV splitting too slow: {elapsed*1000:.1f}ms"

    def test_cv_memory_efficient(self, sample_data):
        """Test that CV doesn't create unnecessary data copies."""
        cv_splitter = TimeSeriesCrossValidator(n_splits=5, test_size=0.15)

        cv_folds, test_indices = cv_splitter.split(sample_data)

        # Folds should contain indices, not data copies
        for fold in cv_folds:
            assert isinstance(fold.train_indices, np.ndarray)
            assert isinstance(fold.val_indices, np.ndarray)

            # Indices should be small
            assert fold.train_indices.nbytes < 10000  # <10KB
            assert fold.val_indices.nbytes < 10000  # <10KB


class TestBackwardCompatibility:
    """Dedicated tests for backward compatibility."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        return pd.DataFrame({
            'Close': np.random.randn(1000).cumsum() + 100
        }, index=dates)

    def test_existing_code_unaffected(self, sample_data):
        """Test that existing code using train_validation_test_split still works."""
        storage = DataStorage(base_path="data")

        # Old code (no use_cv parameter)
        splits = storage.train_validation_test_split(sample_data)

        # Should work exactly as before
        assert len(splits) == 3
        assert 'training' in splits
        assert 'validation' in splits
        assert 'testing' in splits

        # Total samples should equal input
        total = len(splits['training']) + len(splits['validation']) + len(splits['testing'])
        assert total == len(sample_data)

    def test_custom_ratios_still_work(self, sample_data):
        """Test that custom train/val ratios still work."""
        storage = DataStorage(base_path="data")

        # Custom ratios
        splits = storage.train_validation_test_split(
            sample_data,
            train_ratio=0.6,
            val_ratio=0.2
        )

        n = len(sample_data)
        assert abs(len(splits['training']) / n - 0.6) < 0.01
        assert abs(len(splits['validation']) / n - 0.2) < 0.01
        assert abs(len(splits['testing']) / n - 0.2) < 0.01
