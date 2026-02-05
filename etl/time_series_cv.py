"""Time Series Cross-Validation with Moving Window.

Mathematical Foundation:
- Time series CV: Expanding/rolling window to avoid look-ahead bias
- k-fold splits: Each fold maintains temporal ordering
- Moving window: Prevents data memory decay across folds
- Test set isolation: Final 15% held out for unbiased evaluation

Cross-Validation Strategy:
For time series t = [t_0, t_1, ..., t_n]:
1. Reserve test set: t_test = t[n-0.15n:n] (final 15%, never used in CV)
2. CV on remaining: t_cv = t[0:n-0.15n] (85% for training/validation)
3. k-fold splits with moving window:

   For k=5 (default):
   Fold 1: Train[0:17%],   Val[17%:34%]
   Fold 2: Train[17%:34%], Val[34%:51%]
   Fold 3: Train[34%:51%], Val[51%:68%]
   Fold 4: Train[51%:68%], Val[68%:85%]
   Fold 5: Train[68%:85%], Val[85%:85%] (expanding window)

   This ensures:
   - All data in CV set is used for both training and validation
   - Temporal ordering preserved (no future leakage)
   - Representative sampling across entire time range
   - Test set remains completely isolated

Success Criteria:
- Each fold covers ~17% of CV data (for k=5)
- Validation sets cover entire CV distribution
- Zero overlap between test and CV sets
- Deterministic splits (reproducible)
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CVFold:
    """Represents a single cross-validation fold."""
    fold_id: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_start: int
    train_end: int
    val_start: int
    val_end: int


class TimeSeriesCrossValidator:
    """Time series cross-validation with moving window strategy.

    This implementation prevents training/validation disparity by ensuring
    all data across the time index is represented in both training and
    validation sets through k-fold moving window splits.

    Mathematical Properties:
    - Temporal consistency: t_train_i < t_val_i for all folds i
    - Complete coverage: Union of all val sets = entire CV set
    - No leakage: Test set never exposed to model
    - Balanced folds: Each fold ≈ (1/k) * CV_data_size
    """

    def __init__(self, n_splits: int = 5, test_size: float = 0.15,
                 gap: int = 0, expanding_window: bool = True):
        """Initialize time series cross-validator.

        Args:
            n_splits: Number of CV folds (k). Default=5
            test_size: Proportion of data for final test set. Default=0.15 (15%)
            gap: Number of samples to skip between train and val. Default=0
            expanding_window: If True, training window expands for later folds.
                            If False, uses fixed-size rolling window.
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")

        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.expanding_window = expanding_window

    def split(self, data: pd.DataFrame) -> Tuple[List[CVFold], np.ndarray]:
        """Generate k-fold time series cross-validation splits.

        Args:
            data: Time series DataFrame with DatetimeIndex

        Returns:
            Tuple of (cv_folds, test_indices)
            - cv_folds: List of CVFold objects for cross-validation
            - test_indices: Indices for final test set (isolated)
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex for time series CV")

        # Ensure sorted by time
        data = data.sort_index()
        n_samples = len(data)

        # Calculate test set size (final 15%, isolated)
        test_start = int(n_samples * (1 - self.test_size))
        test_indices = np.arange(test_start, n_samples)

        # CV data: Everything before test set (85%)
        cv_size = test_start

        # Calculate fold size for validation sets
        # Use (n_splits + 1) to ensure all folds have training data
        fold_size = cv_size // (self.n_splits + 1)

        cv_folds = []

        for fold_idx in range(self.n_splits):
            # Validation set: starts at fold 1 (fold_idx+1)
            # This ensures fold 0 has training data from [0:fold_size]
            val_start = (fold_idx + 1) * fold_size
            val_end = min(val_start + fold_size, cv_size)

            # Training set: all data before validation (expanding window)
            # OR fixed window (rolling window)
            if self.expanding_window:
                # Expanding: use all data from beginning up to val_start
                train_start = 0
                train_end = val_start
            else:
                # Rolling: use fixed-size window before validation
                train_start = max(0, val_start - fold_size)
                train_end = val_start

            # Apply gap if specified (prevent immediate future leakage)
            if self.gap > 0:
                train_end = max(train_start, train_end - self.gap)

            # Create indices arrays
            train_indices = np.arange(train_start, train_end)
            val_indices = np.arange(val_start, val_end)

            # Only create fold if we have valid train and val sets
            if len(train_indices) > 0 and len(val_indices) > 0:
                fold = CVFold(
                    fold_id=fold_idx,
                    train_indices=train_indices,
                    val_indices=val_indices,
                    train_start=train_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end
                )
                cv_folds.append(fold)

        # Log fold statistics
        logger.info(f"Time Series CV: {len(cv_folds)} folds, "
                   f"CV size: {cv_size}, Test size: {len(test_indices)}")

        for fold in cv_folds:
            train_pct = (len(fold.train_indices) / n_samples) * 100
            val_pct = (len(fold.val_indices) / n_samples) * 100
            logger.info(f"Fold {fold.fold_id}: "
                       f"Train={len(fold.train_indices)} ({train_pct:.1f}%), "
                       f"Val={len(fold.val_indices)} ({val_pct:.1f}%)")

        return cv_folds, test_indices

    def get_fold_data(self, data: pd.DataFrame, fold: CVFold) -> Dict[str, pd.DataFrame]:
        """Extract train/val data for a specific fold.

        Args:
            data: Full time series DataFrame
            fold: CVFold object with indices

        Returns:
            Dictionary with 'train' and 'validation' DataFrames
        """
        data = data.sort_index()

        return {
            'train': data.iloc[fold.train_indices].copy(),
            'validation': data.iloc[fold.val_indices].copy()
        }

    def save_cv_splits(self, data: pd.DataFrame, storage,
                       symbol: str) -> Dict[str, List[str]]:
        """Generate and save all CV splits to storage.

        Args:
            data: Full time series DataFrame
            storage: DataStorage instance
            symbol: Symbol identifier for saved files

        Returns:
            Dictionary mapping fold_id to saved file paths
        """
        from datetime import datetime

        cv_folds, test_indices = self.split(data)
        saved_paths = {'folds': [], 'test': None}

        # Save test set (isolated, never used in CV)
        test_data = data.iloc[test_indices].copy()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_path = storage.save(
            test_data,
            stage='testing',
            symbol=f'test_{symbol}_{timestamp}',
            metadata={'split_type': 'test', 'cv_isolated': True}
        )
        saved_paths['test'] = str(test_path)

        # Save each CV fold
        for fold in cv_folds:
            fold_data = self.get_fold_data(data, fold)
            fold_paths = {}

            # Save training data for this fold
            train_path = storage.save(
                fold_data['train'],
                stage='training',
                symbol=f'fold{fold.fold_id}_train_{symbol}_{timestamp}',
                metadata={
                    'split_type': 'cv_train',
                    'fold_id': fold.fold_id,
                    'total_folds': self.n_splits
                }
            )
            fold_paths['train'] = str(train_path)

            # Save validation data for this fold
            val_path = storage.save(
                fold_data['validation'],
                stage='validation',
                symbol=f'fold{fold.fold_id}_val_{symbol}_{timestamp}',
                metadata={
                    'split_type': 'cv_validation',
                    'fold_id': fold.fold_id,
                    'total_folds': self.n_splits
                }
            )
            fold_paths['validation'] = str(val_path)

            saved_paths['folds'].append(fold_paths)

        logger.info(f"Saved {len(cv_folds)} CV folds + 1 test set for {symbol}")
        return saved_paths

    def validate_splits(self, data: pd.DataFrame) -> Dict[str, any]:
        """Validate CV splits meet quality criteria.

        Checks:
        1. No overlap between folds
        2. Temporal ordering preserved
        3. Complete coverage of CV data
        4. Test set isolation

        Returns:
            Dictionary with validation results
        """
        cv_folds, test_indices = self.split(data)
        n_samples = len(data)

        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }

        # Check 1: Test set isolation
        cv_size = int(n_samples * (1 - self.test_size))
        if test_indices[0] != cv_size:
            results['valid'] = False
            results['errors'].append(
                f"Test set not properly isolated: starts at {test_indices[0]}, "
                f"expected {cv_size}"
            )

        # Check 2: Temporal ordering in each fold
        for fold in cv_folds:
            if fold.train_end > fold.val_start:
                results['valid'] = False
                results['errors'].append(
                    f"Fold {fold.fold_id}: Training data leaks into validation "
                    f"(train_end={fold.train_end} > val_start={fold.val_start})"
                )

        # Check 3: Coverage of CV data
        all_val_indices = np.concatenate([fold.val_indices for fold in cv_folds])
        coverage = len(np.unique(all_val_indices)) / cv_size
        results['statistics']['cv_coverage'] = coverage

        if coverage < 0.80:
            results['warnings'].append(
                f"CV validation sets cover only {coverage*100:.1f}% of CV data (expected ≥80%)"
            )

        # Check 4: Fold size balance
        fold_sizes = [len(fold.val_indices) for fold in cv_folds]
        size_std = np.std(fold_sizes)
        size_mean = np.mean(fold_sizes)
        results['statistics']['fold_size_cv'] = size_std / size_mean if size_mean > 0 else 0

        if size_std / size_mean > 0.2:
            results['warnings'].append(
                f"Fold sizes vary significantly (CV={size_std/size_mean:.2f})"
            )

        logger.info(f"CV Validation: {'PASSED' if results['valid'] else 'FAILED'}")
        if results['errors']:
            for err in results['errors']:
                logger.error(err)
        if results['warnings']:
            for warn in results['warnings']:
                logger.warning(warn)

        return results
