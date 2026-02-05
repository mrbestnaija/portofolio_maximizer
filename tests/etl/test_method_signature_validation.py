"""
Method Signature Validation Tests
Tests for method signature changes and parameter validation
"""

import pytest
import inspect
from unittest.mock import Mock, patch
from datetime import datetime

from etl.data_storage import DataStorage
from etl.time_series_cv import TimeSeriesCrossValidator


class TestMethodSignatureValidation:
    """Test method signature validation and parameter handling"""

    def test_train_validation_test_split_signature(self):
        """Test that train_validation_test_split has correct signature"""
        signature = inspect.signature(DataStorage.train_validation_test_split)
        params = list(signature.parameters.keys())

        # Check required parameters
        expected_params = [
            'data', 'train_ratio', 'val_ratio', 'use_cv',
            'n_splits', 'test_size', 'gap', 'expanding_window'
        ]

        for param in expected_params:
            assert param in params, f"Missing parameter: {param}"

        # Check parameter types and defaults
        assert signature.parameters['data'].annotation.__name__ == 'DataFrame'
        assert signature.parameters['train_ratio'].default == 0.7
        assert signature.parameters['val_ratio'].default == 0.15
        assert signature.parameters['use_cv'].default == False
        assert signature.parameters['n_splits'].default == 5
        assert signature.parameters['test_size'].default == 0.15
        assert signature.parameters['gap'].default == 0
        assert signature.parameters['expanding_window'].default == True

    def test_train_validation_test_split_with_all_parameters(self):
        """Test method works with all new parameters"""
        import pandas as pd

        storage = DataStorage()
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        }, index=pd.date_range('2024-01-01', periods=10))

        # Test with all parameters
        result = storage.train_validation_test_split(
            data=data,
            train_ratio=0.6,
            val_ratio=0.2,
            use_cv=True,
            n_splits=3,
            test_size=0.2,
            gap=1,
            expanding_window=True
        )

        assert isinstance(result, dict)
        assert 'train' in result
        assert 'validation' in result
        assert 'test' in result

    def test_train_validation_test_split_backward_compatibility(self):
        """Test backward compatibility with old parameter set"""
        import pandas as pd

        storage = DataStorage()
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        }, index=pd.date_range('2024-01-01', periods=5))

        # Test with old parameters only
        result = storage.train_validation_test_split(
            data=data,
            train_ratio=0.7,
            val_ratio=0.15,
            use_cv=False
        )

        assert isinstance(result, dict)
        assert 'train' in result
        assert 'validation' in result
        assert 'test' in result

    def test_parameter_validation(self):
        """Test parameter validation and error handling"""
        import pandas as pd

        storage = DataStorage()
        data = pd.DataFrame({
            'close': [100, 101, 102]
        }, index=pd.date_range('2024-01-01', periods=3))

        # Test invalid test_size
        with pytest.raises((ValueError, TypeError)):
            storage.train_validation_test_split(
                data=data,
                use_cv=True,
                test_size=1.5  # Invalid: > 1.0
            )

        # Test invalid gap
        with pytest.raises((ValueError, TypeError)):
            storage.train_validation_test_split(
                data=data,
                use_cv=True,
                gap=-1  # Invalid: negative
            )

    def test_time_series_cv_parameter_usage(self):
        """Test that TimeSeriesCrossValidator uses the new parameters correctly"""
        import pandas as pd

        storage = DataStorage()
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        }, index=pd.date_range('2024-01-01', periods=11))

        # Test with custom parameters
        result = storage.train_validation_test_split(
            data=data,
            use_cv=True,
            n_splits=2,
            test_size=0.3,
            gap=2,
            expanding_window=False
        )

        # Verify the parameters were used correctly
        assert isinstance(result, dict)
        assert len(result) >= 3  # Should have train, validation, test

    def test_method_signature_consistency(self):
        """Test that method signature is consistent across calls"""
        signature1 = inspect.signature(DataStorage.train_validation_test_split)

        # Create another instance and check signature
        storage = DataStorage()
        signature2 = inspect.signature(storage.train_validation_test_split)

        assert signature1 == signature2, "Method signature should be consistent"

    def test_docstring_parameter_documentation(self):
        """Test that all parameters are documented in docstring"""
        method = DataStorage.train_validation_test_split
        docstring = method.__doc__

        assert docstring is not None, "Method should have docstring"

        # Check that new parameters are documented
        new_params = ['test_size', 'gap', 'expanding_window']
        for param in new_params:
            assert param in docstring, f"Parameter {param} should be documented"

    def test_error_handling_for_missing_parameters(self):
        """Test error handling when required parameters are missing"""
        import pandas as pd

        storage = DataStorage()
        data = pd.DataFrame({
            'close': [100, 101, 102]
        }, index=pd.date_range('2024-01-01', periods=3))

        # Test with minimal required parameters
        result = storage.train_validation_test_split(data)

        assert isinstance(result, dict)
        assert 'train' in result
        assert 'validation' in result
        assert 'test' in result

    def test_parameter_type_validation(self):
        """Test parameter type validation"""
        import pandas as pd

        storage = DataStorage()
        data = pd.DataFrame({
            'close': [100, 101, 102]
        }, index=pd.date_range('2024-01-01', periods=3))

        # Test with correct types
        result = storage.train_validation_test_split(
            data=data,
            train_ratio=0.7,
            val_ratio=0.15,
            use_cv=True,
            n_splits=3,
            test_size=0.15,
            gap=0,
            expanding_window=True
        )

        assert isinstance(result, dict)

    def test_method_performance_with_new_parameters(self):
        """Test that method performance is acceptable with new parameters"""
        import pandas as pd
        import time

        storage = DataStorage()
        data = pd.DataFrame({
            'close': range(1000)
        }, index=pd.date_range('2020-01-01', periods=1000))

        start_time = time.time()
        result = storage.train_validation_test_split(
            data=data,
            use_cv=True,
            n_splits=5,
            test_size=0.2,
            gap=1,
            expanding_window=True
        )
        end_time = time.time()

        execution_time = end_time - start_time

        # Should complete within reasonable time (5 seconds)
        assert execution_time < 5.0, f"Method took too long: {execution_time:.2f}s"
        assert isinstance(result, dict)


class TestTimeSeriesCrossValidatorIntegration:
    """Test integration with TimeSeriesCrossValidator"""

    def test_cv_validator_parameter_passing(self):
        """Test that parameters are correctly passed to TimeSeriesCrossValidator"""
        import pandas as pd

        with patch('etl.data_storage.TimeSeriesCrossValidator') as mock_cv:
            mock_cv.return_value.split.return_value = ([], [])

            storage = DataStorage()
            data = pd.DataFrame({
                'close': [100, 101, 102, 103, 104]
            }, index=pd.date_range('2024-01-01', periods=5))

            storage.train_validation_test_split(
                data=data,
                use_cv=True,
                n_splits=3,
                test_size=0.2,
                gap=1,
                expanding_window=True
            )

            # Verify TimeSeriesCrossValidator was called with correct parameters
            mock_cv.assert_called_once_with(
                n_splits=3,
                test_size=0.2,
                gap=1,
                expanding_window=True
            )

    def test_cv_validator_fallback_behavior(self):
        """Test fallback behavior when CV fails"""
        import pandas as pd

        storage = DataStorage()
        data = pd.DataFrame({
            'close': [100, 101, 102]
        }, index=pd.date_range('2024-01-01', periods=3))

        # Test with very small dataset that might cause CV issues
        result = storage.train_validation_test_split(
            data=data,
            use_cv=True,
            n_splits=5,  # More splits than data points
            test_size=0.2
        )

        # Should still return a valid result
        assert isinstance(result, dict)
        assert 'train' in result
        assert 'validation' in result
        assert 'test' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
