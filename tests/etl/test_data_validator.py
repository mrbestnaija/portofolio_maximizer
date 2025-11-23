import pandas as pd

from etl.data_validator import DataValidator


class TestDataValidator:
    def setup_method(self):
        self.validator = DataValidator()

    def test_positive_prices_required(self):
        df = pd.DataFrame(
            {
                "Open": [1.0, -2.0],
                "High": [2.0, 3.0],
                "Low": [0.5, 0.2],
                "Close": [1.5, -1.0],
                "Volume": [100, 200],
            }
        )

        report = self.validator.validate_ohlcv(df)

        assert report["passed"] is False
        assert any("negative/zero prices" in err for err in report["errors"])

    def test_volume_must_be_non_negative(self):
        df = pd.DataFrame(
            {
                "Open": [1.0, 2.0],
                "High": [2.0, 3.0],
                "Low": [0.5, 0.2],
                "Close": [1.5, 1.1],
                "Volume": [100, -50],
            }
        )

        report = self.validator.validate_ohlcv(df)

        assert report["passed"] is False
        assert any("Volume" in err and "negative values" in err for err in report["errors"])

    def test_missing_data_above_threshold_triggers_warning(self):
        df = pd.DataFrame(
            {
                "Open": [1.0, None, None, None],
                "High": [2.0, 3.0, 4.0, 5.0],
                "Low": [0.5, 0.6, 0.7, 0.8],
                "Close": [1.5, 1.6, 1.7, 1.8],
                "Volume": [100, 200, 300, 400],
            }
        )

        report = self.validator.validate_ohlcv(df)

        assert report["passed"] is True
        assert any("missing data" in warn for warn in report["warnings"])

