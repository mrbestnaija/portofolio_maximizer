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

    def test_validate_dataframe_with_custom_price_columns(self):
        df = pd.DataFrame({"Price": [1.0, 0.0, 2.0]})

        report = self.validator.validate_dataframe(df, price_columns=["Price"])

        assert report["passed"] is False
        assert any("Price" in err for err in report["errors"])

    def test_validate_ohlcv_statistics_exposed_for_clean_data(self):
        df = pd.DataFrame(
            {
                "Open": [10.0, 10.5, 11.0],
                "High": [10.5, 11.0, 11.5],
                "Low": [9.5, 10.0, 10.5],
                "Close": [10.2, 10.8, 11.1],
                "Volume": [1000, 1200, 900],
            }
        )

        report = self.validator.validate_ohlcv(df)

        assert report["passed"] is True
        assert report["errors"] == []
        missing_stats = report["statistics"].get("missing_ratio", {})
        assert set(missing_stats.keys()) == set(df.columns)
        assert all(value == 0 for value in missing_stats.values())
