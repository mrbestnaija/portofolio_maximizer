import pandas as pd

from etl.synthetic_extractor import SyntheticExtractor


def test_microstructure_channels_present() -> None:
    extractor = SyntheticExtractor(
        config_path="config/synthetic_data_config.yml",
        name="synthetic",
    )
    data = extractor.extract_ohlcv(
        tickers=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-01-10",
    )
    assert not data.empty
    for col in ["Spread", "Slippage", "Depth", "OrderImbalance"]:
        assert col in data.columns
    # Ensure reasonable numeric content
    assert pd.api.types.is_numeric_dtype(data["Depth"])
    assert pd.api.types.is_numeric_dtype(data["OrderImbalance"])
