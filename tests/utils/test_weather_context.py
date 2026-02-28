from __future__ import annotations

import pandas as pd

from utils.weather_context import extract_weather_context, hydrate_signal_weather_context


def test_extract_weather_context_prefers_ticker_specific_attrs() -> None:
    frame = pd.DataFrame({"Close": [100.0, 101.0]})
    frame.attrs["weather_context"] = {"event_type": "storm", "severity": "low"}
    frame.attrs["weather_context_by_ticker"] = {
        "SOYBEAN": {"event_type": "drought", "severity": "severe", "days_to_event": 2}
    }

    payload = extract_weather_context(frame, ticker="SOYBEAN")

    assert payload["event_type"] == "drought"
    assert payload["severity"] == "severe"
    assert payload["days_to_event"] == 2


def test_hydrate_signal_weather_context_merges_conservatively() -> None:
    frame = pd.DataFrame(
        {
            "Close": [100.0, 101.0],
            "weather_event_type": ["rain", "heatwave"],
            "weather_severity": ["low", "high"],
            "weather_days_to_event": [8, 3],
            "weather_impact_direction": ["favorable", "adverse"],
        }
    )

    signal = {
        "ticker": "CORN",
        "weather_context": {"event_type": "rain", "severity": "low", "days_to_event": 10},
        "provenance": {"weather_context": {"severity": "moderate", "days_to_event": 6}},
    }

    merged = hydrate_signal_weather_context(signal, frame, ticker="CORN")

    assert merged["severity"] == "high"
    assert merged["days_to_event"] == 3
    assert merged["impact_direction"] == "adverse"
    assert signal["weather_context"] == merged
    assert signal["provenance"]["weather_context"] == merged
