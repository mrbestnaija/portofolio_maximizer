import json

from scripts.track_llm_signals import LLMSignalTracker


def test_record_validator_result_updates_metadata(tmp_path):
    tracking_path = tmp_path / "tracking.json"
    tracker = LLMSignalTracker(str(tracking_path))

    signal_id = tracker.register_signal(
        ticker="AAPL",
        date="2025-10-25",
        signal={"action": "BUY", "confidence": 0.8, "reasoning": "unit test"},
    )

    tracker.record_validator_result(
        signal_id,
        {
            "validator_version": "v2",
            "confidence_score": 0.9,
            "recommendation": "EXECUTE",
            "warnings": [],
            "quality_metrics": {"statistical": True},
        },
        status="validated",
    )

    tracker.flush()

    payload = json.loads(tracking_path.read_text())

    assert payload["metadata"]["total_signals"] == 1
    assert payload["metadata"]["validated_signals"] == 1
    assert payload["signals"][signal_id]["validation_status"] == "validated"
    assert (
        payload["signals"][signal_id]["validation_results"]["validator_details"]["validator_version"]
        == "v2"
    )
