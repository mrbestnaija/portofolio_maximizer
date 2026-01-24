import math

from forcester_ts.ensemble import derive_model_confidence


def test_confidence_includes_all_models_and_is_monotonic():
    summaries = {
        "SARIMAX": {
            "regression_metrics": {
                "rmse": 1.2,
                "smape": 0.12,
                "tracking_error": 0.25,
                "directional_accuracy": 0.55,
                "n_observations": 50,
            }
        },
        "garch": {
            "regression_metrics": {
                "rmse": 1.6,
                "smape": 0.18,
                "tracking_error": 0.30,
                "directional_accuracy": 0.52,
                "n_observations": 50,
            }
        },
        "samossa": {
            "regression_metrics": {
                "rmse": 0.9,
                "smape": 0.10,
                "tracking_error": 0.20,
                "directional_accuracy": 0.58,
                "n_observations": 50,
            }
        },
        "mssa_rl": {
            "regression_metrics": {
                "rmse": 1.0,
                "smape": 0.11,
                "tracking_error": 0.22,
                "directional_accuracy": 0.57,
                "n_observations": 50,
            }
        },
    }

    confidence = derive_model_confidence(summaries)

    # All active models should appear with bounded, non-saturated scores.
    assert set(["sarimax", "garch", "samossa", "mssa_rl"]).issubset(confidence.keys())
    assert all(0.05 <= v <= 0.95 for v in confidence.values())
    assert not math.isclose(max(confidence.values()), 1.0)

    # Monotonicity sanity: better RMSE gets higher score.
    assert confidence["samossa"] > confidence["garch"]
    assert confidence["samossa"] > confidence["sarimax"]
