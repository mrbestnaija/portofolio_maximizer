from __future__ import annotations

import numpy as np
import pandas as pd


def _active_window(end_date: str = "2026-01-07") -> dict:
    return {
        "dataset": {
            "ticker": "AAPL",
            "start": "2020-01-01",
            "end": end_date,
            "length": 210,
            "forecast_horizon": 30,
        },
        "artifacts": {
            "residual_experiment": {
                "residual_status": "active",
                "y_hat_anchor": [100.0] * 30,
                "y_hat_residual_ensemble": [101.0] * 30,
            }
        },
    }


def test_backfill_returns_zero_for_no_realized_skips(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    from scripts import residual_experiment_phase3_backfill as mod

    monkeypatch.setattr(
        mod,
        "_load_all_realized_series",
        lambda: pd.Series(
            [100.0, 101.0],
            index=pd.to_datetime(["2023-12-28", "2023-12-29"]),
        ),
    )
    monkeypatch.setattr(
        mod,
        "collect_unique_active_audits",
        lambda: {"fp": (tmp_path / "audit.json", _active_window())},
    )
    monkeypatch.setattr(
        mod,
        "load_realized_prices",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("Only 0 realized prices after 2026-01-07; need 30")
        ),
    )

    assert mod.main(dry_run=True) == 0


def test_backfill_returns_one_for_compute_failures(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    from scripts import residual_experiment_phase3_backfill as mod

    monkeypatch.setattr(
        mod,
        "_load_all_realized_series",
        lambda: pd.Series(
            [100.0, 101.0],
            index=pd.to_datetime(["2023-12-28", "2023-12-29"]),
        ),
    )
    monkeypatch.setattr(
        mod,
        "collect_unique_active_audits",
        lambda: {"fp": (tmp_path / "audit.json", _active_window(end_date="2023-12-01"))},
    )
    monkeypatch.setattr(
        mod,
        "load_realized_prices",
        lambda *_args, **_kwargs: np.array([100.0] * 30),
    )
    monkeypatch.setattr(
        mod,
        "compute_phase3_metrics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert mod.main(dry_run=True) == 1


def test_backfill_returns_zero_for_skipped_bad_signal_windows(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    from scripts import residual_experiment_phase3_backfill as mod

    monkeypatch.setattr(
        mod,
        "_load_all_realized_series",
        lambda: pd.Series(
            [100.0, 101.0],
            index=pd.to_datetime(["2023-12-28", "2023-12-29"]),
        ),
    )
    monkeypatch.setattr(
        mod,
        "collect_unique_active_audits",
        lambda: {
            "fp": (
                tmp_path / "audit.json",
                {
                    "dataset": {
                        "ticker": "AAPL",
                        "start": "2020-01-01",
                        "end": "2023-12-01",
                        "length": 210,
                        "forecast_horizon": 30,
                    },
                    "artifacts": {
                        "residual_experiment": {
                            "residual_status": "inactive",
                            "reason_code": "PREDICTED_RESIDUAL_TOO_CONSTANT",
                            "residual_signal_valid": False,
                            "correction_applied": False,
                        }
                    },
                },
            )
        },
    )

    assert mod.main(dry_run=True) == 0
