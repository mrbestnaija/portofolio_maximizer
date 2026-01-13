from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from forcester_ts.instrumentation import ModelInstrumentation


def test_instrumentation_dump_json_handles_common_types(tmp_path: Path) -> None:
    inst = ModelInstrumentation()
    series = pd.Series([1.0, 2.0, np.nan], index=pd.date_range("2024-01-01", periods=3, freq="D"))
    inst.set_dataset_metadata(start=series.index.min(), end=series.index.max(), length=len(series))
    inst.record_series_snapshot("price_series", series)
    inst.record_artifact("example", {"when": datetime(2024, 1, 1), "today": date(2024, 1, 2), "path": tmp_path})
    inst.record_model_metrics("sarimax", {"rmse": 0.1, "tracking_error": np.float64(0.2)})

    out_file = tmp_path / "instrumentation.json"
    inst.dump_json(out_file)

    loaded = json.loads(out_file.read_text())
    assert "dataset" in loaded
    assert "artifacts" in loaded
    # Confirm datetime/date/Path/NumPy types were serialized.
    assert loaded["artifacts"]["example"]["when"].startswith("2024-01-01")
    assert loaded["artifacts"]["example"]["today"].startswith("2024-01-02")
    assert isinstance(loaded["artifacts"]["example"]["path"], str)
