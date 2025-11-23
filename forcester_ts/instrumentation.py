"""
Model instrumentation utilities for transparent, interpretable forecasting.

The brutal/optimization docs require every forecast cycle to expose timing,
configuration, and diagnostic metadata so the AI stack remains auditable.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import json
import time

import numpy as np
import pandas as pd


@dataclass
class ModelRunStats:
    """Single model run entry with duration and metadata."""

    model: str
    phase: str
    started_at: str
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        # Drop empty metadata for compactness.
        if not payload["metadata"]:
            payload.pop("metadata")
        return payload


class ModelInstrumentation:
    """Capture per-model telemetry and interpretable artifacts."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._runs: List[ModelRunStats] = []
        self._artifacts: Dict[str, Any] = {}
        self._dataset_meta: Dict[str, Any] = {}
        self._dataset_snapshots: List[Dict[str, Any]] = []
        self._model_benchmarks: List[Dict[str, Any]] = []

    def set_dataset_metadata(self, **metadata: Any) -> None:
        cleaned = {k: v for k, v in metadata.items() if v is not None}
        self._dataset_meta.update(cleaned)

    def record_artifact(self, name: str, payload: Any) -> None:
        self._artifacts[name] = payload

    def record_series_snapshot(self, name: str, series: pd.Series) -> None:
        if not isinstance(series, pd.Series) or series.empty:
            return
        snapshot = describe_series(series)
        snapshot["name"] = name
        self._dataset_snapshots.append(snapshot)

    def record_dataframe_snapshot(
        self, name: str, frame: pd.DataFrame, columns: Optional[Sequence[str]] = None
    ) -> None:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return
        snapshot = describe_dataframe(frame, columns=columns)
        snapshot["name"] = name
        self._dataset_snapshots.append(snapshot)

    def record_model_metrics(
        self,
        model: str,
        metrics: Dict[str, Any],
        **context: Any,
    ) -> None:
        if not metrics:
            return
        entry = {
            "model": model,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
        }
        if context:
            entry["context"] = context
        self._model_benchmarks.append(entry)

    @contextmanager
    def track(self, model: str, phase: str, **metadata: Any):
        """
        Context manager that records duration and metadata for model phases.

        Usage:
            with instrumentation.track("sarimax", "fit", points=len(series)) as meta:
                ... # do work
                meta["order"] = order
        """
        started_at = datetime.utcnow().isoformat()
        start = time.perf_counter()
        phase_metadata = dict(metadata)
        try:
            yield phase_metadata
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            self._runs.append(
                ModelRunStats(
                    model=model,
                    phase=phase,
                    started_at=started_at,
                    duration_ms=duration_ms,
                    metadata=phase_metadata,
                )
            )

    def export(self) -> Dict[str, Any]:
        summary: Dict[str, Dict[str, Any]] = {}
        for run in self._runs:
            entry = summary.setdefault(
                run.model, {"count": 0, "duration_ms": 0.0, "phases": {}}
            )
            entry["count"] += 1
            entry["duration_ms"] += run.duration_ms
            phase_bucket = entry["phases"].setdefault(run.phase, {"count": 0, "duration_ms": 0.0})
            phase_bucket["count"] += 1
            phase_bucket["duration_ms"] += run.duration_ms

        return {
            "dataset": dict(self._dataset_meta),
            "dataset_snapshots": list(self._dataset_snapshots),
            "runs": [run.to_dict() for run in self._runs],
            "artifacts": dict(self._artifacts),
            "model_benchmarks": list(self._model_benchmarks),
            "benchmark_summary": _summarize_benchmarks(self._model_benchmarks),
            "summary": summary,
        }

    def dump_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.export(), handle, indent=2)


def _safe_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, pd.Timestamp):
            return None
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def describe_series(series: pd.Series) -> Dict[str, Any]:
    """Return mathematical/statistical summary of a pandas Series."""
    cleaned = series.dropna()
    summary: Dict[str, Any] = {
        "type": "series",
        "length": int(series.shape[0]),
        "missing_pct": float(series.isna().mean() * 100.0),
        "index_start": str(series.index.min()) if isinstance(series.index, pd.DatetimeIndex) else None,
        "index_end": str(series.index.max()) if isinstance(series.index, pd.DatetimeIndex) else None,
        "frequency": getattr(series.index, "freqstr", None),
        "stats": {},
    }
    if not cleaned.empty:
        summary["stats"] = {
            "mean": _safe_value(cleaned.mean()),
            "std": _safe_value(cleaned.std()),
            "min": _safe_value(cleaned.min()),
            "max": _safe_value(cleaned.max()),
            "median": _safe_value(cleaned.median()),
            "iqr": _safe_value(cleaned.quantile(0.75) - cleaned.quantile(0.25)),
        }
        summary["shape"] = (int(len(cleaned)), 1)
    else:
        summary["stats"] = {}
    return summary


def describe_dataframe(
    frame: pd.DataFrame, columns: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """Return shape + column-level statistics for the supplied DataFrame."""
    subset = frame[columns] if columns else frame
    snapshot: Dict[str, Any] = {
        "type": "dataframe",
        "shape": (int(subset.shape[0]), int(subset.shape[1])),
        "columns": list(subset.columns),
        "dtypes": {col: str(dtype) for col, dtype in subset.dtypes.items()},
        "index_start": str(subset.index.min()) if isinstance(subset.index, pd.DatetimeIndex) else None,
        "index_end": str(subset.index.max()) if isinstance(subset.index, pd.DatetimeIndex) else None,
        "frequency": getattr(subset.index, "freqstr", None),
        "missing_pct": float(subset.isna().mean().mean() * 100.0)
        if subset.size
        else 0.0,
        "column_summaries": [],
    }
    for col in subset.columns:
        series = subset[col]
        if not np.issubdtype(series.dtype, np.number):
            continue
        stats = describe_series(series)
        snapshot["column_summaries"].append(
            {
                "name": col,
                "missing_pct": stats.get("missing_pct"),
                "stats": stats.get("stats"),
            }
        )
    return snapshot


def _summarize_benchmarks(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not entries:
        return {}
    metric_data: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        model = entry.get("model")
        metrics = entry.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for metric, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            bucket = metric_data.setdefault(
                metric,
                {
                    "best_model": model,
                    "best_value": value,
                    "worst_model": model,
                    "worst_value": value,
                    "values": [],
                },
            )
            bucket["values"].append(value)
            if value < bucket["best_value"]:
                bucket["best_value"] = value
                bucket["best_model"] = model
            if value > bucket["worst_value"]:
                bucket["worst_value"] = value
                bucket["worst_model"] = model
    summary = {}
    for metric, bucket in metric_data.items():
        values = bucket.pop("values", [])
        bucket["mean_value"] = float(sum(values) / len(values)) if values else None
        summary[metric] = bucket
    return summary
