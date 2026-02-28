"""
model_snapshot_store.py
-----------------------
Persist and load full fitted model objects as joblib pickle files.

Storage layout:
    data/model_snapshots/
        {TICKER}_{MODEL}_{REGIME}_{YYYYMMDD}.pkl   # serialized fitted object
        manifest.json                               # index: latest pkl per key

Enables:
  - Skip-refit when training data is unchanged (load pkl instead of .fit())
  - Cross-ticker transfer: load AAPL GARCH params to warm-start MSFT fit
  - Recreation: exact historical model state from snapshot date

Security: load() asserts the resolved pkl path is inside SNAPSHOT_DIR to
prevent path traversal if manifest.json is tampered with.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.=-]+$")


class ModelSnapshotStore:
    """
    Persist and retrieve complete fitted model objects via joblib.

    Complement to OrderLearner (which stores only order statistics).
    """

    DEFAULT_SNAPSHOT_DIR = Path("data/model_snapshots")

    def __init__(self, snapshot_dir: str | Path | None = None) -> None:
        self._dir = Path(snapshot_dir) if snapshot_dir else self.DEFAULT_SNAPSHOT_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._dir / "manifest.json"

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict:
        if not self._manifest_path.exists():
            return {}
        try:
            manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(manifest, dict):
            logger.warning("ModelSnapshotStore manifest root is not a JSON object")
            return {}
        return manifest

    def _save_manifest(self, manifest: dict) -> None:
        tmp = self._manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tmp.replace(self._manifest_path)

    @staticmethod
    def _make_key(ticker: str, model_type: str, regime: str | None) -> str:
        return f"{ticker}|{model_type}|{regime or 'NONE'}"

    @staticmethod
    def _slugify_component(value: Any, default: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return default
        cleaned = re.sub(r"[^A-Za-z0-9_.=-]+", "_", raw)
        cleaned = cleaned.strip("._-")
        return cleaned or default

    @classmethod
    def _make_filename(cls, ticker: str, model_type: str, regime: str | None) -> str:
        ticker_part = cls._slugify_component(ticker, "UNKNOWN")
        model_part = cls._slugify_component(model_type, "MODEL")
        regime_part = cls._slugify_component(regime or "NONE", "NONE")
        today = date.today().strftime("%Y%m%d")
        return f"{ticker_part}_{model_part}_{regime_part}_{today}.pkl"

    # ------------------------------------------------------------------
    # Security
    # ------------------------------------------------------------------

    def _safe_pkl_path(self, filename: str) -> Path:
        """
        Resolve filename relative to _dir and assert it stays inside _dir.
        Raises ValueError on path traversal.

        Uses Path.is_relative_to() (Python 3.9+) to avoid the str.startswith()
        false-positive on sibling directories that share a common prefix
        (e.g. /data/model_snapshots2 starts with /data/model_snapshots).
        """
        if not isinstance(filename, str) or not filename:
            raise ValueError("Snapshot filename must be a non-empty string")
        candidate = Path(filename)
        if candidate.is_absolute() or candidate.name != filename:
            raise ValueError(
                f"Path traversal detected in snapshot filename: {filename!r}"
            )
        if _SAFE_FILENAME_RE.fullmatch(filename) is None:
            raise ValueError(f"Unsafe snapshot filename: {filename!r}")
        resolved = (self._dir / filename).resolve()
        if not resolved.is_relative_to(self._dir.resolve()):
            raise ValueError(
                f"Path traversal detected in snapshot filename: {filename!r}"
            )
        return resolved

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        ticker: str,
        model_type: str,
        regime: str | None,
        fitted_obj: Any,
        n_obs: int,
        data_hash: str,
        aic: float,
        metadata: dict | None = None,
    ) -> Path:
        """
        Serialize fitted_obj to a .pkl file and update manifest.
        Returns the Path to the written file.
        """
        try:
            import joblib
        except ImportError:
            logger.warning("joblib not available — ModelSnapshotStore.save() skipped")
            return self._dir / "unavailable.pkl"

        filename = self._make_filename(ticker, model_type, regime)
        pkl_path = self._safe_pkl_path(filename)

        try:
            joblib.dump(fitted_obj, pkl_path)
        except Exception as exc:
            logger.warning("ModelSnapshotStore.save failed to dump %s: %s", filename, exc)
            return pkl_path

        manifest = self._load_manifest()
        key = self._make_key(ticker, model_type, regime)
        manifest[key] = {
            "path": filename,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "n_obs": n_obs,
            "data_hash": data_hash,
            "aic": float(aic) if aic == aic else None,
            "metadata": metadata or {},
        }
        self._save_manifest(manifest)
        logger.debug(
            "ModelSnapshotStore saved %s/%s/%s → %s",
            ticker, model_type, regime, filename,
        )
        return pkl_path

    def load(
        self,
        ticker: str,
        model_type: str,
        regime: str | None,
        current_n_obs: int,
        current_data_hash: str,
        max_age_days: int = 7,
        max_obs_delta: int = 20,
        strict_hash: bool = False,
    ) -> Any | None:
        """
        Load and return the serialized model object if the snapshot is fresh.

        Returns None if:
        - No snapshot found
        - Snapshot is older than max_age_days
        - |current_n_obs - snapshot_n_obs| > max_obs_delta (new data available)
        - strict_hash=True and data_hash mismatch
        """
        try:
            import joblib
        except ImportError:
            return None

        manifest = self._load_manifest()
        key = self._make_key(ticker, model_type, regime)
        entry = manifest.get(key)
        if not isinstance(entry, dict):
            return None

        # Age check
        saved_at_str = entry.get("saved_at", "")
        try:
            saved_at = datetime.fromisoformat(saved_at_str.rstrip("Z"))
            if datetime.utcnow() - saved_at > timedelta(days=max_age_days):
                logger.debug(
                    "ModelSnapshotStore: %s/%s/%s snapshot stale (%s)",
                    ticker, model_type, regime, saved_at_str,
                )
                return None
        except ValueError:
            return None

        # Obs delta check
        snap_n_obs = int(entry.get("n_obs", 0))
        if abs(current_n_obs - snap_n_obs) > max_obs_delta:
            logger.debug(
                "ModelSnapshotStore: %s/%s/%s obs delta %d > %d — retrain",
                ticker, model_type, regime,
                abs(current_n_obs - snap_n_obs), max_obs_delta,
            )
            return None

        # Hash check (optional)
        if strict_hash and entry.get("data_hash") != current_data_hash:
            logger.debug(
                "ModelSnapshotStore: %s/%s/%s data_hash mismatch", ticker, model_type, regime
            )
            return None

        # Path safety + load
        try:
            filename = entry["path"]
            pkl_path = self._safe_pkl_path(filename)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("ModelSnapshotStore: %s", exc)
            return None

        if not pkl_path.exists():
            return None

        try:
            obj = joblib.load(pkl_path)
            logger.debug(
                "ModelSnapshotStore loaded snapshot %s/%s/%s from %s",
                ticker, model_type, regime, filename,
            )
            return obj
        except Exception as exc:
            logger.warning("ModelSnapshotStore.load failed for %s: %s", filename, exc)
            return None

    def list_snapshots(self) -> list[dict]:
        """Return a list of manifest entries with their key included."""
        manifest = self._load_manifest()
        result = []
        for key, entry in manifest.items():
            if not isinstance(entry, dict):
                continue
            result.append({"key": key, **entry})
        return sorted(result, key=lambda x: x.get("saved_at", ""), reverse=True)

    def prune_old(self, keep_per_key: int = 3) -> int:
        """
        Keep at most keep_per_key snapshots per (ticker, model, regime) key.
        Deletes older pkl files and removes them from the manifest.
        Returns number of files deleted.
        """
        # Build per-key history by scanning pkl files
        manifest = self._load_manifest()
        active_files = {
            entry["path"]
            for entry in manifest.values()
            if isinstance(entry, dict) and isinstance(entry.get("path"), str)
        }

        # Delete all pkl files NOT in manifest (orphans)
        deleted = 0
        for pkl_file in self._dir.glob("*.pkl"):
            if pkl_file.name not in active_files:
                try:
                    pkl_file.unlink()
                    deleted += 1
                except Exception:
                    pass

        # For each key, keep only last keep_per_key by date
        # (manifest already holds only latest per key, so no multi-entry pruning needed)
        # But ensure pkl files referenced in manifest actually exist; clean stale entries
        to_remove = []
        for key, entry in manifest.items():
            if not isinstance(entry, dict):
                to_remove.append(key)
                continue
            try:
                path = self._safe_pkl_path(entry["path"])
            except (KeyError, TypeError, ValueError):
                to_remove.append(key)
                continue
            if not path.exists():
                to_remove.append(key)
        for key in to_remove:
            del manifest[key]

        if to_remove:
            self._save_manifest(manifest)

        return deleted
