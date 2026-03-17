"""
tests/forcester_ts/test_model_snapshot_store.py
-------------------------------------------------
Unit tests for ModelSnapshotStore — joblib pkl persistence with manifest.
"""
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from forcester_ts.model_snapshot_store import ModelSnapshotStore


# ---------------------------------------------------------------------------
# Fixture: isolated temp snapshot dir
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return ModelSnapshotStore(snapshot_dir=tmp_path / "snapshots")


# ---------------------------------------------------------------------------
# save + load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_and_load_returns_same_object(self, store):
        obj = {"a": 1, "b": [2, 3], "c": 0.5}
        store.save("AAPL", "GARCH", "MODERATE", obj, n_obs=100,
                   data_hash="abc", aic=287.6)
        loaded = store.load("AAPL", "GARCH", "MODERATE",
                            current_n_obs=100, current_data_hash="abc")
        assert loaded == obj

    def test_save_numpy_array(self, store):
        import numpy as np
        obj = np.array([1.0, 2.0, 3.0])
        store.save("MSFT", "SAMOSSA", None, obj, n_obs=50,
                   data_hash="xyz", aic=100.0)
        loaded = store.load("MSFT", "SAMOSSA", None,
                            current_n_obs=50, current_data_hash="xyz")
        assert loaded is not None
        np.testing.assert_array_equal(loaded, obj)

    def test_load_returns_none_when_no_snapshot(self, store):
        result = store.load("NVDA", "GARCH", None,
                            current_n_obs=100, current_data_hash="hash")
        assert result is None

    def test_load_returns_none_when_stale(self, store, tmp_path):
        """Snapshot older than max_age_days returns None."""
        obj = {"key": "val"}
        store.save("AAPL", "GARCH", None, obj, n_obs=80,
                   data_hash="h1", aic=200.0)

        # Manually backdating the manifest
        manifest = json.loads(store._manifest_path.read_text())
        key = "AAPL|GARCH|NONE"
        old_time = (datetime.utcnow() - timedelta(days=10)).isoformat() + "Z"
        manifest[key]["saved_at"] = old_time
        store._manifest_path.write_text(json.dumps(manifest))

        result = store.load("AAPL", "GARCH", None,
                            current_n_obs=80, current_data_hash="h1",
                            max_age_days=7)
        assert result is None

    def test_load_returns_none_when_obs_delta_exceeded(self, store):
        obj = {"model": "fitted"}
        store.save("TSLA", "SARIMAX", "CRISIS", obj, n_obs=100,
                   data_hash="d1", aic=350.0)
        # Current obs is way more than snapshot — triggers retrain
        result = store.load("TSLA", "SARIMAX", "CRISIS",
                            current_n_obs=200, current_data_hash="d1",
                            max_obs_delta=20)
        assert result is None

    def test_load_returns_result_when_obs_delta_within_limit(self, store):
        obj = {"model": "fitted"}
        store.save("TSLA", "SARIMAX", "CRISIS", obj, n_obs=100,
                   data_hash="d1", aic=350.0)
        result = store.load("TSLA", "SARIMAX", "CRISIS",
                            current_n_obs=105, current_data_hash="d1",
                            max_obs_delta=20)
        assert result is not None

    def test_strict_hash_returns_none_on_mismatch(self, store):
        obj = {"x": 42}
        store.save("AAPL", "GARCH", None, obj, n_obs=50,
                   data_hash="correct_hash", aic=0.0)
        result = store.load("AAPL", "GARCH", None,
                            current_n_obs=50, current_data_hash="wrong_hash",
                            strict_hash=True)
        assert result is None

    def test_strict_hash_loads_when_matching(self, store):
        obj = {"x": 99}
        store.save("AAPL", "GARCH", None, obj, n_obs=50,
                   data_hash="correct_hash", aic=0.0)
        result = store.load("AAPL", "GARCH", None,
                            current_n_obs=50, current_data_hash="correct_hash",
                            strict_hash=True)
        assert result == obj

    def test_load_returns_none_when_manifest_path_missing(self, store):
        store.save("AAPL", "GARCH", None, {"ok": 1}, n_obs=10, data_hash="h", aic=0.0)
        manifest = json.loads(store._manifest_path.read_text())
        manifest["AAPL|GARCH|NONE"].pop("path", None)
        store._manifest_path.write_text(json.dumps(manifest))

        result = store.load("AAPL", "GARCH", None, current_n_obs=10, current_data_hash="h")
        assert result is None

    def test_load_returns_none_when_manifest_path_not_string(self, store):
        store.save("AAPL", "GARCH", None, {"ok": 1}, n_obs=10, data_hash="h", aic=0.0)
        manifest = json.loads(store._manifest_path.read_text())
        manifest["AAPL|GARCH|NONE"]["path"] = 123
        store._manifest_path.write_text(json.dumps(manifest))

        result = store.load("AAPL", "GARCH", None, current_n_obs=10, current_data_hash="h")
        assert result is None

    def test_load_returns_none_when_manifest_root_is_not_object(self, store):
        store.save("AAPL", "GARCH", None, {"ok": 1}, n_obs=10, data_hash="h", aic=0.0)
        store._manifest_path.write_text(json.dumps(["not", "a", "dict"]))

        result = store.load("AAPL", "GARCH", None, current_n_obs=10, current_data_hash="h")
        assert result is None


# ---------------------------------------------------------------------------
# Path traversal guard
# ---------------------------------------------------------------------------

class TestPathTraversalGuard:
    def test_path_traversal_raises_valueerror(self, store):
        """
        Manually inject a malicious path into the manifest and verify load()
        refuses to load it.
        """
        # Create a legit snapshot first so manifest exists
        store.save("AAPL", "GARCH", None, {"ok": 1}, n_obs=10,
                   data_hash="h", aic=0.0)
        # Tamper manifest
        manifest = json.loads(store._manifest_path.read_text())
        key = "AAPL|GARCH|NONE"
        manifest[key]["path"] = "../../evil.pkl"
        store._manifest_path.write_text(json.dumps(manifest))

        # load() should detect traversal and return None (not raise to caller)
        result = store.load("AAPL", "GARCH", None,
                            current_n_obs=10, current_data_hash="h")
        assert result is None

    def test_safe_pkl_path_raises_on_traversal(self, store):
        with pytest.raises(ValueError, match="traversal"):
            store._safe_pkl_path("../../evil.pkl")

    def test_safe_pkl_path_raises_on_sibling_directory(self, store, tmp_path):
        """
        Sibling directories share the same string prefix.
        /data/model_snapshots2/evil.pkl starts with /data/model_snapshots
        as a raw string, but is NOT inside the snapshot dir.
        The is_relative_to() guard must reject it.
        """
        # Create the sibling directory so Path.resolve() can resolve it
        sibling = tmp_path / "snapshots2"
        sibling.mkdir(parents=True, exist_ok=True)
        evil_path = sibling / "evil.pkl"
        evil_path.write_bytes(b"evil")
        # Inject relative path that resolves to the sibling dir
        relative_escape = "../snapshots2/evil.pkl"
        with pytest.raises(ValueError, match="traversal"):
            store._safe_pkl_path(relative_escape)

    def test_safe_pkl_path_raises_on_nested_subdirectory(self, store):
        with pytest.raises(ValueError, match="traversal"):
            store._safe_pkl_path("nested/evil.pkl")

    def test_safe_pkl_path_accepts_normal_filename(self, store):
        path = store._safe_pkl_path("AAPL_GARCH_NONE_20260227.pkl")
        assert str(store._dir.resolve()) in str(path)

    def test_make_filename_sanitizes_components(self, store):
        filename = store._make_filename("../AAPL", "GARCH/ALT", "HIGH VOL")
        assert ".." not in filename
        assert "/" not in filename
        assert "\\" not in filename
        assert filename.endswith(".pkl")


# ---------------------------------------------------------------------------
# list_snapshots + prune_old
# ---------------------------------------------------------------------------

class TestListAndPrune:
    def test_list_snapshots_returns_entries(self, store):
        store.save("AAPL", "GARCH", None, {"a": 1}, n_obs=10, data_hash="h1", aic=0.0)
        store.save("MSFT", "GARCH", None, {"b": 2}, n_obs=20, data_hash="h2", aic=0.0)
        snaps = store.list_snapshots()
        assert len(snaps) == 2
        keys = {s["key"] for s in snaps}
        assert "AAPL|GARCH|NONE" in keys
        assert "MSFT|GARCH|NONE" in keys

    def test_list_snapshots_empty_store(self, store):
        assert store.list_snapshots() == []

    def test_prune_removes_orphaned_pkl(self, store):
        """PKL files not referenced by manifest are pruned."""
        store.save("AAPL", "GARCH", None, {"x": 1}, n_obs=5, data_hash="h", aic=0.0)
        # Create an orphan pkl
        orphan = store._dir / "ORPHAN_20260101.pkl"
        orphan.write_bytes(b"fake")
        deleted = store.prune_old()
        assert deleted >= 1
        assert not orphan.exists()

    def test_prune_leaves_manifest_entries_intact(self, store):
        store.save("AAPL", "GARCH", None, {"x": 1}, n_obs=5, data_hash="h", aic=0.0)
        store.prune_old()
        snaps = store.list_snapshots()
        assert len(snaps) == 1

    def test_metadata_stored_in_manifest(self, store):
        store.save("AAPL", "GARCH", "TRENDING", {"v": 1}, n_obs=42,
                   data_hash="abc", aic=123.4, metadata={"p": 1, "q": 2})
        snaps = store.list_snapshots()
        assert len(snaps) == 1
        snap = snaps[0]
        assert snap["n_obs"] == 42
        assert snap["aic"] == pytest.approx(123.4)
        assert snap["metadata"]["p"] == 1


# ---------------------------------------------------------------------------
# Phase 8.5 — version fingerprint sidecar (.meta.json)
# ---------------------------------------------------------------------------

import sys as _sys


class TestVersionFingerprint:
    """save() writes .meta.json; load() validates Python version compatibility."""

    def test_save_writes_meta_sidecar(self, store):
        store.save("AAPL", "GARCH", None, {"x": 1}, n_obs=10, data_hash="h", aic=0.0)
        pkl_files = list(store._dir.glob("*.pkl"))
        assert len(pkl_files) == 1
        meta_path = store._meta_path_for(pkl_files[0])
        assert meta_path.exists(), "Expected .meta.json sidecar alongside .pkl"

    def test_meta_sidecar_contains_python_version(self, store):
        store.save("AAPL", "GARCH", None, {"x": 1}, n_obs=10, data_hash="h", aic=0.0)
        pkl_path = next(store._dir.glob("*.pkl"))
        meta = json.loads(store._meta_path_for(pkl_path).read_text(encoding="utf-8"))
        assert "python_version" in meta
        assert "python_major_minor" in meta
        assert meta["python_major_minor"] == f"{_sys.version_info.major}.{_sys.version_info.minor}"

    def test_meta_sidecar_contains_library_versions(self, store):
        store.save("AAPL", "GARCH", None, {"x": 1}, n_obs=10, data_hash="h", aic=0.0)
        pkl_path = next(store._dir.glob("*.pkl"))
        meta = json.loads(store._meta_path_for(pkl_path).read_text(encoding="utf-8"))
        assert "libraries" in meta
        assert isinstance(meta["libraries"], dict)
        # At minimum numpy and joblib should be present
        assert "numpy" in meta["libraries"]
        assert "joblib" in meta["libraries"]

    def test_meta_sidecar_contains_model_provenance(self, store):
        store.save("MSFT", "SAMOSSA", "TRENDING", {"v": 2}, n_obs=50, data_hash="h2", aic=1.0)
        pkl_path = next(store._dir.glob("*.pkl"))
        meta = json.loads(store._meta_path_for(pkl_path).read_text(encoding="utf-8"))
        assert meta["ticker"] == "MSFT"
        assert meta["model_type"] == "SAMOSSA"
        assert meta["regime"] == "TRENDING"
        assert meta.get("schema_version") == 1

    def test_load_succeeds_with_matching_meta(self, store):
        obj = {"ok": True}
        store.save("AAPL", "GARCH", None, obj, n_obs=10, data_hash="h", aic=0.0)
        loaded = store.load("AAPL", "GARCH", None, current_n_obs=10, current_data_hash="h")
        assert loaded == obj

    def test_load_warns_but_succeeds_for_legacy_snapshot_without_meta(self, store, caplog):
        """Snapshot with no .meta.json → warning emitted, load proceeds."""
        import logging
        obj = {"legacy": True}
        store.save("AAPL", "GARCH", None, obj, n_obs=10, data_hash="h", aic=0.0)
        # Remove the meta sidecar to simulate a legacy snapshot
        pkl_path = next(store._dir.glob("*.pkl"))
        meta_path = store._meta_path_for(pkl_path)
        meta_path.unlink()

        with caplog.at_level(logging.WARNING, logger="forcester_ts.model_snapshot_store"):
            loaded = store.load("AAPL", "GARCH", None, current_n_obs=10, current_data_hash="h")
        assert loaded == obj
        assert any("legacy" in r.message or "meta sidecar" in r.message for r in caplog.records)

    def test_load_blocked_on_python_major_version_mismatch(self, store, monkeypatch):
        """Different Python major version → load returns None."""
        obj = {"critical": True}
        store.save("AAPL", "GARCH", None, obj, n_obs=10, data_hash="h", aic=0.0)
        pkl_path = next(store._dir.glob("*.pkl"))
        meta_path = store._meta_path_for(pkl_path)
        # Overwrite meta with a different major version
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["python_major_minor"] = "2.7"  # incompatible major
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        loaded = store.load("AAPL", "GARCH", None, current_n_obs=10, current_data_hash="h")
        assert loaded is None, "Should refuse to load when Python major version mismatches"

    def test_load_warns_but_proceeds_on_minor_version_mismatch(self, store, caplog, monkeypatch):
        """Different Python minor version within same major → warning, load proceeds."""
        import logging
        obj = {"data": 42}
        store.save("AAPL", "GARCH", None, obj, n_obs=10, data_hash="h", aic=0.0)
        pkl_path = next(store._dir.glob("*.pkl"))
        meta_path = store._meta_path_for(pkl_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # Use same major, different minor
        current_major = _sys.version_info.major
        different_minor = _sys.version_info.minor + 1
        meta["python_major_minor"] = f"{current_major}.{different_minor}"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="forcester_ts.model_snapshot_store"):
            loaded = store.load("AAPL", "GARCH", None, current_n_obs=10, current_data_hash="h")
        assert loaded == obj
        assert any("minor" in r.message for r in caplog.records)

    def test_prune_removes_orphan_meta_json(self, store):
        """prune_old() deletes .meta.json files with no corresponding .pkl."""
        store.save("AAPL", "GARCH", None, {"x": 1}, n_obs=5, data_hash="h", aic=0.0)
        # Create an orphan .meta.json (no .pkl counterpart)
        orphan_meta = store._dir / "ORPHAN_20260101.meta.json"
        orphan_meta.write_text('{"orphan": true}', encoding="utf-8")
        store.prune_old()
        assert not orphan_meta.exists(), "Orphan .meta.json should have been pruned"

    def test_prune_removes_meta_sidecar_alongside_orphan_pkl(self, store):
        """When an orphan .pkl is pruned, its .meta.json is also deleted."""
        store.save("AAPL", "GARCH", None, {"x": 1}, n_obs=5, data_hash="h", aic=0.0)
        # Create an orphan pkl + sidecar not referenced by manifest
        orphan_pkl = store._dir / "ORPHAN_20260101.pkl"
        orphan_pkl.write_bytes(b"fake")
        orphan_meta = store._meta_path_for(orphan_pkl)
        orphan_meta.write_text('{"orphan": true}', encoding="utf-8")

        store.prune_old()
        assert not orphan_pkl.exists()
        assert not orphan_meta.exists()
