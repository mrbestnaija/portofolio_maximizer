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
