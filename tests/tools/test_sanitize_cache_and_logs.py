import os
from datetime import datetime, timedelta
from pathlib import Path

from scripts.sanitize_cache_and_logs import sanitize_cache_and_logs


def _touch(path: Path, days_ago: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    ts = (datetime.now() - timedelta(days=days_ago)).timestamp()
    os.utime(path, (ts, ts))


def test_sanitize_removes_old_and_keeps_recent(tmp_path: Path) -> None:
    old_file = tmp_path / "data" / "raw" / "old.parquet"
    recent_file = tmp_path / "data" / "raw" / "recent.parquet"
    other_file = tmp_path / "logs" / "keep.log"

    _touch(old_file, days_ago=20)
    _touch(recent_file, days_ago=5)
    _touch(other_file, days_ago=3)

    deleted = sanitize_cache_and_logs(
        retention_days=14,
        data_dirs=[str(tmp_path / "data" / "raw")],
        log_dirs=[str(tmp_path / "logs")],
        patterns=["*.parquet", "*.log"],
        exclude_prefixes=[],
    )

    assert deleted == 1
    assert not old_file.exists()
    assert recent_file.exists()
    assert other_file.exists()
