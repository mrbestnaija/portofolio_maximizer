"""
Contract tests for the GPU-parallel runner (dry-run mode).

These tests validate shard-to-GPU mapping, trade-count gates, and
energy-saving defaults without executing heavy workloads.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import json
import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "bash" / "run_gpu_parallel.sh"


def _run_runner(env: dict) -> str:
    payload = os.environ.copy()
    payload.update(env)
    result = subprocess.run(
        ["bash", str(RUNNER)],
        cwd=str(ROOT),
        env=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout + result.stderr


def test_round_robin_gpu_mapping_synthetic() -> None:
    output = _run_runner(
        {
            "MODE": "synthetic",
            "DRY_RUN": "1",
            "PMX_SKIP_RUNTIME_GUARD": "1",
            "GPU_LIST": "0 1",
            "SHARD1": "AAPL,MSFT",
            "SHARD2": "MTN",
            "SYN_VALIDATE": "0",
        }
    )

    lines = [line for line in output.splitlines() if "[dry-run]" in line and "mode=synthetic" in line]
    assert len(lines) == 2

    mapping = {}
    for line in lines:
        parts = dict(part.split("=", 1) for part in line.split() if "=" in part)
        mapping[parts["shard"]] = parts["gpu"]

    assert mapping == {"AAPL,MSFT": "0", "MTN": "1"}


def test_trade_count_gate_skips_auto_trader_shards() -> None:
    output = _run_runner(
        {
            "MODE": "auto_trader",
            "DRY_RUN": "1",
            "PMX_SKIP_RUNTIME_GUARD": "1",
            "GPU_LIST": "0 1",
            "SHARD1": "AAPL",
            "SHARD2": "MSFT",
            "TARGET_TRADES": "30",
            "DRY_RUN_TRADE_COUNT": "30",
        }
    )

    assert "target met (30 >= 30); skipping" in output


def test_energy_defaults_logged_for_auto_trader() -> None:
    output = _run_runner(
        {
            "MODE": "auto_trader",
            "DRY_RUN": "1",
            "PMX_SKIP_RUNTIME_GUARD": "1",
            "GPU_LIST": "0",
            "SHARD1": "AAPL",
            "TARGET_TRADES": "30",
            "DRY_RUN_TRADE_COUNT": "0",
            "CYCLES": "2",
            "SLEEP_SECONDS": "7",
        }
    )

    assert "mode=auto_trader" in output
    assert "llm=0" in output
    assert "cycles=2" in output
    assert "sleep=7" in output


def test_ticker_shards_env_config_present() -> None:
    output = _run_runner(
        {
            "MODE": "synthetic",
            "DRY_RUN": "1",
            "PMX_SKIP_RUNTIME_GUARD": "1",
            "GPU_LIST": "0",
            "SHARD1": "AAPL,MSFT",
            "SHARD2": "CL=F",
            "SYN_VALIDATE": "0",
        }
    )

    assert "shard=AAPL,MSFT" in output
    assert "shard=CL=F" in output


def test_per_shard_db_mirror_stub_emitted(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    mirror_dir = tmp_path / "db_mirror"
    output = _run_runner(
        {
            "MODE": "auto_trader",
            "DRY_RUN": "1",
            "PMX_SKIP_RUNTIME_GUARD": "1",
            "GPU_LIST": "0",
            "SHARD1": "AAPL",
            "SHARD2": ",",
            "DRY_RUN_TRADE_COUNT": "0",
            "TELEMETRY_DIR": str(telemetry_dir),
            "SHARD_DB_MIRROR_DIR": str(mirror_dir),
        }
    )

    assert f"db_mirror={mirror_dir}" in output
    assert telemetry_dir.exists()
    assert mirror_dir.exists()


def test_energy_latency_telemetry_stub_written(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    output = _run_runner(
        {
            "MODE": "synthetic",
            "DRY_RUN": "1",
            "PMX_SKIP_RUNTIME_GUARD": "1",
            "GPU_LIST": "0",
            "SHARD1": "AAPL",
            "SHARD2": ",",
            "SYN_VALIDATE": "0",
            "TELEMETRY_DIR": str(telemetry_dir),
        }
    )

    telemetry_line = next(
        (
            line
            for line in output.splitlines()
            if "telemetry=" in line and "mode=synthetic" in line and "shard=AAPL" in line
        ),
        None,
    )
    assert telemetry_line is not None
    telemetry_path = telemetry_line.split("telemetry=", 1)[1].strip()
    path = Path(telemetry_path)
    assert path.exists()

    payload = json.loads(path.read_text())
    assert payload["mode"] == "synthetic"
    assert payload["run_label"]
    assert payload["shard"] == "AAPL"
    assert payload["energy_kwh"] is None
    assert payload["latency_ms"] is None
