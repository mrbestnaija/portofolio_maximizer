from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from scripts import check_forecast_audits as audit_mod
from scripts import dashboard_db_bridge as bridge_mod


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "config" / "telemetry_contract.yml"
SCHEMA2_REQUIRED_IDS = ["TCON-01", "TCON-02", "TCON-03", "TCON-04", "TCON-05"]
SCHEMA3_REQUIRED_IDS = [
    "TCON-01",
    "TCON-02",
    "TCON-03",
    "TCON-04",
    "TCON-05",
    "TCON-06",
    "TCON-07",
    "TCON-08",
]
SCHEMA2_REQUIRED_WINDOW_COUNT_KEYS = [
    "n_raw_windows",
    "n_parseable_windows",
    "n_deduped_windows",
    "n_rmse_windows_processed",
    "n_rmse_windows_usable",
    "n_outcome_windows_eligible",
    "n_outcome_windows_matched",
]


def _load_contract() -> Dict[str, Any]:
    payload = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8")) or {}
    contract = payload.get("telemetry_contract")
    assert isinstance(contract, dict), "config/telemetry_contract.yml missing telemetry_contract root block"
    return contract


def _active_schema_definition(contract: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    current_version = int(contract.get("current_schema_version", 0))
    definitions = contract.get("schema_definitions")
    assert isinstance(definitions, dict), "schema_definitions must be a mapping"
    active = definitions.get(current_version, definitions.get(str(current_version)))
    assert isinstance(active, dict), f"schema_definitions missing active schema {current_version}"
    return current_version, active


def _path_exists(obj: Dict[str, Any], dotted_path: str) -> bool:
    cur: Any = obj
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def _write_minimal_audit(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset": {
                    "start": "2024-01-01",
                    "end": "2024-01-05",
                    "length": 180,
                    "forecast_horizon": 30,
                    "ticker": "AAPL",
                },
                "artifacts": {
                    "ensemble_weights": {"sarimax": 1.0},
                    "evaluation_metrics": {
                        "sarimax": {"rmse": 2.0},
                        "ensemble": {"rmse": 2.0},
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _write_minimal_monitoring_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "forecaster_monitoring:",
                "  regression_metrics:",
                "    baseline_model: BEST_SINGLE",
                "    holding_period_audits: 1",
                "    disable_ensemble_if_no_lift: false",
                "    max_rmse_ratio_vs_baseline: 1.1",
                "    max_violation_rate: 0.25",
            ]
        ),
        encoding="utf-8",
    )


def test_schema_version_policy_and_schema2_baseline_are_pinned() -> None:
    contract = _load_contract()
    current_version, active = _active_schema_definition(contract)
    assert current_version >= 2

    if current_version == 2:
        assert active.get("required_adversarial_ids") == SCHEMA2_REQUIRED_IDS
        assert active.get("required_window_count_keys") == SCHEMA2_REQUIRED_WINDOW_COUNT_KEYS
    if current_version == 3:
        assert active.get("required_adversarial_ids") == SCHEMA3_REQUIRED_IDS


def test_required_adversarial_ids_exist_in_runner_output() -> None:
    contract = _load_contract()
    _, active = _active_schema_definition(contract)
    required_ids = active.get("required_adversarial_ids") or []
    assert required_ids, "required_adversarial_ids must not be empty"

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "adversarial_diagnostic_runner.py"),
        "--json",
        "--severity",
        "LOW",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    assert proc.returncode in (0, 1), proc.stderr
    payload = json.loads(proc.stdout)
    ids = {str(item.get("id")) for item in payload.get("findings", []) if isinstance(item, dict)}
    for finding_id in required_ids:
        assert finding_id in ids, f"Missing adversarial finding id: {finding_id}"


def test_forecast_summary_emits_contract_fields_and_schema_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _load_contract()
    current_version, active = _active_schema_definition(contract)
    required_fields = active.get("required_contract_fields") or []
    required_window_keys = active.get("required_window_count_keys") or []
    forbidden_window_keys = set(active.get("forbidden_window_count_keys") or [])

    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    _write_minimal_audit(audit_dir / "forecast_audit_one.json")

    cfg = tmp_path / "forecaster_monitoring.yml"
    _write_minimal_monitoring_config(cfg)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_forecast_audits.py",
            "--audit-dir",
            str(audit_dir),
            "--config-path",
            str(cfg),
            "--max-files",
            "10",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        audit_mod.main()
    assert excinfo.value.code == 0

    summary_path = tmp_path / "logs" / "forecast_audits_cache" / "latest_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    for field_path in required_fields:
        assert _path_exists(summary, str(field_path)), f"Missing required telemetry field: {field_path}"

    contract_block = summary.get("telemetry_contract")
    assert isinstance(contract_block, dict)
    assert int(contract_block.get("schema_version")) == current_version

    window_counts = summary.get("window_counts")
    assert isinstance(window_counts, dict)
    for key in required_window_keys:
        assert key in window_counts, f"Missing window_counts key: {key}"
    for key in forbidden_window_keys:
        assert key not in window_counts, f"Forbidden legacy key present: {key}"


def test_dashboard_bridge_warns_on_legacy_telemetry_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    elig = tmp_path / "elig.json"
    ctx = tmp_path / "ctx.json"
    perf = tmp_path / "perf.json"
    summary = tmp_path / "summary.json"

    elig.write_text(json.dumps({"summary": {"HEALTHY": 1, "WEAK": 0, "LAB_ONLY": 0}, "tickers": {}}), encoding="utf-8")
    ctx.write_text(
        json.dumps(
            {
                "n_total_trades": 10,
                "n_trades_no_confidence": 0,
                "partial_data": False,
                "regime_quality": {"R1": {}},
                "confidence_bin_quality": {"0.70-0.75": {}},
            }
        ),
        encoding="utf-8",
    )
    perf.write_text(
        json.dumps(
            {
                "status": "OK",
                "warnings": [],
                "sufficiency": {"status": "SUFFICIENT"},
                "coverage_ratio": 0.8,
                "chart_paths": {},
            }
        ),
        encoding="utf-8",
    )
    summary.write_text(
        json.dumps(
            {
                "telemetry_contract": {"schema_version": 1, "outcomes_loaded": False, "outcome_join_attempted": False},
                "window_counts": {"n_raw_windows": 1},
                "window_diversity": {"regime_count": 1},
                "cache_status": {"write_ok": True},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(bridge_mod, "DEFAULT_ELIGIBILITY_PATH", elig)
    monkeypatch.setattr(bridge_mod, "DEFAULT_CONTEXT_QUALITY_PATH", ctx)
    monkeypatch.setattr(bridge_mod, "DEFAULT_PERFORMANCE_METRICS_PATH", perf)
    monkeypatch.setattr(bridge_mod, "DEFAULT_FORECAST_SUMMARY_PATH", summary)

    robustness = bridge_mod._robustness_payload()
    assert robustness["status"] == "WARN"
    assert "telemetry_contract_legacy_schema" in robustness["warnings"]
