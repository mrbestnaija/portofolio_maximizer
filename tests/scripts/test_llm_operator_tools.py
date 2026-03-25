from __future__ import annotations

import json
import subprocess
from pathlib import Path

import scripts.llm_multi_model_orchestrator as orch
import scripts.llm_operator_tools as tools


def _cp(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["git"], returncode=returncode, stdout=stdout, stderr=stderr)


def test_review_changed_files_reports_dirty_worktree(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    untracked = project_root / "notes.md"
    untracked.write_text("# note\nnew file\n", encoding="utf-8")

    def fake_run_git(_project_root: Path, args: list[str], *, timeout_seconds: float = 15.0):
        if args[:2] == ["status", "--porcelain=v1"]:
            return _cp(stdout=" M scripts/llm_multi_model_orchestrator.py\n?? notes.md\n")
        if args and args[0] == "diff":
            return _cp(stdout="diff --git a/scripts/llm_multi_model_orchestrator.py b/scripts/llm_multi_model_orchestrator.py\n+added line\n")
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr(tools, "_run_git", fake_run_git)

    payload = tools.review_changed_files(project_root=project_root, max_files=10, max_diff_chars=4000)

    assert payload["status"] == "PASS"
    assert payload["worktree_state"] == "dirty"
    assert payload["total_changed_files"] == 2
    assert "llm" in payload["tag_counts"]
    assert payload["untracked_previews"][0]["path"] == "notes.md"
    assert "added line" in payload["diff_preview"]


def test_summarize_test_failure_parses_pytest_output() -> None:
    output_text = """
=========================== short test summary info ===========================
FAILED tests/scripts/test_example.py::test_gate_path - AssertionError: expected PASS
ERROR tests/scripts/test_other.py::test_runtime - ValueError: broken config
E   AssertionError: expected PASS
E   ValueError: broken config
tests/scripts/test_example.py:42: AssertionError
tests/scripts/test_other.py:8: ValueError
2 failed, 10 passed in 1.23s
""".strip()

    payload = tools.summarize_test_failure(output_text=output_text)

    assert payload["status"] == "PASS"
    assert payload["total_failures"] == 2
    assert payload["failures"][0]["nodeid"] == "tests/scripts/test_example.py::test_gate_path"
    assert any(row["type"] == "AssertionError" for row in payload["error_types"])
    assert payload["summary_line"] == "2 failed, 10 passed in 1.23s"


def test_triage_gate_failure_generates_component_breakdown(tmp_path: Path) -> None:
    artifact = tmp_path / "production_gate_latest.json"
    artifact.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-03-09T21:31:07Z",
                "phase3_ready": False,
                "phase3_reason": "GATES_FAIL,THIN_LINKAGE,EVIDENCE_HYGIENE_FAIL",
                "inputs": {
                    "audit_dir": "logs/forecast_audits/production",
                    "max_files": 50,
                    "include_research": False,
                },
                "lift_gate": {
                    "pass": False,
                    "violation_rate": 0.72,
                    "max_violation_rate": 0.35,
                    "lift_fraction": 0.08,
                    "min_lift_fraction": 0.25,
                },
                "profitability_proof": {
                    "pass": False,
                    "is_proof_valid": True,
                    "is_profitable": False,
                    "profit_factor": 0.60,
                    "win_rate": 0.39,
                    "total_pnl": -986.14,
                    "closed_trades": 41,
                    "trading_days": 11,
                },
                "readiness": {
                    "gates_pass": False,
                    "linkage_pass": False,
                    "evidence_hygiene_pass": False,
                    "outcome_matched": 0,
                    "outcome_eligible": 1,
                    "matched_over_eligible": 0.0,
                    "non_trade_context_count": 33,
                    "invalid_context_count": 30,
                    "linkage_waterfall": {
                        "raw_candidates": 82,
                        "production_only": 49,
                        "linked": 1,
                        "hygiene_pass": 19,
                        "matched": 0,
                        "excluded_non_trade_context": 33,
                        "excluded_invalid_context": 30,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    summary = tmp_path / "latest_summary.json"
    summary.write_text(
        json.dumps(
            {
                "audit_dir": "logs/forecast_audits/production",
                "max_files": 50,
                "scope": {"include_research": False},
                "dataset_windows": [
                    {"outcome_status": "INVALID_CONTEXT", "outcome_reason": "MISSING_EXECUTION_METADATA"},
                    {"outcome_status": "NON_TRADE_CONTEXT", "outcome_reason": "MISSING_TICKER"},
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = tools.triage_gate_failure(
        gate_artifact=str(artifact),
        summary_cache=str(summary),
        output_json_path=str(tmp_path / "decomposition.json"),
        output_md_path=str(tmp_path / "decomposition.md"),
        force_refresh=True,
    )

    assert payload["status"] == "PASS"
    assert payload["gate_status"] == "FAIL"
    assert "PERFORMANCE_BLOCKER" in payload["failed_components"]
    invalid_reasons = {row["reason_code"] for row in payload["top_invalid_context_reasons"]}
    assert "MISSING_EXECUTION_METADATA" in invalid_reasons


def test_execute_tool_call_supports_operator_review_aliases(monkeypatch) -> None:
    monkeypatch.setattr(
        orch,
        "_review_changed_files_tool",
        lambda args, budget_seconds=None: json.dumps({"status": "PASS", "action": "review_changed_files", "arguments": args}),
    )
    payload = json.loads(orch.execute_tool_call("review_diff", {}))
    assert payload["action"] == "review_changed_files"


def test_gate_triage_fast_path_request_matches_prompt() -> None:
    tool_name, args = orch._extract_gate_triage_fast_path_request("Why did the production gate fail?") or ("", {})
    assert tool_name == "triage_gate_failure"
    assert args["force_refresh"] is False


def test_change_review_fast_path_matches_prompt() -> None:
    tool_name, args = orch._extract_review_changed_fast_path_request("Review current changes in scripts/llm_multi_model_orchestrator.py") or ("", {})
    assert tool_name == "review_changed_files"
    assert "scripts/llm_multi_model_orchestrator.py" in args["paths"]
