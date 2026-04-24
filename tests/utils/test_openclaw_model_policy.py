from __future__ import annotations

import json
from pathlib import Path

from utils.openclaw_model_policy import load_qwen35_policy


def test_load_qwen35_policy_approves_primary_when_benchmark_allows_it(tmp_path: Path) -> None:
    policy_path = tmp_path / "openclaw_model_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "preferred_primary": "qwen3.5:27b",
                "primary_allowed": True,
            }
        ),
        encoding="utf-8",
    )

    policy = load_qwen35_policy(base_dir=tmp_path, policy_path=policy_path)

    assert policy.preferred_primary == "qwen3.5:27b"
    assert policy.primary_allowed is True
    assert policy.fallback_allowed is True


def test_load_qwen35_policy_keeps_primary_disabled_without_qwen35_approval(tmp_path: Path) -> None:
    policy_path = tmp_path / "openclaw_model_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "preferred_primary": "qwen3:8b",
                "primary_allowed": True,
                "fallback_allowed": True,
            }
        ),
        encoding="utf-8",
    )

    policy = load_qwen35_policy(base_dir=tmp_path, policy_path=policy_path)

    assert policy.preferred_primary == "qwen3:8b"
    assert policy.primary_allowed is False
    assert policy.fallback_allowed is True


def test_load_qwen35_policy_allows_env_fallback_without_policy_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENCLAW_ALLOW_QWEN35_FALLBACK", "1")

    policy = load_qwen35_policy(base_dir=tmp_path, policy_path=tmp_path / "missing-policy.json")

    assert policy.primary_allowed is False
    assert policy.fallback_allowed is True
    assert policy.source.startswith("env")
