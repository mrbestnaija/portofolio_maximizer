from __future__ import annotations

import os

import pytest

from tools import secrets_guard


def test_mask_url_userinfo_no_userinfo() -> None:
    url = "https://github.com/org/repo.git"
    assert secrets_guard._mask_url_userinfo(url) == url


def test_mask_url_userinfo_token_userinfo() -> None:
    url = "https://token@github.com/org/repo.git"
    assert secrets_guard._mask_url_userinfo(url) == "https://***@github.com/org/repo.git"


def test_mask_url_userinfo_user_pass_userinfo() -> None:
    url = "https://user:token@github.com/org/repo.git"
    assert secrets_guard._mask_url_userinfo(url) == "https://***@github.com/org/repo.git"


def test_looks_like_placeholder() -> None:
    assert secrets_guard._looks_like_placeholder("")
    assert secrets_guard._looks_like_placeholder("   ")
    assert secrets_guard._looks_like_placeholder("your_api_key_here")
    assert secrets_guard._looks_like_placeholder("<folder_id>")
    assert not secrets_guard._looks_like_placeholder("MyRealValue123")


def test_parse_added_lines_from_diff_tracks_path() -> None:
    diff = "\n".join(
        [
            "diff --git a/foo.txt b/foo.txt",
            "index 0000000..1111111 100644",
            "--- a/foo.txt",
            "+++ b/foo.txt",
            "@@ -0,0 +1 @@",
            "+hello",
        ]
    )
    added = secrets_guard._parse_added_lines_from_diff(diff)
    assert added == [("foo.txt", "hello")]


def test_scan_line_detects_high_confidence_patterns_runtime_built() -> None:
    # Build token-like strings at runtime so they are not committed verbatim.
    gh_pat = "ghp_" + ("A" * 36)
    anth = "sk-ant-" + "api03-" + ("B" * 32)

    f1 = secrets_guard._scan_line_for_secrets("some_file.py", f"TOKEN={gh_pat}")
    assert any(x.rule == "github_pat_classic" and x.severity == "ERROR" for x in f1)

    f2 = secrets_guard._scan_line_for_secrets("some_file.py", f"ANTHROPIC_API_KEY={anth}")
    assert any(x.rule == "anthropic_key" and x.severity == "ERROR" for x in f2)


def test_scan_line_env_template_non_placeholder_sensitive_assignment_is_error() -> None:
    findings = secrets_guard._scan_line_for_secrets(".env.template", "OPENAI_API_KEY=MyRealValue123")
    assert any(x.rule == "suspicious_assignment" and x.severity == "ERROR" for x in findings)


def test_build_redactor_redacts_sensitive_env_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_PASSWORD", "SuperSecretValue123")
    monkeypatch.setenv("NOT_SENSITIVE", "SuperSecretValue123")  # should not be used for redaction

    redact = secrets_guard._build_redactor()
    out = redact("hello SuperSecretValue123 world")
    assert "SuperSecretValue123" not in out
    assert "***REDACTED***" in out

