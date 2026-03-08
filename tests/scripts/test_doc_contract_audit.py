from __future__ import annotations

from pathlib import Path

from scripts import doc_contract_audit as mod


def test_validate_docs_reports_missing_header_fields(tmp_path: Path) -> None:
    doc = tmp_path / "Documentation" / "sample.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("# Sample\n\nOwner: Agent C\n", encoding="utf-8")

    issues = mod.validate_docs(root=tmp_path, docs=["Documentation/sample.md"])

    messages = {issue.message for issue in issues}
    assert "missing header field 'Doc Type:'" in messages
    assert "missing header field 'Last Verified:'" in messages


def test_validate_docs_passes_with_complete_header(tmp_path: Path) -> None:
    doc = tmp_path / "Documentation" / "sample.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(
        "\n".join(
            [
                "# Sample",
                "",
                "Doc Type: handoff_note",
                "Authority: temporary coordination doc",
                "Owner: Agent C",
                "Last Verified: 2026-03-08",
                "Verification Commands:",
                "Artifacts:",
                "Supersedes: none",
                "Expires When: superseded",
                "",
                "Body",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    issues = mod.validate_docs(root=tmp_path, docs=["Documentation/sample.md"])

    assert issues == []
