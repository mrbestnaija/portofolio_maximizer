from __future__ import annotations

from pathlib import Path

import pytest

from utils import email_inbox


def test_extract_uid_from_fetch_meta() -> None:
    assert email_inbox._extract_uid(b"1 (UID 12345 BODY[HEADER] {342}") == "12345"


def test_decode_mime_header_plain() -> None:
    assert email_inbox._decode_mime_header("Hello") == "Hello"


def test_truncate_adds_ellipsis() -> None:
    assert email_inbox._truncate("abcdef", 4) == "a..."


def test_scan_account_disabled_is_noop(tmp_path: Path) -> None:
    config = {
        "accounts": {
            "gmail": {
                "enabled": False,
                "label": "Gmail",
            }
        }
    }
    res = email_inbox.scan_account(project_root=tmp_path, config=config, account_name="gmail")
    assert res.account == "gmail"
    assert res.fetched == 0
    assert res.total_matched == 0
    assert res.messages == []


def test_send_email_blocked_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("PMX_INBOX_ALLOW_SEND", raising=False)

    config = {
        "limits": {"allow_send": False},
        "accounts": {"gmail": {"enabled": True, "smtp": {}, "secrets": {}}},
    }

    with pytest.raises(PermissionError):
        email_inbox.send_email(
            project_root=tmp_path,
            config=config,
            account_name="gmail",
            to_addresses=["a@example.com"],
            subject="s",
            body="b",
        )


def test_send_email_uses_smtp_connect(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PMX_INBOX_ALLOW_SEND", "1")
    monkeypatch.setenv("PMX_EMAIL_USERNAME", "user@example.com")
    monkeypatch.setenv("PMX_EMAIL_PASSWORD", "password123456")
    monkeypatch.setenv("PMX_EMAIL_FROM", "user@example.com")

    sent = {}

    class DummySMTP:
        def __init__(self) -> None:
            self.logged_in = False

        def login(self, username: str, password: str) -> None:
            assert username == "user@example.com"
            assert password == "password123456"
            self.logged_in = True

        def send_message(self, msg) -> None:  # noqa: ANN001
            assert self.logged_in is True
            sent["from"] = msg["From"]
            sent["to"] = msg["To"]
            sent["subject"] = msg["Subject"]
            sent["body"] = msg.get_content()

        def quit(self) -> None:
            return

    def fake_smtp_connect(smtp_cfg, *, timeout_seconds: float):  # noqa: ANN001
        return DummySMTP()

    monkeypatch.setattr(email_inbox, "_smtp_connect", fake_smtp_connect)

    config = {
        "limits": {"allow_send": False},
        "accounts": {
            "gmail": {
                "enabled": True,
                "smtp": {"host": "smtp.example.com", "port": 587, "ssl": False, "starttls": True},
                "secrets": {
                    "username_env": "PMX_EMAIL_USERNAME",
                    "password_env": "PMX_EMAIL_PASSWORD",
                    "from_env": "PMX_EMAIL_FROM",
                },
            }
        },
    }

    email_inbox.send_email(
        project_root=tmp_path,
        config=config,
        account_name="gmail",
        to_addresses=["recipient@example.com"],
        subject="Test",
        body="Hello",
        timeout_seconds=0.1,
    )

    assert sent["from"] == "user@example.com"
    assert sent["to"] == "recipient@example.com"
    assert sent["subject"] == "Test"
    assert "Hello" in sent["body"]

