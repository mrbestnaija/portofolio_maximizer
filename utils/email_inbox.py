"""
Email inbox workflows (IMAP read, SMTP send).

Design goals:
- Keep credentials local (env / *_FILE via etl/secret_loader.py).
- Default to safe limits (small scans, no body fetch, no send).
- Be usable both by humans and automation (OpenClaw can call these scripts).
"""

from __future__ import annotations

import imaplib
import os
import re
import ssl
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email import message_from_bytes
from email.header import decode_header, make_header
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import smtplib
import yaml


_UID_RE = re.compile(r"\bUID\s+(\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class InboxMessage:
    seq: str
    uid: str
    from_address: str
    to_address: str
    subject: str
    date: str
    message_id: str
    snippet: str


@dataclass(frozen=True)
class InboxScanResult:
    account: str
    label: str
    mailbox: str
    search_criteria: List[str]
    total_matched: int
    fetched: int
    timestamp_utc: str
    messages: List[InboxMessage]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _truthy_env(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _decode_mime_header(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    try:
        return str(make_header(decode_header(raw)))
    except Exception:
        return raw


def _extract_uid(meta: bytes) -> str:
    """
    Extract UID from an IMAP FETCH response metadata blob.

    Example meta:
      b'1 (UID 12345 BODY[HEADER] {342}'
    """
    if not meta:
        return ""
    try:
        text = meta.decode("ascii", errors="ignore")
    except Exception:
        return ""
    match = _UID_RE.search(text)
    return match.group(1) if match else ""


def _truncate(text: str, max_chars: int) -> str:
    s = (text or "").strip()
    if max_chars <= 0:
        return ""
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 3)].rstrip() + "..."


def _resolve_config_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_inbox_config(*, project_root: Path, config_path: str = "config/inbox_workflows.yml") -> Dict[str, Any]:
    path = _resolve_config_path(project_root, config_path)
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except FileNotFoundError:
        return {}


def _load_secret_by_env_name(env_name: str) -> str:
    name = (env_name or "").strip()
    if not name:
        return ""
    try:
        from etl.secret_loader import load_secret

        return (load_secret(name) or "").strip()
    except Exception:
        return (os.getenv(name) or "").strip()


def _imap_connect(imap_cfg: Dict[str, Any], *, timeout_seconds: float) -> imaplib.IMAP4:
    host = (imap_cfg.get("host") or "").strip()
    port = int(imap_cfg.get("port") or 993)
    use_ssl = bool(imap_cfg.get("ssl", True))
    use_starttls = bool(imap_cfg.get("starttls", False))

    if use_ssl:
        ctx = ssl.create_default_context()
        return imaplib.IMAP4_SSL(host, port, ssl_context=ctx, timeout=timeout_seconds)

    client = imaplib.IMAP4(host, port, timeout=timeout_seconds)
    if use_starttls:
        ctx = ssl.create_default_context()
        client.starttls(ssl_context=ctx)
    return client


def _smtp_connect(smtp_cfg: Dict[str, Any], *, timeout_seconds: float) -> smtplib.SMTP:
    host = (smtp_cfg.get("host") or "").strip()
    port = int(smtp_cfg.get("port") or 587)
    use_ssl = bool(smtp_cfg.get("ssl", False))
    use_starttls = bool(smtp_cfg.get("starttls", True))

    if use_ssl:
        ctx = ssl.create_default_context()
        return smtplib.SMTP_SSL(host, port, timeout=timeout_seconds, context=ctx)

    client = smtplib.SMTP(host, port, timeout=timeout_seconds)
    client.ehlo()
    if use_starttls:
        ctx = ssl.create_default_context()
        client.starttls(context=ctx)
        client.ehlo()
    return client


def _imap_fetch_header_and_uid(client: imaplib.IMAP4, seq: str) -> Tuple[str, bytes]:
    typ, data = client.fetch(str(seq), "(UID BODY.PEEK[HEADER])")
    if typ != "OK" or not data:
        return "", b""

    # `data` can contain multiple items; find the first tuple(meta, payload).
    for item in data:
        if isinstance(item, tuple) and len(item) >= 2:
            meta = item[0] if isinstance(item[0], (bytes, bytearray)) else b""
            payload = item[1] if isinstance(item[1], (bytes, bytearray)) else b""
            uid = _extract_uid(meta)
            return uid, payload
    return "", b""


def _imap_fetch_body_snippet(
    client: imaplib.IMAP4,
    seq: str,
    *,
    max_chars: int,
    max_bytes: int = 4096,
) -> str:
    if max_chars <= 0:
        return ""

    # Request a small prefix of the TEXT section to avoid fetching large bodies/attachments.
    # Not all servers support partial fetch; failure is non-fatal.
    try:
        typ, data = client.fetch(str(seq), f"(BODY.PEEK[TEXT]<0.{int(max_bytes)}>)")
    except Exception:
        return ""

    if typ != "OK" or not data:
        return ""

    chunk = b""
    for item in data:
        if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], (bytes, bytearray)):
            chunk = bytes(item[1])
            break

    if not chunk:
        return ""

    try:
        text = chunk.decode("utf-8", errors="ignore")
    except Exception:
        return ""

    # Collapse extreme whitespace; keep it readable.
    squashed = " ".join(text.split())
    return _truncate(squashed, max_chars)


def scan_account(
    *,
    project_root: Path,
    config: Dict[str, Any],
    account_name: str,
    timeout_seconds: float = 20.0,
) -> InboxScanResult:
    accounts = config.get("accounts") if isinstance(config, dict) else None
    if not isinstance(accounts, dict) or account_name not in accounts:
        raise KeyError(f"Unknown account: {account_name}")

    account_cfg = accounts.get(account_name) or {}
    if not isinstance(account_cfg, dict):
        raise ValueError(f"Invalid account config for {account_name}")

    label = str(account_cfg.get("label") or account_name)
    enabled = bool(account_cfg.get("enabled", False))
    if not enabled:
        return InboxScanResult(
            account=account_name,
            label=label,
            mailbox="",
            search_criteria=[],
            total_matched=0,
            fetched=0,
            timestamp_utc=_utc_now_iso(),
            messages=[],
        )

    caps = account_cfg.get("capabilities") if isinstance(account_cfg, dict) else None
    caps = caps if isinstance(caps, dict) else {}
    if not bool(caps.get("read", True)):
        return InboxScanResult(
            account=account_name,
            label=label,
            mailbox="",
            search_criteria=[],
            total_matched=0,
            fetched=0,
            timestamp_utc=_utc_now_iso(),
            messages=[],
        )

    scan_cfg = config.get("scan") if isinstance(config, dict) else None
    scan_cfg = scan_cfg if isinstance(scan_cfg, dict) else {}
    mailbox = str(scan_cfg.get("mailbox") or "INBOX")
    criteria = scan_cfg.get("search_criteria")
    if isinstance(criteria, list) and all(isinstance(v, str) for v in criteria):
        search_criteria = [v for v in criteria if v.strip()]
    else:
        search_criteria = ["UNSEEN"]

    limits_cfg = config.get("limits") if isinstance(config, dict) else None
    limits_cfg = limits_cfg if isinstance(limits_cfg, dict) else {}
    max_messages = int(limits_cfg.get("max_messages_per_account") or 10)
    include_snippet = bool(limits_cfg.get("include_body_snippet", False))
    max_snippet_chars = int(limits_cfg.get("max_snippet_chars") or 400)
    mark_seen = bool(limits_cfg.get("mark_seen", False))

    imap_cfg = account_cfg.get("imap") if isinstance(account_cfg, dict) else None
    imap_cfg = imap_cfg if isinstance(imap_cfg, dict) else {}
    secrets_cfg = account_cfg.get("secrets") if isinstance(account_cfg, dict) else None
    secrets_cfg = secrets_cfg if isinstance(secrets_cfg, dict) else {}

    username_env = str(secrets_cfg.get("username_env") or "").strip()
    password_env = str(secrets_cfg.get("password_env") or "").strip()
    username = _load_secret_by_env_name(username_env)
    password = _load_secret_by_env_name(password_env)
    if not username or not password:
        raise RuntimeError(
            f"Missing credentials for {account_name} (set {username_env} / {password_env})."
        )

    messages: List[InboxMessage] = []
    client: Optional[imaplib.IMAP4] = None
    try:
        client = _imap_connect(imap_cfg, timeout_seconds=timeout_seconds)
        client.login(username, password)
        client.select(mailbox, readonly=not mark_seen)
        typ, data = client.search(None, *search_criteria)
        if typ != "OK" or not data:
            return InboxScanResult(
                account=account_name,
                label=label,
                mailbox=mailbox,
                search_criteria=search_criteria,
                total_matched=0,
                fetched=0,
                timestamp_utc=_utc_now_iso(),
                messages=[],
            )

        raw_ids = (data[0] or b"").split()
        total = len(raw_ids)

        # Fetch newest N by taking the tail.
        if max_messages > 0:
            raw_ids = raw_ids[-max_messages:]

        for raw_seq in raw_ids:
            seq = raw_seq.decode("ascii", errors="ignore") if isinstance(raw_seq, (bytes, bytearray)) else str(raw_seq)
            uid, header_bytes = _imap_fetch_header_and_uid(client, seq)
            msg = message_from_bytes(header_bytes or b"")

            from_address = _decode_mime_header(str(msg.get("From", "")))
            to_address = _decode_mime_header(str(msg.get("To", "")))
            subject = _decode_mime_header(str(msg.get("Subject", "")))
            date = _decode_mime_header(str(msg.get("Date", "")))
            message_id = _decode_mime_header(str(msg.get("Message-Id", "")))

            snippet = ""
            if include_snippet:
                snippet = _imap_fetch_body_snippet(client, seq, max_chars=max_snippet_chars)

            if mark_seen:
                try:
                    # IMAP system flag is "\Seen" (single backslash).
                    client.store(seq, "+FLAGS", "\\Seen")
                except Exception:
                    pass

            messages.append(
                InboxMessage(
                    seq=seq,
                    uid=uid,
                    from_address=_truncate(from_address, 200),
                    to_address=_truncate(to_address, 200),
                    subject=_truncate(subject, 200),
                    date=_truncate(date, 100),
                    message_id=_truncate(message_id, 200),
                    snippet=snippet,
                )
            )
    finally:
        try:
            if client is not None:
                client.logout()
        except Exception:
            pass

    return InboxScanResult(
        account=account_name,
        label=label,
        mailbox=mailbox,
        search_criteria=search_criteria,
        total_matched=total,
        fetched=len(messages),
        timestamp_utc=_utc_now_iso(),
        messages=messages,
    )


def send_email(
    *,
    project_root: Path,
    config: Dict[str, Any],
    account_name: str,
    to_addresses: Sequence[str],
    subject: str,
    body: str,
    timeout_seconds: float = 20.0,
    in_reply_to: str = "",
    references: str = "",
) -> None:
    limits_cfg = config.get("limits") if isinstance(config, dict) else None
    limits_cfg = limits_cfg if isinstance(limits_cfg, dict) else {}
    allow_send_cfg = bool(limits_cfg.get("allow_send", False))
    allow_send_env = _truthy_env("PMX_INBOX_ALLOW_SEND")
    if not (allow_send_cfg or allow_send_env):
        raise PermissionError(
            "Send is disabled by default. Set PMX_INBOX_ALLOW_SEND=1 or config.limits.allow_send=true."
        )

    accounts = config.get("accounts") if isinstance(config, dict) else None
    if not isinstance(accounts, dict) or account_name not in accounts:
        raise KeyError(f"Unknown account: {account_name}")

    account_cfg = accounts.get(account_name) or {}
    if not isinstance(account_cfg, dict) or not bool(account_cfg.get("enabled", False)):
        raise RuntimeError(f"Account is disabled: {account_name}")

    caps = account_cfg.get("capabilities") if isinstance(account_cfg, dict) else None
    caps = caps if isinstance(caps, dict) else {}
    if not bool(caps.get("send", True)):
        raise PermissionError(f"Sending is disabled for account {account_name} (capabilities.send=false).")

    smtp_cfg = account_cfg.get("smtp") if isinstance(account_cfg, dict) else None
    smtp_cfg = smtp_cfg if isinstance(smtp_cfg, dict) else {}
    secrets_cfg = account_cfg.get("secrets") if isinstance(account_cfg, dict) else None
    secrets_cfg = secrets_cfg if isinstance(secrets_cfg, dict) else {}

    username_env = str(secrets_cfg.get("username_env") or "").strip()
    password_env = str(secrets_cfg.get("password_env") or "").strip()
    from_env = str(secrets_cfg.get("from_env") or "").strip()

    username = _load_secret_by_env_name(username_env)
    password = _load_secret_by_env_name(password_env)
    from_address = _load_secret_by_env_name(from_env) or ""
    if not username or not password:
        raise RuntimeError(
            f"Missing credentials for {account_name} (set {username_env} / {password_env})."
        )
    if not from_address:
        from_address = username

    recipients = [str(v).strip() for v in to_addresses if str(v).strip()]
    if not recipients:
        raise ValueError("Missing recipient(s).")

    msg = EmailMessage()
    msg["From"] = from_address
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = str(subject or "").strip()
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
    if references:
        msg["References"] = references
    msg.set_content(str(body or "").rstrip() + "\n")

    smtp: Optional[smtplib.SMTP] = None
    try:
        smtp = _smtp_connect(smtp_cfg, timeout_seconds=timeout_seconds)
        smtp.login(username, password)
        smtp.send_message(msg)
    finally:
        try:
            if smtp is not None:
                smtp.quit()
        except Exception:
            pass


def fetch_message_eml(
    *,
    project_root: Path,
    config: Dict[str, Any],
    account_name: str,
    uid: str = "",
    seq: str = "",
    timeout_seconds: float = 20.0,
    mailbox: str = "INBOX",
) -> bytes:
    """
    Fetch a full message as RFC822/EML bytes (read-only, does not mark seen).

    Identify the message either by:
    - IMAP UID (`uid=...`) preferred, or
    - IMAP sequence number (`seq=...`)
    """
    accounts = config.get("accounts") if isinstance(config, dict) else None
    if not isinstance(accounts, dict) or account_name not in accounts:
        raise KeyError(f"Unknown account: {account_name}")

    account_cfg = accounts.get(account_name) or {}
    if not isinstance(account_cfg, dict) or not bool(account_cfg.get("enabled", False)):
        raise RuntimeError(f"Account is disabled: {account_name}")

    caps = account_cfg.get("capabilities") if isinstance(account_cfg, dict) else None
    caps = caps if isinstance(caps, dict) else {}
    if not bool(caps.get("read", True)):
        raise PermissionError(f"Reading is disabled for account {account_name} (capabilities.read=false).")

    imap_cfg = account_cfg.get("imap") if isinstance(account_cfg, dict) else None
    imap_cfg = imap_cfg if isinstance(imap_cfg, dict) else {}
    secrets_cfg = account_cfg.get("secrets") if isinstance(account_cfg, dict) else None
    secrets_cfg = secrets_cfg if isinstance(secrets_cfg, dict) else {}

    username_env = str(secrets_cfg.get("username_env") or "").strip()
    password_env = str(secrets_cfg.get("password_env") or "").strip()
    username = _load_secret_by_env_name(username_env)
    password = _load_secret_by_env_name(password_env)
    if not username or not password:
        raise RuntimeError(
            f"Missing credentials for {account_name} (set {username_env} / {password_env})."
        )

    uid = (uid or "").strip()
    seq = (seq or "").strip()
    if not uid and not seq:
        raise ValueError("Provide uid or seq.")

    client: Optional[imaplib.IMAP4] = None
    try:
        client = _imap_connect(imap_cfg, timeout_seconds=timeout_seconds)
        client.login(username, password)
        client.select(str(mailbox or "INBOX"), readonly=True)

        if uid:
            typ, data = client.uid("FETCH", uid, "(BODY.PEEK[])")
        else:
            typ, data = client.fetch(seq, "(BODY.PEEK[])")

        if typ != "OK" or not data:
            raise RuntimeError("IMAP fetch failed.")

        for item in data:
            if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], (bytes, bytearray)):
                return bytes(item[1])
        raise RuntimeError("IMAP fetch returned no payload.")
    finally:
        try:
            if client is not None:
                client.logout()
        except Exception:
            pass


def scan_result_to_dict(result: InboxScanResult) -> Dict[str, Any]:
    payload = asdict(result)
    payload["messages"] = [asdict(m) for m in result.messages]
    return payload
