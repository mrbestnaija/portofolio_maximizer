#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from etl.secret_loader import load_secret


def _first_present(keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if load_secret(key):
            return key
    return None


def _report_group(name: str, keys: List[str]) -> Tuple[bool, str]:
    found = _first_present(keys)
    if found:
        return True, f"[OK] {name}: present ({found})"
    return False, f"[WARN] {name}: missing ({' / '.join(keys)})"


def main(argv: List[str]) -> int:
    strict = "--strict" in argv

    ok = True
    lines: List[str] = []

    # Market data providers
    for label, keys in [
        ("alpha_vantage", ["ALPHA_VANTAGE_API_KEY"]),
        ("finnhub", ["FINNHUB_API_KEY"]),
    ]:
        group_ok, line = _report_group(label, keys)
        lines.append(line)
        ok = ok and group_ok

    # LLM providers (optional)
    openai_ok, openai_line = _report_group("openai_api_key", ["OPENAI_API_KEY"])
    lines.append(openai_line)
    anthropic_ok, anthropic_line = _report_group("anthropic_api_key", ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"])
    lines.append(anthropic_line)
    qwen_ok, qwen_line = _report_group("qwen_api_key", ["DASHSCOPE_API_KEY", "QWEN_API_KEY"])
    lines.append(qwen_line)

    # cTrader (live execution readiness)
    user_ok, user_line = _report_group(
        "ctrader_username",
        [
            "CTRADER_DEMO_USERNAME",
            "CTRADER_DEMO_EMAIL",
            "CTRADER_LIVE_USERNAME",
            "CTRADER_LIVE_EMAIL",
            "USERNAME_CTRADER",
            "CTRADER_USERNAME",
            "EMAIL_CTRADER",
            "CTRADER_EMAIL",
        ],
    )
    pwd_ok, pwd_line = _report_group(
        "ctrader_password",
        [
            "CTRADER_DEMO_PASSWORD",
            "CTRADER_LIVE_PASSWORD",
            "PASSWORD_CTRADER",
            "CTRADER_PASSWORD",
        ],
    )
    app_ok, app_line = _report_group(
        "ctrader_application_id",
        [
            "CTRADER_DEMO_APPLICATION_ID",
            "CTRADER_DEMO_APP_ID",
            "CTRADER_LIVE_APPLICATION_ID",
            "CTRADER_LIVE_APP_ID",
            "APPLICATION_NAME_CTRADER",
            "CTRADER_APPLICATION_ID",
            "CTRADER_APP_ID",
        ],
    )
    lines.extend([user_line, pwd_line, app_line])

    ctrader_ok = user_ok and pwd_ok and app_ok
    ok = ok and ctrader_ok

    # OpenClaw notifications (optional)
    oc_ok, oc_line = _report_group("openclaw_targets", ["OPENCLAW_TARGETS", "OPENCLAW_TO"])
    lines.append(oc_line)

    # OpenClaw optional remote channels (optional)
    tg_ok, tg_line = _report_group("telegram_bot_token", ["TELEGRAM_BOT_TOKEN"])
    dc_ok, dc_line = _report_group("discord_bot_token", ["DISCORD_BOT_TOKEN"])
    sb_ok, sb_line = _report_group("slack_bot_token", ["SLACK_BOT_TOKEN"])
    sa_ok, sa_line = _report_group("slack_app_token", ["SLACK_APP_TOKEN"])
    lines.extend([tg_line, dc_line, sb_line, sa_line])

    # Email alerts (optional, Gmail supported)
    email_user_ok, email_user_line = _report_group("email_username", ["PMX_EMAIL_USERNAME"])
    email_pwd_ok, email_pwd_line = _report_group("email_password", ["PMX_EMAIL_PASSWORD"])
    email_to_ok, email_to_line = _report_group("email_to", ["PMX_EMAIL_TO", "PMX_EMAIL_RECIPIENTS"])
    lines.extend([email_user_line, email_pwd_line, email_to_line])

    # Inbox workflows (optional) - Proton Mail Bridge
    proton_user_ok, proton_user_line = _report_group("proton_bridge_username", ["PMX_PROTON_BRIDGE_USERNAME"])
    proton_pwd_ok, proton_pwd_line = _report_group("proton_bridge_password", ["PMX_PROTON_BRIDGE_PASSWORD"])
    lines.extend([proton_user_line, proton_pwd_line])

    # GitHub automation (optional)
    gh_ok, gh_line = _report_group("github_projects_token", ["PROJECTS_TOKEN"])
    lines.append(gh_line)

    print("Credential presence check (values are never printed)")
    for line in lines:
        print(line)

    if strict:
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
