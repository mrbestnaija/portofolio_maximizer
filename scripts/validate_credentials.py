#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from etl.secret_loader import load_secret


_PLACEHOLDER_SECRETS = {
    "your_interactions_api_key_here",
    "your_api_key_here",
    "changeme",
    "change_me",
    "replace_me",
    "replace-me",
    "todo",
}


def _looks_like_placeholder_secret(value: str) -> bool:
    v = (value or "").strip().lower()
    if not v:
        return True
    if v in _PLACEHOLDER_SECRETS:
        return True
    if v.startswith("your_") and v.endswith("_here"):
        return True
    return False


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


def _first_env_present(keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if (os.getenv(key) or "").strip():
            return key
    return None


def _all_env_present(keys: Iterable[str]) -> bool:
    for key in keys:
        if not (os.getenv(key) or "").strip():
            return False
    return True


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
    openai_ok, openai_line = _report_group("openai_api_key", ["OPENAI_API_KEY", "OPENAI_SECRET_KEY"])
    lines.append(openai_line)
    anthropic_ok, anthropic_line = _report_group("anthropic_api_key", ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"])
    lines.append(anthropic_line)
    qwen_ok, qwen_line = _report_group("qwen_api_key", ["DASHSCOPE_API_KEY", "QWEN_API_KEY", "QWEN_PASSWORD"])
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

    # Interactions API (optional)
    interactions_key = (load_secret("INTERACTIONS_API_KEY") or "").strip()
    if not interactions_key:
        lines.append("[WARN] interactions_api_key: missing (INTERACTIONS_API_KEY)")
    elif _looks_like_placeholder_secret(interactions_key):
        lines.append("[WARN] interactions_api_key: present but looks like template placeholder (INTERACTIONS_API_KEY)")
    elif len(interactions_key) < 16:
        lines.append("[WARN] interactions_api_key: present but weak (<16 chars) (INTERACTIONS_API_KEY)")
    else:
        lines.append("[OK] interactions_api_key: present (INTERACTIONS_API_KEY)")

    interactions_url_key = _first_env_present(["INTERACTIONS_ENDPOINT_URL", "INTERACTIONS_URL"])
    verify_url_key = _first_env_present(["LINKED_ROLES_VERIFICATION_URL", "VERIFY_ROLES_URL"])
    lines.append(
        f"[OK] interactions_endpoint_url: present ({interactions_url_key})"
        if interactions_url_key
        else "[WARN] interactions_endpoint_url: missing (INTERACTIONS_ENDPOINT_URL / INTERACTIONS_URL)"
    )
    lines.append(
        f"[OK] linked_roles_verification_url: present ({verify_url_key})"
        if verify_url_key
        else "[WARN] linked_roles_verification_url: missing (LINKED_ROLES_VERIFICATION_URL / VERIFY_ROLES_URL)"
    )

    auth0_domain_key = _first_env_present(["AUTH0_DOMAIN", "AUTH0_TENANT_DOMAIN"])
    auth0_audience_key = _first_env_present(["AUTH0_AUDIENCE", "AUTH0_API_AUDIENCE"])
    lines.append(
        "[OK] auth0_jwt: configured "
        f"({auth0_domain_key} / {auth0_audience_key})"
        if (auth0_domain_key and auth0_audience_key)
        else "[WARN] auth0_jwt: missing "
        "(AUTH0_DOMAIN / AUTH0_AUDIENCE or AUTH0_TENANT_DOMAIN / AUTH0_API_AUDIENCE)"
    )

    tos_key = _first_env_present(["TERMS_OF_SERVICE_URL", "TERMS_URL"])
    pp_key = _first_env_present(["PRIVACY_POLICY_URL", "PRIVACY_URL"])
    lines.append(
        f"[OK] terms_of_service_url: present ({tos_key})"
        if tos_key
        else "[WARN] terms_of_service_url: missing (TERMS_OF_SERVICE_URL / TERMS_URL)"
    )
    lines.append(
        f"[OK] privacy_policy_url: present ({pp_key})"
        if pp_key
        else "[WARN] privacy_policy_url: missing (PRIVACY_POLICY_URL / PRIVACY_URL)"
    )

    discord_app_keys = [
        "DISCORD_APP_NAME",
        "DISCORD_APPLICATION_ID",
        "DISCORD_PUBLIC_KEY",
        "DISCORD_APP_INSTALL_LINK",
    ]
    discord_app_ok = _all_env_present(discord_app_keys)
    interactions_for_discord_ok = bool((load_secret("INTERACTIONS_API_KEY") or "").strip())
    lines.append(
        "[OK] discord_interactions_app: configured "
        "(DISCORD_APP_NAME / DISCORD_APPLICATION_ID / DISCORD_PUBLIC_KEY / DISCORD_APP_INSTALL_LINK + INTERACTIONS_API_KEY)"
        if (discord_app_ok and interactions_for_discord_ok)
        else "[WARN] discord_interactions_app: partial/missing "
        "(DISCORD_APP_NAME / DISCORD_APPLICATION_ID / DISCORD_PUBLIC_KEY / DISCORD_APP_INSTALL_LINK + INTERACTIONS_API_KEY)"
    )

    # OpenClaw optional remote channels (optional)
    tg_ok, tg_line = _report_group("telegram_bot_token", ["TELEGRAM_BOT_TOKEN"])
    dc_ok, dc_line = _report_group("discord_bot_token", ["DISCORD_BOT_TOKEN", "DISCORD_TOKEN"])
    sb_ok, sb_line = _report_group("slack_bot_token", ["SLACK_BOT_TOKEN", "SLACK_TOKEN"])
    sa_ok, sa_line = _report_group("slack_app_token", ["SLACK_APP_TOKEN", "SLACK_SOCKET_MODE_TOKEN"])
    lines.extend([tg_line, dc_line, sb_line, sa_line])

    # Email alerts (optional, Gmail supported)
    email_user_ok, email_user_line = _report_group("email_username", ["PMX_EMAIL_USERNAME", "MAIN_EMAIL_GMAIL", "OPENAI_EMAIL"])
    email_pwd_ok, email_pwd_line = _report_group("email_password", ["PMX_EMAIL_PASSWORD", "OPENAI_EMAIL_PASSWORD"])
    email_to_ok, email_to_line = _report_group(
        "email_to",
        ["PMX_EMAIL_TO", "PMX_EMAIL_RECIPIENTS", "MAIN_EMAIL_GMAIL", "ALTERNATIVE_EMAIL_PROTONMAIL"],
    )
    lines.extend([email_user_line, email_pwd_line, email_to_line])

    # Inbox workflows (optional) - Proton Mail Bridge
    proton_user_ok, proton_user_line = _report_group(
        "proton_bridge_username",
        ["PMX_PROTON_BRIDGE_USERNAME", "ALTERNATIVE_EMAIL_PROTONMAIL"],
    )
    proton_pwd_ok, proton_pwd_line = _report_group("proton_bridge_password", ["PMX_PROTON_BRIDGE_PASSWORD"])
    lines.extend([proton_user_line, proton_pwd_line])

    # GitHub automation (optional)
    gh_ok, gh_line = _report_group("github_projects_token", ["PROJECTS_TOKEN", "PROJECTS_SECRET"])
    lines.append(gh_line)

    print("Credential presence check (values are never printed)")
    for line in lines:
        print(line)

    if strict:
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
