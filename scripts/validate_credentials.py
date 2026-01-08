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

    # cTrader (live execution readiness)
    user_ok, user_line = _report_group(
        "ctrader_username",
        ["USERNAME_CTRADER", "CTRADER_USERNAME", "EMAIL_CTRADER", "CTRADER_EMAIL"],
    )
    pwd_ok, pwd_line = _report_group(
        "ctrader_password",
        ["PASSWORD_CTRADER", "CTRADER_PASSWORD"],
    )
    app_ok, app_line = _report_group(
        "ctrader_application_id",
        ["APPLICATION_NAME_CTRADER", "CTRADER_APPLICATION_ID", "CTRADER_APP_ID"],
    )
    lines.extend([user_line, pwd_line, app_line])

    ctrader_ok = user_ok and pwd_ok and app_ok
    ok = ok and ctrader_ok

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
