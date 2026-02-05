#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys


def _open_in_windows_default_browser(url: str) -> bool:
    powershell = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    if not os.path.exists(powershell):
        return False

    safe_url = url.replace("'", "''")
    cmd = [powershell, "-NoProfile", "-Command", f"Start-Process '{safe_url}'"]
    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )
    return True


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        return 0

    url = argv[1]
    if _open_in_windows_default_browser(url):
        return 0

    # Fallback for non-WSL environments.
    subprocess.Popen(
        ["xdg-open", url],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
