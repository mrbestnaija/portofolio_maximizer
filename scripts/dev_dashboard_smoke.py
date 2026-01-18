#!/usr/bin/env python3
"""
Minimal local wiring smoke test for the live dashboard.

Checks:
- Dashboard HTML exists and is configured for real-time polling (no demo payload).
- Dashboard schema + sample JSON parse.

This is intentionally offline and does not start a web server.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_static_files() -> None:
    html_path = ROOT / "visualizations" / "live_dashboard.html"
    schema_path = ROOT / "visualizations" / "dashboard_data.schema.json"
    sample_path = ROOT / "visualizations" / "dashboard_data.sample.json"

    html = html_path.read_text(encoding="utf-8")
    if "fetch('dashboard_data.json?_=' + Date.now())" not in html:
        raise SystemExit(
            "FAIL: live_dashboard.html is not configured to poll dashboard_data.json with cache busting."
        )
    if "return null;" not in html:
        raise SystemExit(
            "FAIL: live_dashboard.html does not use a null/empty-state fallback on missing data."
        )
    if "run_id\\\": \\\"demo\\\"" in html or "T-3" in html:
        raise SystemExit(
            "FAIL: live_dashboard.html still contains demo/fictitious payload content."
        )

    _load_json(schema_path)
    _load_json(sample_path)


def _serve_check(port: int) -> None:
    cmd = [
        "python3",
        "-m",
        "http.server",
        str(port),
        "--directory",
        str(ROOT),
        "--bind",
        "127.0.0.1",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(ROOT),
    )
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 3.0
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"{base}/visualizations/live_dashboard.html", timeout=1.0
                ) as resp:
                    if resp.status == 200:
                        break
            except Exception as exc:  # pragma: no cover - timing dependent
                last_err = exc
                time.sleep(0.1)
        else:
            raise SystemExit(f"FAIL: could not reach http.server on {base} ({last_err})")

        for path in (
            "visualizations/live_dashboard.html",
            "visualizations/dashboard_data.sample.json",
            "visualizations/dashboard_data.schema.json",
        ):
            with urllib.request.urlopen(f"{base}/{path}", timeout=1.0) as resp:
                if resp.status != 200:
                    raise SystemExit(f"FAIL: {path} did not return 200 (got {resp.status})")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:  # pragma: no cover
            proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dashboard wiring smoke test.")
    parser.add_argument(
        "--serve-check",
        action="store_true",
        help="Start a temporary local http.server and verify assets load over HTTP.",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port for --serve-check (default: 8000).")
    args = parser.parse_args()

    _check_static_files()
    if args.serve_check:
        _serve_check(args.port)

    print("OK: dashboard wiring looks sane")


if __name__ == "__main__":
    main()
