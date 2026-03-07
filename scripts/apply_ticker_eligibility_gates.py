"""
apply_ticker_eligibility_gates.py
----------------------------------

Applies per-ticker eligibility decisions (HEALTHY/WEAK/LAB_ONLY from
``compute_ticker_eligibility.py``) to the runtime gating layer.

Modes
-----
autonomous (default)
    Writes ``logs/ticker_gating/active_lab_only.json`` immediately.
    LAB_ONLY tickers picked up on the next signal-generator instantiation.
    This is the production default -- no human intervention required.

recommendation-only (--recommendation-only or --dry-run)
    Prints what would be applied; never writes any file.
    Use for manual review before committing to a gate change.

The runtime override file has a TTL (default 26 h). If the file is older
than the TTL, the signal generator treats it as expired and blocks no
tickers on the basis of this file alone.  This prevents stale gates from
silently persisting if the cron job that refreshes eligibility stops running.

Usage
-----
    # Autonomous (default) -- applies gates immediately
    python scripts/apply_ticker_eligibility_gates.py

    # Recommendation-only -- prints, does not write
    python scripts/apply_ticker_eligibility_gates.py --recommendation-only

    # Override eligibility input path
    python scripts/apply_ticker_eligibility_gates.py \\
        --eligibility logs/ticker_eligibility.json

Exit codes
----------
    0  success (file written in autonomous mode, or recommendations printed)
    1  error (eligibility file missing/unreadable, unexpected exception)
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)

DEFAULT_ELIGIBILITY_PATH = ROOT / "logs" / "ticker_eligibility.json"
DEFAULT_OVERRIDE_PATH = ROOT / "logs" / "ticker_gating" / "active_lab_only.json"
DEFAULT_TTL_HOURS = 26

# Signals that a ticker should be fully blocked in the router.
# min_expected_return of 1.0 (100%) is never reachable in practice.
LAB_ONLY_RETURN_FLOOR = 1.0


def _load_eligibility(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Eligibility file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse eligibility JSON at {path}: {exc}") from exc
    if not isinstance(data, dict) or "tickers" not in data:
        raise ValueError(f"Eligibility file missing 'tickers' key: {path}")
    return data


def _extract_lab_only(eligibility: dict[str, Any]) -> tuple[list[str], dict[str, list[str]]]:
    """Return (lab_only_tickers, reasons_map) from eligibility output."""
    tickers: dict[str, dict] = eligibility.get("tickers") or {}
    lab_only: list[str] = []
    reasons: dict[str, list[str]] = {}
    for ticker, info in tickers.items():
        if not isinstance(info, dict):
            continue
        if str(info.get("status", "")).upper() == "LAB_ONLY":
            lab_only.append(str(ticker).upper())
            reasons[str(ticker).upper()] = info.get("reasons") or []
    return sorted(lab_only), reasons


def build_override_payload(
    lab_only_tickers: list[str],
    reasons: dict[str, list[str]],
    mode: str,
    ttl_hours: int,
    eligibility_path: Path,
) -> dict[str, Any]:
    now = datetime.datetime.now(datetime.timezone.utc)
    expires = now + datetime.timedelta(hours=ttl_hours)
    return {
        "generated_utc": now.isoformat(),
        "expires_utc": expires.isoformat(),
        "ttl_hours": ttl_hours,
        "mode": mode,
        "lab_only_tickers": lab_only_tickers,
        "applied_reasons": reasons,
        "lab_only_return_floor": LAB_ONLY_RETURN_FLOOR,
        "source_eligibility": str(eligibility_path),
        "schema_version": 1,
    }


def write_override(payload: dict[str, Any], override_path: Path) -> None:
    """Atomically write the override file."""
    override_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, default=str)
    # Atomic write via tmp → replace
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=override_path.parent, prefix=".lab_only_", suffix=".tmp"
    )
    try:
        with open(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        Path(tmp_name).replace(override_path)
    except Exception:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass
        raise


def apply_eligibility_gates(
    eligibility_path: Path = DEFAULT_ELIGIBILITY_PATH,
    override_path: Path = DEFAULT_OVERRIDE_PATH,
    recommendation_only: bool = False,
    ttl_hours: int = DEFAULT_TTL_HOURS,
) -> dict[str, Any]:
    """
    Core logic — returns result dict regardless of mode.

    Parameters
    ----------
    eligibility_path:
        Path to the JSON produced by ``compute_ticker_eligibility.py``.
    override_path:
        Path where the runtime gate file will be written (autonomous mode).
    recommendation_only:
        If True, print recommendations only; never write the override file.
    ttl_hours:
        Hours before the written override expires.

    Returns
    -------
    dict with keys: mode, lab_only_tickers, applied_reasons, written (bool),
                    override_path (str), payload (dict)
    """
    eligibility = _load_eligibility(eligibility_path)
    lab_only, reasons = _extract_lab_only(eligibility)
    mode = "recommendation-only" if recommendation_only else "autonomous"

    payload = build_override_payload(
        lab_only_tickers=lab_only,
        reasons=reasons,
        mode=mode,
        ttl_hours=ttl_hours,
        eligibility_path=eligibility_path,
    )

    written = False
    if not recommendation_only:
        write_override(payload, override_path)
        written = True

    return {
        "mode": mode,
        "lab_only_tickers": lab_only,
        "applied_reasons": reasons,
        "written": written,
        "override_path": str(override_path),
        "payload": payload,
    }


def _print_result(result: dict[str, Any], *, emit_json: bool) -> None:
    if emit_json:
        print(json.dumps(result["payload"], indent=2, default=str))
        return

    mode = result["mode"]
    lab_only = result["lab_only_tickers"]
    written = result["written"]

    print(f"[apply_ticker_eligibility_gates] mode={mode}")
    if not lab_only:
        print("  No LAB_ONLY tickers -- no routing gates applied.")
    else:
        print(f"  LAB_ONLY tickers ({len(lab_only)}): {', '.join(lab_only)}")
        for ticker in lab_only:
            reasons = result["applied_reasons"].get(ticker, [])
            print(f"    {ticker}: {'; '.join(reasons) or 'no_reason_recorded'}")

    if written:
        print(f"  Gate file written: {result['override_path']}")
        expires = result["payload"].get("expires_utc", "?")
        print(f"  Expires (UTC): {expires}")
    else:
        print("  [recommendation-only] No file written.")
        if lab_only:
            print(
                "  To apply, re-run without --recommendation-only "
                "(autonomous mode writes the gate file automatically)."
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--eligibility",
        type=Path,
        default=DEFAULT_ELIGIBILITY_PATH,
        help="Path to ticker_eligibility.json (default: logs/ticker_eligibility.json)",
    )
    parser.add_argument(
        "--override-path",
        type=Path,
        default=DEFAULT_OVERRIDE_PATH,
        help="Path to write active_lab_only.json gate file (default: logs/ticker_gating/active_lab_only.json)",
    )
    parser.add_argument(
        "--recommendation-only",
        "--dry-run",
        action="store_true",
        help="Print recommendations only; never write the gate file.",
    )
    parser.add_argument(
        "--ttl-hours",
        type=int,
        default=DEFAULT_TTL_HOURS,
        help=f"Hours before the gate file expires (default: {DEFAULT_TTL_HOURS})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="emit_json",
        help="Emit gate payload to stdout as JSON.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    try:
        result = apply_eligibility_gates(
            eligibility_path=args.eligibility,
            override_path=args.override_path,
            recommendation_only=args.recommendation_only,
            ttl_hours=args.ttl_hours,
        )
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        print(
            "  Run: python scripts/compute_ticker_eligibility.py  to generate eligibility data first.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        log.error("apply_ticker_eligibility_gates failed: %s", exc, exc_info=True)
        print(f"[ERROR] Unexpected error: {exc}", file=sys.stderr)
        return 1

    _print_result(result, emit_json=args.emit_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
