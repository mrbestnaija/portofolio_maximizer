"""
Apply eligibility statuses to a gate sidecar consumed by downstream automation.

This script is read-only with respect to strategy/routing config. It only
translates ticker eligibility output into an explicit gate artifact.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INPUT = ROOT / "logs" / "ticker_eligibility.json"
DEFAULT_OUTPUT = ROOT / "logs" / "ticker_eligibility_gates.json"

log = logging.getLogger(__name__)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _load_eligibility(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "eligibility_missing"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, "eligibility_unreadable"
    if not isinstance(raw, dict):
        return None, "eligibility_invalid_payload"
    return raw, None


def apply_eligibility_gates(
    *,
    eligibility_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    eligibility, load_error = _load_eligibility(Path(eligibility_path))
    if load_error:
        warnings.append(load_error)
        tickers: dict[str, Any] = {}
    else:
        tickers = eligibility.get("tickers", {}) if isinstance(eligibility, dict) else {}
        if not isinstance(tickers, dict):
            warnings.append("eligibility_tickers_invalid")
            tickers = {}

    healthy_tickers: list[str] = []
    weak_tickers: list[str] = []
    lab_only_tickers: list[str] = []
    for ticker, info in tickers.items():
        status = str((info or {}).get("status", "")).upper()
        symbol = str(ticker).upper()
        if status == "HEALTHY":
            healthy_tickers.append(symbol)
        elif status == "WEAK":
            weak_tickers.append(symbol)
        elif status == "LAB_ONLY":
            lab_only_tickers.append(symbol)

    healthy_tickers.sort()
    weak_tickers.sort()
    lab_only_tickers.sort()

    payload: dict[str, Any] = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "eligibility_path": str(eligibility_path),
        "eligibility_window": (
            eligibility.get("window")
            if isinstance(eligibility, dict) and isinstance(eligibility.get("window"), dict)
            else (eligibility.get("meta", {}).get("window") if isinstance(eligibility, dict) and isinstance(eligibility.get("meta"), dict) else {})
        ),
        "status": "ERROR" if errors else ("WARN" if warnings else "PASS"),
        "gate_written": True,
        "healthy_tickers": healthy_tickers,
        "weak_tickers": weak_tickers,
        "lab_only_tickers": lab_only_tickers,
        "summary": {
            "HEALTHY": len(healthy_tickers),
            "WEAK": len(weak_tickers),
            "LAB_ONLY": len(lab_only_tickers),
        },
        "warnings": warnings,
        "errors": errors,
    }

    _write_json(Path(output_path), payload)
    payload["output"] = str(output_path)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply ticker eligibility statuses to a gate sidecar.")
    parser.add_argument("--eligibility", type=Path, default=DEFAULT_INPUT, help="Eligibility JSON input.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Eligibility gate output path.")
    parser.add_argument("--json", action="store_true", dest="emit_json", help="Print result as JSON.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    result = apply_eligibility_gates(
        eligibility_path=args.eligibility,
        output_path=args.output,
    )

    if args.emit_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Eligibility gate written to {args.output}")
        print(
            "  HEALTHY={HEALTHY} WEAK={WEAK} LAB_ONLY={LAB_ONLY}".format(
                **result["summary"]
            )
        )
        if result["warnings"]:
            print(f"  warnings: {', '.join(result['warnings'])}")
    return 0 if result["status"] in {"PASS", "WARN"} else 1


if __name__ == "__main__":
    sys.exit(main())
