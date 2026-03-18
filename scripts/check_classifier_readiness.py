"""
check_classifier_readiness.py
------------------------------
Gate readiness monitor for the directional classifier activation.

Tracks progress toward the 500 fully-featured, outcome-linked labeled examples
required to activate the classifier as a hard execution gate.

Reports:
  - Total labeled examples by source (parquet_scan vs outcome_linked)
  - Feature fill rates for all 20 features
  - Milestone progress (100 / 250 / 500)
  - Projected date to reach 500 examples at current accumulation rate
  - Recommendation: NOT_READY / APPROACHING / READY

Exit codes:
  0 = NOT_READY  (< 250 outcome-linked examples)
  1 = APPROACHING (250-499 outcome-linked examples)
  2 = READY       (>= 500 outcome-linked examples, key features >= 70% fill)

Usage:
  python scripts/check_classifier_readiness.py [--json] [--dataset PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASET_PATH = _REPO_ROOT / "data" / "training" / "directional_dataset.parquet"

# Features that must be >= MIN_FILL_RATE for gate activation
_KEY_FEATURES = [
    "ensemble_pred_return",
    "snr",
    "model_agreement",
    "hurst_exponent",
    "trend_strength",
    "realized_vol_annualized",
]
_MIN_FILL_RATE = 0.70  # 70% non-NaN required for key features
_GATE_MIN_EXAMPLES = 500
_APPROACHING_THRESHOLD = 250

_FEATURE_NAMES: List[str] = [
    "ensemble_pred_return", "ci_width_normalized", "snr", "model_agreement",
    "directional_vote_fraction", "garch_conf", "samossa_conf", "mssa_rl_conf",
    "igarch_fallback_flag", "samossa_evr", "hurst_exponent", "trend_strength",
    "realized_vol_annualized", "adf_pvalue", "regime_liquid_rangebound",
    "regime_moderate_trending", "regime_high_vol_trending", "regime_crisis",
    "recent_return_5d", "recent_vol_ratio",
]

_MILESTONES = [100, 250, 500]


def _days_to_milestone(
    current: int,
    target: int,
    daily_rate: float,
) -> Optional[int]:
    if daily_rate <= 0 or current >= target:
        return None
    return int(np.ceil((target - current) / daily_rate))


def check_readiness(
    dataset_path: Path = _DATASET_PATH,
) -> Dict[str, Any]:
    if not dataset_path.exists():
        return {
            "status": "NO_DATASET",
            "verdict": "NOT_READY",
            "exit_code": 0,
            "n_total": 0,
            "n_outcome_linked": 0,
            "n_parquet_scan": 0,
            "feature_fill_rates": {},
            "key_feature_fill_rates": {},
            "milestones": {},
            "blockers": ["directional_dataset.parquet not found"],
        }

    df = pd.read_parquet(dataset_path)
    n_total = len(df)

    # Source breakdown
    n_by_source: Dict[str, int] = {}
    if "label_source" in df.columns:
        n_by_source = df["label_source"].value_counts().to_dict()
    n_outcome_linked = n_by_source.get("outcome_linked", 0)
    # label_source may be "parquet_scan" or "price_parquet_scan" (legacy)
    n_parquet_scan = n_by_source.get("parquet_scan", 0) + n_by_source.get("price_parquet_scan", 0)

    # Feature fill rates
    fill_rates: Dict[str, float] = {}
    for fname in _FEATURE_NAMES:
        if fname in df.columns:
            fill_rates[fname] = float(1.0 - df[fname].isna().mean())
        else:
            fill_rates[fname] = 0.0

    key_fill_rates = {k: fill_rates.get(k, 0.0) for k in _KEY_FEATURES}

    # Ticker distribution
    ticker_dist: Dict[str, int] = {}
    if "ticker" in df.columns:
        ticker_dist = df["ticker"].value_counts().to_dict()

    # Outcome-linked: by ticker and recent accumulation rate
    outcome_df = df[df.get("label_source", pd.Series()) == "outcome_linked"] \
        if "label_source" in df.columns else pd.DataFrame()

    # Estimate daily rate from last 14 days of outcome_linked entries
    daily_rate = 0.0
    if not outcome_df.empty and "entry_ts" in outcome_df.columns:
        try:
            ts = pd.to_datetime(outcome_df["entry_ts"], utc=True, errors="coerce").dropna()
            if len(ts) >= 2:
                span_days = max(1.0, (ts.max() - ts.min()).total_seconds() / 86400)
                daily_rate = len(ts) / span_days
        except Exception:
            pass

    # Milestone progress
    milestones: Dict[str, Any] = {}
    for m in _MILESTONES:
        days = _days_to_milestone(n_outcome_linked, m, daily_rate)
        milestones[str(m)] = {
            "reached": n_outcome_linked >= m,
            "current": n_outcome_linked,
            "days_remaining": days,
        }

    # Blockers for gate activation
    blockers: List[str] = []
    if n_outcome_linked < _GATE_MIN_EXAMPLES:
        blockers.append(
            f"Need {_GATE_MIN_EXAMPLES - n_outcome_linked} more outcome-linked examples "
            f"(have {n_outcome_linked}/{_GATE_MIN_EXAMPLES})"
        )
    for k in _KEY_FEATURES:
        rate = key_fill_rates.get(k, 0.0)
        if rate < _MIN_FILL_RATE:
            blockers.append(f"Feature '{k}' fill rate {rate:.0%} < {_MIN_FILL_RATE:.0%} required")

    # Verdict
    if n_outcome_linked >= _GATE_MIN_EXAMPLES and not blockers:
        verdict = "READY"
        exit_code = 2
    elif n_outcome_linked >= _APPROACHING_THRESHOLD:
        verdict = "APPROACHING"
        exit_code = 1
    else:
        verdict = "NOT_READY"
        exit_code = 0

    # Days to READY estimate
    days_to_ready: Optional[int] = None
    if verdict != "READY" and daily_rate > 0:
        days_to_ready = _days_to_milestone(n_outcome_linked, _GATE_MIN_EXAMPLES, daily_rate)

    return {
        "status": "OK",
        "verdict": verdict,
        "exit_code": exit_code,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_total": n_total,
        "n_outcome_linked": n_outcome_linked,
        "n_parquet_scan": n_parquet_scan,
        "n_by_source": n_by_source,
        "ticker_distribution": ticker_dist,
        "daily_accumulation_rate": round(daily_rate, 3),
        "days_to_ready_estimate": days_to_ready,
        "feature_fill_rates": {k: round(v, 3) for k, v in fill_rates.items()},
        "key_feature_fill_rates": {k: round(v, 3) for k, v in key_fill_rates.items()},
        "milestones": milestones,
        "gate_min_examples": _GATE_MIN_EXAMPLES,
        "blockers": blockers,
    }


def _print_report(r: Dict[str, Any]) -> None:
    verdict = r["verdict"]
    verdict_marker = {"READY": "[READY]", "APPROACHING": "[APPROACHING]", "NOT_READY": "[NOT READY]"}.get(
        verdict, f"[{verdict}]"
    )

    print("=" * 60)
    print(f"Classifier Gate Readiness  {verdict_marker}")
    print("=" * 60)
    print(f"  Total labeled examples  : {r['n_total']}")
    print(f"  Outcome-linked          : {r['n_outcome_linked']}  (gate target: {r['gate_min_examples']})")
    print(f"  Parquet-scan (proxy)    : {r['n_parquet_scan']}")
    print(f"  Tickers                 : {r.get('ticker_distribution', {})}")
    print(f"  Daily rate              : {r['daily_accumulation_rate']:.2f} examples/day")
    if r.get("days_to_ready_estimate"):
        print(f"  Estimated days to READY : {r['days_to_ready_estimate']} days")
    print()

    print("Milestones:")
    for m, info in r.get("milestones", {}).items():
        marker = "[x]" if info["reached"] else "[ ]"
        days = f"  ({info['days_remaining']}d remaining)" if info.get("days_remaining") else ""
        print(f"  {marker} {m} examples{days}")
    print()

    print("Key feature fill rates:")
    for k, v in r.get("key_feature_fill_rates", {}).items():
        bar = "#" * int(v * 10) + "." * (10 - int(v * 10))
        ok = " OK" if v >= 0.70 else " LOW"
        print(f"  {k:<30} [{bar}] {v:.0%}{ok}")
    print()

    if r.get("blockers"):
        print("Blockers for gate activation:")
        for b in r["blockers"]:
            print(f"  - {b}")
    else:
        print("No blockers — gate activation criteria met.")
    print("=" * 60)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Classifier gate readiness monitor")
    parser.add_argument("--dataset", type=Path, default=_DATASET_PATH)
    parser.add_argument("--json", action="store_true", help="Emit JSON to stdout")
    args = parser.parse_args(argv)

    result = check_readiness(dataset_path=args.dataset)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_report(result)

    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
