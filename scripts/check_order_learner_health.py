#!/usr/bin/env python3
"""
check_order_learner_health.py
------------------------------
CI health gate for the Phase 7.16 auto-learning pipeline.

Checks:
  1. Coverage: model_order_stats has entries with n_fits >= 3 (qualified suggestions)
  2. AIC drift: mean AIC in cache vs recent time_series_forecasts (alerts if cache is >10% worse)
  3. Stale entries: rows where last_used < today - max_age_days
  4. Snapshot store: manifest is readable, pkl files exist
  5. VaR + Shapley smoke: runs a 30-bar smoke test and prints fold stats

Exit codes:
  0 = healthy (all checks pass)
  1 = any warning/error (fail closed by default; opt out with --allow-warn-pass)

Usage:
    python scripts/check_order_learner_health.py [--db PATH] [--allow-warn-pass]
"""

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "data" / "portfolio_maximizer.db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _connect(db_path: str):
    from integrity.sqlite_guardrails import guarded_sqlite_connect
    conn = guarded_sqlite_connect(db_path, timeout=5.0, enable_guardrails=False)
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _row_count(conn, table: str) -> int:
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Check 1: Coverage
# ---------------------------------------------------------------------------

def check_coverage(conn, min_fits: int = 3) -> dict:
    """Ensure qualified entries exist after >0 pipeline runs."""
    try:
        total = conn.execute("SELECT COUNT(*) FROM model_order_stats").fetchone()[0]
        qualified = conn.execute(
            "SELECT COUNT(*) FROM model_order_stats WHERE n_fits >= ? AND best_aic IS NOT NULL",
            (min_fits,),
        ).fetchone()[0]
        by_model = conn.execute(
            "SELECT model_type, COUNT(*) FROM model_order_stats "
            "WHERE n_fits >= ? AND best_aic IS NOT NULL GROUP BY model_type",
            (min_fits,),
        ).fetchall()
    except Exception as exc:
        return {"status": "ERROR", "detail": str(exc)}

    status = "OK" if qualified > 0 else "WARN"
    return {
        "status": status,
        "total_entries": total,
        "qualified_entries": qualified,
        "by_model": {r[0]: r[1] for r in by_model},
    }


# ---------------------------------------------------------------------------
# Check 2: AIC drift
# ---------------------------------------------------------------------------

def check_aic_drift(conn, lookback_days: int = 30, drift_threshold: float = 0.10) -> dict:
    """
    Compare mean AIC in cache vs recent time_series_forecasts.
    Alerts if the cached order's mean AIC is > drift_threshold worse than recent fits.
    """
    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
    try:
        recent = conn.execute(
            """
            SELECT model_type, AVG(aic) AS avg_aic
            FROM time_series_forecasts
            WHERE aic IS NOT NULL
              AND DATE(created_at) >= ?
              AND model_type IN ('GARCH', 'SARIMAX', 'SAMOSSA')
            GROUP BY model_type
            """,
            (cutoff,),
        ).fetchall()

        cached = conn.execute(
            """
            SELECT model_type, MIN(aic_sum / NULLIF(n_fits, 0)) AS mean_aic
            FROM model_order_stats
            WHERE n_fits > 0
              AND best_aic IS NOT NULL
              AND last_used >= ?
            GROUP BY model_type
            """,
            (cutoff,),
        ).fetchall()
    except Exception as exc:
        return {"status": "ERROR", "detail": str(exc)}

    recent_by_model = {r[0]: float(r[1]) for r in recent if r[1] is not None}
    alerts = []
    for model_type, cached_mean in cached:
        recent_mean = recent_by_model.get(model_type)
        if recent_mean is None or recent_mean <= 0:
            continue
        drift = (cached_mean - recent_mean) / abs(recent_mean)
        if drift > drift_threshold:
            alerts.append({
                "model_type": model_type,
                "cached_mean_aic": round(cached_mean, 2),
                "recent_mean_aic": round(recent_mean, 2),
                "drift_pct": round(drift * 100, 1),
            })

    return {
        "status": "WARN" if alerts else "OK",
        "alerts": alerts,
        "recent_models": recent_by_model,
    }


# ---------------------------------------------------------------------------
# Check 3: Stale entries
# ---------------------------------------------------------------------------

def check_stale(conn, max_age_days: int = 90) -> dict:
    cutoff = (date.today() - timedelta(days=max_age_days)).isoformat()
    try:
        stale = conn.execute(
            "SELECT COUNT(*) FROM model_order_stats WHERE last_used < ?", (cutoff,)
        ).fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM model_order_stats").fetchone()[0]
    except Exception as exc:
        return {"status": "ERROR", "detail": str(exc)}

    pct = (stale / total * 100) if total > 0 else 0.0
    return {
        "status": "WARN" if pct > 30 else "OK",
        "stale_entries": stale,
        "total_entries": total,
        "stale_pct": round(pct, 1),
    }


# ---------------------------------------------------------------------------
# Check 4: Snapshot store
# ---------------------------------------------------------------------------

def check_snapshot_store() -> dict:
    try:
        from forcester_ts.model_snapshot_store import ModelSnapshotStore
        store = ModelSnapshotStore()
        if store._manifest_path.exists():
            raw_manifest = json.loads(store._manifest_path.read_text(encoding="utf-8"))
            if not isinstance(raw_manifest, dict):
                raise ValueError("snapshot manifest is not a JSON object")
        snaps = store.list_snapshots()
        missing_files = []
        invalid_paths = []
        for snap in snaps:
            raw_path = snap.get("path")
            try:
                snap_path = store._safe_pkl_path(raw_path)
            except ValueError:
                invalid_paths.append(str(raw_path))
                continue
            if not snap_path.exists():
                missing_files.append(str(snap_path))
        return {
            "status": "ERROR" if (missing_files or invalid_paths) else "OK",
            "snapshot_count": len(snaps),
            "dir": str(store._dir),
            "invalid_paths": invalid_paths,
            "missing_files": missing_files,
        }
    except Exception as exc:
        return {"status": "ERROR", "detail": str(exc)}


# ---------------------------------------------------------------------------
# Check 5: VaR + Shapley smoke test
# ---------------------------------------------------------------------------

def check_var_shapley_smoke() -> dict:
    try:
        import numpy as np
        import pandas as pd
        from forcester_ts.walk_forward_learner import WalkForwardLearner

        rng = np.random.default_rng(99)
        prices = 100 * np.exp(np.cumsum(rng.standard_normal(200) * 0.01))
        series = pd.Series(prices)

        wfl = WalkForwardLearner(
            forecaster_config={},
            min_train_length=80,
            fold_step=20,
            forecast_horizon=5,
        )
        result = wfl.run(series, ticker="SMOKE")

        agg = result.aggregate
        return {
            "status": "OK",
            "n_folds": result.n_folds,
            "rmse_mean": round(agg.get("rmse_mean", float("nan")), 4),
            "dir_acc_mean": round(agg.get("dir_acc_mean", float("nan")), 3),
            "var_violation_rate_mean": round(agg.get("var_violation_rate_mean", float("nan")), 3),
            "shapley_mean_keys": list(agg.get("shapley_mean", {}).keys()),
        }
    except Exception as exc:
        return {"status": "ERROR", "detail": str(exc)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_health_check(db_path: str, strict: bool = True) -> int:
    """Run all checks. Returns exit code (0=OK, 1=warn)."""
    print(f"[check_order_learner_health] DB: {db_path}")
    print(f"[check_order_learner_health] Date: {date.today().isoformat()}")
    print()

    # DB-level checks
    db_exists = Path(db_path).exists()
    if not db_exists:
        print(f"[ERROR] DB not found: {db_path}")
        return 1

    conn = None
    try:
        conn = _connect(db_path)
    except Exception as exc:
        print(f"[ERROR] Cannot connect to DB: {exc}")
        return 1
    results = {}

    results["coverage"] = check_coverage(conn)
    results["aic_drift"] = check_aic_drift(conn)
    results["stale"] = check_stale(conn)
    conn.close()

    results["snapshot_store"] = check_snapshot_store()
    results["var_shapley_smoke"] = check_var_shapley_smoke()

    # Print results
    any_warn = False
    any_error = False
    for check_name, r in results.items():
        status = r.get("status", "UNKNOWN")
        if status == "OK":
            print(f"  [OK]   {check_name}")
        elif status in ("WARN",):
            any_warn = True
            print(f"  [WARN] {check_name}")
        elif status in ("SKIP",):
            any_warn = True
            print(f"  [SKIP] {check_name}: {r.get('detail', '')}")
        else:
            any_error = True
            print(f"  [ERR]  {check_name}: {r.get('detail', '')}")

        # Print detail fields
        for k, v in r.items():
            if k == "status":
                continue
            if isinstance(v, (dict, list)) and v:
                print(f"         {k}: {json.dumps(v, default=str)}")
            elif v not in (None, "", []):
                print(f"         {k}: {v}")
    print()

    # Overall status
    if any_error:
        print("[check_order_learner_health] RESULT: ERROR (see above)")
        return 1
    if any_warn:
        print("[check_order_learner_health] RESULT: WARN")
        return 0 if not strict else 1
    print("[check_order_learner_health] RESULT: HEALTHY")
    return 0


def main():
    parser = argparse.ArgumentParser(description="OrderLearner health gate")
    parser.add_argument("--db", default=str(DB_PATH), help="Path to SQLite DB")
    parser.add_argument("--strict", action="store_true",
                        help="Deprecated: strict mode is now the default.")
    parser.add_argument(
        "--allow-warn-pass",
        action="store_true",
        help="Exit 0 on warnings (legacy fail-open behavior).",
    )
    args = parser.parse_args()
    strict_mode = not bool(args.allow_warn_pass)
    sys.exit(run_health_check(db_path=args.db, strict=strict_mode))


if __name__ == "__main__":
    main()
