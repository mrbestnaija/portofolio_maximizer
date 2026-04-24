"""One-shot script to add 5 live-ops cron jobs to ~/.openclaw/cron/jobs.json."""
import json
from pathlib import Path

from scripts.openclaw_cron_contract import DEFAULT_SESSION_TARGET
from utils.repo_python import resolve_repo_python

JOBS_PATH = Path.home() / ".openclaw" / "cron" / "jobs.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = resolve_repo_python(PROJECT_ROOT)
TELEGRAM_FALLBACK_TO = "+2348061573767"

NEW_JOBS = [
    {
        "id": "c1d2e3f4-a5b6-7c8d-9e0f-1a2b3c4d5e6f",
        "agentId": "training",
        "name": "[P1] JSONL Label Accumulation",
        "description": (
            "Daily JSONL-path label accumulator. Joins quant_validation.jsonl "
            "classifier_features to production_closed_trades outcomes and appends "
            "new outcome-linked rows to directional_dataset.parquet."
        ),
        "enabled": True,
        "createdAtMs": 1773878400000,
        "updatedAtMs": 1773878400000,
        "schedule": {"kind": "cron", "expr": "22 23 * * *"},
        "sessionTarget": DEFAULT_SESSION_TARGET,
        "wakeMode": "now",
        "payload": {
            "kind": "agentTurn",
            "message": (
                "Run JSONL label accumulation. Use exec to run exactly this single command:\n"
                f"{PYTHON_EXE} scripts\\accumulate_classifier_labels.py --json\n\n"
                "Parse the JSON output.\n\n"
                "Rules:\n"
                "- If n_new > 0: announce a short summary: "
                "'Classifier labels +{n_new} new (outcome-linked: {n_by_source[outcome_linked]}, total: {n_total})'."
                " Include any features < 50% fill rate.\n"
                "- If n_new == 0 and n_candidates > 0 and n_outcome_map == 0: "
                "announce '[WARN] No outcome-linked trades in DB yet -- live cycles needed'.\n"
                "- If n_new == 0 and n_candidates == 0: "
                "announce '[WARN] No JSONL BUY/SELL entries with classifier_features -- pipeline may not be writing features'.\n"
                "- If n_new == 0 and n_outcome_map > 0 and n_candidates > 0: respond NO_REPLY.\n"
                "- PowerShell rule: never chain commands with &&."
            ),
        },
        "delivery": {
            "mode": "announce",
            "channel": "whatsapp",
            "to": "+2348061573767",
            "fallback": {"channel": "telegram", "to": TELEGRAM_FALLBACK_TO},
        },
        "state": {"consecutiveErrors": 0, "nextRunAtMs": 1773957720000, "lastStatus": "pending"},
    },
    {
        "id": "d2e3f4a5-b6c7-8d9e-0f1a-2b3c4d5e6f7a",
        "agentId": "training",
        "name": "[P1] Classifier Retrain Trigger",
        "description": (
            "Nightly conditional classifier retrain at 23:48. "
            "Retrains when outcome-linked labels cross 50/150/300/500 thresholds "
            "or model is > 7 days stale with new labels available."
        ),
        "enabled": True,
        "createdAtMs": 1773878400000,
        "updatedAtMs": 1773878400000,
        "schedule": {"kind": "cron", "expr": "48 23 * * *"},
        "sessionTarget": DEFAULT_SESSION_TARGET,
        "wakeMode": "now",
        "payload": {
            "kind": "agentTurn",
            "message": (
                "Check if directional classifier needs retraining.\n\n"
                "Step 1 - Dataset stats. Use exec:\n"
                f"{PYTHON_EXE} scripts\\check_classifier_readiness.py --json\n\n"
                "Step 2 - Model freshness. Use exec:\n"
                f"{PYTHON_EXE} -c \""
                "import json,pathlib,time; "
                "meta=pathlib.Path('models/directional_v1.meta.json'); "
                "age=(time.time()-meta.stat().st_mtime)/86400 if meta.exists() else -1; "
                "d=json.loads(meta.read_text()) if meta.exists() else {}; "
                "print(json.dumps({'model_exists':meta.exists(),'stale':age>7,"
                "'age_days':round(age,1),'n_train':d.get('n_train',0)}))\"\n\n"
                "Retrain conditions (any one triggers Step 3):\n"
                "  - model_exists is False\n"
                "  - stale is True AND outcome_linked >= 10\n"
                "  - outcome_linked >= 50 AND n_train < 50\n"
                "  - outcome_linked >= 150 AND n_train < 150\n"
                "  - outcome_linked >= 300 AND n_train < 300\n"
                "  - outcome_linked >= 500 AND n_train < 500\n\n"
                "Step 3 - Retrain (only if condition met). Use exec:\n"
                f"{PYTHON_EXE} scripts\\train_directional_classifier.py\n\n"
                "Step 4 - Evaluate (only after successful retrain). Use exec:\n"
                f"{PYTHON_EXE} scripts\\evaluate_directional_classifier.py\n\n"
                "Rules:\n"
                "- No retrain needed: respond NO_REPLY.\n"
                "- Retrain ran: announce 'Classifier retrained: n_train={n_train} "
                "DA={walk_forward_da:.3f} gate_lift={gate_lift_buy:+.3f}'.\n"
                "- cold_start=True: respond NO_REPLY.\n"
                "- Error: announce [ERROR] with summary.\n"
                "- PowerShell rule: never chain commands with &&."
            ),
        },
        "delivery": {
            "mode": "announce",
            "channel": "whatsapp",
            "to": "+2348061573767",
            "fallback": {"channel": "telegram", "to": TELEGRAM_FALLBACK_TO},
        },
        "state": {"consecutiveErrors": 0, "nextRunAtMs": 1773958080000, "lastStatus": "pending"},
    },
    {
        "id": "e3f4a5b6-c7d8-9e0f-1a2b-3c4d5e6f7a8b",
        "agentId": "training",
        "name": "[P2] Classifier Gate Readiness",
        "description": (
            "Weekly Sunday readiness progress toward 500 outcome-linked examples. "
            "Tracks milestones (100/250/500) and announces blockers for gate activation."
        ),
        "enabled": True,
        "createdAtMs": 1773878400000,
        "updatedAtMs": 1773878400000,
        "schedule": {"kind": "cron", "expr": "17 9 * * 0"},
        "sessionTarget": DEFAULT_SESSION_TARGET,
        "wakeMode": "now",
        "payload": {
            "kind": "agentTurn",
            "message": (
                "Run weekly classifier gate readiness check. Use exec to run:\n"
                f"{PYTHON_EXE} scripts\\check_classifier_readiness.py --json\n\n"
                "Always announce a brief digest (never NO_REPLY for this job):\n"
                "'Classifier gate [{verdict}]: {n_outcome_linked}/{gate_min_examples} outcome-linked. "
                "Rate: {daily_accumulation_rate:.1f}/day. Est {days_to_ready_estimate}d to READY.'\n"
                "Then show milestone progress: [x] 100  [x] 250  [ ] 500\n"
                "Then list top 2 blockers (or 'No blockers' if none).\n\n"
                "If verdict changed to APPROACHING or READY: prefix with [MILESTONE].\n"
                "If READY and all blockers cleared: announce '[GATE READY] Activate via "
                "directional_gate_enabled: true in config/signal_routing_config.yml'.\n"
                "PowerShell rule: never chain commands with &&."
            ),
        },
        "delivery": {
            "mode": "announce",
            "channel": "whatsapp",
            "to": "+2348061573767",
            "fallback": {"channel": "telegram", "to": TELEGRAM_FALLBACK_TO},
        },
        "state": {"consecutiveErrors": 0, "nextRunAtMs": 1774004400000, "lastStatus": "pending"},
    },
    {
        "id": "f4a5b6c7-d8e9-0f1a-2b3c-4d5e6f7a8b9c",
        "agentId": "trading",
        "name": "[P0] ATR Stop-Loss Path Audit",
        "description": (
            "Weekly Monday check that ATR-based stops are primary (>70% of trades). "
            "Alerts if vol*0.5 fallback rate exceeds 30%."
        ),
        "enabled": True,
        "createdAtMs": 1773878400000,
        "updatedAtMs": 1773878400000,
        "schedule": {"kind": "cron", "expr": "23 9 * * 1"},
        "sessionTarget": DEFAULT_SESSION_TARGET,
        "wakeMode": "now",
        "payload": {
            "kind": "agentTurn",
            "message": (
                "Audit ATR vs fallback stop-loss path usage. Use exec to run:\n"
                f"{PYTHON_EXE} -c \""
                "import sqlite3,json; conn=sqlite3.connect('data/portfolio_maximizer.db'); "
                "total=conn.execute('SELECT COUNT(*) FROM trade_executions "
                "WHERE is_close=0 AND COALESCE(is_synthetic,0)=0 AND COALESCE(is_diagnostic,0)=0').fetchone()[0]; "
                "atr=conn.execute('SELECT COUNT(*) FROM trade_executions "
                "WHERE is_close=0 AND COALESCE(is_synthetic,0)=0 AND COALESCE(is_diagnostic,0)=0 "
                "AND bar_high IS NOT NULL AND bar_low IS NOT NULL').fetchone()[0]; "
                "fb=total-atr; rate=round(fb/total,3) if total else 0; "
                "print(json.dumps({'total_opens':total,'atr_path':atr,"
                "'fallback_path':fb,'fallback_rate':rate})); conn.close()\"\n\n"
                "Rules:\n"
                "- fallback_rate <= 0.30: respond NO_REPLY.\n"
                "- fallback_rate > 0.30: announce '[WARN] ATR stop underutilised: "
                "{atr_path}/{total_opens} using ATR ({fallback_rate:.0%} fallback). "
                "Verify OHLC market_data is passed to signal generator'.\n"
                "- fallback_rate >= 0.80: escalate to '[CRITICAL] Stop-loss ATR path "
                "failing -- 80%+ using vol*0.5 cap. NVDA/high-vol positions at risk'.\n"
                "- total_opens == 0: respond NO_REPLY.\n"
                "- PowerShell rule: never chain commands with &&."
            ),
        },
        "delivery": {
            "mode": "announce",
            "channel": "whatsapp",
            "to": "+2348061573767",
            "fallback": {"channel": "telegram", "to": TELEGRAM_FALLBACK_TO},
        },
        "state": {"consecutiveErrors": 0, "nextRunAtMs": 1774256400000, "lastStatus": "pending"},
    },
    {
        "id": "a5b6c7d8-e9f0-1a2b-3c4d-5e6f7a8b9c0d",
        "agentId": "training",
        "name": "[P1] Live Ops Weekly Digest",
        "description": (
            "Friday 18:47 weekly digest covering label accumulation pace, "
            "stop-loss performance (14d), classifier metrics, and gate blockers."
        ),
        "enabled": True,
        "createdAtMs": 1773878400000,
        "updatedAtMs": 1773878400000,
        "schedule": {"kind": "cron", "expr": "47 18 * * 5"},
        "sessionTarget": DEFAULT_SESSION_TARGET,
        "wakeMode": "now",
        "payload": {
            "kind": "agentTurn",
            "message": (
                "Generate weekly live-ops digest. Run in sequence:\n\n"
                "1. Label readiness. Use exec:\n"
                f"{PYTHON_EXE} scripts\\check_classifier_readiness.py --json\n\n"
                "2. Stop-loss performance (14d). Use exec:\n"
                f"{PYTHON_EXE} -c \""
                "import sqlite3,json; conn=sqlite3.connect('data/portfolio_maximizer.db'); "
                "s=conn.execute(\\\"SELECT COUNT(*),COALESCE(SUM(realized_pnl),0) FROM production_closed_trades "
                "WHERE exit_reason LIKE '%stop%' AND trade_date>=date('now','-14 days')\\\").fetchone(); "
                "t=conn.execute(\\\"SELECT COUNT(*),COALESCE(SUM(realized_pnl),0),"
                "COALESCE(AVG(CASE WHEN realized_pnl>0 THEN 1.0 ELSE 0.0 END),0) "
                "FROM production_closed_trades WHERE trade_date>=date('now','-14 days')\\\").fetchone(); "
                "print(json.dumps({'stop_n':s[0],'stop_pnl':round(s[1],2),"
                "'total_14d':t[0],'pnl_14d':round(t[1],2),'wr_14d':round(t[2],3)})); conn.close()\"\n\n"
                "3. Classifier eval. Use exec:\n"
                f"{PYTHON_EXE} scripts\\evaluate_directional_classifier.py --json\n\n"
                "Always announce (never NO_REPLY). Format:\n"
                "'[Weekly Digest {date}]\\n"
                "- Labels: {n_outcome_linked}/500 outcome-linked ({daily_rate:.1f}/day, ~{days}d to gate)\\n"
                "- Trades (14d): {total_14d} trades WR={wr_14d:.0%} PnL=${pnl_14d:+.2f}\\n"
                "- Stops (14d): {stop_n} exits ${stop_pnl:+.2f}\\n"
                "- Classifier: DA={walk_forward_da:.3f} gate_lift={gate_lift_buy:+.3f}\\n"
                "- Top blocker: {blockers[0] or None}'\n"
                "PowerShell rule: never chain commands with &&."
            ),
        },
        "delivery": {
            "mode": "announce",
            "channel": "whatsapp",
            "to": "+2348061573767",
            "fallback": {"channel": "telegram", "to": TELEGRAM_FALLBACK_TO},
        },
        "state": {"consecutiveErrors": 0, "nextRunAtMs": 1774000020000, "lastStatus": "pending"},
    },
]

def main():
    with open(JOBS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    existing_ids = {j["id"] for j in data["jobs"]}
    added = 0
    for job in NEW_JOBS:
        if job["id"] not in existing_ids:
            data["jobs"].append(job)
            added += 1
            print(f"  Added: {job['name']}")
        else:
            print(f"  Skipped (exists): {job['name']}")

    with open(JOBS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\nTotal jobs: {len(data['jobs'])} ({added} new)")

if __name__ == "__main__":
    main()
