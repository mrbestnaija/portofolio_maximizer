#!/usr/bin/env bash
# Live/auto orchestration for Portfolio Maximizer ETL pipeline.
# Executes with network data first (live mode by default), with optional
# synthetic/auto modes controlled via EXECUTION_MODE.

set -euo pipefail

# Production-safe defaults: clear diagnostic shortcuts so live runs keep
# quant validation and latency guards enabled.
unset DIAGNOSTIC_MODE TS_DIAGNOSTIC_MODE EXECUTION_DIAGNOSTIC_MODE LLM_FORCE_FALLBACK || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/simpleTrader_env/bin/python"
PIPELINE_SCRIPT="$ROOT_DIR/scripts/run_etl_pipeline.py"
LOG_DIR="$ROOT_DIR/logs/live_runs"
DASH_PATH="$ROOT_DIR/visualizations/dashboard_data.json"

# Enable TS forecast audit logs for live runs when forecaster instrumentation is active.
export TS_FORECAST_AUDIT_DIR="${TS_FORECAST_AUDIT_DIR:-$ROOT_DIR/logs/forecast_audits}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found at $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$PIPELINE_SCRIPT" ]]; then
  echo "Pipeline runner missing at $PIPELINE_SCRIPT" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/pipeline_live_${RUN_STAMP}.log"

TICKERS="${TICKERS:-AAPL,MSFT}"
START_DATE="${START_DATE:-2020-01-01}"
END_DATE="${END_DATE:-$(date +%Y-%m-%d)}"
DATA_SOURCE="${DATA_SOURCE:-}"
USE_CV="${USE_CV:-0}"
ENABLE_LLM="${ENABLE_LLM:-1}"
LLM_MODEL="${LLM_MODEL:-}"
# Default to true live mode for this launcher; callers can override to
# 'auto' or 'synthetic' via EXECUTION_MODE if needed.
EXECUTION_MODE="${EXECUTION_MODE:-live}"
INCLUDE_FRONTIER_TICKERS="${INCLUDE_FRONTIER_TICKERS:-1}"

CMD=("$PYTHON_BIN" "$PIPELINE_SCRIPT"
  --tickers "$TICKERS"
  --start "$START_DATE"
  --end "$END_DATE"
  --execution-mode "$EXECUTION_MODE"
)

if [[ -n "$DATA_SOURCE" ]]; then
  CMD+=(--data-source "$DATA_SOURCE")
fi

if [[ "$USE_CV" == "1" ]]; then
  CMD+=(--use-cv)
fi

if [[ "$ENABLE_LLM" == "1" ]]; then
  CMD+=(--enable-llm)
fi

if [[ -n "$LLM_MODEL" ]]; then
  CMD+=(--llm-model "$LLM_MODEL")
fi

if [[ "$INCLUDE_FRONTIER_TICKERS" == "1" ]]; then
  CMD+=(--include-frontier-tickers)
fi

# Allow callers to append additional overrides.
CMD+=("$@")

echo "Running pipeline (${EXECUTION_MODE}) @ ${RUN_STAMP}"
echo "Command: ${CMD[*]}"
echo "Streaming output to $LOG_FILE"

set +e
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
EXIT_CODE=$?
set -e

if [[ "$EXIT_CODE" -ne 0 ]]; then
  echo "Pipeline run failed (exit code $EXIT_CODE). See $LOG_FILE for details." >&2
  exit "$EXIT_CODE"
fi

PIPELINE_ID="$(grep -oE 'pipeline_[0-9]{8}_[0-9]{6}' "$LOG_FILE" | tail -n 1 || true)"

if [[ -z "$PIPELINE_ID" ]]; then
  echo "Unable to determine pipeline ID from $LOG_FILE" >&2
  exit 2
fi

echo ""
echo "Pipeline ID: $PIPELINE_ID"
echo "Stage timing summary:"

ROOT_DIR="$ROOT_DIR" PIPELINE_ID="$PIPELINE_ID" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

import yaml
from etl.database_manager import DatabaseManager

root = Path(os.environ["ROOT_DIR"])
pipeline_id = os.environ["PIPELINE_ID"]
events_path = root / "logs" / "events" / "events.log"
quant_log_path = root / "logs" / "signals" / "quant_validation.jsonl"
config_path = root / "config" / "quant_success_config.yml"

tail_entries = 10
if config_path.exists():
    try:
        cfg = yaml.safe_load(config_path.read_text()) or {}
        tail_entries = int(
            cfg.get("quant_validation", {})
            .get("logging", {})
            .get("tail_entries", tail_entries)
        )
    except Exception:
        tail_entries = 10

stage_rows = []
with events_path.open() as fh:
    for line in fh:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("pipeline_id") != pipeline_id:
            continue
        if record.get("event_type") == "stage_complete":
            stage = record.get("stage")
            duration = record.get("metadata", {}).get("duration_seconds")
            if stage and duration is not None:
                stage_rows.append((stage, duration))

if not stage_rows:
    print("  (no stage metrics found)")
else:
    for stage, duration in stage_rows:
        print(f"  - {stage}: {duration:0.2f}s")

print("\nLatest dataset artifacts:")
data_root = root / "data"
for split in ("training", "validation", "testing"):
    split_dir = data_root / split
    if not split_dir.exists():
        continue
    candidates = sorted(split_dir.glob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    latest = candidates[0] if candidates else None
    if latest:
        size_kb = latest.stat().st_size / 1024
        print(f"  - {split}: {latest.name} ({size_kb:0.1f} KiB)")

print("\nQuant validation summary:")
if not quant_log_path.exists():
    print("  (quant validation log not found)")
else:
    entries = []
    with quant_log_path.open() as fh:
        for line in fh:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            entries.append(record)

    matching = [r for r in entries if r.get("pipeline_id") == pipeline_id and r.get("pipeline_id")]
    target = matching if matching else entries
    target = target[-tail_entries:]

    if not target:
        print("  (no quant validation entries yet)")
    else:
        for record in target:
            ticker = record.get("ticker", "UNKNOWN")
            status = record.get("status") or record.get("quant_validation", {}).get("status", "UNKNOWN")
            confidence = record.get("confidence")
            expected_return = record.get("expected_return")
            conf_str = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "n/a"
            exp_str = f"{expected_return:.2%}" if isinstance(expected_return, (int, float)) else "n/a"
            viz = record.get("visualization_path") or "n/a"
            print(f"  - {ticker}: {status} (conf={conf_str}, exp={exp_str}) viz={viz}")

print("\nPortfolio performance summary:")
start_env = os.getenv("MVS_START_DATE")
end_env = os.getenv("MVS_END_DATE")
window_days = os.getenv("MVS_WINDOW_DAYS")

start_date = start_env
end_date = end_env

if not start_date and window_days:
    try:
        days = int(window_days)
        end = datetime.utcnow().date()
        start = end - timedelta(days=days)
        start_date = start.isoformat()
        end_date = end.isoformat()
    except ValueError:
        start_date = None
        end_date = None

db = DatabaseManager(db_path=str(root / "data" / "portfolio_maximizer.db"))
perf = db.get_performance_summary(start_date=start_date, end_date=end_date)
db.close()

total_trades = perf.get("total_trades", 0)
total_profit = perf.get("total_profit", 0.0) or 0.0
win_rate = perf.get("win_rate", 0.0) or 0.0
profit_factor = perf.get("profit_factor", 0.0) or 0.0

window_label = "full history"
if start_date or end_date:
    window_label = f"{start_date or '...'} -> {end_date or '...'}"

print(f"  Window         : {window_label}")
print(f"  Total trades   : {total_trades}")
print(f"  Total profit   : {total_profit:.2f} USD")
print(f"  Win rate       : {win_rate:.1%}")
print(f"  Profit factor  : {profit_factor:.2f}")

mvs_passed = (
    total_profit > 0.0
    and win_rate > 0.45
    and profit_factor > 1.0
    and total_trades >= 30
)
print(f"  MVS Status     : {'PASS' if mvs_passed else 'FAIL'}")
print("")
PY

echo ""
echo "Data source snapshot (from $LOG_FILE):"
grep -E "Primary:" "$LOG_FILE" | tail -n 1 || echo "  (no primary source line found)"
grep -E "OK Successfully extracted" "$LOG_FILE" | tail -n 1 || echo "  (no extraction success line found)"

echo "Log captured at: $LOG_FILE"
if [[ -f "$DASH_PATH" ]]; then
  echo "Dashboard JSON available at: $DASH_PATH"
fi
echo "Bestman's Portfolio Maximizer v45 Live pipeline run complete."
