# Morning Check 6 AM — 2026-02-24

Overnight refresh started: 2026-02-23 ~22:27
Job task ID: b63acb2
Script: bash/overnight_refresh.sh
Purpose: clear legacy pre-7.10b entries from rolling window + accumulate Platt scaling data

---

## 1. Find log and check completion

```powershell
# Find log file
Get-ChildItem logs\run_audit\overnight_refresh_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1

# Check last 40 lines (completion status + final health check output)
Get-Content (Get-ChildItem logs\run_audit\overnight_refresh_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName) -Tail 40
```

**Completed if last lines show:**
- `[HH:MM:SS] OVERNIGHT REFRESH COMPLETE`
- `Errors: 0`  (or small number from non-critical ticker failures)

**Still running if last line is a ticker pipeline step.**

---

## 2. Re-run health checks (takes ~10 seconds)

```bash
python scripts/check_quant_validation_health.py
python scripts/quant_validation_headroom.py --json
```

**Expected results after refresh:**

| Metric | Before overnight | Expected after |
|--------|-----------------|----------------|
| `check_quant_validation_health` FAIL fraction | 0.277 | <= 0.30 (stable) |
| `quant_validation_headroom` fail_rate_pct | 71.7% | ~40-55% |
| headroom_to_red_gate | 23.3% | >= 35% |
| Non-AAPL tickers at 100% FAIL | 9/9 | 0-2 (cleared by fresh runs) |

---

## 3. Check Platt scaling data readiness

```bash
python -c "
import json, pathlib
log = pathlib.Path('logs/signals/quant_validation.jsonl')
entries = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
pairs = [(e.get('confidence'), e.get('outcome'))
         for e in entries
         if e.get('confidence') is not None and e.get('outcome') is not None]
print(f'Platt scaling pairs available: {len(pairs)}')
print('Status:', '[READY]' if len(pairs) >= 30 else '[NOT YET - need 30+]')
"
```

---

## 4. If a ticker failed — re-run manually

```bash
# Replace TICKER with failed ticker name
python scripts/run_etl_pipeline.py --tickers TICKER --start 2024-01-01 --end 2026-01-01 --execution-mode synthetic
```

---

## 5. Next code tasks (in order)

### P1 — signal_id NULL fix (~1-2h)
Every trade_executions row has signal_id=NULL. No model attribution.
- File: execution/paper_trading_engine.py — add signal_id to Trade dataclass + DB write
- File: models/time_series_signal_generator.py — ensure signal_id flows through

### P2 — B5 Platt scaling (~2-3h, only if >= 30 pairs available)
Confidence 0.9 currently -> 41% actual win rate. Fix: logistic regression calibration.
- File: models/time_series_signal_generator.py — add _calibrate_confidence() method

### P3 — Validate directional accuracy improvement
Run adversarial suite and compare ensemble_under_best_rate vs pre-7.10b baseline (92%).
```bash
python scripts/run_adversarial_forecaster_suite.py --json
```
