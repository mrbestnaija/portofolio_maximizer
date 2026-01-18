# Runtime Guardrails (WSL `simpleTrader_env` Only)

This repo is **validated only** when run under **WSL** using the **Linux virtualenv** at `simpleTrader_env/` (i.e. `simpleTrader_env/bin/python`).

If you run the code under any other interpreter (Windows Python, `py`, `.venv`, system `python`, or `simpleTrader_env\\Scripts\\python.exe`), results are **not comparable** and must be treated as **invalid**.

## Time-Series Parameter Policy (Hard Requirement)

- **No manual orders**: SARIMAX `(p,d,q,P,D,Q,s,trend)` and GARCH `(p,q)` parameters are **learned from data by default**. Manual overrides are **unsupported** and will raise.
- **Performance controls are caps/search modes only** (example: `max_p/max_q/max_P/max_Q`, `order_search_mode`, `order_search_maxiter`) configured in `config/forecasting_config.yml`.
- **SARIMAX‑X is enabled by default**: `forcester_ts/forecaster.py` builds a small exogenous feature set and supplies it to SARIMAX fit + forecast. Include the `sarimax_exogenous` artifact (feature list) when reporting results.

## Required Pre-Run Fingerprint (Always Paste With Results)

Run this **before** any ETL/backtest/auto-trader execution and include the **command + output** in logs/issues/PRs:

```bash
cd /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45
source simpleTrader_env/bin/activate

which python
python -V

python -c "import torch; print({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'device_count': torch.cuda.device_count(), 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"
```

Acceptance criteria:
- `which python` points to `.../simpleTrader_env/bin/python`
- `torch` imports successfully
- `cuda_available` is `True` for “FULL GPU” runs

## Live-Data Evaluation Guard (No Synthetic)

For evaluation runs using live data, ensure synthetic flags are **unset/disabled**:

```bash
unset ENABLE_SYNTHETIC_PROVIDER ENABLE_SYNTHETIC_DATA_SOURCE SYNTHETIC_ONLY
export EXECUTION_MODE=live
```

## Dashboard Guard (No Fictitious UI)

- `visualizations/live_dashboard.html` is a **real-time view of run artifacts** and must not ship with embedded demo/fake values.
- The dashboard polls `visualizations/dashboard_data.json` every 5 seconds; if the file is missing, it shows empty states.
- To view it reliably, serve the repo root over HTTP (avoids `file://` fetch restrictions):

```bash
cd /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45
python3 -m http.server 8000
```

Then open `http://localhost:8000/visualizations/live_dashboard.html`.

If you cannot `cd` into the repo first, use `--directory`:

```bash
python3 -m http.server 8000 --directory /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45
```

### Audit-Grade Persistence (Default in Orchestrators)

- Bash entrypoints (`bash/run_auto_trader.sh`, `bash/run_pipeline.sh`, `bash/run_end_to_end.sh`, `bash/run_post_eval.sh`) start a DB-backed dashboard bridge via `bash/lib/common.sh:pmx_ensure_dashboard`.
- The bridge (`scripts/dashboard_db_bridge.py`) renders `visualizations/dashboard_data.json` from the SQLite trading DB and, by default, persists snapshots to `data/dashboard_audit.db` (`--persist-snapshot`).
- Controls:
  - `DASHBOARD_PERSIST=1` (default) enables `--persist-snapshot` → `data/dashboard_audit.db`.
  - `DASHBOARD_PERSIST=0` disables audit persistence (view-only).
  - `DASHBOARD_PORT=8000` sets the HTTP port (default 8000).
  - `DASHBOARD_KEEP_ALIVE=1` (default) leaves the bridge/server running after the script exits.
- Provenance auditing: `scripts/audit_dashboard_payload_sources.py` can be run to validate the latest `visualizations/dashboard_data.json` and the latest persisted snapshot in `data/dashboard_audit.db` for non-fictitious content and source consistency.

## Fast Guard Script

For convenience, use:

```bash
bash bash/runtime_check.sh
```

This prints the fingerprint and exits non-zero if it detects the wrong runtime.

## Failure Handling Rules

- If the fingerprint fails, **stop** and fix the runtime. Do not “guess” or infer correctness from `pip freeze`.
- If a command times out or fails, **rerun under the correct runtime** and report the **exit code**.
- If you are unsure which environment was used for a past run, treat the result as **untrusted** and rerun with the fingerprint attached.
- Do not report a task as complete unless you can cite the exact command(s) and outputs; if verification is missing or inconclusive, say so and rerun.
