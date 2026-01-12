# Runtime Guardrails (WSL `simpleTrader_env` Only)

This repo is **validated only** when run under **WSL** using the **Linux virtualenv** at `simpleTrader_env/` (i.e. `simpleTrader_env/bin/python`).

If you run the code under any other interpreter (Windows Python, `py`, `.venv`, system `python`, or `simpleTrader_env\\Scripts\\python.exe`), results are **not comparable** and must be treated as **invalid**.

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
