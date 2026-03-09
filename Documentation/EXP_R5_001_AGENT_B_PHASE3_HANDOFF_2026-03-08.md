# EXP-R5-001: Agent B Phase 3 Handoff

**Date**: 2026-03-08
**From**: Agent A
**To**: Agent B
**Prerequisite**: Phase 2 COMPLETE, M1 ACHIEVED (`n_active_audits=7`, `n_windows_with_residual_metrics=3`)

---

## What Phase 3 Is

Phase 3 populates the currently-null metrics in each active audit:

| Field | Phase 2 | Phase 3 |
|-------|---------|---------|
| `rmse_anchor` | `null` | RMSE of `y_hat_anchor` vs realized prices |
| `rmse_residual_ensemble` | `null` | RMSE of `y_hat_residual_ensemble` vs realized prices |
| `rmse_ratio` | `null` | `rmse_residual_ensemble / rmse_anchor` |
| `da_anchor` | `null` | Directional accuracy of anchor (sign correct fraction) |
| `da_residual_ensemble` | `null` | Directional accuracy of residual ensemble |
| `corr_anchor_residual` | phase-2 proxy | `corr(epsilon[t], epsilon_hat[t])` — correlation between realized anchor errors and residual forecasts |

**Note on `corr_anchor_residual` in Phase 2**: The field currently holds `corr(y_hat_anchor, y_hat_residual_ensemble)` — a proxy that is always near ±1 and NOT the promotion criterion. Phase 3 must **overwrite** this with the correct `corr(epsilon[t], epsilon_hat[t])` computation.

---

## Realized Price Source

The audit forecasts are produced by `scripts/run_etl_pipeline.py --execution-mode synthetic`. Synthetic prices are deterministic (seeded from AAPL historical data). For realized price matching:

1. **Primary**: Read the actual OHLCV data from `data/testing/` parquet files — the test split covers the forecast window.
2. **Secondary**: Re-run the synthetic extractor with the same seed to regenerate the price series.

The audit's `dataset` block contains:
```json
{
  "ticker": "AAPL",
  "start": "2020-01-01",
  "end": "2023-11-20",
  "length": 1014,
  "forecast_horizon": 30
}
```

The forecast covers `forecast_horizon` steps **after** `dataset.end`. Match realized prices starting at the day after `dataset.end`.

---

## Audit File Structure

Active audit files are in `logs/forecast_audits/forecast_audit_*.json`. Each has:

```json
{
  "dataset": {
    "ticker": "AAPL",
    "start": "...",
    "end": "...",
    "length": 1014,
    "forecast_horizon": 30
  },
  "artifacts": {
    "residual_experiment": {
      "experiment_id": "EXP-R5-001",
      "anchor_model_id": "mssa_rl",
      "phase": 2,
      "residual_status": "active",
      "residual_active": true,
      "y_hat_anchor": [206.44, 206.45, ..., 206.71],   // 30 values
      "y_hat_residual_ensemble": [212.51, 211.58, ..., 206.61],  // 30 values
      "rmse_anchor": null,               // Phase 3 fills these
      "rmse_residual_ensemble": null,
      "rmse_ratio": null,
      "da_anchor": null,
      "da_residual_ensemble": null,
      "corr_anchor_residual": -0.856,    // Phase 3 OVERWRITES with corr(eps, eps_hat)
      "residual_mean": 1.206,
      "residual_std": 1.662,
      "n_corrected": 30
    }
  }
}
```

---

## Computation Protocol

### Step 1 — Identify active audit files

```python
import json, pathlib

audit_dir = pathlib.Path("logs/forecast_audits")
active_audits = []
for f in sorted(audit_dir.glob("forecast_audit_*.json")):
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
        re = d.get("artifacts", {}).get("residual_experiment", {})
        if re.get("residual_status") == "active" and re.get("residual_active"):
            active_audits.append((f, d, re))
    except Exception:
        pass
```

### Step 2 — Load realized prices for each audit window

```python
import pandas as pd

def load_realized_prices(ticker: str, forecast_start_date: str, n_steps: int) -> pd.Series:
    """Load n_steps realized closing prices starting from forecast_start_date."""
    # Option A: from test parquet
    test_files = sorted(pathlib.Path("data/testing").glob(f"*{ticker}*.parquet"))
    if test_files:
        df = pd.read_parquet(test_files[-1])
        df.index = pd.to_datetime(df.index, utc=True)
        cutoff = pd.Timestamp(forecast_start_date, tz="UTC")
        future = df[df.index > cutoff]["Close"].iloc[:n_steps]
        if len(future) >= n_steps:
            return future.values  # numpy array of length n_steps
    # Option B: re-generate synthetic prices with same seed
    # (use etl/synthetic_extractor.py with ticker=AAPL)
    raise ValueError(f"Could not load realized prices for {ticker} from {forecast_start_date}")
```

### Step 3 — Compute metrics

```python
import numpy as np

def compute_phase3_metrics(
    y_hat_anchor: list,
    y_hat_resid_ens: list,
    realized: np.ndarray,
) -> dict:
    y_a = np.array(y_hat_anchor)
    y_r = np.array(y_hat_resid_ens)
    y_true = realized[:len(y_a)]

    # RMSE
    rmse_anchor = float(np.sqrt(np.mean((y_a - y_true) ** 2)))
    rmse_resid = float(np.sqrt(np.mean((y_r - y_true) ** 2)))
    rmse_ratio = rmse_resid / rmse_anchor if rmse_anchor > 0 else None

    # Directional accuracy (fraction of steps where sign(forecast_change) == sign(actual_change))
    actual_delta = np.diff(np.concatenate([[y_true[0]], y_true]))
    anchor_delta = np.diff(np.concatenate([[y_a[0]], y_a]))
    resid_delta = np.diff(np.concatenate([[y_r[0]], y_r]))
    da_anchor = float(np.mean(np.sign(anchor_delta) == np.sign(actual_delta)))
    da_resid = float(np.mean(np.sign(resid_delta) == np.sign(actual_delta)))

    # corr(epsilon[t], epsilon_hat[t])
    # epsilon[t] = realized[t] - y_hat_anchor[t]  (anchor's actual OOS errors)
    # epsilon_hat[t] = y_hat_resid_ens[t] - y_hat_anchor[t]  (residual correction applied)
    epsilon = y_true - y_a          # realized anchor errors
    epsilon_hat = y_r - y_a        # residual model's predictions of those errors
    if len(epsilon) >= 2 and np.std(epsilon) > 0 and np.std(epsilon_hat) > 0:
        corr_eps = float(np.corrcoef(epsilon, epsilon_hat)[0, 1])
    else:
        corr_eps = None

    return {
        "rmse_anchor": rmse_anchor,
        "rmse_residual_ensemble": rmse_resid,
        "rmse_ratio": rmse_ratio,
        "da_anchor": da_anchor,
        "da_residual_ensemble": da_resid,
        "corr_anchor_residual": corr_eps,  # OVERWRITES the phase-2 proxy
        "phase": 3,
    }
```

### Step 4 — Write back to audit files (in-place patch)

```python
def patch_audit_file(audit_path: pathlib.Path, metrics: dict) -> None:
    """Patch residual_experiment fields in-place; preserve all other fields."""
    d = json.loads(audit_path.read_text(encoding="utf-8"))
    re = d.setdefault("artifacts", {}).setdefault("residual_experiment", {})
    re.update(metrics)
    audit_path.write_text(json.dumps(d, indent=2), encoding="utf-8")
```

### Step 5 — Re-run quality pipeline to aggregate

```bash
python scripts/run_quality_pipeline.py \
    --audit-dir logs/forecast_audits \
    --enable-residual-experiment
```

Check `visualizations/performance/residual_experiment_summary.json`:
- `rmse_ratio_mean` should now be non-null
- `corr_anchor_residual_mean` should be the genuine `corr(ε, ε_hat)` values
- `early_abort_signal` field indicates if abort threshold is hit

---

## What Triggers Early Abort

The quality pipeline now checks for ≥5 consecutive windows with `rmse_ratio > 1.02`. If triggered:
- `early_abort_signal: true` in summary JSON
- Warning string: `EARLY_ABORT_SIGNAL:rmse_ratio>1.02_for_N_consecutive_windows`
- Action: halt accumulation, file redesign proposal — do NOT continue to 20 windows

---

## Promotion Contract (reminder)

All three must be met at ≥20 windows before promotion:

| Criterion | Threshold |
|-----------|-----------|
| `mean(rmse_ratio)` | `<= 0.98` |
| `mean(corr_anchor_residual)` | `>= 0.30` (the genuine `corr(ε, ε_hat)`) |
| `N_effective_audits` | `>= 20` |

**Promotion decision requires human + Claude Code review. Agent B does not promote unilaterally.**

---

## Files Relevant to Phase 3

| File | Role |
|------|------|
| `logs/forecast_audits/forecast_audit_*.json` | Source: read `y_hat_anchor`, `y_hat_residual_ensemble`; write Phase 3 metrics |
| `data/testing/*.parquet` | Realized price source (primary) |
| `etl/synthetic_extractor.py` | Realized price source (fallback — regenerate synthetic) |
| `scripts/run_quality_pipeline.py` | Aggregate Phase 3 metrics after patching |
| `visualizations/performance/residual_experiment_summary.json` | Output: aggregated metrics, early_abort_signal |
| `Documentation/EXP_R5_001_M3_DECISION_TEMPLATE_2026-03-08.md` | Decision framework for ≥10 windows |
