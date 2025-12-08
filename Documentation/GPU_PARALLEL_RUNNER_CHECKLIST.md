# GPU-Parallel, Energy-Efficient Multi-Asset Runner Checklist

**Goal**: Institution-grade, GPU-aware, energy-efficient orchestration for multi-asset execution without degrading PnL.

## Institution-Level Parallelism
- [ ] Shard tickers by asset class/liquidity; isolate high-notional names into dedicated shards.
- [ ] Map shards to GPUs one-to-one (or capped slices); avoid GPU oversubscription.
- [ ] Keep extraction/routing/execution on CPU; reserve GPUs for TS/backtest workloads.
- [ ] Per-shard DB mirror (temp SQLite) to prevent cross-shard corruption; roll up metrics post-run.
- [ ] Independent logs per shard; include run_id, GPU id, shard tickers.
- [ ] Health checks: DB integrity before launch; quant-validation gates on; no-trade windows enforced.

## Energy Efficiency
- [ ] Skip trading if realised trades for shard tickers >= target (trade-count gate).
- [ ] Short cycles + low sleep in diagnostics; increase only when evidence is scarce.
- [ ] Disable LLM fallback/redundancy for sampling runs; TS-only for speed/energy savings.
- [ ] Exclude high-notional names from default diagnostics unless capital is scaled.
- [ ] Auto-sleep between cycles; cap concurrent shards to GPU count.
- [ ] Use mid-price logging and slippage windows to identify/avoid costly time slots.

## Future Expansion
- [ ] Add GPU_LIST/TICKER_SHARDS config to scale GPUs or shards without code edits.
- [ ] Integrate with a scheduler (K8s/Slurm/Airflow) for queued shard runs and preemption.
- [ ] Add regime-aware min_expected_return bands per asset class driven by recent slippage/costs.
- [ ] Extend to crypto/EM shards with stricter caps and separate capital buckets.
- [ ] Add optional mixed precision for forecasters/backtests where numerically safe.
- [ ] Persist shard-level energy/latency telemetry for optimization.

## Runbook (local)
- [ ] Ensure `simpleTrader_env` and GPUs are available (`nvidia-smi`).
- [ ] Configure shards/GPU list and thresholds in `bash/run_gpu_parallel.sh` (or env overrides).
- [ ] Run `bash/bash/run_gpu_parallel.sh`.
- [ ] After completion: review `logs/auto_runs/auto_trader_*.log`, `logs/automation/slippage_windows.json`, `logs/automation/config_proposals.json`.

### Recommended defaults for current box (RTX 4060 Ti 16GB, CUDA 12.9)
- GPU list: `GPU_LIST=0` (single GPU, low util headroom).
- Shards: keep two light shards to reduce per-run load: `SHARD1="MTN,MSFT,AAPL"`, `SHARD2="CL=F"`.
- Concurrency: one shard at a time (current script maps round-robin; with one GPU it stays serial).
- Caps: `TARGET_TRADES=30`, `CYCLES=4-6`, `INITIAL_CAPITAL=25000-50000`, `SLEEP_SECONDS=10`.
- LLM: keep disabled for sampling runs to avoid extra GPU load.
- Monitoring: watch `nvidia-smi` for power/thermals; GPU util should stay low given TS workloads are mostly CPU-bound.

## Pointers
- Orchestrator script: `bash/run_gpu_parallel.sh`.
- Trade-count aware helper: `bash/auto_rebuild_and_sweep.sh` (skips runs if targets met).
- Routing thresholds: `config/signal_routing_config.yml` (per-ticker overrides, no-trade windows).
- Slippage analysis: `scripts/analyze_slippage_windows.py` (consumes `logs/automation/execution_log.jsonl`).
- Quant gates: `config/forecaster_monitoring.yml` + `scripts/check_quant_validation_health.py`.
