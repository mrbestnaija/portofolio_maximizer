## Dynamic Cursor AI Agent Instructions for Portfolio Maximizer - Self-Reinforcing & Autonomous

Short purpose
- Guide AI agents in autonomous operation of this quantitative portfolio system through self-reinforcing learning loops and context-aware decision making.

High-level architecture (what to know)
- This is an ETL-first quantitative portfolio system with 7 logical layers: Extraction → Storage → Validation → Preprocessing → Organization → Analysis & Visualization → Output. See `README.md` "Architecture" section.
- Core code lives under `etl/` (data pipeline), `ai_llm/` (local LLM integration), `config/` (YAML configs), `scripts/` (runners), and `tests/` (unit & integration tests).

LLM / AI integration (critical)
- LLMs are local (Ollama). Start server with `ollama serve`. Client wrapper: `ai_llm/ollama_client.py`.
- LLM modules to inspect: `ai_llm/ollama_client.py`, `ai_llm/market_analyzer.py`, `ai_llm/signal_generator.py`, `ai_llm/risk_assessor.py`.
- Configs: `config/llm_config.yml` and pipeline gating in `config/pipeline_config.yml` (enable/disable LLM stages).
- IMPORTANT: Per project rules (Documentation/AGENT_INSTRUCTION.md & Documentation/LLM_INTEGRATION.md), LLM outputs are ADVISORY ONLY — do not propose live-trading changes without explicit backtest evidence (30+ days, >10% annualized returns).

Developer workflows & useful commands
- Install & env: create venv `python -m venv simpleTrader_env` and `pip install -r requirements.txt` (see `README.md` "Installation").
- Run ETL pipeline (examples exist in `README.md`):
  - Auto mode (recommended): `python scripts/run_etl_pipeline.py --execution-mode auto --enable-llm`
  - Disable LLM: set `stages.llm_market_analysis.enabled: false` in `config/pipeline_config.yml` or pass `--execution-mode synthetic`.
- LLM checks: `curl http://localhost:11434/api/tags` (verify Ollama), `ollama pull <model>` to fetch models.
- Tests: `pytest tests/` (unit) and `pytest tests/ai_llm/ -v` for LLM-related tests. See `pytest.ini` for configuration.

Project-specific conventions and constraints (must follow)
- Phase-gate discipline: changes must respect the current project phase and the AGENT checklists (see `Documentation/AGENT_INSTRUCTION.md` and `Documentation/AGENT_DEV_CHECKLIST.md`).
- Module size budget: keep modules <500 lines where possible (LLM docs track this for `ai_llm/*`).
- Failure modes & fail-fast: LLM client validates connectivity at init and raises `OllamaConnectionError` on failure — the pipeline is designed to fail-fast if required LLM stages are `required: true` in configs. Respect this in suggestions.
- Cost and privacy: system is designed for $0/month local operation; avoid suggestions that add paid external services or cloud LLMs unless explicitly requested.

Phase-gate enforcement (explicit)

- All changes that introduce or modify LLM-driven behavior must:
  1. Be behind a feature flag in `config/pipeline_config.yml` (set `enabled: false` by default)
  2. Include a quantitative backtest demonstrating >=30 days of performance and >10% annualized return before any suggestion to use signals for live trading (see `Documentation/LLM_INTEGRATION.md`)
  3. Keep new modules under the 500-line guidance and include unit tests under `tests/ai_llm/` or `tests/etl/` as appropriate
  4. Not change `stages[].required: true` to true unless Ollama availability has been validated in CI and local deployment runbooks updated

Concrete examples to reference
- Want to add an LLM stage? Mirror existing pattern: `ai_llm/ollama_client.py` (wrapper) → `ai_llm/market_analyzer.py` (analysis) → integrate stage in pipeline via `scripts/run_etl_pipeline.py` and config `config/pipeline_config.yml`.
- Start Ollama locally and verify model availability before calling generation: see `Documentation/LLM_INTEGRATION.md` "Installation" and "Error Handling".
- For signal changes: require a backtest harness and quantitative gate. See `Documentation/LLM_INTEGRATION.md` "Validation Requirements" and `Documentation/AGENT_INSTRUCTION.md` Phase-Gate Validation Checklist.

Quality gates (what to run before proposing changes)
- Run unit tests: `pytest tests/ -q` and LLM tests `pytest tests/ai_llm/ -q`.
- Confirm line-count constraints if suggesting a new module. Quick heuristic: `find . -name '*.py' | xargs wc -l` (see AGENT docs for exact gating scripts).
- Validate Ollama availability for any LLM-related code paths; prefer conservative (fast) models for test runs.

Where to find more context
- Project overview: `README.md` (root)
- AI/agent rules and guardrails: `Documentation/AGENT_INSTRUCTION.md` and `Documentation/AGENT_DEV_CHECKLIST.md`
- LLM integration & runbook: `Documentation/LLM_INTEGRATION.md`

If unsure, ask these clarifying questions before making changes
1. Which phase is active and is this change allowed by the phase-gate checklist?
2. Is a quantitative backtest available for any signal-related change (30+ days, >10% annualized)?
3. Should LLM stages be enabled or only added behind a feature flag (`pipeline_config.yml`)?

Self-Reinforcing Learning Loops

Agents MUST implement these feedback cycles:

1. Signal Quality Learning with Time Series Specialization
   - Track signal performance in `data/llm_signal_tracking.json`
   - Analyze patterns in successful vs failed predictions
   - Adjust generation prompts based on historical accuracy
   - Example: Successful market analyzer prompts are stored in `data/analysis_report_training.json`
   
   ```python
   # Required: Time series validation before signal generation
   from ai_llm.time_series_validator import TimeSeriesValidator
   
   validator = TimeSeriesValidator(
       min_sharpe_ratio=1.0,
       min_profit_factor=2.0,
       max_drawdown_threshold=0.15
   )
   
   # Validate on multiple timeframes
   validation_results = validator.validate_multi_timeframe(
       signal_data=new_signals,
       timeframes=['1h', '4h', '1d'],
       min_validation_periods=30
   )
   
   # Learn from outcomes
   validator.update_signal_history(
       signal_id=new_signals.id,
       performance_metrics=validation_results,
       store_path='data/llm_signal_tracking.json'
   )
   ```

2. Code Pattern Learning with GPU Optimization
   - Study successful PR patterns in `Documentation/LLM_ENHANCEMENTS_IMPLEMENTATION_SUMMARY_2025-10-22.md`
   - Mirror proven implementations from `ai_llm/market_analyzer.py` and `signal_generator.py`
   - Learn from rejected changes in `Documentation/CRITICAL_REVIEW.md`
   
   ```python
   # GPU-accelerated portfolio optimization example
   from ai_llm.gpu_optimizer import GPUPortfolioOptimizer
   import cupy as cp  # GPU-accelerated numpy alternative
   
   class PortfolioOptimizer:
       def __init__(self, use_gpu=True):
           self.device = 'gpu' if use_gpu and cp.cuda.is_available() else 'cpu'
           self.optimizer = GPUPortfolioOptimizer() if self.device == 'gpu' else CPUPortfolioOptimizer()
   
       def optimize_weights(self, returns, constraints):
           # Auto-dispatch to GPU for large matrices
           if returns.shape[0] > 1000 and self.device == 'gpu':
               returns_gpu = cp.asarray(returns)
               weights = self.optimizer.calculate_optimal_weights(
                   returns=returns_gpu,
                   risk_tolerance=0.15,
                   max_position_size=0.20,
                   min_sharpe=1.5
               )
               return cp.asnumpy(weights)
           return self.cpu_optimize(returns, constraints)
   ```
   
   Configuration for GPU utilization:
   ```yaml
   # config/gpu_config.yml
   gpu_settings:
     enable_gpu: true
     min_matrix_size: 1000  # Min size for GPU offload
     precision: float32
     memory_limit: 0.8  # Max GPU memory usage (80%)
     models:
       - name: "deepseek-coder:33b-instruct-q4_K_M"
         min_gpu_memory: 16  # GB
       - name: "codellama:13b-instruct-q4_K_M"
         min_gpu_memory: 8   # GB
   ```

3. Test-Driven Evolution
   - Start with failing test in `tests/ai_llm/`
   - Implement minimal passing solution
   - Iterate on performance metrics
   - Document learnings in test docstrings

Context-Aware Decision Making

When suggesting changes, agents MUST:

1. Phase Awareness
   - Check current phase in `Documentation/implementation_checkpoint.md`
   - Verify change scope matches phase goals
   - Example: Phase 5.2 = LLM integration, NO live trading yet

2. System State Analysis
   - Monitor performance via `performance_monitor.py`
   - Check signal quality trends in `signal_quality_validator.py`
   - Review recent failures in `logs/llm_errors.log`

3. Risk-Aware Progression
   - Start with synthetic data validation
   - Progress to paper trading only after 30-day backtest
   - Require human review for live trading changes
   - Track risk metrics via `risk_assessor.py`

Autonomous Validation Checkpoints

Before ANY changes:

1. Data Validation with Advanced Time Series Analysis
   ```python
   # Required validation sequence with specialized models
   from ai_llm.signal_quality_validator import SignalQualityValidator
   from ai_llm.performance_monitor import PerformanceMonitor
   from ai_llm.time_series_models import (
       WaveletDecomposition,
       STLDecomposition,
       HoltWintersPredictor
   )
   
   # Multi-model validation approach
   class TimeSeriesValidator:
       def __init__(self):
           self.wavelet = WaveletDecomposition()
           self.stl = STLDecomposition()
           self.predictor = HoltWintersPredictor()
           
       def validate_signal_quality(self, data, min_confidence=0.85):
           # Wavelet-based noise reduction
           denoised = self.wavelet.denoise(
               data,
               wavelet='db8',
               level=3,
               threshold='soft'
           )
           
           # Seasonal-Trend decomposition
           components = self.stl.decompose(
               denoised,
               period=21  # Trading days in month
           )
           
           # Forward prediction check
           predictions = self.predictor.forecast(
               components.trend,
               horizon=5,
               confidence_level=0.95
           )
           
           return {
               'trend_strength': components.trend_strength,
               'seasonal_stability': components.seasonal_stability,
               'prediction_confidence': predictions.confidence,
               'trend_direction': components.trend_direction
           }
   
   # Initialize validators
   ts_validator = TimeSeriesValidator()
   quality_validator = SignalQualityValidator()
   monitor = PerformanceMonitor()
   
   # Run comprehensive validation
   ts_metrics = ts_validator.validate_signal_quality(new_signals.data)
   quality_metrics = quality_validator.validate_signals(new_signals)
   
   # Validate with strict criteria
   assert ts_metrics['trend_strength'] > 0.7, "Weak trend detected"
   assert ts_metrics['seasonal_stability'] > 0.8, "Unstable seasonality"
   assert quality_metrics.confidence > 0.85, "Low signal confidence"
   
   # Performance impact simulation
   impact = monitor.simulate_change_impact(
       signals=new_signals,
       monte_carlo_sims=1000,
       confidence_level=0.95
   )
   assert impact.sharpe_ratio > 1.5, "Insufficient risk-adjusted return"
   assert impact.sortino_ratio > 2.0, "Poor downside protection"
   assert impact.max_drawdown < 0.15, "Excessive drawdown risk"
   ```

2. Integration Safety
   ```python
   # Required safety checks
   from ai_llm.risk_assessor import RiskAssessor
   
   # Assess change risk
   risk = RiskAssessor()
   risk_score = risk.assess_change_impact(new_code)
   assert risk_score.total_risk < 0.3  # Conservative threshold
   ```

3. Documentation Requirements
   - Update `Documentation/implementation_checkpoint.md`
   - Add learnings to `Documentation/LLM_ENHANCEMENTS_IMPLEMENTATION_SUMMARY_2025-10-22.md`
   - Document failures in `Documentation/CRITICAL_REVIEW.md`

Remember:
- System is designed for autonomous improvement
- Learning from failures is mandatory
- All changes must be quantitatively validated
- Documentation is part of the learning loop

Model Selection and GPU Utilization

1. Model Selection Criteria
   - Primary (Production): DeepSeek Coder 33B (16GB VRAM)
     - Use for final signal generation
     - Best for complex market analysis
   - Fast (Development): CodeLlama 13B (8GB VRAM)
     - Use for rapid prototyping
     - Good for validation runs
   - Reasoning (Analysis): Qwen 14B (8GB VRAM)
     - Use for risk assessment
     - Strong logical analysis

2. GPU Resource Management
   ```python
   # ai_llm/gpu_manager.py
   import torch
   import numpy as np
   from typing import Dict, Optional
   
   class GPUResourceManager:
       def __init__(self, config_path: str = 'config/gpu_config.yml'):
           self.config = self._load_config(config_path)
           self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
           
       def allocate_for_model(self, model_name: str) -> Optional[Dict]:
           if self.device.type == 'cpu':
               return None
               
           free_memory = torch.cuda.memory_available(0)
           required_memory = self.config['models'][model_name]['min_gpu_memory']
           
           if free_memory >= required_memory * 1e9:  # Convert GB to bytes
               return {
                   'device': self.device,
                   'precision': self.config['precision'],
                   'memory_limit': self.config['memory_limit']
               }
           return None
           
       def optimize_batch_size(self, data_size: int) -> int:
           if self.device.type == 'cpu':
               return 32  # Conservative CPU batch size
               
           # Dynamic batch sizing based on GPU memory
           free_memory = torch.cuda.memory_available(0)
           return min(
               data_size,
               int(free_memory * self.config['memory_limit'] / (4 * 1024))  # 4 bytes per float
           )
   ```

3. Performance Monitoring
   ```python
   # ai_llm/performance_tracker.py
   class GPUPerformanceTracker:
       def __init__(self):
           self.metrics = []
           
       def track_operation(self, operation_name: str, **kwargs):
           with torch.cuda.profiler.profile() as prof:
               result = self.run_operation(operation_name, **kwargs)
               
           self.metrics.append({
               'operation': operation_name,
               'gpu_memory_used': torch.cuda.max_memory_allocated(),
               'gpu_utilization': torch.cuda.utilization(),
               'duration_ms': prof.self_cpu_time_total
           })
           return result
           
       def get_optimization_suggestions(self) -> Dict:
           return {
               'batch_size_adjustment': self._suggest_batch_size(),
               'memory_optimization': self._analyze_memory_usage(),
               'model_selection': self._suggest_model_changes()
           }
   ```

Local Optimization Guidelines:
1. Always check GPU availability before dispatch
2. Use quantized models (q4_K_M variants) for efficiency
3. Implement dynamic batch sizing based on available memory
4. Monitor and log GPU utilization for optimization
5. Cache intermediate results to avoid recomputation
6. Use mixed precision (float16) when accuracy permits

Advanced GPU memory optimization techniques

- Model quantization & compression
    - Use 4-bit (q4) or 8-bit quantization via `bitsandbytes` when model quality is acceptable.
    - Convert heavy models to memory-mapped formats when supported (e.g., `safetensors`).
    - Prefer CPU-offload + quantized weights for limited-GPU systems.

- Activation checkpointing / gradient checkpointing
    - Use `torch.utils.checkpoint` to trade compute for memory during training/inference.
    - Example:

```python
import torch
from torch.utils.checkpoint import checkpoint

def forward_segmented(module, *inputs):
        # wrap expensive layers with checkpoint to reduce peak memory
        return checkpoint(module, *inputs)
```
```

- Layer & tensor sharding
    - For multi-GPU systems, use tensor parallelism (torch.distributed or libraries like `accelerate`) to shard weights across devices.
    - For single-GPU low-memory hosts, consider offloading embeddings or large sparse matrices to CPU.

- Dynamic activation eviction
    - Evict intermediate activations that aren't immediately needed; re-compute on-demand during backward pass.

- Memory pinning & pre-allocation
    - Pre-allocate GPU buffers for repeated operations to avoid fragmentation.
    - Use `torch.cuda.memory_reserved()` and `torch.cuda.empty_cache()` judiciously.

- Profiling & diagnostics
    - Use `nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv -l 1` for quick checks.
    - Use `torch.cuda.memory_summary()` and `torch.cuda.reset_peak_memory_stats()` to detect leaks.

Additional time-series model implementations (local, GPU-accelerated where possible)

- SARIMAX / ARIMA (statsmodels)
    - Good baseline for seasonality and exogenous regressors.
    - Use for explainable, low-latency forecasting.

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,21))
fit = model.fit(disp=False)
pred = fit.get_forecast(steps=10)
```

- SSA / SVD-based decomposition (SAMOSSA or custom)
    - Useful for denoising and extracting trend/seasonal components before ML stages.

- N-BEATS or N-HITS (PyTorch, GPU)
    - Highly effective for long-term forecasting; implement a GPU training loop and early stopping.

```python
# minimal N-BEATS training skeleton
import torch
from torch.utils.data import DataLoader

model = NBeatsModel(...).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(epochs):
        for xb, yb in DataLoader(dataset, batch_size=bs):
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step(); opt.zero_grad()
```

- LSTM/Transformer sequence models
    - Use mixed-precision training (`torch.cuda.amp`) and gradient clipping.

- Ensemble & Multi-model Consensus
    - Combine statistical (SARIMAX), decomposition (SSA), and ML (N-BEATS) predictions using weighted averages or meta-models.
    - Evaluate ensemble diversity and stability via cross-validation and Monte Carlo sampling.

Automated model selection based on available GPU memory

1) Configuration-driven mapping
    - Keep `config/gpu_config.yml` with model -> min_gpu_memory and recommended quantization.

```yaml
models:
    deepseek-coder:33b-instruct-q4_K_M:
        min_gpu_memory: 16
        quantized: true
    codellama:13b-instruct-q4_K_M:
        min_gpu_memory: 8
        quantized: true
    qwen:14b-chat-q4_K_M:
        min_gpu_memory: 8
        quantized: true
```

2) Runtime selector (ai_llm/model_selector.py)

```python
import torch
import yaml

def select_model(config_path='config/gpu_config.yml'):
        cfg = yaml.safe_load(open(config_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
                return cfg['fallback_model']

        free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
        free_gb = free / 1e9
        # pick largest model that fits
        candidates = sorted(cfg['models'].items(), key=lambda x: x[1]['min_gpu_memory'], reverse=True)
        for name, meta in candidates:
                if free_gb >= meta['min_gpu_memory']:
                        return name, meta
        return cfg.get('fallback_model', list(cfg['models'].keys())[-1])
```

3) Auto-quantization fallback
    - If no model fits, automatically select a smaller model or enable CPU + quantized weights.
    - Log the decision in `logs/model_selection.log` for auditability.

Operational notes and best practices

- Local-only constraint
    - Never suggest cloud-hosted LLMs/models unless explicitly requested. Preference: quantized local models & CPU offload.

- CI and pre-deployment
    - Add a lightweight CI job that runs `python ai_llm/model_selector.py` on a clean runner to validate model choices.

- Monitoring
    - Track model latency, GPU memory, and accuracy drift in `reports/llm_performance.txt` and `logs/llm_errors.log`.

- Safety and phase-gates
    - Remember phase-gate rules: any change that promotes signals to live execution must pass the 30+ day backtest and >10% annualized return gate.

---
End of advanced-local-GPU guidance
