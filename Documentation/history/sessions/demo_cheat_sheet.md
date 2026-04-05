Demo Cheat Sheet
Recommended file name:
C:\Users\Bestman\personal_projects\portfolio_maximizer_v45\portfolio_maximizer_v45\Documentation\history\sessions\SESSION_DEMO_CHEAT_SHEET_2026_04_04.md

Preparation
Open terminal in:
portfolio_maximizer_v45
Use Python environment with dependencies from requirements.txt / requirements-ml.txt
Confirm repo status:
python -m pytest tests/ --tb=short -q -m "not slow and not integration"
Validation demo
Validate pipeline inputs:
python scripts/validate_pipeline_inputs.py
Machine-readable mode:
python scripts/validate_pipeline_inputs.py --json
Purpose: show pre-flight checks before model training or trading
Model health demo
Run unified model improvement audit:
python scripts/check_model_improvement.py
For JSON artifact:
python scripts/check_model_improvement.py --json
Use baseline snapshot if available:
python scripts/check_model_improvement.py --save-baseline logs/baseline_$(date +%Y%m%d).json
Production gate demo
Run the production readiness gate:
python scripts/production_audit_gate.py
For machine-readable output:
python scripts/production_audit_gate.py --json
Focus: how forecast reliability + profitability proof combine into a readiness result
Autonomous trading demo
Quick synthetic run:
python scripts/run_auto_trader.py --tickers AAPL,MSFT --cycles 1 --execution-mode synthetic --proof-mode --verbose
If you want live/stock data and local environment ready:
python scripts/run_auto_trader.py --tickers AAPL,MSFT --cycles 1 --proof-mode --verbose
Key points:
auto pipeline from data to signal to trade decision
safety controls with --proof-mode
resume state with --resume / fresh start with --no-resume
Forecast & classifier demo
Evaluate directional classifier:
python scripts/evaluate_directional_classifier.py
If available, inspect classifier artifacts or metrics
Visualization demo
Render a dataset overview:
python scripts/visualize_dataset.py --ticker AAPL --overview --save
Purpose: show data quality and stationarity checks visually
Orchestration / operations notes
Scheduled Windows/PowerShell workflows:
bash/overnight_classifier_bootstrap.ps1
scripts/run_openclaw_maintenance.ps1
OpenClaw readiness:
python scripts/openclaw_production_readiness.py --json
Dashboard manager:
python scripts/windows_dashboard_manager.py ensure --port 8000
Narrative bullets for demo
Show how validate_pipeline_inputs.py prevents bad data from poisoning the ML pipeline
Show how check_model_improvement.py quantifies ensemble quality and gate health
Show how production_audit_gate.py creates a production-ready pass/fail artifact
Show how run_auto_trader.py converts forecasts into an end-to-end loop