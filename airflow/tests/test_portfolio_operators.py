import os

import pytest

from airflow.plugins.operators.portfolio_operators import (
    AutoTraderOperator,
    PortfolioETLOperator,
)


def test_etl_operator_merges_env_and_context(monkeypatch):
    calls = []
    log_events = []

    class FakeLogger:
        def __init__(self, *args, **kwargs):
            pass

        def log_event(self, **kwargs):
            log_events.append(kwargs)

    def fake_run(cmd, env, check, text):
        calls.append((cmd, env, check, text))

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(
        "airflow.plugins.operators.portfolio_operators.subprocess.run", fake_run
    )
    monkeypatch.setattr(
        "airflow.plugins.operators.portfolio_operators.PipelineLogger", FakeLogger
    )
    op = PortfolioETLOperator(
        task_id="t",
        python_callable_path="scripts/run_etl_pipeline.py",
        env_vars={"PIPELINE_DEVICE": "cuda", "ENABLE_GPU_PARALLEL": "1"},
    )
    context = {"run_id": "airflow_run_123", "ds": "2026-01-08"}

    op.execute(context)

    assert calls, "subprocess.run should be invoked"
    cmd, env, check_flag, text_flag = calls[0]
    assert cmd[:2] == ["python", "scripts/run_etl_pipeline.py"]
    assert env["PIPELINE_DEVICE"] == "cuda"
    assert env["ENABLE_GPU_PARALLEL"] == "1"
    assert env["AIRFLOW_RUN_ID"] == "airflow_run_123"
    assert env["EXECUTION_DATE"] == "2026-01-08"
    assert check_flag is True
    assert text_flag is True
    assert any(e["event_type"] == "stage_start" for e in log_events)
    assert any(e["event_type"] == "stage_complete" for e in log_events)


def test_auto_trader_operator_keeps_script_args(monkeypatch):
    calls = []

    def fake_run(cmd, env, check, text):
        calls.append(cmd)

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(
        "airflow.plugins.operators.portfolio_operators.subprocess.run", fake_run
    )

    op = AutoTraderOperator(
        task_id="auto",
        script_args=["--core-mode"],
        env_vars={"PIPELINE_DEVICE": "cpu"},
    )

    op.execute({"run_id": "rid", "ds": "2026-01-08"})

    assert calls, "subprocess.run should be invoked"
    assert calls[0][-1] == "--core-mode"
