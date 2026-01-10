"""
Auto Trader DAG scaffold.

Notes:
- Uses AutoTraderOperator wrapper; preserves bar-aware, TS-first behaviour.
- Safe import when Airflow is absent.
"""

from __future__ import annotations

from datetime import timedelta

try:
    from airflow import DAG
    from airflow.utils.dates import days_ago
    from airflow.plugins.operators.portfolio_operators import AutoTraderOperator
except ImportError:  # pragma: no cover
    DAG = None
    AutoTraderOperator = None  # type: ignore[misc]


dag = None

if DAG and AutoTraderOperator:
    default_args = {
        "owner": "portfolio-team",
        "depends_on_past": False,
        "email_on_failure": True,
        "email_on_retry": False,
        "retries": 3,
        "retry_delay": timedelta(minutes=5),
    }

    dag = DAG(
        "auto_trader",
        default_args=default_args,
        description="Bar-aware auto-trader (TS-primary, LLM-fallback gated).",
        schedule_interval="*/30 7-20 * * 1-5",
        start_date=days_ago(1),
        catchup=False,
        tags=["trading", "auto", "production"],
    )

    _ = AutoTraderOperator(
        task_id="run_auto_trader",
        dag=dag,
        python_callable_path="scripts/run_auto_trader.py",
        script_args=["--core-mode"],
        env_vars={
            "PIPELINE_DEVICE": "{{ var.value.get('PIPELINE_DEVICE', 'cpu') }}",
            "ENABLE_GPU_PARALLEL": "{{ var.value.get('ENABLE_GPU_PARALLEL', '1') }}",
        },
    )
