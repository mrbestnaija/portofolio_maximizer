"""
Health monitoring DAG scaffold.

Notes:
- Wraps scripts/monitor_llm_system.py via MonitoringOperator.
- Intended to pair with quant health checks and dashboard snapshot triggers.
"""

from __future__ import annotations

from datetime import timedelta

try:
    from airflow import DAG
    from airflow.utils.dates import days_ago
    from airflow.plugins.operators.portfolio_operators import MonitoringOperator
except ImportError:  # pragma: no cover
    DAG = None
    MonitoringOperator = None  # type: ignore[misc]


dag = None

if DAG and MonitoringOperator:
    default_args = {
        "owner": "portfolio-team",
        "depends_on_past": False,
        "email_on_failure": True,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }

    dag = DAG(
        "health_monitoring",
        default_args=default_args,
        description="LLM/system monitoring with alerting hooks.",
        schedule_interval="5 * * * *",
        start_date=days_ago(1),
        catchup=False,
        tags=["monitoring", "hourly"],
    )

    _ = MonitoringOperator(
        task_id="run_monitoring",
        dag=dag,
        python_callable_path="scripts/monitor_llm_system.py",
        env_vars={
            "PIPELINE_DEVICE": "{{ var.value.get('PIPELINE_DEVICE', 'cpu') }}",
            "ENABLE_GPU_PARALLEL": "{{ var.value.get('ENABLE_GPU_PARALLEL', '0') }}",
        },
    )
