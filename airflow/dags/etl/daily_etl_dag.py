"""
Daily ETL DAG scaffold.

Notes:
- TS-primary, uses PortfolioETLOperator wrapper.
- Safe to import when Airflow is absent: dag will be None.
"""

from __future__ import annotations

from datetime import timedelta

try:
    from airflow import DAG
    from airflow.utils.dates import days_ago
    from airflow.plugins.operators.portfolio_operators import PortfolioETLOperator
except ImportError:  # pragma: no cover
    DAG = None
    PortfolioETLOperator = None  # type: ignore[misc]


dag = None

if DAG and PortfolioETLOperator:
    default_args = {
        "owner": "portfolio-team",
        "depends_on_past": False,
        "email_on_failure": True,
        "email_on_retry": False,
        "retries": 3,
        "retry_delay": timedelta(minutes=5),
    }

    dag = DAG(
        "daily_etl",
        default_args=default_args,
        description="Daily ETL pipeline refresh (TS-first architecture)",
        schedule_interval="15 5 * * 1-5",
        start_date=days_ago(1),
        catchup=False,
        tags=["etl", "daily", "production", "ts-first"],
    )

    _ = PortfolioETLOperator(
        task_id="run_daily_etl",
        dag=dag,
        python_callable_path="scripts/run_etl_pipeline.py",
        env_vars={
            "PIPELINE_DEVICE": "{{ var.value.get('PIPELINE_DEVICE', 'cpu') }}",
            "ENABLE_GPU_PARALLEL": "{{ var.value.get('ENABLE_GPU_PARALLEL', '0') }}",
        },
        script_args=[],
    )
