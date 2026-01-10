import importlib


def test_daily_etl_dag_import_succeeds():
    mod = importlib.import_module("airflow.dags.etl.daily_etl_dag")
    assert hasattr(mod, "dag")
    # dag may be None when Airflow is not installed; import should still succeed.


def test_auto_trader_dag_import_succeeds():
    mod = importlib.import_module("airflow.dags.trading.auto_trader_dag")
    assert hasattr(mod, "dag")


def test_health_monitoring_dag_import_succeeds():
    mod = importlib.import_module("airflow.dags.monitoring.health_monitoring_dag")
    assert hasattr(mod, "dag")
