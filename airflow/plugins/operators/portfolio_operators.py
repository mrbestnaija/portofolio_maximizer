"""
Airflow operators for Portfolio Maximizer.

Design goals:
- TS-primary: wrap existing scripts without changing routing or bar-aware behaviour.
- Guardrails: respect PIPELINE_DEVICE, quant-health gates, synthetic/live separation.
- Compatibility: imports succeed even when Airflow is not installed (e.g., during lightweight CI),
  while still executing normally inside Airflow.
"""

from __future__ import annotations

import os
import subprocess
from typing import Dict, Iterable, List, Optional

try:
    from etl.pipeline_logger import PipelineLogger
except Exception:  # pragma: no cover
    PipelineLogger = None  # type: ignore

try:
    from airflow.models import BaseOperator
except ImportError:  # pragma: no cover
    class BaseOperator:  # type: ignore
        """Fallback for environments without Airflow; execute will raise."""

        def __init__(self, *args, **kwargs):
            self.task_id = kwargs.get("task_id")

        def execute(self, context):
            raise RuntimeError("Airflow is not installed; this is a stub BaseOperator.")


class ScriptOperatorBase(BaseOperator):
    """
    Minimal operator that executes a Python script with merged env and Airflow context.
    """

    template_fields = ("script_args", "env_vars")

    def __init__(
        self,
        task_id: str,
        python_callable_path: str,
        script_args: Optional[Iterable[str]] = None,
        env_vars: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(task_id=task_id, **kwargs)
        self.python_callable_path = python_callable_path
        self.script_args = list(script_args) if script_args else []
        self.env_vars = env_vars or {}
        self._logger = PipelineLogger(log_dir="logs/cron") if PipelineLogger else None

    def _build_command(self) -> List[str]:
        return ["python", self.python_callable_path, *self.script_args]

    def _build_env(self, context) -> Dict[str, str]:
        env = os.environ.copy()
        env.update(self.env_vars)
        # Run-scoped context for logging/DB writes.
        run_id = context.get("run_id")
        if not run_id and context.get("dag_run"):
            run_id = getattr(context["dag_run"], "run_id", None)
        if run_id:
            env["AIRFLOW_RUN_ID"] = str(run_id)
        if context.get("ds"):
            env["EXECUTION_DATE"] = str(context["ds"])
        elif context.get("execution_date"):
            env["EXECUTION_DATE"] = str(context["execution_date"])
        return env

    def execute(self, context):
        cmd = self._build_command()
        env = self._build_env(context)
        pipeline_id = env.get("AIRFLOW_RUN_ID", "airflow_unknown")
        stage = self.task_id
        if self._logger:
            try:
                self._logger.log_event(
                    event_type="stage_start",
                    pipeline_id=pipeline_id,
                    stage=stage,
                    status="info",
                    metadata={"cmd": cmd},
                )
            except Exception:
                pass
        try:
            result = subprocess.run(cmd, env=env, check=True, text=True)
            if self._logger:
                try:
                    self._logger.log_event(
                        event_type="stage_complete",
                        pipeline_id=pipeline_id,
                        stage=stage,
                        status="success",
                        metadata={"cmd": cmd},
                    )
                except Exception:
                    pass
            return result
        except Exception as exc:
            if self._logger:
                try:
                    self._logger.log_event(
                        event_type="stage_error",
                        pipeline_id=pipeline_id,
                        stage=stage,
                        status="error",
                        metadata={"cmd": cmd, "error": str(exc)},
                    )
                except Exception:
                    pass
            raise


class PortfolioETLOperator(ScriptOperatorBase):
    """
    Wraps scripts/run_etl_pipeline.py with guardrails intact.
    """

    def __init__(
        self,
        python_callable_path: str = "scripts/run_etl_pipeline.py",
        env_vars: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            python_callable_path=python_callable_path,
            env_vars=env_vars,
            **kwargs,
        )


class AutoTraderOperator(ScriptOperatorBase):
    """
    Wraps scripts/run_auto_trader.py (bar-aware, TS-primary).
    """

    def __init__(
        self,
        script_args: Optional[Iterable[str]] = None,
        python_callable_path: str = "scripts/run_auto_trader.py",
        env_vars: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            python_callable_path=python_callable_path,
            script_args=script_args,
            env_vars=env_vars,
            **kwargs,
        )


class MonitoringOperator(ScriptOperatorBase):
    """
    Wraps scripts/monitor_llm_system.py and related monitoring hooks.
    """

    def __init__(
        self,
        python_callable_path: str = "scripts/monitor_llm_system.py",
        env_vars: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            python_callable_path=python_callable_path,
            env_vars=env_vars,
            **kwargs,
        )
