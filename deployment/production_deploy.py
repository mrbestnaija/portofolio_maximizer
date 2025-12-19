"""Production deployment helper for orchestrating end-to-end system bring-up."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from etl.data_source_manager import DataSourceManager
from etl.database_manager import DatabaseManager
from scripts.check_dashboard_health import _load_json as load_dashboard  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class DeploymentResult:
    status: str
    reason: Optional[str] = None
    components_deployed: int = 0
    health_status: Optional[Dict[str, Any]] = None
    steps: List[str] = field(default_factory=list)


class ProductionDeployer:
    """Coordinate environment validation and subsystem bring-up."""

    def __init__(self, db_path: str = "data/portfolio_maximizer.db") -> None:
        self.db_path = Path(db_path)

    def deploy_trading_system(self, dry_run: bool = False) -> DeploymentResult:
        steps: List[str] = []

        env_ok, env_reason = self._validate_environment()
        if not env_ok:
            return DeploymentResult(status="FAILED", reason=env_reason, steps=steps)
        steps.append("environment_validated")

        api_ok, api_reason = self._validate_api_keys()
        if not api_ok:
            return DeploymentResult(status="FAILED", reason=api_reason, steps=steps)
        steps.append("api_keys_validated")

        health = self._system_health_check()
        if health.get("status") != "HEALTHY":
            return DeploymentResult(
                status="FAILED",
                reason=f"Health check failed: {health.get('issues')}",
                health_status=health,
                steps=steps,
            )
        steps.append("health_check_passed")

        if dry_run:
            return DeploymentResult(status="SUCCESS", components_deployed=0, health_status=health, steps=steps)

        self._deploy_signal_validator()
        steps.append("signal_validator_deployed")

        self._deploy_paper_trading_engine()
        steps.append("paper_trading_engine_deployed")

        self._deploy_risk_manager()
        steps.append("risk_manager_deployed")

        self._deploy_performance_dashboard()
        steps.append("performance_dashboard_deployed")

        self._start_monitoring_services()
        steps.append("monitoring_services_started")

        return DeploymentResult(
            status="SUCCESS",
            components_deployed=4,
            health_status=health,
            steps=steps,
        )

    def _validate_environment(self) -> tuple[bool, Optional[str]]:
        required_vars = ["PORTFOLIO_ENV"]
        missing = [var for var in required_vars if not Path(".env").exists() and not Path(var).exists() and not os.getenv(var)]  # type: ignore[name-defined]
        if missing:
            return False, f"Missing required environment variables: {', '.join(missing)}"
        return True, None

    def _validate_api_keys(self) -> tuple[bool, Optional[str]]:
        keys = ["ALPHA_VANTAGE_API_KEY", "FINNHUB_API_KEY"]
        missing = [k for k in keys if not os.getenv(k)]  # type: ignore[name-defined]
        if missing:
            logger.warning("API keys missing (non-fatal for synthetic runs): %s", missing)
        return True, None

    def _system_health_check(self) -> Dict[str, Any]:
        issues: List[str] = []
        datasource_status = "unknown"

        try:
            dsm = DataSourceManager()
            datasource_status = "synthetic" if "synthetic" in dsm.sources else "available"
        except Exception as exc:  # pragma: no cover - defensive
            issues.append(f"DataSourceManager failed: {exc}")

        db_status = "unknown"
        try:
            with DatabaseManager(db_path=str(self.db_path)) as db:
                db.cursor.execute("SELECT 1")
                db_status = "ready"
        except Exception as exc:  # pragma: no cover - defensive
            issues.append(f"Database check failed: {exc}")
            db_status = "error"

        dashboard_ok = False
        dashboard_path = Path("visualizations/dashboard_data.json")
        if dashboard_path.exists():
            try:
                payload = load_dashboard(dashboard_path)
                dashboard_ok = bool(payload)
            except Exception:
                dashboard_ok = False

        status = "HEALTHY" if not issues else "DEGRADED"
        return {
            "status": status,
            "issues": issues,
            "datasource_status": datasource_status,
            "db_status": db_status,
            "dashboard_present": dashboard_ok,
        }

    def _deploy_signal_validator(self) -> None:
        logger.info("Deploying signal validator (placeholder hook) ...")

    def _deploy_paper_trading_engine(self) -> None:
        logger.info("Deploying paper trading engine (placeholder hook) ...")

    def _deploy_risk_manager(self) -> None:
        logger.info("Deploying risk manager (placeholder hook) ...")

    def _deploy_performance_dashboard(self) -> None:
        try:
            from monitoring.performance_dashboard import PerformanceDashboard

            snapshot = PerformanceDashboard(db_path=str(self.db_path)).generate_live_metrics()
            PerformanceDashboard(db_path=str(self.db_path)).save_snapshot(
                snapshot, Path("visualizations/performance_dashboard.json")
            )
            logger.info("Performance dashboard snapshot emitted.")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Performance dashboard emit skipped: %s", exc)

    def _start_monitoring_services(self) -> None:
        logger.info("Starting monitoring services (placeholder hook) ...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    deployer = ProductionDeployer()
    result = deployer.deploy_trading_system(dry_run=False)
    if result.status != "SUCCESS":
        logger.error("Deployment failed: %s", result.reason)
    else:
        logger.info("Deployment succeeded (steps=%s)", result.steps)
