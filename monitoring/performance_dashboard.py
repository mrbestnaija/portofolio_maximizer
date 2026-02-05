"""Performance dashboard data builder for live monitoring."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from etl.database_manager import DatabaseManager
from etl import portfolio_math

logger = logging.getLogger(__name__)


@dataclass
class DashboardSnapshot:
    metrics: Dict[str, Any]
    charts: Dict[str, Any]
    alerts: List[str]
    generated_at: str
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics,
            "charts": self.charts,
            "alerts": self.alerts,
            "generated_at": self.generated_at,
            "provenance": self.provenance,
        }


class PerformanceDashboard:
    """Generate a JSON-friendly snapshot of live performance."""

    def __init__(self, db_path: str = "data/portfolio_maximizer.db"):
        self.db_path = db_path

    def generate_live_metrics(
        self,
        lookback_days: int = 30,
        equity_limit: int = 500,
        signal_limit: int = 50,
    ) -> DashboardSnapshot:
        now = datetime.now(timezone.utc)
        start_date = (now - timedelta(days=lookback_days)).date().isoformat()

        with DatabaseManager(db_path=self.db_path) as db:
            provenance = db.get_data_provenance_summary()
            performance = db.get_performance_summary(start_date=start_date)
            equity_curve = db.get_equity_curve(start_date=start_date)
            pnl_history = self._fetch_realized_pnl_history(db, limit=equity_limit, start_date=start_date)
            signals = db.get_latest_signals(limit=signal_limit)
            quality_summary = self._load_quality_summary(db, start_date)
            latency_summary = self._load_latency_summary(db, start_date)

        portfolio_metrics, current_drawdown = self._compute_portfolio_metrics(equity_curve, pnl_history)
        signal_accuracy = self._calculate_signal_accuracy(signals)
        avg_confidence = self._average_confidence(signals)

        last_run = provenance.get("last_run_provenance") if isinstance(provenance, dict) else None
        active_run_id = last_run.get("run_id") if isinstance(last_run, dict) else None
        active_dataset = None
        generator_version = None
        if isinstance(last_run, dict):
            active_dataset = last_run.get("synthetic_dataset_id") or last_run.get("dataset_id")
            generator_version = last_run.get("synthetic_generator_version")

        metrics = {
            "total_trades": int(performance.get("total_trades", 0) or 0),
            "win_rate": float(performance.get("win_rate", 0.0) or 0.0),
            "profit_factor": float(performance.get("profit_factor", 0.0) or 0.0),
            "total_profit": float(performance.get("total_profit", 0.0) or 0.0),
            "avg_profit_per_trade": float(performance.get("avg_profit_per_trade", 0.0) or 0.0),
            "signal_accuracy": signal_accuracy,
            "avg_confidence": avg_confidence,
            "signal_count": len(signals),
            "current_drawdown": current_drawdown,
            "portfolio_volatility": portfolio_metrics.get("volatility"),
            "sharpe_ratio": portfolio_metrics.get("sharpe_ratio"),
            "max_drawdown": portfolio_metrics.get("max_drawdown"),
            "var_95": portfolio_metrics.get("var_95"),
            "data_quality_score": quality_summary.get("avg_quality"),
            "avg_latency_ms": latency_summary.get("avg_total_ms"),
            "data_origin": provenance.get("origin"),
            "active_run_id": active_run_id,
            "active_dataset_id": active_dataset,
            "synthetic_generator_version": generator_version,
        }
        metrics["profitability_proof"] = bool(metrics.get("data_origin") == "live")

        charts = {
            "equity_curve": equity_curve,
            "realized_pnl": pnl_history,
            "signals": self._summarize_signals(signals),
        }

        alerts = self._build_alerts(metrics, quality_summary, latency_summary)
        origin = metrics.get("data_origin")
        if origin in {"synthetic", "mixed"}:
            alerts.insert(0, "Synthetic data present: metrics are NOT profitability proof artifacts")
            metrics["profitability_proof"] = False

        return DashboardSnapshot(
            metrics=metrics,
            charts=charts,
            alerts=alerts,
            generated_at=now.isoformat(),
            provenance=provenance or {},
        )

    def save_snapshot(self, snapshot: DashboardSnapshot, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot.to_dict(), indent=2))
        return path

    def export_metrics_csv(self, snapshot: DashboardSnapshot, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            for key, value in snapshot.metrics.items():
                writer.writerow([key, value])
        return path

    def export_equity_curve_csv(self, snapshot: DashboardSnapshot, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        curve = snapshot.charts.get("equity_curve", [])
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", "equity"])
            for row in curve:
                writer.writerow([row.get("date"), row.get("equity")])
        return path

    def _fetch_realized_pnl_history(
        self,
        db: DatabaseManager,
        limit: int = 500,
        start_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT trade_date, realized_pnl, realized_pnl_pct, ticker, action
            FROM trade_executions
            WHERE realized_pnl IS NOT NULL
        """
        params: List[Any] = []
        if start_date:
            query += " AND trade_date >= ?"
            params.append(start_date)
        query += " ORDER BY trade_date DESC, id DESC LIMIT ?"
        params.append(limit)

        db.cursor.execute(query, params)
        rows = [dict(row) for row in db.cursor.fetchall()]
        return list(reversed(rows))

    def _load_quality_summary(self, db: DatabaseManager, start_date: str) -> Dict[str, Optional[float]]:
        query = """
            SELECT AVG(quality_score) AS avg_quality,
                   AVG(missing_pct) AS avg_missing_pct,
                   AVG(outlier_frac) AS avg_outlier_frac
            FROM data_quality_snapshots
            WHERE window_end >= ?
        """
        try:
            db.cursor.execute(query, (start_date,))
            row = db.cursor.fetchone()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Unable to load data quality summary: %s", exc)
            return {}

        if not row:
            return {}
        return {
            "avg_quality": row["avg_quality"],
            "avg_missing_pct": row["avg_missing_pct"],
            "avg_outlier_frac": row["avg_outlier_frac"],
        }

    def _load_latency_summary(self, db: DatabaseManager, start_date: str) -> Dict[str, Optional[float]]:
        query = """
            SELECT AVG(ts_ms) AS avg_ts_ms,
                   AVG(llm_ms) AS avg_llm_ms
            FROM latency_metrics
            WHERE recorded_at >= ?
        """
        try:
            db.cursor.execute(query, (start_date,))
            row = db.cursor.fetchone()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Unable to load latency summary: %s", exc)
            return {}

        if not row:
            return {}
        avg_ts = row["avg_ts_ms"]
        avg_llm = row["avg_llm_ms"]
        avg_total = None
        if avg_ts is not None or avg_llm is not None:
            avg_total = float((avg_ts or 0.0) + (avg_llm or 0.0))
        return {
            "avg_ts_ms": avg_ts,
            "avg_llm_ms": avg_llm,
            "avg_total_ms": avg_total,
        }

    def _compute_portfolio_metrics(
        self,
        equity_curve: List[Dict[str, Any]],
        pnl_history: List[Dict[str, Any]],
    ) -> tuple[Dict[str, Any], Optional[float]]:
        equity = np.array([row.get("equity", 0.0) for row in equity_curve], dtype=float)
        returns = None
        current_drawdown = None

        if len(equity) > 1:
            returns = self._equity_returns(equity)
            current_drawdown = self._current_drawdown(equity)
        else:
            pnl_returns = [
                row.get("realized_pnl_pct")
                for row in pnl_history
                if row.get("realized_pnl_pct") is not None
            ]
            if pnl_returns:
                returns = np.asarray(pnl_returns, dtype=float)

        metrics: Dict[str, Any] = {}
        if returns is not None and len(returns) > 0:
            returns = np.asarray(returns, dtype=float).reshape(-1, 1)
            try:
                metrics = portfolio_math.calculate_enhanced_portfolio_metrics(returns, np.array([1.0]))
            except ValueError as exc:
                logger.debug("Portfolio metrics unavailable: %s", exc)

        return metrics, current_drawdown

    @staticmethod
    def _equity_returns(equity: np.ndarray) -> np.ndarray:
        prev = equity[:-1]
        delta = equity[1:] - equity[:-1]
        returns = np.zeros_like(delta)
        mask = prev != 0
        returns[mask] = delta[mask] / prev[mask]
        return returns

    @staticmethod
    def _current_drawdown(equity: np.ndarray) -> float:
        if equity.size == 0:
            return 0.0
        running_max = np.maximum.accumulate(equity)
        drawdowns = np.zeros_like(equity)
        mask = running_max != 0
        drawdowns[mask] = 1 - (equity[mask] / running_max[mask])
        return float(drawdowns[-1])

    @staticmethod
    def _calculate_signal_accuracy(signals: List[Dict[str, Any]]) -> Optional[float]:
        correct = 0
        scored = 0
        for signal in signals:
            action = (signal.get("action") or signal.get("signal_type") or "").upper()
            actual_return = signal.get("actual_return")
            if actual_return is None or action not in {"BUY", "SELL", "HOLD"}:
                continue
            scored += 1
            if action == "BUY" and actual_return > 0:
                correct += 1
            elif action == "SELL" and actual_return < 0:
                correct += 1
            elif action == "HOLD" and abs(actual_return) < 1e-6:
                correct += 1
        if scored == 0:
            return None
        return correct / scored

    @staticmethod
    def _average_confidence(signals: List[Dict[str, Any]]) -> Optional[float]:
        values = []
        for signal in signals:
            confidence = signal.get("confidence")
            if confidence is None:
                continue
            try:
                values.append(float(confidence))
            except (TypeError, ValueError):
                continue
        if not values:
            return None
        return float(sum(values) / len(values))

    @staticmethod
    def _summarize_signals(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        summary = []
        for signal in signals[:10]:
            summary.append(
                {
                    "ticker": signal.get("ticker"),
                    "action": signal.get("action"),
                    "confidence": signal.get("confidence"),
                    "signal_date": signal.get("signal_date"),
                    "actual_return": signal.get("actual_return"),
                }
            )
        return summary

    @staticmethod
    def _build_alerts(
        metrics: Dict[str, Any],
        quality_summary: Dict[str, Any],
        latency_summary: Dict[str, Any],
    ) -> List[str]:
        alerts: List[str] = []
        data_quality = metrics.get("data_quality_score")
        if data_quality is not None and data_quality < 0.9:
            alerts.append("Data quality score below 0.90")
        if metrics.get("profit_factor") is not None and metrics["profit_factor"] < 1.0:
            alerts.append("Profit factor below 1.0")
        if metrics.get("current_drawdown") is not None and metrics["current_drawdown"] > 0.2:
            alerts.append("Current drawdown above 20%")

        avg_total = latency_summary.get("avg_total_ms")
        if avg_total is not None and avg_total > 5000:
            alerts.append("Average latency above 5s")

        if quality_summary.get("avg_missing_pct") is not None and quality_summary["avg_missing_pct"] > 0.05:
            alerts.append("Missing data above 5% in quality snapshots")

        return alerts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a performance dashboard snapshot.")
    parser.add_argument("--db-path", default="data/portfolio_maximizer.db", help="Database path.")
    parser.add_argument("--lookback-days", type=int, default=30, help="Lookback window in days.")
    parser.add_argument("--out-json", default="visualizations/performance_dashboard.json", help="Output JSON path.")
    parser.add_argument("--out-metrics-csv", default="visualizations/performance_metrics.csv", help="Output metrics CSV path.")
    parser.add_argument("--out-equity-csv", default="visualizations/performance_equity_curve.csv", help="Output equity curve CSV path.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = _parse_args()
    dashboard = PerformanceDashboard(db_path=args.db_path)
    snapshot = dashboard.generate_live_metrics(lookback_days=args.lookback_days)
    json_path = dashboard.save_snapshot(snapshot, Path(args.out_json))
    dashboard.export_metrics_csv(snapshot, Path(args.out_metrics_csv))
    dashboard.export_equity_curve_csv(snapshot, Path(args.out_equity_csv))
    logger.info("Performance dashboard snapshot written to %s", json_path)


if __name__ == "__main__":
    main()
