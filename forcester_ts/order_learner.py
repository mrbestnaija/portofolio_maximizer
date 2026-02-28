"""
order_learner.py
----------------
AIC-weighted order cache for GARCH, SARIMAX, and SAMOSSA residual models.

Stores per (ticker, model_type, regime) order statistics in the
model_order_stats SQLite table and provides warm-start suggestions
to reduce or skip grid search on subsequent fits.

Usage:
    learner = OrderLearner(db_path="data/portfolio_maximizer.db")
    # After a fit:
    learner.record_fit("AAPL", "GARCH", "MODERATE_TRENDING",
                       {"p": 1, "q": 1, "dist": "skewt", "mean": "AR"},
                       aic=287.6, bic=307.4, n_obs=240)
    # Before next fit:
    suggestion = learner.suggest("AAPL", "GARCH", "MODERATE_TRENDING")
    # suggestion = {"p": 1, "q": 1, "dist": "skewt", "mean": "AR"}
"""
from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# sentinel for "regime-agnostic" fallback rows
_NO_REGIME = "__none__"
_MIN_FITS_FLOOR = 3
_SKIP_GRID_THRESHOLD_FLOOR = 5
_MAX_ORDER_AGE_DAYS_FLOOR = 1


def _regime_key(regime: str | None) -> str:
    return regime if regime else _NO_REGIME


def _canonical_json(d: dict) -> str:
    """Stable JSON key for an order_params dict."""
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def _candidate_regimes(regime: str | None) -> tuple[str, ...]:
    """
    Return the lookup order for regime-specific cache rows.

    Backward compatibility:
    - Old backfills used "UNKNOWN" for missing regime.
    - Current runtime stores the explicit sentinel `_NO_REGIME`.
    """
    primary = _regime_key(regime)
    ordered: list[str] = [primary]
    if primary != _NO_REGIME:
        ordered.append(_NO_REGIME)
        ordered.append("UNKNOWN")
    else:
        ordered.append("UNKNOWN")
    # Preserve order while de-duplicating.
    return tuple(dict.fromkeys(ordered))


def _coerce_floor(value: Any, floor: int) -> int:
    """Clamp config values so runtime callers cannot lower production guardrails."""
    try:
        parsed = int(value)
    except Exception:
        parsed = floor
    return max(floor, parsed)


class OrderLearner:
    """
    Persist and retrieve best-fit model orders per (ticker, model_type, regime).

    Backend: model_order_stats SQLite table (created by
    scripts/migrate_add_model_order_stats.py).

    Thread-safety: one UPSERT write per fit; reads are SELECT only.
    """

    def __init__(self, db_path: str, config: dict | None = None) -> None:
        self._db_path = str(db_path)
        cfg = config or {}
        self._min_fits = _coerce_floor(
            cfg.get("min_fits_to_suggest", _MIN_FITS_FLOOR),
            _MIN_FITS_FLOOR,
        )
        self._max_age_days = _coerce_floor(
            cfg.get("max_order_age_days", 90),
            _MAX_ORDER_AGE_DAYS_FLOOR,
        )
        self._skip_grid_threshold = _coerce_floor(
            cfg.get("skip_grid_threshold", _SKIP_GRID_THRESHOLD_FLOOR),
            _SKIP_GRID_THRESHOLD_FLOOR,
        )
        self._skip_grid_threshold = max(self._skip_grid_threshold, self._min_fits)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _connect_rw(self):
        import sys
        ROOT = Path(self._db_path).resolve().parent.parent
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from integrity.sqlite_guardrails import guarded_sqlite_connect, apply_sqlite_guardrails
        conn = guarded_sqlite_connect(self._db_path, timeout=5.0, enable_guardrails=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        apply_sqlite_guardrails(conn, allow_schema_changes=False)
        return conn

    def _connect_ro(self):
        import sys
        ROOT = Path(self._db_path).resolve().parent.parent
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from integrity.sqlite_guardrails import guarded_sqlite_connect, apply_sqlite_guardrails
        conn = guarded_sqlite_connect(self._db_path, timeout=2.0, enable_guardrails=False)
        conn.execute("PRAGMA busy_timeout=2000")
        apply_sqlite_guardrails(conn, allow_schema_changes=False)
        return conn

    def _cutoff_date(self) -> str:
        return (date.today() - timedelta(days=self._max_age_days)).isoformat()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_fit(
        self,
        ticker: str,
        model_type: str,
        regime: str | None,
        order_params: dict[str, Any],
        aic: float,
        bic: float,
        n_obs: int,
    ) -> None:
        """Upsert one fit record; updates n_fits, aic_sum, bic_sum, best_aic."""
        if not ticker or not model_type:
            return
        try:
            aic_val = float(aic)
        except Exception:
            aic_val = float("nan")
        if not math.isfinite(aic_val):
            logger.debug(
                "OrderLearner.record_fit skipped %s/%s due to non-finite AIC: %r",
                ticker,
                model_type,
                aic,
            )
            return
        regime_key = _regime_key(regime)
        op_json = _canonical_json(order_params)
        today = date.today().isoformat()
        try:
            bic_val = float(bic)
        except Exception:
            bic_val = float("nan")
        if not math.isfinite(bic_val):
            bic_val = aic_val

        try:
            conn = self._connect_rw()
            conn.execute(
                """
                INSERT INTO model_order_stats
                    (ticker, model_type, regime, order_params,
                     n_fits, aic_sum, bic_sum, best_aic, last_used, first_seen)
                VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, model_type, regime, order_params) DO UPDATE SET
                    n_fits    = n_fits + 1,
                    aic_sum   = aic_sum + excluded.aic_sum,
                    bic_sum   = bic_sum + excluded.bic_sum,
                    best_aic  = MIN(COALESCE(best_aic, excluded.best_aic), excluded.best_aic),
                    last_used = excluded.last_used
                """,
                (ticker, model_type, regime_key, op_json,
                 aic_val, bic_val,
                 aic_val,          # best_aic initial value
                 today, today),
            )
            conn.commit()
            conn.close()
            logger.debug(
                "OrderLearner recorded %s/%s/%s order=%s aic=%.3f",
                ticker, model_type, regime_key, op_json, aic_val,
            )
        except Exception as exc:
            logger.warning("OrderLearner.record_fit failed: %s", exc)

    def suggest(
        self,
        ticker: str,
        model_type: str,
        regime: str | None,
    ) -> dict[str, Any] | None:
        """
        Return the order_params dict with the lowest mean AIC for this
        (ticker, model_type, regime) key, if n_fits >= min_fits and the
        entry was used within max_age_days.

        Falls back to regime=None if no regime-specific qualifying entry.
        Returns None if no qualifying entry found.
        """
        cutoff = self._cutoff_date()

        try:
            conn = self._connect_ro()
            for rk in _candidate_regimes(regime):
                row = conn.execute(
                    """
                    SELECT order_params
                    FROM model_order_stats
                    WHERE ticker = ?
                      AND model_type = ?
                      AND regime = ?
                      AND n_fits >= ?
                      AND best_aic IS NOT NULL
                      AND last_used >= ?
                    ORDER BY best_aic ASC, (aic_sum / n_fits) ASC
                    LIMIT 1
                    """,
                    (ticker, model_type, rk, self._min_fits, cutoff),
                ).fetchone()
                if row is not None:
                    conn.close()
                    result = json.loads(row[0])
                    logger.debug(
                        "OrderLearner.suggest %s/%s/%s -> %s",
                        ticker, model_type, rk, result,
                    )
                    return result
            conn.close()
        except Exception as exc:
            logger.warning("OrderLearner.suggest failed: %s", exc)
        return None

    def should_skip_grid(
        self,
        ticker: str,
        model_type: str,
        regime: str | None,
    ) -> bool:
        """
        True if the best cached order for this key has n_fits >= skip_grid_threshold.
        When True the caller should use the cached order directly, bypassing grid search.
        """
        cutoff = self._cutoff_date()
        required_fits = max(self._min_fits, self._skip_grid_threshold)

        try:
            conn = self._connect_ro()
            for rk in _candidate_regimes(regime):
                row = conn.execute(
                    """
                    SELECT n_fits
                    FROM model_order_stats
                    WHERE ticker = ?
                      AND model_type = ?
                      AND regime = ?
                      AND n_fits >= ?
                      AND best_aic IS NOT NULL
                      AND last_used >= ?
                    ORDER BY best_aic ASC, (aic_sum / n_fits) ASC
                    LIMIT 1
                    """,
                    (ticker, model_type, rk, required_fits, cutoff),
                ).fetchone()
                if row is not None:
                    conn.close()
                    return int(row[0]) >= required_fits
            conn.close()
        except Exception as exc:
            logger.warning("OrderLearner.should_skip_grid failed: %s", exc)
        return False

    def prune_stale(self) -> int:
        """
        Delete rows not used within max_age_days.
        Returns the number of rows deleted.
        """
        cutoff = self._cutoff_date()
        try:
            conn = self._connect_rw()
            conn.execute(
                "DELETE FROM model_order_stats WHERE last_used < ?",
                (cutoff,),
            )
            deleted = conn.execute("SELECT changes()").fetchone()[0]
            conn.commit()
            conn.close()
            if deleted:
                logger.info("OrderLearner pruned %d stale rows", deleted)
            return deleted
        except Exception as exc:
            logger.warning("OrderLearner.prune_stale failed: %s", exc)
            return 0

    def coverage_stats(self) -> dict[str, int]:
        """Return {total_entries, qualified_entries} for health monitoring."""
        try:
            conn = self._connect_ro()
            total = conn.execute("SELECT COUNT(*) FROM model_order_stats").fetchone()[0]
            qualified = conn.execute(
                "SELECT COUNT(*) FROM model_order_stats WHERE n_fits >= ? AND best_aic IS NOT NULL",
                (self._min_fits,),
            ).fetchone()[0]
            conn.close()
            return {"total_entries": total, "qualified_entries": qualified}
        except Exception as exc:
            logger.warning("OrderLearner.coverage_stats failed: %s", exc)
            return {"total_entries": 0, "qualified_entries": 0}
