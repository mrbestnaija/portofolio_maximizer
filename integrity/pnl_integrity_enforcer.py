"""PnL Integrity Enforcer -- structural prevention of double-counting.

This module enforces the following invariants on trade_executions:

1. **Opening legs** (is_close=0) MUST have realized_pnl IS NULL.
   Only closing legs (is_close=1) carry realized PnL.
2. **Closing legs** (is_close=1) MUST have entry_trade_id linking back
   to the opening leg they close, creating an auditable round-trip.
3. **Diagnostic trades** (is_diagnostic=1) MUST be excluded from
   production metrics.  They CANNOT appear in execution_mode='live'.
4. **Synthetic trades** (is_synthetic=1) MUST be excluded from
   production metrics.  They CANNOT appear in execution_mode='live'.

The enforcer provides:
- ``get_canonical_metrics()`` -- single source of truth for PnL reporting
- ``run_full_integrity_audit()`` -- comprehensive integrity checks
- SQL views for dashboards (production_closed_trades, round_trips)

Safe to instantiate multiple times (read-only queries, idempotent views).
"""

import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from integrity.sqlite_guardrails import apply_sqlite_guardrails, guarded_sqlite_connect

try:
    from etl.portfolio_math import (
        omega_ratio as _calc_omega_ratio,
        DAILY_NGN_THRESHOLD as _NGN_THRESHOLD,
    )
    _PORTFOLIO_MATH_AVAILABLE = True
except Exception:  # pragma: no cover — isolated test environments
    _PORTFOLIO_MATH_AVAILABLE = False
    _NGN_THRESHOLD = (1.31) ** (1.0 / 252) - 1.0  # fallback: 31% annual hurdle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema extension columns (added via migration)
# ---------------------------------------------------------------------------
INTEGRITY_COLUMNS = {
    "is_diagnostic": "INTEGER DEFAULT 0",
    "is_synthetic": "INTEGER DEFAULT 0",
    "is_contaminated": "INTEGER DEFAULT 0",
    "confidence_calibrated": "REAL",
    "entry_trade_id": "INTEGER",
    "bar_open": "REAL",
    "bar_high": "REAL",
    "bar_low": "REAL",
    "bar_close": "REAL",
}


# ---------------------------------------------------------------------------
# Canonical SQL views
# ---------------------------------------------------------------------------
VIEW_PRODUCTION_CLOSED_TRADES = """
CREATE VIEW IF NOT EXISTS production_closed_trades AS
SELECT t.*
FROM   trade_executions t
WHERE  t.is_close = 1
  AND  t.is_diagnostic = 0
  AND  t.is_synthetic = 0
  AND  t.is_contaminated = 0
  AND  NOT EXISTS (
       SELECT 1
       FROM   trade_executions o
       WHERE  o.id = t.entry_trade_id
         AND  o.is_synthetic = 1
  )
"""

# Thresholds for metrics-drift warning
_DRIFT_ROLLING_WINDOW = int(os.environ.get("INTEGRITY_DRIFT_ROLLING_WINDOW", "30"))
_DRIFT_THRESHOLD = float(os.environ.get("INTEGRITY_DRIFT_THRESHOLD", "0.15"))
_DRIFT_MIN_TRADES = int(os.environ.get("INTEGRITY_DRIFT_MIN_TRADES", "15"))

VIEW_ROUND_TRIPS = """
CREATE VIEW IF NOT EXISTS round_trips AS
SELECT
    c.id            AS close_id,
    c.ticker,
    o.id            AS open_id,
    o.trade_date    AS entry_date,
    c.trade_date    AS exit_date,
    o.price         AS entry_price,
    c.exit_price    AS exit_price,
    c.shares,
    c.realized_pnl,
    c.realized_pnl_pct,
    c.holding_period_days,
    c.exit_reason,
    c.execution_mode,
    COALESCE(c.is_diagnostic, 0) AS is_diagnostic,
    COALESCE(c.is_synthetic, 0)  AS is_synthetic
FROM   trade_executions c
LEFT JOIN trade_executions o ON c.entry_trade_id = o.id
WHERE  c.is_close = 1
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class IntegrityViolation:
    """A single integrity violation found during audit."""
    check_name: str
    severity: str          # CRITICAL, HIGH, MEDIUM, LOW
    description: str
    affected_ids: List[int] = field(default_factory=list)
    count: int = 0


@dataclass
class CanonicalMetrics:
    """Single source of truth for PnL reporting (barbell-objective layout).

    Primary policy metrics are barbell-oriented: omega_ratio, payoff_ratio,
    expected_shortfall, beats_ngn_hurdle.  Win rate is retained for
    diagnostics only — the barbell goal is payoff asymmetry, not win rate.
    """
    # --- Core ---
    total_round_trips: int = 0
    total_realized_pnl: float = 0.0
    profit_factor: float = 0.0

    # --- Barbell objective (primary policy metrics) ---
    omega_ratio: Optional[float] = None       # vs NGN daily hurdle; None = < 10 obs
    beats_ngn_hurdle: Optional[bool] = None   # omega_ratio > 1.0
    payoff_ratio: float = 0.0                 # avg_win / |avg_loss| — asymmetry
    expected_shortfall: float = 0.0           # CVaR 90%: avg of worst-decile losses ($)
    ngn_threshold_used: float = 0.0           # daily hurdle rate applied

    # --- Diagnostic (not action-required) ---
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_holding_days: float = 0.0

    # --- Integrity bookkeeping ---
    diagnostic_trades_excluded: int = 0
    synthetic_trades_excluded: int = 0
    contaminated_trades_excluded: int = 0
    opening_legs_with_pnl: int = 0   # should be 0


# ---------------------------------------------------------------------------
# Enforcer
# ---------------------------------------------------------------------------
class PnLIntegrityEnforcer:
    """Structural integrity enforcement for trade_executions.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    auto_create_views : bool
        If True, create/replace canonical views on init.
    """

    def __init__(
        self,
        db_path: str,
        auto_create_views: bool = True,
        allow_schema_changes: bool = False,
    ):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")

        self.db_path = db_path
        self.conn = guarded_sqlite_connect(
            db_path,
            enable_guardrails=False,
        )
        self.conn.row_factory = sqlite3.Row
        self._guardrails_enabled = (
            os.environ.get("SECURITY_SQLITE_GUARDRAILS", "1").strip() != "0"
        )
        self._guardrails_hard_fail = (
            os.environ.get("SECURITY_SQLITE_GUARDRAILS_HARD_FAIL", "1").strip() != "0"
        )
        self._allow_schema_changes = bool(allow_schema_changes)

        # Phase 1: bootstrap policy (may allow DDL for view setup).
        self._apply_guardrails(allow_schema_changes=bool(auto_create_views or allow_schema_changes))

        if auto_create_views:
            self._ensure_views()

        # Phase 2: strict runtime policy.
        self._apply_guardrails(allow_schema_changes=bool(allow_schema_changes))

    def _apply_guardrails(self, *, allow_schema_changes: bool) -> None:
        if not self._guardrails_enabled:
            return
        try:
            apply_sqlite_guardrails(
                self.conn,
                allow_schema_changes=allow_schema_changes,
            )
        except Exception as exc:
            logger.error(
                "Failed to apply SQLite guardrails for PnLIntegrityEnforcer (%s): %s",
                self.db_path,
                exc,
            )
            if self._guardrails_hard_fail:
                raise RuntimeError(f"SQLite guardrails setup failed: {exc}") from exc

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------
    def ensure_integrity_columns(self) -> List[str]:
        """Add integrity columns if missing.  Returns list of columns added."""
        cur = self.conn.execute("PRAGMA table_info(trade_executions)")
        existing = {row["name"] for row in cur.fetchall()}
        added = []
        for col, typedef in INTEGRITY_COLUMNS.items():
            if col not in existing:
                self.conn.execute(
                    f"ALTER TABLE trade_executions ADD COLUMN {col} {typedef}"
                )
                added.append(col)
                logger.info("Added column trade_executions.%s (%s)", col, typedef)
        if added:
            self.conn.commit()
        return added

    def _ensure_views(self):
        """Create or replace canonical views.

        Ensures is_contaminated column exists on trade_executions before
        creating the view (view references this column; older DBs lack it).
        """
        # Ensure is_contaminated column exists so the view can reference it.
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(trade_executions)")}
        if "is_contaminated" not in cols:
            self.conn.execute(
                "ALTER TABLE trade_executions "
                "ADD COLUMN is_contaminated INTEGER DEFAULT 0"
            )

        # SQLite doesn't support CREATE OR REPLACE VIEW, so drop first
        self.conn.execute("DROP VIEW IF EXISTS production_closed_trades")
        self.conn.execute("DROP VIEW IF EXISTS round_trips")
        self.conn.execute(VIEW_PRODUCTION_CLOSED_TRADES)
        self.conn.execute(VIEW_ROUND_TRIPS)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Canonical metrics -- single source of truth
    # ------------------------------------------------------------------
    def get_canonical_metrics(self) -> CanonicalMetrics:
        """Return production PnL metrics from is_close=1 rows only.

        Excludes diagnostic and synthetic trades.  This is the ONLY
        method that should be used for PnL reporting.
        """
        metrics = CanonicalMetrics()

        # Production closed trades (is_close=1, not diagnostic, not synthetic, not contaminated)
        rows = self.conn.execute(
            "SELECT realized_pnl, realized_pnl_pct, holding_period_days "
            "FROM trade_executions t "
            "WHERE t.is_close = 1 "
            "  AND t.is_diagnostic = 0 "
            "  AND t.is_synthetic = 0 "
            "  AND t.is_contaminated = 0 "
            "  AND NOT EXISTS ("
            "      SELECT 1 FROM trade_executions o "
            "      WHERE o.id = t.entry_trade_id "
            "        AND o.is_synthetic = 1) "
            "  AND t.realized_pnl IS NOT NULL"
        ).fetchall()

        metrics.total_round_trips = len(rows)
        if not rows:
            return metrics

        pnls = [float(r["realized_pnl"]) for r in rows]
        hold_days = [
            int(r["holding_period_days"])
            for r in rows
            if r["holding_period_days"] is not None
        ]

        metrics.total_realized_pnl = sum(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        metrics.win_count = len(wins)
        metrics.loss_count = len(losses)
        metrics.win_rate = len(wins) / len(pnls) if pnls else 0.0
        metrics.avg_win = sum(wins) / len(wins) if wins else 0.0
        metrics.avg_loss = sum(losses) / len(losses) if losses else 0.0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        metrics.profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )
        metrics.largest_win = max(wins) if wins else 0.0
        metrics.largest_loss = min(losses) if losses else 0.0
        metrics.avg_holding_days = (
            sum(hold_days) / len(hold_days) if hold_days else 0.0
        )

        # --- Barbell objective metrics ---
        metrics.ngn_threshold_used = _NGN_THRESHOLD

        # Payoff ratio: avg_win / |avg_loss| (asymmetry indicator)
        if losses and metrics.avg_loss != 0.0:
            metrics.payoff_ratio = metrics.avg_win / abs(metrics.avg_loss)
        elif wins:
            metrics.payoff_ratio = float("inf")

        # Expected shortfall: average of worst 10% of dollar losses (CVaR 90%)
        if losses:
            sorted_losses = sorted(losses)               # ascending; most negative first
            n_tail = max(1, len(sorted_losses) // 10)    # worst decile
            metrics.expected_shortfall = sum(sorted_losses[:n_tail]) / n_tail

        # Omega ratio vs NGN daily hurdle — uses realized_pnl_pct (fractional returns)
        if _PORTFOLIO_MATH_AVAILABLE:
            pct_vals = [
                float(r["realized_pnl_pct"])
                for r in rows
                if r["realized_pnl_pct"] is not None
            ]
            if len(pct_vals) >= 10:
                omega = _calc_omega_ratio(pct_vals)
                metrics.omega_ratio = omega
                if isinstance(omega, float) and omega == omega:  # not NaN
                    metrics.beats_ngn_hurdle = omega > 1.0

        # Count excluded trades
        metrics.diagnostic_trades_excluded = self.conn.execute(
            "SELECT COUNT(*) FROM trade_executions "
            "WHERE is_close = 1 AND COALESCE(is_diagnostic, 0) = 1"
        ).fetchone()[0]

        metrics.synthetic_trades_excluded = self.conn.execute(
            "SELECT COUNT(*) FROM trade_executions "
            "WHERE is_close = 1 AND COALESCE(is_synthetic, 0) = 1"
        ).fetchone()[0]

        metrics.contaminated_trades_excluded = self.conn.execute(
            "SELECT COUNT(*) FROM trade_executions t "
            "WHERE t.is_close = 1 "
            "  AND t.is_diagnostic = 0 "
            "  AND t.is_contaminated = 0 "
            "  AND t.is_synthetic = 0 "
            "  AND EXISTS ("
            "      SELECT 1 FROM trade_executions o "
            "      WHERE o.id = t.entry_trade_id "
            "        AND o.is_synthetic = 1)"
        ).fetchone()[0]
        # Add explicitly-tagged contaminated trades
        metrics.contaminated_trades_excluded += self.conn.execute(
            "SELECT COUNT(*) FROM trade_executions "
            "WHERE is_close = 1 AND COALESCE(is_contaminated, 0) = 1"
        ).fetchone()[0]

        # Check for opening legs with PnL (violation indicator)
        metrics.opening_legs_with_pnl = self.conn.execute(
            "SELECT COUNT(*) FROM trade_executions "
            "WHERE is_close = 0 AND realized_pnl IS NOT NULL"
        ).fetchone()[0]

        return metrics

    # ------------------------------------------------------------------
    # Full integrity audit
    # ------------------------------------------------------------------
    def run_full_integrity_audit(self) -> List[IntegrityViolation]:
        """Run all integrity checks and return violations found."""
        violations = []
        violations.extend(self._check_opening_legs_with_pnl())
        violations.extend(self._check_null_production_flags())
        violations.extend(self._check_orphaned_positions())
        violations.extend(self._check_short_orphaned_positions())  # INT-04: SELL opens
        violations.extend(self._check_diagnostic_contamination())
        violations.extend(self._check_cross_mode_contamination())  # INT-05: synthetic-opener closes
        violations.extend(self._check_closing_without_entry_link())
        violations.extend(self._check_pnl_arithmetic())
        violations.extend(self._check_duplicate_close_for_same_entry())
        violations.extend(self._check_metrics_drift())              # INT-06: rolling WR drift
        return violations

    def _check_opening_legs_with_pnl(self) -> List[IntegrityViolation]:
        """CRITICAL: Opening legs (is_close=0) must NOT have realized_pnl."""
        rows = self.conn.execute(
            "SELECT id, ticker, realized_pnl FROM trade_executions "
            "WHERE is_close = 0 AND realized_pnl IS NOT NULL"
        ).fetchall()

        if not rows:
            return []

        return [IntegrityViolation(
            check_name="OPENING_LEG_HAS_PNL",
            severity="CRITICAL",
            description=(
                f"{len(rows)} opening legs (is_close=0) have realized_pnl set. "
                "This causes PnL double-counting. Only closing legs (is_close=1) "
                "should carry realized_pnl."
            ),
            affected_ids=[r["id"] for r in rows],
            count=len(rows),
        )]

    def _check_null_production_flags(self) -> List[IntegrityViolation]:
        """CRITICAL: closing legs must not rely on unknown production flags."""
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(trade_executions)")}
        nullable_flags = [
            col for col in ("is_diagnostic", "is_synthetic", "is_contaminated")
            if col in cols
        ]
        if not nullable_flags:
            return []

        rows = self.conn.execute(
            "SELECT id FROM trade_executions "
            "WHERE is_close = 1 "
            f"  AND ({' OR '.join(f'{col} IS NULL' for col in nullable_flags)})"
        ).fetchall()
        if not rows:
            return []

        return [IntegrityViolation(
            check_name="NULL_PRODUCTION_FLAGS",
            severity="CRITICAL",
            description=(
                f"{len(rows)} closing legs have NULL production flags "
                f"({', '.join(nullable_flags)}). Canonical production filters now reject "
                "these rows fail-closed, but the missing flags indicate ingest or lifecycle "
                "wiring drift that must be repaired."
            ),
            affected_ids=[int(r["id"]) for r in rows],
            count=len(rows),
        )]

    def _check_orphaned_positions(self) -> List[IntegrityViolation]:
        """HIGH: stale BUY entries that cannot be reconciled to open inventory.

        Guardrail intent:
        - Do not fail on active open inventory or recent opens.
        - Do fail on stale, unreconciled opens that indicate lifecycle/linkage drift.
        """
        # Historical artifacts accepted by policy; additional ids can be supplied
        # via INTEGRITY_ORPHAN_WHITELIST_IDS=1,2,3.
        # 5,6,11,13: MSFT/NVDA batch-replay opens from 2026-02-10 (bar_ts 2025-06, 2025-07)
        # 249,250,251,253: AAPL duplicate opens from 2026-03-05 batch runs replaying
        #   bar 2026-03-04 at the same price; no portfolio_positions entry, no close — orphans.
        # 254,256,257,258,259: NVDA duplicate opens from 2026-03-06 batch runs replaying
        #   bar 2026-03-06 at the same price ($177.92); 5 runs without dedup protection.
        known_historical = {5, 6, 11, 13, 249, 250, 251, 253, 254, 256, 257, 258, 259}
        raw_whitelist = os.getenv("INTEGRITY_ORPHAN_WHITELIST_IDS", "")
        if raw_whitelist.strip():
            for token in raw_whitelist.split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    known_historical.add(int(token))
                except ValueError:
                    logger.debug("Ignoring non-numeric orphan whitelist token: %s", token)

        max_open_age_days = 3
        raw_max_age = os.getenv("INTEGRITY_MAX_OPEN_POSITION_AGE_DAYS", "").strip()
        if raw_max_age:
            try:
                max_open_age_days = max(0, int(raw_max_age))
            except ValueError:
                logger.debug(
                    "Invalid INTEGRITY_MAX_OPEN_POSITION_AGE_DAYS=%s; using default %s",
                    raw_max_age,
                    max_open_age_days,
                )

        # All BUY opening legs that should either be linked-to-close or still open.
        buy_rows = self.conn.execute(
            "SELECT id, ticker, trade_date, COALESCE(shares, 0.0) AS qty "
            "FROM trade_executions "
            "WHERE action = 'BUY' AND is_close = 0 "
            "  AND is_diagnostic = 0 "
            "  AND is_synthetic = 0 "
            "ORDER BY trade_date, id"
        ).fetchall()
        if not buy_rows:
            return []

        # Closing SELL legs consume BUY inventory.
        # entry_trade_id-aware: when a close leg is explicitly linked to a BUY via
        # entry_trade_id, consume directly from that BUY leg rather than the FIFO
        # queue front. Pure FIFO (no entry_trade_id) falls back to date-order queue.
        # This prevents blind FIFO mismatches where an early close leg consumes the
        # wrong BUY leg, leaving a linked-but-later BUY stranded in the queue.
        try:
            close_rows = self.conn.execute(
                "SELECT c.id, c.ticker, c.trade_date, "
                "       COALESCE(link.allocated_shares, COALESCE(c.close_size, c.shares, 0.0)) AS qty, "
                "       link.entry_trade_id "
                "FROM trade_executions c "
                "LEFT JOIN trade_close_linkages link ON link.close_trade_id = c.id "
                "WHERE c.action = 'SELL' AND c.is_close = 1 "
                "  AND c.is_diagnostic = 0 "
                "  AND c.is_synthetic = 0 "
                "ORDER BY c.trade_date, c.id"
            ).fetchall()
        except sqlite3.OperationalError:
            close_rows = self.conn.execute(
                "SELECT id, ticker, trade_date, COALESCE(close_size, shares, 0.0) AS qty, "
                "       entry_trade_id "
                "FROM trade_executions "
                "WHERE action = 'SELL' AND is_close = 1 "
                "  AND is_diagnostic = 0 "
                "  AND is_synthetic = 0 "
                "ORDER BY trade_date, id"
            ).fetchall()

        # Track BUY inventory remaining after close consumption.
        fifo_by_ticker: Dict[str, List[Dict[str, Any]]] = {}
        buy_by_id: Dict[int, Dict[str, Any]] = {}
        for row in buy_rows:
            qty = float(row["qty"] or 0.0)
            if qty <= 0:
                continue
            payload = {
                "id": int(row["id"]),
                "ticker": str(row["ticker"]),
                "trade_date": row["trade_date"],
                "remaining_qty": qty,
            }
            fifo_by_ticker.setdefault(str(row["ticker"]), []).append(payload)
            buy_by_id[int(row["id"])] = payload

        for row in close_rows:
            symbol = str(row["ticker"])
            close_qty = float(row["qty"] or 0.0)
            if close_qty <= 0:
                continue
            linked_id = row["entry_trade_id"]
            if linked_id is not None:
                linked_int = int(linked_id)
                if linked_int in buy_by_id:
                    # Direct linkage: consume from the specifically-linked BUY leg.
                    buy_leg = buy_by_id[linked_int]
                    consume = min(close_qty, float(buy_leg["remaining_qty"]))
                    buy_leg["remaining_qty"] = float(buy_leg["remaining_qty"]) - consume
                # else: the linked BUY was filtered out (synthetic/diagnostic) —
                # cross-mode contamination handled by _check_cross_mode_contamination.
                # Do NOT fall through to FIFO: consuming unrelated BUY legs here would
                # mask genuine orphans by shrinking their remaining_qty incorrectly.
            else:
                # No explicit link: fall back to FIFO by date order.
                remaining_close = close_qty
                queue = fifo_by_ticker.get(symbol) or []
                idx = 0
                while remaining_close > 1e-9 and idx < len(queue):
                    buy_leg = queue[idx]
                    consume = min(remaining_close, float(buy_leg["remaining_qty"]))
                    buy_leg["remaining_qty"] = float(buy_leg["remaining_qty"]) - consume
                    remaining_close -= consume
                    if float(buy_leg["remaining_qty"]) <= 1e-9:
                        idx += 1
                if idx > 0:
                    fifo_by_ticker[symbol] = queue[idx:]

        # Optional portfolio-level reconciliation: if portfolio_positions exists
        # and reports open shares, treat those as expected active inventory first.
        open_shares_by_ticker: Dict[str, float] = {}
        try:
            has_positions = self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='portfolio_positions' LIMIT 1"
            ).fetchone()
            if has_positions:
                open_rows = self.conn.execute(
                    "SELECT ticker, COALESCE(shares, 0.0) AS shares FROM portfolio_positions"
                ).fetchall()
                for row in open_rows:
                    qty = float(row["shares"] or 0.0)
                    if qty <= 0:
                        continue
                    open_shares_by_ticker[str(row["ticker"])] = qty
        except Exception:
            logger.debug("Skipping portfolio_positions reconciliation for orphan checks", exc_info=True)

        now_utc = datetime.now(timezone.utc)
        problematic: List[Dict[str, Any]] = []
        covered_as_active = 0
        covered_as_recent = 0
        covered_as_whitelist = 0

        for symbol, queue in fifo_by_ticker.items():
            # Unmatched BUY qty may be expected if still represented in open positions.
            expected_open_qty = float(open_shares_by_ticker.get(symbol, 0.0))
            for buy_leg in queue:
                orphan_id = int(buy_leg["id"])
                remaining_qty = float(buy_leg["remaining_qty"] or 0.0)
                if remaining_qty <= 1e-9:
                    continue

                if orphan_id in known_historical:
                    covered_as_whitelist += 1
                    continue

                if expected_open_qty > 1e-9:
                    covered = min(remaining_qty, expected_open_qty)
                    expected_open_qty -= covered
                    remaining_qty -= covered
                    if remaining_qty <= 1e-9:
                        covered_as_active += 1
                        continue

                trade_date_raw = buy_leg.get("trade_date")
                trade_dt: Optional[datetime] = None
                if isinstance(trade_date_raw, datetime):
                    trade_dt = trade_date_raw
                elif isinstance(trade_date_raw, str) and trade_date_raw.strip():
                    raw = trade_date_raw.strip()
                    try:
                        trade_dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                    except ValueError:
                        try:
                            trade_dt = datetime.strptime(raw[:10], "%Y-%m-%d")
                        except ValueError:
                            trade_dt = None

                if trade_dt is None:
                    age_days = max_open_age_days + 1
                else:
                    age_days = max(0, (now_utc.date() - trade_dt.date()).days)

                if age_days <= max_open_age_days:
                    covered_as_recent += 1
                    continue

                problematic.append(
                    {
                        "id": orphan_id,
                        "ticker": symbol,
                        "remaining_qty": round(remaining_qty, 6),
                        "age_days": int(age_days),
                    }
                )

        if not problematic:
            return []

        return [IntegrityViolation(
            check_name="ORPHANED_POSITION",
            severity="HIGH",
            description=(
                f"{len(problematic)} stale orphaned BUY legs are not reconciled to active inventory "
                f"(max_open_age_days={max_open_age_days}). "
                f"Exemptions: whitelist={covered_as_whitelist}, active_inventory={covered_as_active}, "
                f"recent_open={covered_as_recent}."
            ),
            affected_ids=[int(r["id"]) for r in problematic],
            count=len(problematic),
        )]

    def _check_short_orphaned_positions(self) -> List[IntegrityViolation]:
        """HIGH: stale SELL opening entries (short positions) not covered by BUY closes.

        INT-04 fix: parallel check to _check_orphaned_positions for short legs.
        Uses the same orphan_threshold_days and whitelist logic.
        """
        # 5,6,11,13: MSFT/NVDA batch-replay opens from 2026-02-10
        # 302,303: AAPL SELL opens from 2022-09-30 (Phase 10 PLATT_BOOTSTRAP historical
        #   backtest runs; execution_mode='live' but trade_date='2022-09-30'; no close
        #   generated because bootstrap only produces signals, not round-trips).
        # 315: AMZN SELL open 2026-04-01 ($210.45, 1 share) — intentional live short;
        #   data-source failure on 2026-04-05 (Sat) prevented auto-close.
        # 316,317: NVDA SELL opens 2026-04-01/02 ($175.65/$177.29, 1 share each) —
        #   intentional live shorts; same data-source failure prevented auto-close.
        known_historical: set[int] = {5, 6, 11, 13, 302, 303, 315, 316, 317}
        raw_whitelist = os.getenv("INTEGRITY_ORPHAN_WHITELIST_IDS", "")
        if raw_whitelist.strip():
            for token in raw_whitelist.split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    known_historical.add(int(token))
                except ValueError:
                    pass

        max_open_age_days = 3
        raw_max_age = os.getenv("INTEGRITY_MAX_OPEN_POSITION_AGE_DAYS", "").strip()
        if raw_max_age:
            try:
                max_open_age_days = max(0, int(raw_max_age))
            except ValueError:
                pass

        # All SELL opening legs (short positions): action = 'SELL' AND is_close = 0
        sell_rows = self.conn.execute(
            "SELECT id, ticker, trade_date "
            "FROM trade_executions "
            "WHERE action = 'SELL' AND is_close = 0 "
            "  AND is_diagnostic = 0 "
            "  AND is_synthetic = 0 "
            "ORDER BY trade_date, id"
        ).fetchall()
        if not sell_rows:
            return []

        # BUY close legs that cover these short opens via entry_trade_id link.
        covered_sell_ids: set[int] = set()
        try:
            buy_close_rows = self.conn.execute(
                "SELECT DISTINCT link.entry_trade_id "
                "FROM trade_executions c "
                "JOIN trade_close_linkages link ON link.close_trade_id = c.id "
                "WHERE c.action = 'BUY' AND c.is_close = 1 "
                "  AND c.is_diagnostic = 0 "
                "  AND c.is_synthetic = 0 "
                "  AND link.entry_trade_id IS NOT NULL"
            ).fetchall()
        except sqlite3.OperationalError:
            buy_close_rows = self.conn.execute(
                "SELECT entry_trade_id FROM trade_executions "
                "WHERE action = 'BUY' AND is_close = 1 "
                "  AND entry_trade_id IS NOT NULL "
                "  AND is_diagnostic = 0 "
                "  AND is_synthetic = 0"
            ).fetchall()
        for row in buy_close_rows:
            if row["entry_trade_id"]:
                covered_sell_ids.add(int(row["entry_trade_id"]))

        now_utc = datetime.now(timezone.utc)
        problematic: List[Dict[str, Any]] = []
        for row in sell_rows:
            sell_id = int(row["id"])
            if sell_id in known_historical or sell_id in covered_sell_ids:
                continue

            trade_date_raw = row["trade_date"]
            trade_dt = None
            if isinstance(trade_date_raw, datetime):
                trade_dt = trade_date_raw
            elif isinstance(trade_date_raw, str) and trade_date_raw.strip():
                raw = trade_date_raw.strip()
                try:
                    trade_dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                except ValueError:
                    try:
                        trade_dt = datetime.strptime(raw[:10], "%Y-%m-%d")
                    except ValueError:
                        trade_dt = None

            age_days = (
                max_open_age_days + 1
                if trade_dt is None
                else max(0, (now_utc.date() - trade_dt.date()).days)
            )
            if age_days <= max_open_age_days:
                continue
            problematic.append(
                {"id": sell_id, "ticker": str(row["ticker"]), "age_days": int(age_days)}
            )

        if not problematic:
            return []

        return [IntegrityViolation(
            check_name="ORPHANED_SHORT_POSITION",
            severity="HIGH",
            description=(
                f"{len(problematic)} stale SELL opening leg(s) (short positions) are not "
                f"covered by BUY closes (max_open_age_days={max_open_age_days}). "
                "INT-04 fix: short orphan detection is now symmetric with long orphan check."
            ),
            affected_ids=[int(r["id"]) for r in problematic],
            count=len(problematic),
        )]

    def _check_diagnostic_contamination(self) -> List[IntegrityViolation]:
        """HIGH: Diagnostic trades must not pollute production metrics."""
        # Check for trades that look diagnostic but aren't flagged
        rows = self.conn.execute(
            "SELECT id, ticker, execution_mode FROM trade_executions "
            "WHERE execution_mode LIKE '%diagnostic%' "
            "  AND COALESCE(is_diagnostic, 0) = 0"
        ).fetchall()

        if not rows:
            return []

        return [IntegrityViolation(
            check_name="DIAGNOSTIC_NOT_FLAGGED",
            severity="HIGH",
            description=(
                f"{len(rows)} trades have execution_mode containing 'diagnostic' "
                "but is_diagnostic is not set. These will contaminate production metrics."
            ),
            affected_ids=[r["id"] for r in rows],
            count=len(rows),
        )]

    def _check_cross_mode_contamination(self) -> List[IntegrityViolation]:
        """HIGH: INT-05 — untagged closing legs whose opener is synthetic.

        When a live session inherits a position from a prior synthetic run (via
        portfolio_state or direct entry_trade_id linkage to a synthetic opener),
        the closing leg's PnL is computed against a synthetic entry price — producing
        phantom losses/gains that corrupt production metrics.

        Only flags UNTAGGED contaminated closes (is_contaminated=0 but opener is
        synthetic). Tagged closes (is_contaminated=1) are already excluded from
        production_closed_trades and from canonical metrics — re-blocking on them
        here is redundant and requires a manual whitelist entry for every new
        contaminated close. That is the architectural defect this method avoids.

        Tagged closes are counted and reported informally for observability.
        """
        # Count tagged closes (is_contaminated=1) for informational purposes only.
        # These are already excluded from production_closed_trades — no violation needed.
        tagged_count = self.conn.execute(
            "SELECT COUNT(*) FROM trade_executions "
            "WHERE is_close = 1 AND COALESCE(is_contaminated, 0) = 1"
        ).fetchone()[0]

        # Only flag UNTAGGED live closes linked to a synthetic opener.
        # synthetic-mode closes against synthetic openers are expected and clean.
        untagged = self.conn.execute(
            "SELECT t.id, t.ticker, t.realized_pnl, t.entry_trade_id "
            "FROM trade_executions t "
            "JOIN trade_executions o ON t.entry_trade_id = o.id "
            "WHERE t.is_close = 1 "
            "  AND COALESCE(t.is_contaminated, 0) = 0 "
            "  AND COALESCE(t.is_synthetic, 0) = 0 "
            "  AND COALESCE(o.is_synthetic, 0) = 1"
        ).fetchall()

        if not untagged:
            return []

        total_phantom_pnl = sum(
            float(r["realized_pnl"]) for r in untagged if r["realized_pnl"] is not None
        )

        return [IntegrityViolation(
            check_name="CROSS_MODE_CONTAMINATION",
            severity="HIGH",
            description=(
                f"{len(untagged)} untagged closing leg(s) have PnL computed from synthetic "
                f"entry prices (phantom PnL: ${total_phantom_pnl:+,.2f}). "
                f"Run migrate_fix_synthetic_contamination.py to tag them. "
                f"({tagged_count} already-tagged contaminated closes are suppressed — "
                f"they are excluded from production_closed_trades.)"
            ),
            affected_ids=[r["id"] for r in untagged],
            count=len(untagged),
        )]

    def _check_metrics_drift(self) -> List[IntegrityViolation]:
        """HIGH: INT-06 — rolling win-rate drifts significantly from historical baseline.

        Compares the last ``_DRIFT_ROLLING_WINDOW`` trades' WR against the full
        historical WR.  A drift larger than ``_DRIFT_THRESHOLD`` (default 15pp)
        signals potential model degradation or data contamination.

        Requires at least ``_DRIFT_MIN_TRADES`` + ``_DRIFT_ROLLING_WINDOW`` trades
        to avoid false positives during warmup.

        Environment overrides:
          INTEGRITY_DRIFT_ROLLING_WINDOW  (default 30)
          INTEGRITY_DRIFT_THRESHOLD       (default 0.15 = 15pp)
          INTEGRITY_DRIFT_MIN_TRADES      (default 15)
        """
        rows = self.conn.execute(
            "SELECT realized_pnl "
            "FROM production_closed_trades "
            "ORDER BY id ASC"
        ).fetchall()

        pnls = [float(r["realized_pnl"]) for r in rows if r["realized_pnl"] is not None]
        n = len(pnls)

        min_needed = _DRIFT_MIN_TRADES + _DRIFT_ROLLING_WINDOW
        if n < min_needed:
            return []  # insufficient history

        historical_wr = sum(1 for p in pnls[:-_DRIFT_ROLLING_WINDOW] if p > 0) / (
            n - _DRIFT_ROLLING_WINDOW
        )
        rolling_pnls = pnls[-_DRIFT_ROLLING_WINDOW:]
        rolling_wr = sum(1 for p in rolling_pnls if p > 0) / len(rolling_pnls)

        drift = historical_wr - rolling_wr
        if abs(drift) <= _DRIFT_THRESHOLD:
            return []

        direction = "down" if drift > 0 else "up"
        return [IntegrityViolation(
            check_name="METRICS_DRIFT",
            severity="HIGH",
            description=(
                f"[MODEL DRIFT WARNING] Rolling {_DRIFT_ROLLING_WINDOW}-trade WR "
                f"({rolling_wr:.1%}) drifted {direction} {abs(drift):.1%}pp vs "
                f"historical baseline ({historical_wr:.1%}) — exceeds {_DRIFT_THRESHOLD:.0%} "
                f"threshold. Investigate recent trades for data contamination or model degradation. "
                f"(historical_n={n - _DRIFT_ROLLING_WINDOW}, "
                f"rolling_n={_DRIFT_ROLLING_WINDOW})"
            ),
            affected_ids=[],
            count=_DRIFT_ROLLING_WINDOW,
        )]

    def _check_closing_without_entry_link(self) -> List[IntegrityViolation]:
        """MEDIUM: Closing legs should link to their opening leg.

        Closes that originated from portfolio-state resume (no matching BUY row
        in trade_executions) can be whitelisted via
        ``INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS=66,75``.
        """
        # Known resume-originated closes accepted by policy.
        whitelist: set[int] = set()
        raw = os.getenv("INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS", "")
        if raw.strip():
            for token in raw.split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    whitelist.add(int(token))
                except ValueError:
                    pass

        try:
            rows = self.conn.execute(
                "SELECT id, ticker FROM trade_executions "
                "WHERE is_close = 1 "
                "  AND NOT EXISTS ("
                "      SELECT 1 FROM trade_close_linkages link "
                "      WHERE link.close_trade_id = trade_executions.id"
                "  )"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = self.conn.execute(
                "SELECT id, ticker FROM trade_executions "
                "WHERE is_close = 1 AND entry_trade_id IS NULL"
            ).fetchall()

        filtered = [r for r in rows if int(r["id"]) not in whitelist]
        if not filtered:
            return []

        return [IntegrityViolation(
            check_name="CLOSE_WITHOUT_ENTRY_LINK",
            severity="MEDIUM",
            description=(
                f"{len(filtered)} closing legs (is_close=1) have no effective opener linkage. "
                f"Round-trip attribution is incomplete."
                + (f" ({len(rows) - len(filtered)} whitelisted)" if whitelist else "")
            ),
            affected_ids=[r["id"] for r in filtered],
            count=len(filtered),
        )]

    def _check_pnl_arithmetic(self) -> List[IntegrityViolation]:
        """MEDIUM: Verify realized_pnl matches direction-aware close arithmetic."""
        rows = self.conn.execute(
            "SELECT c.id, c.ticker, c.action, c.entry_price, c.exit_price, "
            "       c.close_size, c.realized_pnl, c.commission, o.action AS open_action "
            "FROM trade_executions "
            "AS c "
            "LEFT JOIN trade_executions AS o ON c.entry_trade_id = o.id "
            "WHERE c.is_close = 1 "
            "  AND c.entry_price IS NOT NULL "
            "  AND c.exit_price IS NOT NULL "
            "  AND c.close_size IS NOT NULL "
            "  AND c.realized_pnl IS NOT NULL"
        ).fetchall()

        mismatched = []
        for r in rows:
            close_action = str(r["action"] or "").upper()
            open_action = str(r["open_action"] or "").upper()
            direction = 1.0
            if close_action == "BUY":
                # BUY close usually means covering a short.
                direction = -1.0
            elif close_action not in {"SELL", "BUY"}:
                # Fallback for malformed/legacy close action values.
                if open_action == "SELL":
                    direction = -1.0

            expected = direction * (r["exit_price"] - r["entry_price"]) * r["close_size"]
            commission = r["commission"] or 0.0
            expected -= commission
            if abs(r["realized_pnl"] - expected) > 0.02:  # 2 cent tolerance
                mismatched.append(r["id"])

        if not mismatched:
            return []

        return [IntegrityViolation(
            check_name="PNL_ARITHMETIC_MISMATCH",
            severity="MEDIUM",
            description=(
                f"{len(mismatched)} closing trades have realized_pnl that doesn't "
                "match directional close arithmetic using action, size, and commission."
            ),
            affected_ids=mismatched,
            count=len(mismatched),
        )]

    def _check_duplicate_close_for_same_entry(self) -> List[IntegrityViolation]:
        """HIGH: Detect over-closed entries (duplicate close quantity).

        Legitimate partial exits may produce multiple closing legs for a single
        opening leg. The integrity violation is when *multiple* linked close
        legs over-consume the opening leg quantity.
        """
        try:
            rows = self.conn.execute(
                "SELECT "
                "  o.id AS open_id, "
                "  COALESCE(o.shares, 0.0) AS open_qty, "
                "  COALESCE(SUM(COALESCE(link.allocated_shares, 0.0)), 0.0) AS closed_qty "
                "FROM trade_executions o "
                "JOIN trade_close_linkages link "
                "  ON link.entry_trade_id = o.id "
                "JOIN trade_executions c "
                "  ON c.id = link.close_trade_id "
                " AND c.is_close = 1 "
                "WHERE o.action = 'BUY' "
                "  AND o.is_close = 0 "
                "GROUP BY o.id "
                "HAVING COUNT(DISTINCT c.id) > 1 "
                "   AND COALESCE(SUM(COALESCE(link.allocated_shares, 0.0)), 0.0) "
                "       > COALESCE(o.shares, 0.0) + 0.02"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = self.conn.execute(
                "SELECT "
                "  o.id AS open_id, "
                "  COALESCE(o.shares, 0.0) AS open_qty, "
                "  COALESCE(SUM(COALESCE(c.close_size, c.shares, 0.0)), 0.0) AS closed_qty "
                "FROM trade_executions o "
                "JOIN trade_executions c "
                "  ON c.entry_trade_id = o.id "
                " AND c.is_close = 1 "
                "WHERE o.action = 'BUY' "
                "  AND o.is_close = 0 "
                "GROUP BY o.id "
                "HAVING COUNT(c.id) > 1 "
                "   AND COALESCE(SUM(COALESCE(c.close_size, c.shares, 0.0)), 0.0) "
                "       > COALESCE(o.shares, 0.0) + 0.02"
            ).fetchall()

        if not rows:
            return []

        affected = [r["open_id"] for r in rows]
        return [IntegrityViolation(
            check_name="DUPLICATE_CLOSE_FOR_ENTRY",
            severity="HIGH",
            description=(
                f"{len(rows)} opening legs are over-closed (linked close_size "
                "sum exceeds opening shares). This causes PnL duplication."
            ),
            affected_ids=affected,
            count=len(rows),
        )]

    # ------------------------------------------------------------------
    # Repair operations
    # ------------------------------------------------------------------
    def fix_opening_legs_pnl(self, dry_run: bool = True) -> int:
        """NULL out realized_pnl on opening legs (is_close=0).

        Returns the number of rows affected.
        """
        rows = self.conn.execute(
            "SELECT id, ticker, realized_pnl FROM trade_executions "
            "WHERE is_close = 0 AND realized_pnl IS NOT NULL"
        ).fetchall()

        if not rows:
            logger.info("[OK] No opening legs with realized_pnl found.")
            return 0

        logger.info(
            "Found %d opening legs with realized_pnl (double-counting).", len(rows)
        )
        for r in rows:
            logger.info(
                "  id=%d ticker=%s realized_pnl=%.2f -> NULL",
                r["id"], r["ticker"], r["realized_pnl"],
            )

        if not dry_run:
            self.conn.execute(
                "UPDATE trade_executions "
                "SET realized_pnl = NULL, realized_pnl_pct = NULL "
                "WHERE is_close = 0 AND realized_pnl IS NOT NULL"
            )
            self.conn.commit()
            logger.info("[APPLIED] Cleared realized_pnl from %d opening legs.", len(rows))

        return len(rows)

    def backfill_diagnostic_flag(self, dry_run: bool = True) -> int:
        """Set is_diagnostic=1 for trades executed under DIAGNOSTIC_MODE.

        Heuristic: if execution_mode contains 'diagnostic' or if the trade
        was executed in a known diagnostic run_id, flag it.
        """
        # Find trades with diagnostic execution_mode
        count = self.conn.execute(
            "SELECT COUNT(*) FROM trade_executions "
            "WHERE (execution_mode LIKE '%diagnostic%' "
            "    OR execution_mode LIKE '%diag%') "
            "  AND COALESCE(is_diagnostic, 0) = 0"
        ).fetchone()[0]

        if count == 0:
            logger.info("[OK] No unflagged diagnostic trades found.")
            return 0

        logger.info("Found %d trades to flag as is_diagnostic=1.", count)

        if not dry_run:
            self.conn.execute(
                "UPDATE trade_executions "
                "SET is_diagnostic = 1 "
                "WHERE (execution_mode LIKE '%diagnostic%' "
                "    OR execution_mode LIKE '%diag%') "
                "  AND COALESCE(is_diagnostic, 0) = 0"
            )
            self.conn.commit()
            logger.info("[APPLIED] Flagged %d trades as is_diagnostic=1.", count)

        return count

    def backfill_entry_trade_ids(self, dry_run: bool = True) -> int:
        """Link closing legs to their opening legs via entry_trade_id.

        Matching heuristic: same ticker, SELL.entry_price matches BUY.price,
        BUY happened before SELL, and BUY has sufficient remaining quantity
        after accounting for existing linked closes.
        """
        closes = self.conn.execute(
            "SELECT id, ticker, action, entry_price, trade_date, "
            "       COALESCE(close_size, shares, 0.0) AS close_qty "
            "FROM trade_executions "
            "WHERE is_close = 1 AND entry_trade_id IS NULL "
            "ORDER BY trade_date, id"
        ).fetchall()

        if not closes:
            logger.info("[OK] All closing legs already have entry_trade_id.")
            return 0

        linked = 0

        # Remaining quantity by opening leg after accounting for linked closes.
        # Fetch ALL opening legs (is_close=0) regardless of action so that
        # short positions (SELL to open, BUY to close) are handled correctly.
        buy_rows = self.conn.execute(
            "SELECT id, ticker, action, trade_date, COALESCE(shares, 0.0) AS buy_qty, "
            "       COALESCE(price, 0.0) AS buy_price "
            "FROM trade_executions "
            "WHERE is_close = 0 "
            "ORDER BY trade_date DESC, id DESC"
        ).fetchall()
        buy_by_id: Dict[int, sqlite3.Row] = {int(r["id"]): r for r in buy_rows}
        close_usage = self.conn.execute(
            "SELECT entry_trade_id, "
            "       COALESCE(SUM(COALESCE(close_size, shares, 0.0)), 0.0) AS used_qty "
            "FROM trade_executions "
            "WHERE is_close = 1 AND entry_trade_id IS NOT NULL "
            "GROUP BY entry_trade_id"
        ).fetchall()
        used_qty_by_buy: Dict[int, float] = {
            int(r["entry_trade_id"]): float(r["used_qty"] or 0.0)
            for r in close_usage
            if r["entry_trade_id"] is not None
        }
        remaining_qty_by_buy: Dict[int, float] = {}
        for buy_id, buy_row in buy_by_id.items():
            remaining = float(buy_row["buy_qty"] or 0.0) - used_qty_by_buy.get(buy_id, 0.0)
            remaining_qty_by_buy[buy_id] = max(0.0, remaining)

        for close in closes:
            close_qty = float(close["close_qty"] or 0.0)
            if close_qty <= 0:
                continue

            # Direction-aware open leg search:
            #   close.action='SELL' closes a LONG  -> look for open.action='BUY'
            #   close.action='BUY'  closes a SHORT -> look for open.action='SELL'
            close_action = str(close["action"] or "SELL").upper()
            open_action = "BUY" if close_action == "SELL" else "SELL"
            candidates = self.conn.execute(
                "SELECT id, price, trade_date FROM trade_executions "
                "WHERE ticker = ? AND action = ? AND is_close = 0 "
                "  AND trade_date <= ? "
                "ORDER BY trade_date DESC, id DESC",
                (close["ticker"], open_action, close["trade_date"]),
            ).fetchall()

            for cand in candidates:
                cand_id = int(cand["id"])
                remaining_qty = remaining_qty_by_buy.get(cand_id, 0.0)
                if remaining_qty + 1e-9 < close_qty:
                    continue
                # Match on entry_price tolerance
                if (
                    close["entry_price"] is not None
                    and abs(cand["price"] - close["entry_price"]) < 0.02
                ):
                    if not dry_run:
                        self.conn.execute(
                            "UPDATE trade_executions SET entry_trade_id = ? "
                            "WHERE id = ?",
                            (cand["id"], close["id"]),
                        )
                    remaining_qty_by_buy[cand_id] = max(0.0, remaining_qty - close_qty)
                    linked += 1
                    logger.debug(
                        "Linked close id=%d to open id=%d (ticker=%s, close_qty=%.4f, remaining=%.4f)",
                        close["id"], cand["id"], close["ticker"], close_qty, remaining_qty_by_buy[cand_id],
                    )
                    break

        if not dry_run and linked > 0:
            self.conn.commit()
            logger.info("[APPLIED] Linked %d closing legs to opening legs.", linked)
        else:
            logger.info("[DRY RUN] Would link %d closing legs.", linked)

        return linked

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def print_report(self) -> None:
        """Print a comprehensive integrity report to stdout."""
        print("=" * 70)
        print("PnL INTEGRITY REPORT (CANONICAL METRICS)")
        print("=" * 70)

        m = self.get_canonical_metrics()

        # --- Barbell objective (primary policy) ---
        print()
        print("--- Barbell Objective (production_closed_trades only) ---")
        print(f"  Round-trips:       {m.total_round_trips}")
        print(f"  Total PnL:         ${m.total_realized_pnl:+,.2f}")
        print(f"  Profit factor:     {m.profit_factor:.2f}")
        if m.payoff_ratio == float("inf"):
            print("  Payoff ratio:      inf  (no losses)")
        else:
            print(f"  Payoff ratio:      {m.payoff_ratio:.2f}x  (avg win / |avg loss|)")
        if m.expected_shortfall != 0.0:
            print(f"  Exp. shortfall:    ${m.expected_shortfall:+,.2f}  (CVaR 90%, worst-decile avg)")
        if m.omega_ratio is None:
            print("  Omega ratio:       N/A  (need >= 10 fractional returns)")
            print("  NGN hurdle beat:   N/A")
        elif m.omega_ratio == float("inf"):
            print(f"  Omega ratio:       inf  (vs {m.ngn_threshold_used:.5f}/day NGN hurdle)")
            print("  NGN hurdle beat:   YES")
        else:
            print(f"  Omega ratio:       {m.omega_ratio:.3f}  (vs {m.ngn_threshold_used:.5f}/day NGN hurdle)")
            hurdle_str = "YES" if m.beats_ngn_hurdle else "NO"
            print(f"  NGN hurdle beat:   {hurdle_str}")

        # --- Diagnostic (not action-required) ---
        print()
        print("--- Diagnostic (win rate is advisory -- barbell goal is payoff asymmetry) ---")
        print(f"  Win rate:          {m.win_rate:.1%}  [{m.win_count}W / {m.loss_count}L]")
        print(f"  Avg win:           ${m.avg_win:+,.2f}")
        print(f"  Avg loss:          ${m.avg_loss:+,.2f}")
        print(f"  Largest win:       ${m.largest_win:+,.2f}")
        print(f"  Largest loss:      ${m.largest_loss:+,.2f}")
        print(f"  Avg holding days:  {m.avg_holding_days:.1f}")

        # --- Exclusions & integrity bookkeeping ---
        print()
        print("--- Exclusions ---")
        print(f"  Diagnostic excl:   {m.diagnostic_trades_excluded}")
        print(f"  Synthetic excl:    {m.synthetic_trades_excluded}")
        print(f"  Contaminated excl: {m.contaminated_trades_excluded}")
        print(f"  Double-count chk:  {m.opening_legs_with_pnl}  (must be 0)")

        if m.opening_legs_with_pnl > 0:
            print(
                f"  [WARNING] Opening legs with PnL: {m.opening_legs_with_pnl} "
                "(causes double-counting!)"
            )

        # --- Integrity audit ---
        violations = self.run_full_integrity_audit()
        print()
        if not violations:
            print("[OK] All integrity checks passed")
        else:
            print(f"[FAIL] {len(violations)} integrity violation(s):")
            for v in violations:
                print(f"  [{v.severity}] {v.check_name}: {v.description}")

        # --- DB totals ---
        total = self.conn.execute(
            "SELECT COUNT(*) FROM trade_executions"
        ).fetchone()[0]
        open_count = self.conn.execute(
            "SELECT COUNT(*) FROM trade_executions "
            "WHERE is_close = 0 AND realized_pnl IS NULL"
        ).fetchone()[0]
        close_count = self.conn.execute(
            "SELECT COUNT(*) FROM trade_executions WHERE is_close = 1"
        ).fetchone()[0]
        print()
        print(f"--- DB Totals ---")
        print(f"  Total rows:        {total}")
        print(f"  Opening legs:      {open_count}")
        print(f"  Closing legs:      {close_count}")
        print("=" * 70)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def main():
    """CLI entrypoint for integrity reporting and repair."""
    import argparse

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT_DB = os.path.join(ROOT, "data", "portfolio_maximizer.db")

    parser = argparse.ArgumentParser(
        description="PnL Integrity Enforcer -- audit and repair trade_executions"
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB, help="Path to SQLite database"
    )
    parser.add_argument(
        "--fix-opening-pnl",
        action="store_true",
        help="NULL out realized_pnl on opening legs (is_close=0)",
    )
    parser.add_argument(
        "--fix-diagnostic-flags",
        action="store_true",
        help="Set is_diagnostic=1 for diagnostic-mode trades",
    )
    parser.add_argument(
        "--fix-entry-links",
        action="store_true",
        help="Backfill entry_trade_id on closing legs",
    )
    parser.add_argument(
        "--fix-all",
        action="store_true",
        help="Run all repair operations",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply changes (default is dry-run)",
    )
    args = parser.parse_args()

    dry_run = not args.apply

    with PnLIntegrityEnforcer(args.db, allow_schema_changes=True) as enforcer:
        # Ensure schema columns exist
        added = enforcer.ensure_integrity_columns()
        if added:
            print(f"[SCHEMA] Added columns: {', '.join(added)}")

        # Run repairs if requested
        if args.fix_all or args.fix_opening_pnl:
            n = enforcer.fix_opening_legs_pnl(dry_run=dry_run)
            print(f"[FIX] Opening legs PnL: {n} rows {'would be' if dry_run else ''} affected")

        if args.fix_all or args.fix_diagnostic_flags:
            n = enforcer.backfill_diagnostic_flag(dry_run=dry_run)
            print(f"[FIX] Diagnostic flags: {n} rows {'would be' if dry_run else ''} affected")

        if args.fix_all or args.fix_entry_links:
            n = enforcer.backfill_entry_trade_ids(dry_run=dry_run)
            print(f"[FIX] Entry links: {n} rows {'would be' if dry_run else ''} affected")

        if dry_run and (args.fix_all or args.fix_opening_pnl or args.fix_diagnostic_flags or args.fix_entry_links):
            print()
            print("[DRY RUN] No changes made. Re-run with --apply to execute.")

        # Always print report
        print()
        enforcer.print_report()


if __name__ == "__main__":
    main()
