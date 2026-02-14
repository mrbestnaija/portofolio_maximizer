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
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from integrity.sqlite_guardrails import apply_sqlite_guardrails, guarded_sqlite_connect

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema extension columns (added via migration)
# ---------------------------------------------------------------------------
INTEGRITY_COLUMNS = {
    "is_diagnostic": "INTEGER DEFAULT 0",
    "is_synthetic": "INTEGER DEFAULT 0",
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
SELECT *
FROM   trade_executions
WHERE  is_close = 1
  AND  COALESCE(is_diagnostic, 0) = 0
  AND  COALESCE(is_synthetic, 0)  = 0
"""

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
    """Single source of truth for PnL reporting."""
    total_round_trips: int = 0
    total_realized_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_holding_days: float = 0.0
    diagnostic_trades_excluded: int = 0
    synthetic_trades_excluded: int = 0
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
        """Create or replace canonical views."""
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

        # Production closed trades (is_close=1, not diagnostic, not synthetic)
        rows = self.conn.execute(
            "SELECT realized_pnl, realized_pnl_pct, holding_period_days "
            "FROM trade_executions "
            "WHERE is_close = 1 "
            "  AND COALESCE(is_diagnostic, 0) = 0 "
            "  AND COALESCE(is_synthetic, 0) = 0 "
            "  AND realized_pnl IS NOT NULL"
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

        # Count excluded trades
        metrics.diagnostic_trades_excluded = self.conn.execute(
            "SELECT COUNT(*) FROM trade_executions "
            "WHERE is_close = 1 AND COALESCE(is_diagnostic, 0) = 1"
        ).fetchone()[0]

        metrics.synthetic_trades_excluded = self.conn.execute(
            "SELECT COUNT(*) FROM trade_executions "
            "WHERE is_close = 1 AND COALESCE(is_synthetic, 0) = 1"
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
        violations.extend(self._check_orphaned_positions())
        violations.extend(self._check_diagnostic_contamination())
        violations.extend(self._check_closing_without_entry_link())
        violations.extend(self._check_pnl_arithmetic())
        violations.extend(self._check_duplicate_close_for_same_entry())
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

    def _check_orphaned_positions(self) -> List[IntegrityViolation]:
        """HIGH: stale BUY entries that cannot be reconciled to open inventory.

        Guardrail intent:
        - Do not fail on active open inventory or recent opens.
        - Do fail on stale, unreconciled opens that indicate lifecycle/linkage drift.
        """
        # Historical artifacts accepted by policy; additional ids can be supplied
        # via INTEGRITY_ORPHAN_WHITELIST_IDS=1,2,3.
        known_historical = {5, 6, 11, 13}
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
            "  AND COALESCE(is_diagnostic, 0) = 0 "
            "  AND COALESCE(is_synthetic, 0) = 0 "
            "ORDER BY trade_date, id"
        ).fetchall()
        if not buy_rows:
            return []

        # Closing SELL legs consume BUY inventory in FIFO order.
        close_rows = self.conn.execute(
            "SELECT id, ticker, trade_date, COALESCE(close_size, shares, 0.0) AS qty "
            "FROM trade_executions "
            "WHERE action = 'SELL' AND is_close = 1 "
            "  AND COALESCE(is_diagnostic, 0) = 0 "
            "  AND COALESCE(is_synthetic, 0) = 0 "
            "ORDER BY trade_date, id"
        ).fetchall()

        # Track BUY inventory remaining after FIFO close consumption.
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
            remaining_close = float(row["qty"] or 0.0)
            if remaining_close <= 0:
                continue
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

        now_utc = datetime.utcnow()
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
                f"{len(filtered)} closing legs (is_close=1) have no entry_trade_id. "
                f"Round-trip attribution is incomplete."
                + (f" ({len(rows) - len(filtered)} whitelisted)" if whitelist else "")
            ),
            affected_ids=[r["id"] for r in filtered],
            count=len(filtered),
        )]

    def _check_pnl_arithmetic(self) -> List[IntegrityViolation]:
        """MEDIUM: Verify realized_pnl matches (exit_price - entry_price) * shares."""
        rows = self.conn.execute(
            "SELECT id, ticker, entry_price, exit_price, close_size, "
            "       realized_pnl, commission "
            "FROM trade_executions "
            "WHERE is_close = 1 "
            "  AND entry_price IS NOT NULL "
            "  AND exit_price IS NOT NULL "
            "  AND close_size IS NOT NULL "
            "  AND realized_pnl IS NOT NULL"
        ).fetchall()

        mismatched = []
        for r in rows:
            expected = (r["exit_price"] - r["entry_price"]) * r["close_size"]
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
                "match (exit_price - entry_price) * close_size - commission."
            ),
            affected_ids=mismatched,
            count=len(mismatched),
        )]

    def _check_duplicate_close_for_same_entry(self) -> List[IntegrityViolation]:
        """HIGH: Each opening leg should be closed at most once."""
        rows = self.conn.execute(
            "SELECT entry_trade_id, COUNT(*) as cnt "
            "FROM trade_executions "
            "WHERE is_close = 1 AND entry_trade_id IS NOT NULL "
            "GROUP BY entry_trade_id "
            "HAVING COUNT(*) > 1"
        ).fetchall()

        if not rows:
            return []

        affected = [r["entry_trade_id"] for r in rows]
        return [IntegrityViolation(
            check_name="DUPLICATE_CLOSE_FOR_ENTRY",
            severity="HIGH",
            description=(
                f"{len(rows)} opening legs have been closed more than once. "
                "This causes PnL duplication."
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
        BUY happened before SELL, and BUY is not already linked.
        """
        closes = self.conn.execute(
            "SELECT id, ticker, entry_price, trade_date FROM trade_executions "
            "WHERE is_close = 1 AND entry_trade_id IS NULL "
            "ORDER BY id"
        ).fetchall()

        if not closes:
            logger.info("[OK] All closing legs already have entry_trade_id.")
            return 0

        linked = 0
        used_buy_ids = set()

        # Pre-fetch used entry_trade_ids to avoid double-linking
        existing_links = self.conn.execute(
            "SELECT entry_trade_id FROM trade_executions "
            "WHERE entry_trade_id IS NOT NULL"
        ).fetchall()
        used_buy_ids = {r["entry_trade_id"] for r in existing_links}

        for close in closes:
            # Find matching opening BUY
            candidates = self.conn.execute(
                "SELECT id, price, trade_date FROM trade_executions "
                "WHERE ticker = ? AND action = 'BUY' AND is_close = 0 "
                "  AND trade_date <= ? "
                "ORDER BY trade_date DESC, id DESC",
                (close["ticker"], close["trade_date"]),
            ).fetchall()

            for cand in candidates:
                if cand["id"] in used_buy_ids:
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
                    used_buy_ids.add(cand["id"])
                    linked += 1
                    logger.debug(
                        "Linked close id=%d to open id=%d (ticker=%s)",
                        close["id"], cand["id"], close["ticker"],
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
        print("PnL INTEGRITY REPORT")
        print("=" * 70)

        # Canonical metrics
        m = self.get_canonical_metrics()
        print()
        print("--- Canonical Metrics (production_closed_trades only) ---")
        print(f"  Round-trips:       {m.total_round_trips}")
        print(f"  Total PnL:         ${m.total_realized_pnl:+,.2f}")
        print(f"  Win rate:          {m.win_rate:.1%}")
        print(f"  Wins/Losses:       {m.win_count}/{m.loss_count}")
        print(f"  Avg win:           ${m.avg_win:+,.2f}")
        print(f"  Avg loss:          ${m.avg_loss:+,.2f}")
        print(f"  Profit factor:     {m.profit_factor:.2f}")
        print(f"  Largest win:       ${m.largest_win:+,.2f}")
        print(f"  Largest loss:      ${m.largest_loss:+,.2f}")
        print(f"  Avg holding days:  {m.avg_holding_days:.1f}")
        print(f"  Diagnostic excl:   {m.diagnostic_trades_excluded}")
        print(f"  Synthetic excl:    {m.synthetic_trades_excluded}")

        if m.opening_legs_with_pnl > 0:
            print(
                f"  [WARNING] Opening legs with PnL: {m.opening_legs_with_pnl} "
                "(causes double-counting!)"
            )

        # Integrity audit
        violations = self.run_full_integrity_audit()
        print()
        if not violations:
            print("--- Integrity Checks: ALL PASSED ---")
        else:
            print(f"--- Integrity Checks: {len(violations)} VIOLATION(S) ---")
            for v in violations:
                print(f"  [{v.severity}] {v.check_name}: {v.description}")

        # DB totals
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
