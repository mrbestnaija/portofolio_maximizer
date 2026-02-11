"""PnL Integrity Enforcement Framework.

Structural prevention of PnL double-counting, orphaned positions,
diagnostic contamination, and artificial trade legs.
"""

from integrity.pnl_integrity_enforcer import PnLIntegrityEnforcer

__all__ = ["PnLIntegrityEnforcer"]
