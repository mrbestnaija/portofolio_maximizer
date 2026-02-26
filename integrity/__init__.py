"""PnL Integrity Enforcement Framework.

Structural prevention of PnL double-counting, orphaned positions,
diagnostic contamination, and artificial trade legs.

Submodules are imported lazily so that importing any single submodule
(e.g. integrity.sqlite_guardrails) does not load all heavy dependencies
(pandas, numpy) from sibling modules. Callers should import directly:
    from integrity.pnl_integrity_enforcer import PnLIntegrityEnforcer
    from integrity.sqlite_guardrails import guarded_sqlite_connect
"""
