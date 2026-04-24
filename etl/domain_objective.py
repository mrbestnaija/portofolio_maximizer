"""Repo-wide domain objective constants.

These constants encode the TAKE_PROFIT-first policy in one importable place so
selection, labeling, ranking, and monitoring code do not drift apart.
"""
from __future__ import annotations


SYSTEM_OBJECTIVE = "TAKE_PROFIT_CAPTURE"
DOMAIN_OBJECTIVE_VERSION = "v1.0.0"
MIN_OMEGA_VS_HURDLE = 1.0
MIN_TAKE_PROFIT_FREQUENCY = 0.095
TARGET_AMPLITUDE_MULTIPLIER = 2.0
TAKE_PROFIT_FILTER_THRESHOLD_FALLBACK = 0.15
