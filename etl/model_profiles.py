"""
Model profile selection helpers
-------------------------------

Config-driven selection of time-series model profiles based on sleeve and
regime diagnostics.

This wires together:
- `config/model_profiles.yml` (profile definitions),
- `etl.regime_detector.RegimeDetector` (volatility/trend regimes),
so that callers (TS forecaster runs, TS model search scripts) can request a
profile for a (sleeve, returns) pair without hard-coding per-script grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import yaml

from etl.regime_detector import RegimeDetector, RegimeState

ROOT_PATH = Path(__file__).resolve().parent.parent
MODEL_PROFILES_PATH = ROOT_PATH / "config" / "model_profiles.yml"
TS_MODEL_OVERRIDES_PATH = ROOT_PATH / "config" / "ts_model_overrides.yml"


@dataclass
class ModelProfile:
    name: str
    sleeves: Tuple[str, ...]
    regimes: Tuple[str, ...]
    payload: Dict[str, Any]


@dataclass
class TSModelOverride:
    """
    Explicit TS model selection override for a (ticker, regime) pair.

    This mirrors the schema in config/ts_model_overrides.yml and is intended
    to be driven by evidence from ts_model_candidates + config proposals.
    """

    ticker: str
    regime: str
    candidate_name: str
    profile_hint: Optional[str] = None
    notes: Optional[str] = None


def _normalise_list(values: Any) -> Tuple[str, ...]:
    if not values:
        return tuple()
    return tuple(str(v).strip().lower() for v in values if str(v).strip())


def load_model_profiles(path: Path = MODEL_PROFILES_PATH) -> Dict[str, ModelProfile]:
    """
    Load model profiles from config/model_profiles.yml.

    The YAML structure is expected to be:

    model_profiles:
      profile_name:
        sleeves: [safe, core, ...]
        regimes: [default, high_vol, ...]
        sarimax: {enabled: true, ...}
        samossa: {...}
        garch: {...}
        mssa_rl: {...}
    """
    if not path.exists():
        return {}

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    block = raw.get("model_profiles") or {}
    profiles: Dict[str, ModelProfile] = {}
    for name, cfg in block.items():
        sleeves = _normalise_list(cfg.get("sleeves"))
        regimes = _normalise_list(cfg.get("regimes"))
        payload = {k: v for k, v in cfg.items() if k not in {"sleeves", "regimes"}}
        profiles[name] = ModelProfile(
            name=str(name),
            sleeves=sleeves,
            regimes=regimes,
            payload=payload,
        )
    return profiles


def load_ts_model_overrides(
    path: Path = TS_MODEL_OVERRIDES_PATH,
) -> Dict[Tuple[str, str], TSModelOverride]:
    """
    Load TS model overrides from config/ts_model_overrides.yml.

    Returns a mapping keyed by (TICKER_UPPER, regime_lower).
    """
    if not path.exists():
        return {}

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("overrides") or []
    overrides: Dict[Tuple[str, str], TSModelOverride] = {}

    for item in entries:
        if not isinstance(item, dict):
            continue

        ticker_raw = item.get("ticker")
        candidate_raw = item.get("candidate_name")
        if not ticker_raw or not candidate_raw:
            # Require at least ticker + candidate_name to consider an entry.
            continue

        ticker = str(ticker_raw).strip().upper()
        regime_raw = item.get("regime") or "default"
        regime = str(regime_raw).strip().lower() or "default"
        candidate_name = str(candidate_raw).strip()
        if not ticker or not candidate_name:
            continue

        profile_hint_raw = item.get("profile_hint")
        profile_hint = str(profile_hint_raw).strip() if profile_hint_raw else None
        notes_raw = item.get("notes")
        notes = str(notes_raw).strip() if notes_raw else None

        key = (ticker, regime)
        overrides[key] = TSModelOverride(
            ticker=ticker,
            regime=regime,
            candidate_name=candidate_name,
            profile_hint=profile_hint,
            notes=notes,
        )

    return overrides


def lookup_ts_model_override(
    ticker: str,
    regime: str,
    *,
    overrides: Optional[Dict[Tuple[str, str], TSModelOverride]] = None,
    path: Path = TS_MODEL_OVERRIDES_PATH,
) -> Optional[TSModelOverride]:
    """
    Look up a TS model override for (ticker, regime).

    Resolution order:
    1. Exact (ticker, regime) match.
    2. Fallback to (ticker, "default") if present.
    """
    ticker_norm = str(ticker).strip().upper()
    regime_norm = str(regime or "default").strip().lower() or "default"

    if overrides is None:
        overrides = load_ts_model_overrides(path)

    return overrides.get((ticker_norm, regime_norm)) or overrides.get(
        (ticker_norm, "default")
    )


def select_profile_for_sleeve_and_returns(
    sleeve: str,
    returns: Sequence[float],
    *,
    detector: Optional[RegimeDetector] = None,
    profiles_path: Path = MODEL_PROFILES_PATH,
) -> Tuple[Optional[ModelProfile], RegimeState]:
    """
    Select an appropriate model profile for a given (sleeve, returns) pair.

    Selection rules:
    1. Detect volatility regime via RegimeDetector.detect_volatility_regime.
    2. Among profiles whose sleeves include the given sleeve (case-insensitive):
       a) Prefer those whose regimes include the detected regime_type.
       b) Otherwise profiles whose regimes include "default".
    3. If nothing matches, fall back to:
       a) "default" profile if present,
       b) None.

    Returns
    -------
    (profile, regime_state)
        profile: ModelProfile or None if no suitable profile found.
        regime_state: RegimeState returned by the detector (always present).
    """
    sleeve_norm = str(sleeve).strip().lower()
    detector = detector or RegimeDetector()

    arr = np.asarray(returns, dtype=float)
    regime_state = detector.detect_volatility_regime(arr)
    regime_label = regime_state.regime_type.strip().lower()

    profiles = load_model_profiles(profiles_path)
    if not profiles:
        return None, regime_state

    # Step 1: candidates with matching sleeve.
    matching_sleeve = [
        p for p in profiles.values() if not p.sleeves or sleeve_norm in p.sleeves
    ]
    if not matching_sleeve:
        matching_sleeve = list(profiles.values())

    # Step 2: best regime match within sleeve candidates.
    exact = [
        p for p in matching_sleeve if p.regimes and regime_label in p.regimes
    ]
    if exact:
        return exact[0], regime_state

    default_regime = [
        p for p in matching_sleeve if p.regimes and "default" in p.regimes
    ]
    if default_regime:
        return default_regime[0], regime_state

    # Step 3: global default fallback.
    if "default" in profiles:
        return profiles["default"], regime_state

    return matching_sleeve[0], regime_state


def select_profile_with_overrides(
    ticker: str,
    sleeve: str,
    returns: Sequence[float],
    *,
    detector: Optional[RegimeDetector] = None,
    profiles_path: Path = MODEL_PROFILES_PATH,
    overrides_path: Path = TS_MODEL_OVERRIDES_PATH,
) -> Tuple[Optional[ModelProfile], RegimeState, Optional[TSModelOverride]]:
    """
    Extended profile selection that also consults ts_model_overrides.

    Workflow:
    1. Use select_profile_for_sleeve_and_returns to pick a baseline profile
       based on (sleeve, returns)-driven volatility regime.
    2. Look up a TSModelOverride for (ticker, detected_regime) or (ticker, default)
       from config/ts_model_overrides.yml.
    3. If an override is found and it has profile_hint, attempt to swap the
       profile to that hinted profile (when defined in model_profiles.yml).

    Returns
    -------
    (profile, regime_state, override)
        profile: baseline or override-hinted ModelProfile (may be None).
        regime_state: detected volatility regime.
        override: TSModelOverride if one exists for (ticker, regime), else None.
    """
    detector = detector or RegimeDetector()
    # Step 1: baseline profile + regime detection.
    profile, regime_state = select_profile_for_sleeve_and_returns(
        sleeve=sleeve,
        returns=returns,
        detector=detector,
        profiles_path=profiles_path,
    )

    # Step 2: override lookup based on (ticker, regime/default).
    overrides_map = load_ts_model_overrides(overrides_path)
    override = lookup_ts_model_override(
        ticker=ticker,
        regime=regime_state.regime_type,
        overrides=overrides_map,
    )

    # Step 3: optional profile_hint remapping.
    if override is not None and override.profile_hint:
        profiles = load_model_profiles(profiles_path)
        hinted_name = override.profile_hint.strip()
        hinted_profile = profiles.get(hinted_name)
        if hinted_profile is not None:
            profile = hinted_profile

    return profile, regime_state, override


__all__ = [
    "ModelProfile",
    "TSModelOverride",
    "load_model_profiles",
    "load_ts_model_overrides",
    "lookup_ts_model_override",
    "select_profile_for_sleeve_and_returns",
    "select_profile_with_overrides",
]

