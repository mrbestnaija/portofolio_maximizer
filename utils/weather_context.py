from __future__ import annotations

import json
from collections.abc import MutableMapping
from typing import Any, Dict, Optional

import pandas as pd


WEATHER_CONTEXT_COLUMNS = {
    "event_type": "weather_event_type",
    "severity": "weather_severity",
    "impact_score": "weather_impact_score",
    "confidence": "weather_confidence",
    "supply_disruption_pct": "weather_supply_disruption_pct",
    "days_to_event": "weather_days_to_event",
    "impact_direction": "weather_impact_direction",
    "region": "weather_region",
}

_WEATHER_SEVERITY_SCORES = {
    "none": 0.0,
    "low": 0.25,
    "minor": 0.25,
    "moderate": 0.5,
    "medium": 0.5,
    "high": 0.75,
    "severe": 0.9,
    "extreme": 1.0,
}
_ADVERSE_DIRECTIONS = {"adverse", "negative", "harmful", "disruptive"}
_FAVORABLE_DIRECTIONS = {"favorable", "positive", "supportive", "benign"}


def _coerce_mapping(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return dict(parsed)
    return {}


def _safe_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return float(parsed)


def _safe_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return int(parsed)


def _lookup_ticker_context(mapping: Dict[str, Any], ticker: Optional[str]) -> Dict[str, Any]:
    if not ticker:
        return {}
    for candidate in (str(ticker).strip(), str(ticker).strip().upper(), str(ticker).strip().lower()):
        if not candidate:
            continue
        payload = _coerce_mapping(mapping.get(candidate))
        if payload:
            return payload
    return {}


def _extract_weather_context_from_columns(
    market_data: Optional[pd.DataFrame],
    *,
    ticker: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(market_data, pd.DataFrame) or market_data.empty:
        return {}

    frame = market_data
    if ticker and "ticker" in frame.columns:
        ticker_series = frame["ticker"].astype(str).str.upper()
        matches = ticker_series == str(ticker).strip().upper()
        if bool(matches.any()):
            frame = frame.loc[matches]
    if frame.empty:
        return {}

    extracted: Dict[str, Any] = {}
    for field, column in WEATHER_CONTEXT_COLUMNS.items():
        if column not in frame.columns:
            continue
        series = frame[column].dropna()
        if series.empty:
            continue
        value = series.iloc[-1]
        if field in {"impact_score", "confidence", "supply_disruption_pct"}:
            parsed = _safe_float(value)
            if parsed is not None:
                extracted[field] = parsed
        elif field == "days_to_event":
            parsed = _safe_int(value)
            if parsed is not None:
                extracted[field] = parsed
        else:
            text = str(value).strip()
            if text:
                extracted[field] = text
    return extracted


def extract_weather_context(
    market_data: Optional[pd.DataFrame],
    *,
    ticker: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract structured weather context from market-data attrs or columns."""
    if not isinstance(market_data, pd.DataFrame):
        return {}

    attrs = getattr(market_data, "attrs", None)
    attr_payload: Dict[str, Any] = {}
    if isinstance(attrs, dict):
        by_ticker = _coerce_mapping(attrs.get("weather_context_by_ticker"))
        attr_payload = _lookup_ticker_context(by_ticker, ticker)
        if not attr_payload:
            attr_payload = _coerce_mapping(attrs.get("weather_context"))

    column_payload = _extract_weather_context_from_columns(market_data, ticker=ticker)
    return merge_weather_contexts(attr_payload, column_payload)


def merge_weather_contexts(*contexts: Any) -> Dict[str, Any]:
    """Merge weather contexts conservatively so adverse/stronger signals win."""
    merged: Dict[str, Any] = {}
    severity_score = -1.0
    best_severity = None
    days_to_event: Optional[int] = None
    impact_direction = ""

    for raw in contexts:
        ctx = _coerce_mapping(raw)
        if not ctx:
            continue

        event_type = str(ctx.get("event_type") or "").strip()
        if event_type and not merged.get("event_type"):
            merged["event_type"] = event_type

        region = str(ctx.get("region") or "").strip()
        if region and not merged.get("region"):
            merged["region"] = region

        severity = str(ctx.get("severity") or "").strip().lower()
        score = _WEATHER_SEVERITY_SCORES.get(severity)
        if score is not None and score > severity_score:
            severity_score = score
            best_severity = severity

        for key in ("impact_score", "confidence", "supply_disruption_pct"):
            value = _safe_float(ctx.get(key))
            if value is None:
                continue
            prior = _safe_float(merged.get(key))
            merged[key] = value if prior is None else max(prior, value)

        day_value = _safe_int(ctx.get("days_to_event"))
        if day_value is not None:
            if days_to_event is None or day_value < days_to_event:
                days_to_event = day_value

        direction = str(ctx.get("impact_direction") or "").strip().lower()
        if direction in _ADVERSE_DIRECTIONS:
            impact_direction = "adverse"
        elif direction in _FAVORABLE_DIRECTIONS and impact_direction != "adverse":
            impact_direction = "favorable"
        elif direction and not impact_direction:
            impact_direction = direction

    if best_severity is not None:
        merged["severity"] = best_severity
    if days_to_event is not None:
        merged["days_to_event"] = days_to_event
    if impact_direction:
        merged["impact_direction"] = impact_direction

    if not merged:
        return {}
    if "event_type" not in merged:
        merged["event_type"] = "weather_event"
    return merged


def hydrate_signal_weather_context(
    signal: MutableMapping[str, Any],
    market_data: Optional[pd.DataFrame],
    *,
    ticker: Optional[str] = None,
) -> Dict[str, Any]:
    """Attach merged weather context to both root signal and provenance."""
    if not isinstance(signal, MutableMapping):
        return {}

    ticker_value = ticker if ticker is not None else signal.get("ticker")
    provenance = _coerce_mapping(signal.get("provenance"))
    merged = merge_weather_contexts(
        signal.get("weather_context"),
        provenance.get("weather_context"),
        extract_weather_context(market_data, ticker=ticker_value),
    )
    if not merged:
        return {}

    signal["weather_context"] = merged
    provenance["weather_context"] = merged
    signal["provenance"] = provenance
    return merged
