from __future__ import annotations

import pandas as pd


def seasonality_multiplier(ts: pd.Timestamp, season_cfg: dict | None) -> float:
    if not season_cfg:
        return 1.0
    dow = ts.dayofweek
    hour = ts.hour
    weights = season_cfg.get("day_weights", {}) or {}
    hour_weights = season_cfg.get("hour_weights", {}) or {}
    day_mult = float(weights.get(str(dow), 1.0))
    hour_mult = float(hour_weights.get(str(hour), 1.0)) if hour_weights else 1.0
    return day_mult * hour_mult
