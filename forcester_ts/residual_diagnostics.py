"""Shared residual diagnostic utilities for all forecasting models.

Phase 8.2: Run Ljung-Box and Jarque-Bera on model residuals regardless of
model type.  Previously only SARIMAX ran these checks (sarimax.py:912-963);
GARCH, SAMOSSA, and MSSA-RL residuals were never inspected, allowing
silently mis-specified models to pass all gates.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def run_residual_diagnostics(
    residuals: Union[np.ndarray, "pd.Series"],  # noqa: F821
    lags: int = 10,
) -> Dict[str, Any]:
    """Run Ljung-Box + Jarque-Bera on model residuals.

    Parameters
    ----------
    residuals:
        1-D array of model residuals (observed - fitted).
    lags:
        Number of lags to use for the Ljung-Box test.

    Returns
    -------
    dict with keys:
        lb_pvalue    -- Ljung-Box p-value at the last lag (None on failure)
        jb_pvalue    -- Jarque-Bera p-value              (None on failure)
        white_noise  -- True if both lb_pvalue > 0.05 AND jb_pvalue > 0.05
        n            -- number of residual observations used
    """
    result: Dict[str, Any] = {
        "lb_pvalue": None,
        "jb_pvalue": None,
        "white_noise": False,
        "n": 0,
    }

    try:
        import pandas as _pd  # local import to avoid hard dependency at module level
        if hasattr(residuals, "dropna"):
            arr = _pd.to_numeric(residuals, errors="coerce").dropna().to_numpy(dtype=float)
        else:
            arr = np.asarray(residuals, dtype=float)
            arr = arr[np.isfinite(arr)]
    except Exception as exc:
        logger.debug("residual_diagnostics: failed to coerce residuals (%s)", exc)
        return result

    n = len(arr)
    result["n"] = n

    if n < 10:
        logger.debug(
            "residual_diagnostics: too few observations (%d < 10); skipping tests", n
        )
        return result

    # --- Ljung-Box ---
    lb_p: Optional[float] = None
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox  # pylint: disable=import-outside-toplevel
        effective_lags = min(lags, n // 4)
        if effective_lags >= 1:
            lb_df = acorr_ljungbox(arr, lags=effective_lags, return_df=True)
            lb_p = float(lb_df["lb_pvalue"].iloc[-1])
    except Exception as exc:
        logger.debug("residual_diagnostics: Ljung-Box failed (%s)", exc)

    result["lb_pvalue"] = lb_p

    # --- Jarque-Bera ---
    jb_p: Optional[float] = None
    if n >= 5:
        try:
            from scipy.stats import jarque_bera  # pylint: disable=import-outside-toplevel
            jb_result = jarque_bera(arr)
            if isinstance(jb_result, (tuple, list)) and len(jb_result) >= 2:
                jb_p = float(jb_result[1])
        except Exception as exc:
            logger.debug("residual_diagnostics: Jarque-Bera failed (%s)", exc)

    result["jb_pvalue"] = jb_p

    # white_noise = residuals appear uncorrelated (LB) AND normally distributed (JB)
    lb_ok = lb_p is not None and lb_p > 0.05
    jb_ok = jb_p is not None and jb_p > 0.05
    result["white_noise"] = lb_ok and jb_ok

    return result
