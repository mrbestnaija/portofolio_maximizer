"""
GARCH volatility model implementation moved from the ETL layer.

Phase 7.10b improvements:
  - AR(1) conditional mean model for directional signal (was 'Zero' / volatility only)
  - skewt distribution for fat tails + negative skew (was 'normal')
  - ADF stationarity test; auto-difference if unit root detected
  - GJR-GARCH asymmetric vol fallback before EWMA when persistence >= 0.97 (hard_igarch_threshold)
"""

from __future__ import annotations

import logging
import warnings as _warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    from arch import arch_model

    ARCH_AVAILABLE = True
except Exception:  # pragma: no cover - defensive
    arch_model = None  # type: ignore[assignment]
    ARCH_AVAILABLE = False

try:
    from statsmodels.tsa.stattools import adfuller as _adfuller  # type: ignore[import]

    STATSMODELS_ADF_AVAILABLE = True
except Exception:  # pragma: no cover
    _adfuller = None  # type: ignore[assignment]
    STATSMODELS_ADF_AVAILABLE = False

logger = logging.getLogger(__name__)

GARCH_FALLBACK_MODE_NONE = "none"
_GARCH_FALLBACK_MODE_ALIASES = {
    "ewma_backend": "explicit_ewma_backend",
}
VALID_GARCH_FALLBACK_MODES = {
    GARCH_FALLBACK_MODE_NONE,
    "explicit_ewma_backend",
    "insufficient_sample_size",
    "near_igarch",
    "convergence_failure",
    "exploding_variance_ratio",
}


def _coerce_garch_fallback_mode(
    value: Any,
    *,
    default: str = GARCH_FALLBACK_MODE_NONE,
) -> str:
    if value is None:
        return default
    mode = _GARCH_FALLBACK_MODE_ALIASES.get(str(value).strip().lower(), str(value).strip().lower())
    if mode in VALID_GARCH_FALLBACK_MODES:
        return mode
    logger.debug("Unknown GARCH fallback mode '%s'; using default '%s'.", value, default)
    return default


class GARCHForecaster:
    """Wrapper around `arch` to produce volatility forecasts."""

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        vol: str = "GARCH",
        dist: str = "skewt",
        mean: str = "AR",
        backend: Optional[str] = None,
        *,
        auto_select: bool = True,
        max_p: int = 3,
        max_q: int = 3,
        enforce_stationarity: bool = True,
        igarch_fallback: str = "gjr",
        ewma_lambda: float = 0.94,
        min_arch_sample_size: int = 120,
        hard_igarch_threshold: float = 0.97,  # P4 fix: align to documented 0.97 (was 0.99)
        max_volatility_ratio_to_realized: float = 4.0,
    ) -> None:
        self.p = p
        self.q = q
        self.vol = vol
        self.dist = dist
        self.mean = mean  # Phase 7.10b: AR(1) mean model for directional signal
        self.enforce_stationarity = bool(enforce_stationarity)  # ADF pre-check
        self.igarch_fallback = str(igarch_fallback).lower()  # 'gjr' or 'ewma'
        self.ewma_lambda = float(ewma_lambda)
        self.backend = (backend or ("arch" if ARCH_AVAILABLE else "ewma")).lower()
        self.auto_select = bool(auto_select)
        self.max_p = int(max_p)
        self.max_q = int(max_q)
        self.min_arch_sample_size = max(int(min_arch_sample_size), 20)
        self.hard_igarch_threshold = float(hard_igarch_threshold)
        self.max_volatility_ratio_to_realized = float(max_volatility_ratio_to_realized)
        if not 0.0 < self.ewma_lambda < 1.0:
            raise ValueError(
                f"Invalid ewma_lambda={self.ewma_lambda!r}; expected 0 < ewma_lambda < 1."
            )
        if self.backend == "arch" and not ARCH_AVAILABLE:
            logger.warning("arch not installed; falling back to EWMA volatility model.")
            self.backend = "ewma"
        if self.backend not in {"arch", "ewma"}:
            raise ValueError(f"Unsupported GARCH backend '{backend}' (expected 'arch' or 'ewma').")

        self.model = None
        self.fitted_model = None
        self.forecast_results: Optional[Dict[str, Any]] = None
        self._scale_factor = 1.0  # Track scaling factor for rescaling forecasts
        self._fallback_state: Optional[Dict[str, Any]] = None
        self._differenced: bool = False  # Track if series was differenced for stationarity
        self._convergence_ok: bool = True  # Phase 7.14-C: False when optimizer did not converge
        # Phase 8.2: residual diagnostics populated after fit() completes.
        self._residual_diagnostics: Dict[str, Any] = {}
        self._fallback_mode: str = GARCH_FALLBACK_MODE_NONE
        self._last_persistence: Optional[float] = None
        self._volatility_ratio_to_realized: Optional[float] = None
        self._realized_volatility: Optional[float] = None
        self._fit_sample_size: int = 0
        self._training_returns_clean: Optional[pd.Series] = None

    def fit(
        self,
        returns: pd.Series,
        order_learner=None,
        ticker: str = "",
        regime: str | None = None,
    ) -> "GARCHForecaster":
        if not getattr(self, "auto_select", True):
            raise ValueError("Manual GARCH orders are unsupported; set auto_select=True and use max_p/max_q caps.")

        if returns.isna().all():
            raise ValueError("Returns series cannot be all NaNs")

        returns_clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if returns_clean.empty:
            raise ValueError("Returns series required for GARCH is empty after dropna")
        returns_clean = returns_clean.astype(float)
        self._convergence_ok = True
        self._fallback_state = None
        self._fallback_mode = GARCH_FALLBACK_MODE_NONE
        self._last_persistence = None
        self._volatility_ratio_to_realized = None
        self._residual_diagnostics = {}
        self._training_returns_clean = returns_clean.copy()
        self._fit_sample_size = int(len(returns_clean))

        # Guard against numerical overflow by clipping extreme tails.
        abs_returns = returns_clean.abs()
        abs_returns = abs_returns[np.isfinite(abs_returns)]
        if abs_returns.empty:
            raise ValueError("Returns series required for GARCH is empty after sanitization")
        try:
            p995 = float(np.nanpercentile(abs_returns, 99.5))
        except Exception:
            p995 = float(abs_returns.max())
        med = float(np.nanmedian(abs_returns)) if len(abs_returns) else 0.0
        cap = max(p995, med * 10.0) if med > 0 else p995
        if np.isfinite(cap) and cap > 0:
            returns_clean = returns_clean.clip(-cap, cap)
        self._training_returns_clean = returns_clean.copy()
        realized_volatility = float(returns_clean.std(ddof=0))
        if not np.isfinite(realized_volatility) or realized_volatility <= 0.0:
            realized_volatility = float(returns_clean.abs().mean())
        self._realized_volatility = max(realized_volatility, 1e-6)

        backend = getattr(self, "backend", "arch")
        if backend != "arch":
            # Lightweight EWMA fallback when `arch` isn't installed. This keeps
            # the forecasting stack functional in minimal environments and for
            # unit tests while still reacting to recent volatility.
            return self._fit_ewma(returns_clean)

        # Phase 7.10b: ADF stationarity check. Financial returns are generally
        # stationary, but some series (especially level prices fed incorrectly)
        # may have a unit root. Auto-difference once if detected.
        if getattr(self, "enforce_stationarity", True) and STATSMODELS_ADF_AVAILABLE:
            try:
                adf_result = _adfuller(returns_clean.values, autolag="AIC")
                adf_pvalue = float(adf_result[1])
                if adf_pvalue > 0.05:
                    logger.warning(
                        "ADF test: unit root detected (p=%.4f); differencing returns once.",
                        adf_pvalue,
                    )
                    returns_clean = returns_clean.diff().dropna()
                    self._differenced = True
                    if returns_clean.empty:
                        raise ValueError("Returns series empty after ADF-motivated differencing")
            except Exception as adf_exc:  # pragma: no cover
                logger.debug("ADF stationarity check failed (%s); skipping.", adf_exc)

        self._training_returns_clean = returns_clean.copy()
        self._fit_sample_size = int(len(returns_clean))
        realized_volatility = float(returns_clean.std(ddof=0))
        if not np.isfinite(realized_volatility) or realized_volatility <= 0.0:
            realized_volatility = float(returns_clean.abs().mean())
        self._realized_volatility = max(realized_volatility, 1e-6)

        if len(returns_clean) < self.min_arch_sample_size:
            logger.warning(
                "GARCH sample size %d below minimum %d; falling back to EWMA.",
                len(returns_clean),
                self.min_arch_sample_size,
            )
            self.backend = "ewma"
            return self._fit_ewma(
                returns_clean,
                reason="insufficient_sample_size",
            )

        returns_scaled, self._scale_factor = self._scale_returns_for_fit(returns_clean)

        best_model = None
        best_fit = None
        best_aic = np.inf
        best_order = (self.p, self.q)
        _best_convergence_ok = True  # tracks convergence of the SELECTED model only

        # Phase 7.10b: Use configurable mean model (AR for directional signal).
        # 'AR' adds an AR(1) conditional mean; 'Zero' is the old default.
        mean_model = str(getattr(self, "mean", "AR"))
        # skewt may fail if arch version doesn't support it; fall back to 't' then 'normal'.
        dist_candidates = [self.dist]
        if self.dist == "skewt":
            dist_candidates += ["t", "normal"]
        elif self.dist == "t":
            dist_candidates += ["normal"]
        best_dist = dist_candidates[0]

        max_p = max(1, int(getattr(self, "max_p", self.p)))
        max_q = max(1, int(getattr(self, "max_q", self.q)))

        # Phase 7.16: OrderLearner warm-start — prioritize cached (p,q,dist,mean).
        _p_range = list(range(1, max_p + 1))
        _q_range = list(range(1, max_q + 1))
        if order_learner is not None and ticker:
            try:
                _suggestion = order_learner.suggest(ticker, "GARCH", regime)
                if _suggestion is not None:
                    _wp = int(_suggestion.get("p", self.p))
                    _wq = int(_suggestion.get("q", self.q))
                    _wd = str(_suggestion.get("dist", dist_candidates[0]))
                    _wm = str(_suggestion.get("mean", mean_model))
                    if order_learner.should_skip_grid(ticker, "GARCH", regime):
                        _p_range = [_wp]
                        _q_range = [_wq]
                    else:
                        _p_range = sorted({_wp} | set(range(1, max_p + 1)))
                        _q_range = sorted({_wq} | set(range(1, max_q + 1)))
                    dist_candidates = [_wd] + [d for d in dist_candidates if d != _wd]
                    mean_model = _wm
                    logger.debug(
                        "GARCH warm-start %s/%s: p=%d q=%d dist=%s mean=%s",
                        ticker, regime, _wp, _wq, _wd, _wm,
                    )
            except Exception as _we:
                logger.debug("GARCH OrderLearner.suggest failed: %s", _we)

        for p_candidate in _p_range:
            for q_candidate in _q_range:
                for dist_try in dist_candidates:
                    try:
                        model = arch_model(
                            returns_scaled,
                            mean=mean_model,
                            vol=self.vol,
                            p=p_candidate,
                            q=q_candidate,
                            dist=dist_try,
                            rescale=False,  # We handle scaling manually
                        )
                        # Phase 7.14-C: Detect convergence failure via warning capture.
                        # scipy optimizer emits RuntimeWarning containing "convergence" or
                        # "code 9" when SLSQP hits the iteration limit.
                        _convergence_failed = False
                        with _warnings.catch_warnings(record=True) as _caught:
                            _warnings.simplefilter("always")
                            fitted = model.fit(disp="off")
                            _convergence_failed = any(
                                issubclass(w.category, (RuntimeWarning, UserWarning))
                                and (
                                    "convergence" in str(w.message).lower()
                                    or "code 9" in str(w.message).lower()
                                )
                                for w in _caught
                            )
                        if not _convergence_failed and hasattr(fitted, "convergence_flag"):
                            _convergence_failed = fitted.convergence_flag != 0
                        if _convergence_failed:
                            logger.warning(
                                "GARCH(%s,%s, %s) optimizer did not converge; "
                                "considering GJR fallback if selected.",
                                p_candidate, q_candidate, dist_try,
                            )
                        aic = float(getattr(fitted, "aic", np.inf))
                        if np.isfinite(aic) and aic < best_aic:
                            best_aic = aic
                            best_model = model
                            best_fit = fitted
                            best_order = (p_candidate, q_candidate)
                            best_dist = dist_try
                            # Track convergence of the *selected* model only.
                            # Earlier failed orders must not poison this flag —
                            # the stability guard at line ~351 acts on the final
                            # selected model, not on any rejected candidates.
                            _best_convergence_ok = not _convergence_failed
                        break  # Use first dist that fits
                    except Exception:  # pragma: no cover - best-effort search
                        continue

        # Write convergence status for the selected model once, after grid search.
        # Doing this inside the loop would set _convergence_ok=False from any
        # rejected candidate, even when the finally selected model converged.
        self._convergence_ok = _best_convergence_ok

        self.p, self.q = best_order
        self.model = best_model
        self.fitted_model = best_fit

        # Phase 7.16: Record best fit in OrderLearner for future warm-start.
        if order_learner is not None and best_fit is not None and ticker:
            try:
                _aic_r = float(getattr(best_fit, "aic", float("nan")))
                _bic_r = float(getattr(best_fit, "bic", float("nan")))
                order_learner.record_fit(
                    ticker=ticker, model_type="GARCH", regime=regime,
                    order_params={
                        "p": self.p, "q": self.q,
                        "dist": best_dist, "mean": mean_model,
                    },
                    aic=_aic_r, bic=_bic_r,
                    n_obs=len(returns_scaled),
                )
            except Exception as _oe:
                logger.debug("GARCH OrderLearner.record_fit failed: %s", _oe)

        if self.fitted_model is None:
            logger.warning(
                "GARCH auto_select failed to fit any model; falling back to EWMA volatility."
            )
            self.backend = "ewma"
            return self._fit_ewma(returns_clean)
        # Phase 7.10b: IGARCH / unit-root guard -- if alpha+beta >= 0.97 the
        # variance process is near-non-stationary and forecasts diverge.
        # Phase 7.14-C: Also trigger GJR fallback when the optimizer did not converge,
        # because unconverged fits produce unreliable parameter estimates and CIs.
        # Try GJR-GARCH (asymmetric volatility; better for equity fat tails) first;
        # only fall back to EWMA if GJR also degenerates.
        persistence = self._garch_persistence()
        self._last_persistence = persistence
        volatility_ratio = self._conditional_volatility_ratio_to_realized()
        self._volatility_ratio_to_realized = volatility_ratio
        convergence_failed = not getattr(self, "_convergence_ok", True)
        ratio_exploded = (
            volatility_ratio is not None
            and volatility_ratio > self.max_volatility_ratio_to_realized
        )
        if (
            (persistence is not None and persistence >= self.hard_igarch_threshold)
            or convergence_failed
            or ratio_exploded
        ):
            if persistence is not None and persistence >= self.hard_igarch_threshold:
                guard_reason = "near_igarch"
            elif convergence_failed:
                guard_reason = "convergence_failure"
            else:
                guard_reason = "exploding_variance_ratio"
            igarch_fallback = str(getattr(self, "igarch_fallback", "gjr")).lower()
            if igarch_fallback == "gjr" and ARCH_AVAILABLE:
                logger.warning(
                    "GARCH(%s,%s) triggered stability guard (%s, persistence=%s, "
                    "volatility_ratio=%s); attempting GJR-GARCH asymmetric model.",
                    self.p,
                    self.q,
                    guard_reason,
                    f"{persistence:.4f}" if persistence is not None else "n/a",
                    f"{volatility_ratio:.4f}" if volatility_ratio is not None else "n/a",
                )
                try:
                    gjr_model = arch_model(
                        returns_scaled,
                        mean=mean_model,
                        vol="GARCH",
                        p=self.p,
                        o=1,  # GJR leverage term
                        q=self.q,
                        dist=dist_candidates[0],
                        rescale=False,
                    )
                    # GJR is already a fallback path; optimizer warnings here are
                    # expected and handled — suppress to avoid log noise.
                    with _warnings.catch_warnings(record=True) as _gjr_caught:
                        _warnings.simplefilter("always")
                        gjr_fit = gjr_model.fit(disp="off")
                    _gjr_conv_failed = any(
                        issubclass(w.category, (RuntimeWarning, UserWarning))
                        and (
                            "convergence" in str(w.message).lower()
                            or "code 9" in str(w.message).lower()
                        )
                        for w in _gjr_caught
                    )
                    if _gjr_conv_failed:
                        logger.debug(
                            "GJR-GARCH optimizer did not fully converge; "
                            "checking persistence to decide fallback."
                        )
                    if hasattr(gjr_fit, "convergence_flag") and gjr_fit.convergence_flag != 0:
                        _gjr_conv_failed = True
                    gjr_persistence = sum(
                        float(v) for k, v in gjr_fit.params.items()
                        if k.startswith(("alpha[", "beta[")) and np.isfinite(float(v))
                    )
                    self._last_persistence = gjr_persistence
                    gjr_volatility_ratio = self._conditional_volatility_ratio_to_realized(gjr_fit)
                    self._volatility_ratio_to_realized = gjr_volatility_ratio
                    if (
                        not _gjr_conv_failed
                        and gjr_persistence < self.hard_igarch_threshold
                        and (
                            gjr_volatility_ratio is None
                            or gjr_volatility_ratio <= self.max_volatility_ratio_to_realized
                        )
                    ):
                        self.model = gjr_model
                        self.fitted_model = gjr_fit
                        self.vol = "GJR-GARCH"
                        logger.info(
                            "GJR-GARCH fitted (persistence=%.4f, volatility_ratio=%s); "
                            "using asymmetric model.",
                            gjr_persistence,
                            f"{gjr_volatility_ratio:.4f}"
                            if gjr_volatility_ratio is not None
                            else "n/a",
                        )
                        self._residual_diagnostics = self._capture_residual_diagnostics()
                        return self
                except Exception as gjr_exc:
                    logger.debug("GJR-GARCH attempt failed (%s); falling back to EWMA.", gjr_exc)
            logger.warning(
                "GARCH stability guard triggered (%s); GJR unavailable or also unstable; "
                "falling back to EWMA to avoid divergent variance forecasts.",
                guard_reason,
            )
            self.backend = "ewma"
            return self._fit_ewma(
                returns_clean,
                reason=guard_reason,
                persistence=self._last_persistence,
                volatility_ratio_to_realized=self._volatility_ratio_to_realized,
            )

        logger.info(
            "GARCH(%s,%s) model fitted successfully (vol=%s, dist=%s, persistence=%.4f, "
            "volatility_ratio=%s)",
            self.p,
            self.q,
            self.vol,
            self.dist,
            persistence if persistence is not None else 0.0,
            f"{volatility_ratio:.4f}" if volatility_ratio is not None else "n/a",
        )
        self._residual_diagnostics = self._capture_residual_diagnostics()
        return self

    @staticmethod
    def _scale_returns_for_fit(returns: pd.Series) -> tuple[pd.Series, float]:
        """Scale fit inputs into a stable numeric band for the ARCH optimizer."""
        values = returns.dropna().values
        if values.size == 0:
            return returns, 1.0

        max_abs = float(np.nanmax(np.abs(values)))
        if not np.isfinite(max_abs) or max_abs == 0.0:
            return returns, 1.0

        scale_factor = 1.0
        if max_abs < 1.0:
            scale_factor = min(1000.0, 1.0 / max_abs)
        elif max_abs > 1000.0:
            scale_factor = 1000.0 / max_abs

        if scale_factor == 1.0:
            return returns, 1.0
        return returns * scale_factor, float(scale_factor)

    def _capture_residual_diagnostics(self) -> Dict[str, Any]:
        """Phase 8.2: run Ljung-Box + Jarque-Bera on GARCH standardized residuals."""
        try:
            from .residual_diagnostics import run_residual_diagnostics  # pylint: disable=import-outside-toplevel
            if self.fitted_model is None or self.fitted_model is True:
                return {}
            resid = getattr(self.fitted_model, "resid", None)
            if resid is None:
                return {}
            diag = run_residual_diagnostics(resid)
            if not diag.get("white_noise", True):
                logger.warning(
                    "GARCH residuals fail white-noise check "
                    "(lb_p=%.4f, jb_p=%.4f, n=%d) — model may be mis-specified.",
                    diag.get("lb_pvalue") or 0.0,
                    diag.get("jb_pvalue") or 0.0,
                    diag.get("n", 0),
                )
            return diag
        except Exception as exc:
            logger.debug("GARCH _capture_residual_diagnostics failed: %s", exc)
            return {}

    def _garch_persistence(self) -> Optional[float]:
        """Sum of alpha + beta coefficients (persistence parameter).

        Returns None if the model is not fitted or params are unavailable.
        For stationary GARCH, persistence < 1.0; IGARCH has persistence >= 1.0.
        """
        if self.fitted_model is None or self.fitted_model is True:
            return None
        try:
            params = self.fitted_model.params
            alpha_sum = sum(
                float(v) for k, v in params.items()
                if k.startswith("alpha[") and np.isfinite(v)
            )
            beta_sum = sum(
                float(v) for k, v in params.items()
                if k.startswith("beta[") and np.isfinite(v)
            )
            return alpha_sum + beta_sum
        except Exception:
            return None

    def _conditional_volatility_ratio_to_realized(self, fitted_model: Any = None) -> Optional[float]:
        realized = float(getattr(self, "_realized_volatility", 0.0) or 0.0)
        if not np.isfinite(realized) or realized <= 0.0:
            return None
        model = fitted_model if fitted_model is not None else self.fitted_model
        if model is None or model is True:
            return None
        try:
            conditional = getattr(model, "conditional_volatility", None)
            if conditional is None:
                return None
            values = pd.to_numeric(pd.Series(conditional), errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            if values.empty:
                return None
            # Use p95 of |conditional_vol| rather than the terminal bar; a single
            # tail spike at the last in-sample step should not trip the guard.
            return float(np.percentile(values.abs(), 95)) / realized
        except Exception:
            return None

    def _forecast_volatility_ratio_to_realized(self, volatility: Any) -> Optional[float]:
        realized = float(getattr(self, "_realized_volatility", 0.0) or 0.0)
        if not np.isfinite(realized) or realized <= 0.0:
            return None
        try:
            values = pd.to_numeric(pd.Series(volatility), errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            if values.empty:
                return None
            # Use p95 rather than max; forecast vol is monotone-increasing toward
            # unconditional variance so max() always picks the worst step and is
            # easily triggered by a single high-horizon step.
            return float(np.percentile(values, 95)) / realized
        except Exception:
            return None

    def _fit_ewma(
        self,
        returns_clean: pd.Series,
        *,
        reason: str = "explicit_ewma_backend",
        persistence: Optional[float] = None,
        volatility_ratio_to_realized: Optional[float] = None,
    ) -> "GARCHForecaster":
        fallback_mode = _coerce_garch_fallback_mode(reason, default="explicit_ewma_backend")
        lam = float(getattr(self, "ewma_lambda", 0.94))
        squared = np.square(returns_clean.to_numpy(dtype=float, copy=False))
        baseline_var = float(np.var(squared)) if squared.size else 0.0
        variance = float(np.var(returns_clean.to_numpy(dtype=float, copy=False), ddof=0))
        # Floor at 1e-6 (~0.1% daily vol): below any real equity daily variance.
        # 1e-12 (machine-epsilon) collapses the CI → inflates SNR spuriously.
        MIN_EWMA_VARIANCE = 1e-6
        if not np.isfinite(variance) or variance <= 0.0:
            variance = max(MIN_EWMA_VARIANCE, baseline_var)

        ewma_var = variance
        for r2 in squared:
            ewma_var = lam * ewma_var + (1.0 - lam) * float(r2)
        ewma_var = float(max(ewma_var, MIN_EWMA_VARIANCE))

        self._fallback_state = {
            "lambda": lam,
            "unconditional_variance": variance,
            "last_variance": ewma_var,
            "mean": float(returns_clean.mean()),
            "n_obs": int(len(returns_clean)),
            "fallback_mode": fallback_mode,
            "persistence": persistence if persistence is not None else self._last_persistence,
            "volatility_ratio_to_realized": (
                volatility_ratio_to_realized
                if volatility_ratio_to_realized is not None
                else self._volatility_ratio_to_realized
            ),
            "realized_volatility": self._realized_volatility,
        }
        self._fallback_mode = fallback_mode
        self._last_persistence = self._fallback_state.get("persistence")
        self._volatility_ratio_to_realized = self._fallback_state.get(
            "volatility_ratio_to_realized"
        )
        # Sentinel so downstream callers treat the model as "fitted".
        self.fitted_model = True
        logger.info(
            "EWMA volatility model fitted successfully (lambda=%.3f, var=%.6g, fallback_mode=%s)",
            lam,
            ewma_var,
            fallback_mode,
        )
        return self

    def _current_fallback_mode(self) -> str:
        return _coerce_garch_fallback_mode(getattr(self, "_fallback_mode", None))

    def _state_fallback_mode(self, state: Optional[Dict[str, Any]]) -> str:
        payload = state or {}
        return _coerce_garch_fallback_mode(
            payload.get("fallback_mode", payload.get("fallback_reason")),
        )

    def forecast(self, steps: int) -> Dict[str, Any]:
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")

        backend = getattr(self, "backend", "arch")
        if backend != "arch":
            state = self._fallback_state or {}
            fallback_mode = self._state_fallback_mode(state)
            last_var = float(state.get("last_variance") or 0.0)
            last_var = float(max(last_var, 1e-6))
            idx = pd.Index(range(1, int(steps) + 1), name="horizon")
            variance = pd.Series([last_var] * int(steps), index=idx)
            mean = pd.Series([float(state.get("mean") or 0.0)] * int(steps), index=idx)
            volatility = np.sqrt(variance)
            self.forecast_results = {
                "variance_forecast": variance,
                "mean_forecast": mean,
                "volatility": volatility,
                "steps": int(steps),
                "p": self.p,
                "q": self.q,
                "vol": self.vol,
                "dist": self.dist,
                "aic": None,
                "bic": None,
                "convergence_ok": False,  # EWMA fallback means arch fit failed or unavailable
                "residual_diagnostics": {},
                "residual_diagnostics_status": "unavailable",
                "residual_diagnostics_reason": "ewma_fallback",
                "ewma_lambda": state.get("lambda"),
                "fallback_mode": fallback_mode,
                "persistence": state.get("persistence"),
                "volatility_ratio_to_realized": state.get("volatility_ratio_to_realized"),
            }
            return self.forecast_results

        forecast_res = self.fitted_model.forecast(horizon=steps)
        variance = forecast_res.variance.iloc[-1]
        mean = forecast_res.mean.iloc[-1]

        # Rescale back if we scaled during fitting
        scale_factor = getattr(self, "_scale_factor", 1.0)
        if scale_factor != 1.0:
            # Variance scales as scale_factor^2, volatility and mean scale as scale_factor
            variance = variance / (scale_factor**2)
            mean = mean / scale_factor

        volatility = np.sqrt(variance)
        forecast_volatility_ratio = self._forecast_volatility_ratio_to_realized(volatility)
        self._volatility_ratio_to_realized = forecast_volatility_ratio
        if (
            forecast_volatility_ratio is not None
            and forecast_volatility_ratio > self.max_volatility_ratio_to_realized
            and isinstance(getattr(self, "_training_returns_clean", None), pd.Series)
            and not getattr(self, "_training_returns_clean").empty
        ):
            logger.warning(
                "GARCH forecast volatility ratio %.4f exceeds max %.4f; "
                "falling back to EWMA forecast.",
                forecast_volatility_ratio,
                self.max_volatility_ratio_to_realized,
            )
            self.backend = "ewma"
            self._fit_ewma(
                self._training_returns_clean,
                reason="exploding_variance_ratio",
                persistence=self._last_persistence or self._garch_persistence(),
                volatility_ratio_to_realized=forecast_volatility_ratio,
            )
            return self.forecast(steps)

        self.forecast_results = {
            "variance_forecast": variance,
            "mean_forecast": mean,
            "volatility": volatility,
            "steps": steps,
            "p": self.p,
            "q": self.q,
            "vol": self.vol,
            "dist": self.dist,
            "aic": float(self.fitted_model.aic),
            "bic": float(self.fitted_model.bic),
            # Phase 7.14-C: Propagate convergence status so _enrich_garch_forecast
            # can inflate CI width when the optimizer did not converge.
            "convergence_ok": bool(getattr(self, "_convergence_ok", True)),
            # Phase 8.2: residual diagnostics (Ljung-Box + Jarque-Bera)
            "residual_diagnostics": getattr(self, "_residual_diagnostics", {}),
            "residual_diagnostics_status": "available",
            "residual_diagnostics_reason": None,
            "ewma_lambda": None,
            "fallback_mode": self._current_fallback_mode(),
            "persistence": getattr(self, "_last_persistence", None) or self._garch_persistence(),
            "volatility_ratio_to_realized": forecast_volatility_ratio,
        }
        return self.forecast_results

    def get_model_summary(self) -> Dict[str, Any]:
        if self.fitted_model is None:
            return {}
        backend = getattr(self, "backend", "arch")
        if backend != "arch":
            state = self._fallback_state or {}
            return {
                "params": {"lambda": state.get("lambda"), "p": self.p, "q": self.q},
                "aic": None,
                "bic": None,
                "log_likelihood": None,
                "backend": backend,
                # Mirror the arch-path keys so callers have a consistent contract
                "vol_model": "ewma",
                "mean_model": getattr(self, "mean", "AR"),
                "dist": self.dist,
                "differenced": bool(getattr(self, "_differenced", False)),
                "ewma_lambda": state.get("lambda"),
                "residual_diagnostics_status": "unavailable",
                "residual_diagnostics_reason": "ewma_fallback",
                "fallback_mode": self._state_fallback_mode(state),
                "persistence": state.get("persistence"),
                "volatility_ratio_to_realized": state.get("volatility_ratio_to_realized"),
                "fit_sample_size": state.get("n_obs"),
            }
        return {
            "params": self.fitted_model.params.to_dict(),
            "aic": float(self.fitted_model.aic),
            "bic": float(self.fitted_model.bic),
            "log_likelihood": float(self.fitted_model.loglikelihood),
            "persistence": self._last_persistence or self._garch_persistence(),
            "backend": backend,
            "mean_model": getattr(self, "mean", "AR"),
            "dist": self.dist,
            "differenced": bool(getattr(self, "_differenced", False)),
            "ewma_lambda": None,
            "residual_diagnostics_status": "available",
            "residual_diagnostics_reason": None,
            "fallback_mode": self._current_fallback_mode(),
            "volatility_ratio_to_realized": self._volatility_ratio_to_realized,
            "fit_sample_size": self._fit_sample_size,
        }

    def load_fitted(self, snapshot: Any) -> "GARCHForecaster":
        """
        Restore fitted state from a ModelSnapshotStore snapshot (skip-refit path).

        `snapshot` may be:
        - An ARCHModelResult (arch backend) — stored directly as fitted_model
        - A dict with {"fitted_model": ..., "backend": ..., "scale_factor": ...}
        """
        if snapshot is None:
            return self
        try:
            if hasattr(snapshot, "aic") and hasattr(snapshot, "forecast"):
                # Raw ARCHModelResult
                self.fitted_model = snapshot
                self.backend = "arch"
                logger.debug("GARCHForecaster.load_fitted: loaded ARCHModelResult")
            elif isinstance(snapshot, dict) and "fitted_model" in snapshot:
                self.fitted_model = snapshot["fitted_model"]
                self.backend = snapshot.get("backend", self.backend)
                self._scale_factor = float(snapshot.get("scale_factor", 1.0))
                self._fallback_state = snapshot.get("fallback_state")
                if snapshot.get("p") is not None:
                    self.p = int(snapshot["p"])
                if snapshot.get("q") is not None:
                    self.q = int(snapshot["q"])
                if snapshot.get("vol") is not None:
                    self.vol = str(snapshot["vol"])
                if snapshot.get("dist") is not None:
                    self.dist = str(snapshot["dist"])
                if snapshot.get("mean") is not None:
                    self.mean = str(snapshot["mean"])
                logger.debug("GARCHForecaster.load_fitted: loaded from snapshot dict")
            else:
                logger.debug("GARCHForecaster.load_fitted: unrecognized snapshot format")
        except Exception as exc:
            logger.warning("GARCHForecaster.load_fitted failed: %s", exc)
        return self
