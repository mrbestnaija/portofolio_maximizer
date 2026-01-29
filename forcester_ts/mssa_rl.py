"""
Hybrid Multi-Singular Spectrum Analysis (mSSA) with reinforcement learning
style change-point detection.

The implementation is intentionally lightweight: it focuses on producing
diagnostics that can replace LLM-driven analytics in monitoring dashboards.
"""

from __future__ import annotations

import logging
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import svd
from pandas.tseries.frequencies import to_offset

cp = None
_CUPY_AVAILABLE = False
_CUPY_TRIED = False


def _preload_cuda_libraries() -> None:
    """Preload CUDA shared libraries shipped via pip (best-effort).

    In WSL, CuPy may import successfully but later fail to dlopen CUDA libs
    (e.g. `libnvrtc.so.12`) because the dynamic loader cannot see the packaged
    `site-packages/nvidia/**/lib/` directories. Preloading by absolute path via
    `ctypes.CDLL(..., RTLD_GLOBAL)` makes them available for CuPy's subsequent
    loads.
    """
    if not sys.platform.startswith("linux"):
        return

    try:  # pragma: no cover - optional platform behaviour
        import ctypes
    except Exception:
        return

    rtld_global = getattr(ctypes, "RTLD_GLOBAL", None)
    if rtld_global is None:
        return

    # Load order matters: nvrtc -> cudart -> cusolver.
    candidates = (
        ("nvidia.cuda_nvrtc", "libnvrtc.so.12"),
        ("nvidia.cuda_runtime", "libcudart.so.12"),
        ("nvidia.cusolver", "libcusolver.so.11"),
    )

    for module_name, library_name in candidates:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                continue

            base_path = None
            if spec.submodule_search_locations:
                base_path = Path(list(spec.submodule_search_locations)[0])
            elif spec.origin:
                base_path = Path(spec.origin).parent

            if base_path is None:
                continue

            lib_path = base_path / "lib" / library_name
            if lib_path.exists():
                ctypes.CDLL(str(lib_path), mode=rtld_global)
        except Exception:
            continue


def _load_cupy() -> bool:
    """Best-effort CuPy import (optional); caches the result for the process."""
    global cp, _CUPY_AVAILABLE, _CUPY_TRIED
    if _CUPY_TRIED:
        return _CUPY_AVAILABLE
    _CUPY_TRIED = True
    try:  # Optional GPU acceleration via CuPy
        _preload_cuda_libraries()

        import cupy as _cp  # type: ignore

        try:
            if int(_cp.cuda.runtime.getDeviceCount()) < 1:
                raise RuntimeError("No CUDA devices detected by CuPy")

            # Minimal self-test to ensure linalg backends can be loaded.
            test = _cp.eye(2, dtype=_cp.float32)
            _cp.linalg.svd(test, full_matrices=False)
        except Exception as exc:
            cp = None
            _CUPY_AVAILABLE = False
            logger.debug("MSSARL: CuPy unavailable; falling back to CPU (%s).", exc)
            return _CUPY_AVAILABLE

        cp = _cp
        _CUPY_AVAILABLE = True
    except Exception:  # pragma: no cover - optional dependency
        cp = None
        _CUPY_AVAILABLE = False
    return _CUPY_AVAILABLE

logger = logging.getLogger(__name__)


@dataclass
class MSSARLConfig:
    window_length: int = 30
    rank: Optional[int] = None
    change_point_threshold: float = 2.5
    q_learning_alpha: float = 0.3
    q_learning_gamma: float = 0.85
    q_learning_epsilon: float = 0.1
    forecast_horizon: int = 10
    use_gpu: bool = False
    # PHASE 7.3 FIX: Accept additional params from YAML config
    min_series_length: int = 150
    max_forecast_steps: int = 30


class MSSARLForecaster:
    """
    Applies mSSA decomposition, simple CUSUM change-point detection, and a
    tabular Q-learning loop that rewards variance reduction. The reward signal
    is deliberately simple but provides a deterministic alternative to LLM
    reasoning for regime detection.
    """

    def __init__(self, config: Optional[MSSARLConfig] = None) -> None:
        self.config = config or MSSARLConfig()
        self._fitted = False
        self._q_table: Dict[Tuple[int, int], float] = {}
        self._baseline_variance: float = 0.0
        self._last_index: Optional[pd.Timestamp] = None
        self._change_points: Optional[pd.DatetimeIndex] = None
        self._reconstruction: Optional[pd.Series] = None
        self._change_point_density: float = 0.0
        self._recent_change_point_days: Optional[int] = None
        self._freq_hint: Optional[str] = None
        self._last_observed_value: Optional[float] = None
        self._last_reconstruction_error: Optional[float] = None
        self._use_gpu = bool(self.config.use_gpu and _load_cupy())
        if self.config.use_gpu and not self._use_gpu:
            # CuPy is optional; when unavailable we fall back to CPU silently.
            logger.debug("MSSARL: CuPy unavailable; falling back to CPU.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _construct_page_matrix(self, series: pd.Series) -> np.ndarray:
        L = self.config.window_length
        T = len(series)
        K = T - L + 1
        if K <= 0:
            raise ValueError("Window length longer than the series length")

        return np.column_stack([series.values[i : i + L] for i in range(K)])

    def _truncate_svd(self, trajectory: np.ndarray) -> np.ndarray:
        if self._use_gpu:
            xp = cp
            traj = cp.asarray(trajectory)
            U, s, Vt = xp.linalg.svd(traj, full_matrices=False)
            cumulative = cp.asnumpy(cp.cumsum(s) / cp.sum(s))
        else:
            xp = np
            U, s, Vt = svd(trajectory, full_matrices=False)
            cumulative = np.cumsum(s) / np.sum(s)

        rank = self.config.rank
        if rank is None:
            rank = int(np.searchsorted(cumulative, 0.9)) + 1
            rank = max(1, min(rank, trajectory.shape[0]))
            self.config.rank = rank

        U_r = U[:, :rank]
        s_r = s[:rank]
        Vt_r = Vt[:rank, :]

        recon = U_r @ xp.diag(s_r) @ Vt_r
        if self._use_gpu:
            recon = cp.asnumpy(recon)
        return recon

    def _cusum_change_points(self, residuals: pd.Series) -> pd.DatetimeIndex:
        cleaned = residuals.dropna()
        if cleaned.empty:
            return pd.DatetimeIndex([])

        std = float(cleaned.std(ddof=0))
        if std <= 0 or not np.isfinite(std):
            return pd.DatetimeIndex([])

        # Guardrail: for ultra-low residual volatility, standardization can make
        # small noise look like frequent regime shifts. Floor the scale so
        # stable series do not over-trigger change points.
        std = max(std, 1.0)

        # Standardized one-sided CUSUM on residual mean shifts.
        # Reference: Page (1954) CUSUM tests for parameter changes.
        threshold = float(self.config.change_point_threshold)
        centered = (cleaned - float(cleaned.mean())) / (std + 1e-12)

        pos_sum = 0.0
        neg_sum = 0.0
        change_points = []
        for ts, value in zip(centered.index, centered.to_numpy()):
            pos_sum = max(0.0, pos_sum + float(value))
            neg_sum = min(0.0, neg_sum + float(value))
            if pos_sum > threshold or neg_sum < -threshold:
                change_points.append(ts)
                pos_sum = 0.0
                neg_sum = 0.0

        return pd.DatetimeIndex(change_points)

    def _update_q_table(self, variance_ratio: float, state: int, action: int) -> None:
        key = (state, action)
        current_q = self._q_table.get(key, 0.0)
        reward = 1.0 - variance_ratio

        best_future = max(
            (self._q_table.get((action, next_action), 0.0) for next_action in range(3)),
            default=0.0,
        )

        new_q = current_q + self.config.q_learning_alpha * (
            reward + self.config.q_learning_gamma * best_future - current_q
        )
        self._q_table[key] = new_q

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, series: pd.Series) -> "MSSARLForecaster":
        cleaned = series.dropna()
        if len(cleaned) < self.config.window_length + 5:
            raise ValueError("Series length insufficient for mSSA analysis")

        try:
            self._last_observed_value = float(cleaned.iloc[-1])
        except Exception:
            self._last_observed_value = None

        freq_hint = None
        try:
            freq_hint = series.attrs.get("_pm_freq_hint")
        except Exception:
            freq_hint = None
        self._freq_hint = cleaned.index.freqstr or cleaned.index.inferred_freq or freq_hint

        self._last_index = cleaned.index[-1]
        trajectory = self._construct_page_matrix(cleaned)
        recon_matrix = self._truncate_svd(trajectory)

        recon = np.zeros(len(cleaned))
        counts = np.zeros(len(cleaned))
        L, K = recon_matrix.shape
        for i in range(L):
            for j in range(K):
                recon[i + j] += recon_matrix[i, j]
                counts[i + j] += 1
        recon /= counts

        self._reconstruction = pd.Series(recon, index=cleaned.index)
        residuals = cleaned - self._reconstruction
        self._baseline_variance = float(residuals.var(ddof=1))
        try:
            self._last_reconstruction_error = float(abs(residuals.iloc[-1]))
        except Exception:
            self._last_reconstruction_error = None

        change_points = self._cusum_change_points(residuals)
        self._change_points = change_points
        self._change_point_density = len(change_points) / max(len(cleaned), 1)
        if len(change_points) > 0:
            last_cp = pd.to_datetime(change_points[-1])
            self._recent_change_point_days = abs(
                int((cleaned.index[-1] - last_cp).days)
            )
        else:
            self._recent_change_point_days = None

        # Tabular Q-learning over pseudo states (variance buckets)
        variance_ratio = (
            residuals.rolling(window=self.config.window_length // 2, min_periods=5)
            .var()
            .fillna(self._baseline_variance)
            / self._baseline_variance
        )
        state_series = np.digitize(variance_ratio, bins=[0.8, 1.0, 1.2])

        for state, ratio in zip(state_series, variance_ratio):
            action = min(2, state)  # simple policy: act proportionally
            self._update_q_table(float(ratio), int(state), action)

        logger.info(
            "MSSARL fit complete (window=%s, rank=%s, change_points=%s)",
            self.config.window_length,
            self.config.rank,
            len(change_points),
        )
        self._fitted = True
        return self

    def forecast(self, steps: int) -> Dict[str, Any]:
        if not self._fitted or self._reconstruction is None:
            raise ValueError("Model must be fitted before forecasting")

        last_recon = float(self._reconstruction.iloc[-1])
        last_obs = self._last_observed_value
        base_value = last_obs if last_obs is not None else last_recon

        if (
            self._last_reconstruction_error is not None
            and self._baseline_variance is not None
            and np.isfinite(self._baseline_variance)
            and self._change_point_density < 0.005
        ):
            scale = float(np.sqrt(max(self._baseline_variance, 0.0)))
            if scale > 0 and self._last_reconstruction_error <= 1.5 * scale:
                base_value = 0.85 * base_value + 0.15 * last_recon

        baseline_forecast = np.full(steps, base_value)

        if self._change_points is not None and len(self._change_points) > 0:
            apply_decay = False
            if self._recent_change_point_days is not None:
                if self._recent_change_point_days <= max(1, self.config.window_length // 4):
                    apply_decay = True
            if self._change_point_density < 0.1:
                apply_decay = False
            if apply_decay:
                decay = np.linspace(0.998, 0.99, num=steps)
                baseline_forecast = baseline_forecast * decay

        freq = (
            self._freq_hint
            or getattr(self._reconstruction.index, "freqstr", None)
            or getattr(self._reconstruction.index, "inferred_freq", None)
            or "D"
        )
        if self._last_index is not None:
            try:
                offset = to_offset(freq)
                start = self._last_index + offset
                future_index = pd.date_range(start=start, periods=steps, freq=freq)
            except Exception:
                future_index = pd.RangeIndex(start=0, stop=steps, step=1)
        else:
            future_index = pd.RangeIndex(start=0, stop=steps, step=1)

        forecast_series = pd.Series(baseline_forecast, index=future_index)
        noise = np.sqrt(self._baseline_variance)

        return {
            "forecast": forecast_series,
            "lower_ci": forecast_series - noise,
            "upper_ci": forecast_series + noise,
            "change_points": self._change_points,
            "q_table_size": len(self._q_table),
            "baseline_variance": self._baseline_variance,
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "window_length": self.config.window_length,
            "rank": self.config.rank,
            "change_points": (
                self._change_points.to_pydatetime().tolist()
                if self._change_points is not None
                else []
            ),
            "q_table": dict(self._q_table),
            "baseline_variance": self._baseline_variance,
            "change_point_density": self._change_point_density,
            "recent_change_point_days": self._recent_change_point_days,
            "use_gpu": self._use_gpu,
        }
