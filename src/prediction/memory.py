"""
KV-Cache-Aware Memory and VRAM trajectory predictor.

Models memory dynamics during autoregressive decoding, integrating:
- KV-cache growth rate (Delta K / Delta t)
- Token generation rate (r_T)
- Base memory slopes (s_V for VRAM, s_R for RAM)
- Time-to-threshold estimation for memory exhaustion warnings

STRICT CONSTRAINT: When VRAM metrics are unavailable, VRAM forecasts are marked
unavailable. CPU RAM is NEVER substituted for VRAM.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from src.prediction.base import (
    DEFAULT_TARGETS,
    Predictor,
    extract_target_series,
    generate_forecast_timestamps,
)
from src.prediction.constraints import apply_physical_constraints
from src.prediction.types import PredictionResult, TargetForecast
from src.state.types import RuntimeState


def estimate_time_to_threshold(
    current_val: float,
    slope: float,
    threshold: float,
    mode: str = "upper",
) -> Optional[float]:
    """
    Calculate estimated seconds until a metric crosses a safety threshold.

    Args:
        current_val: Current observed value at t0.
        slope: Rate of change per second (units / sec).
        threshold: Critical boundary value.
        mode: 'upper' (e.g. memory allocated exceeding threshold) or
              'lower' (e.g. free memory dropping below threshold).

    Returns:
        Seconds until crossing, 0.0 if already crossed, or None if slope never reaches threshold.
    """
    if mode == "upper":
        if current_val >= threshold:
            return 0.0
        if slope <= 0.0:
            return None
        return float((threshold - current_val) / slope)

    elif mode == "lower":
        if current_val <= threshold:
            return 0.0
        if slope >= 0.0:
            return None
        return float((threshold - current_val) / slope)

    else:
        raise ValueError(f"Unknown threshold mode: '{mode}'. Must be 'upper' or 'lower'.")


class KVCacheAwareMemoryPredictor(Predictor):
    """
    KV-Cache-Aware Memory Forecaster:
    Jointly projects memory expansion and KV-cache accumulation during autoregressive generation.
    """

    def __init__(
        self,
        targets: Optional[Sequence[str]] = None,
        minimum_history_length: int = 3,
        window_size: int = 10,
        vram_danger_threshold_mb: float = 256.0,
        ram_danger_threshold_mb: float = 512.0,
    ) -> None:
        default_targets = (
            "vram_free_mb",
            "vram_allocated_mb",
            "ram_used_mb",
            "ram_available_mb",
            "kv_cache_bytes",
            "bandwidth_mbps",
            "latency_ms",
        )
        super().__init__(
            targets=targets or default_targets,
            minimum_history_length=minimum_history_length,
        )
        self._window_size = window_size
        self._vram_threshold = vram_danger_threshold_mb
        self._ram_threshold = ram_danger_threshold_mb

    @property
    def name(self) -> str:
        return "kv_aware_memory"

    def predict(
        self,
        state_window: Sequence[RuntimeState],
        horizon_seconds: float = 2.0,
        step_interval_seconds: Optional[float] = None,
    ) -> PredictionResult:
        # First let the standard predict() framework handle extraction & bounds
        result = super().predict(
            state_window=state_window,
            horizon_seconds=horizon_seconds,
            step_interval_seconds=step_interval_seconds,
        )

        if not result.is_valid:
            return result

        # Compute additional domain diagnostics: slopes and time-to-threshold
        meta: Dict[str, Any] = dict(result.metadata)

        # 1. KV-cache slope
        _, kv_vals, kv_avail, _ = extract_target_series(state_window, "kv_cache_bytes")
        if kv_avail and len(kv_vals) >= 2:
            dt = state_window[-1].timestamp - state_window[0].timestamp
            if dt > 0 and kv_vals[-1] is not None and kv_vals[0] is not None:
                kv_slope = (kv_vals[-1] - kv_vals[0]) / dt
                meta["kv_cache_growth_rate_bps"] = float(kv_slope)

        # 2. VRAM time to threshold (only when VRAM is available)
        vram_target = result.get_target("vram_free_mb")
        if vram_target and vram_target.is_available and vram_target.values[0] is not None:
            v_curr = vram_target.values[0]
            v_slope = (vram_target.values[-1] - v_curr) / horizon_seconds if horizon_seconds > 0 else 0.0
            ttt_vram = estimate_time_to_threshold(
                current_val=v_curr,
                slope=v_slope,
                threshold=self._vram_threshold,
                mode="lower",
            )
            meta["time_to_vram_exhaustion_s"] = ttt_vram
        else:
            meta["time_to_vram_exhaustion_s"] = None

        # 3. RAM time to threshold
        ram_avail_target = result.get_target("ram_available_mb")
        if ram_avail_target and ram_avail_target.is_available and ram_avail_target.values[0] is not None:
            r_curr = ram_avail_target.values[0]
            r_slope = (ram_avail_target.values[-1] - r_curr) / horizon_seconds if horizon_seconds > 0 else 0.0
            ttt_ram = estimate_time_to_threshold(
                current_val=r_curr,
                slope=r_slope,
                threshold=self._ram_threshold,
                mode="lower",
            )
            meta["time_to_ram_exhaustion_s"] = ttt_ram

        return PredictionResult(
            prediction_timestamp=result.prediction_timestamp,
            horizon_seconds=result.horizon_seconds,
            step_interval_seconds=result.step_interval_seconds,
            forecast_timestamps=result.forecast_timestamps,
            targets=result.targets,
            predictor_name=self.name,
            input_window_length=result.input_window_length,
            is_valid=result.is_valid,
            status_message=result.status_message,
            prediction_latency_ms=result.prediction_latency_ms,
            metadata=meta,
        )

    def forecast_series(
        self,
        timestamps: Sequence[float],
        values: Sequence[Optional[float]],
        forecast_timestamps: Sequence[float],
        target_name: str,
    ) -> Tuple[List[Optional[float]], Optional[float]]:
        # Check availability
        valid_pairs = [(t, v) for t, v in zip(timestamps, values) if v is not None]
        if len(valid_pairs) < self.minimum_history_length:
            return [None] * len(forecast_timestamps), None

        if self._window_size is not None and len(valid_pairs) > self._window_size:
            valid_pairs = valid_pairs[-self._window_size:]

        ts = [p[0] for p in valid_pairs]
        ys = [p[1] for p in valid_pairs]
        n = len(ts)

        t_mean = sum(ts) / n
        y_mean = sum(ys) / n

        num = sum((t - t_mean) * (y - y_mean) for t, y in zip(ts, ys))
        den = sum((t - t_mean) ** 2 for t in ts)

        if den == 0.0:
            slope = 0.0
            intercept = ys[-1]
        else:
            slope = num / den
            intercept = y_mean - slope * t_mean

        preds: List[Optional[float]] = [float(intercept + slope * ft) for ft in forecast_timestamps]
        residuals = [abs(y - (intercept + slope * t)) for t, y in zip(ts, ys)]
        error_est = float(sum(residuals) / n) if n > 0 else 0.0

        return preds, error_est
