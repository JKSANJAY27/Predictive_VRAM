"""
Abstract base class and shared extraction utilities for time-series predictors.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.prediction.constraints import apply_physical_constraints
from src.prediction.types import PredictionResult, TargetForecast
from src.state.types import RuntimeState
from src.telemetry.types import DataSource, TaggedValue

# Canonical mapping from target identifier to RuntimeState accessor
TARGET_ACCESSORS = {
    "bandwidth": lambda s: s.network.bandwidth_mbps,
    "bandwidth_mbps": lambda s: s.network.bandwidth_mbps,
    "latency": lambda s: s.network.latency_ms,
    "latency_ms": lambda s: s.network.latency_ms,
    "packet_loss": lambda s: s.network.packet_loss,
    "jitter": lambda s: s.network.jitter_ms,
    "jitter_ms": lambda s: s.network.jitter_ms,
    "vram_allocated": lambda s: s.memory.vram_allocated_mb,
    "vram_allocated_mb": lambda s: s.memory.vram_allocated_mb,
    "vram_free": lambda s: s.memory.vram_free_mb,
    "vram_free_mb": lambda s: s.memory.vram_free_mb,
    "ram_used": lambda s: s.memory.ram_used_mb,
    "ram_used_mb": lambda s: s.memory.ram_used_mb,
    "ram_available": lambda s: s.memory.ram_available_mb,
    "ram_available_mb": lambda s: s.memory.ram_available_mb,
    "cpu_utilization": lambda s: s.compute.cpu_utilization,
    "gpu_utilization": lambda s: s.compute.gpu_utilization,
    "kv_cache_bytes": lambda s: s.inference.kv_cache_bytes,
    "kv_cache_growth_bytes": lambda s: s.inference.kv_cache_growth_bytes,
}

DEFAULT_TARGETS: Tuple[str, ...] = (
    "bandwidth_mbps",
    "latency_ms",
    "vram_free_mb",
    "kv_cache_bytes",
)


def extract_target_series(
    states: Sequence[RuntimeState],
    target_name: str,
) -> Tuple[List[float], List[Optional[float]], bool, str]:
    """
    Extract timestamps, values, availability flag, and provenance for a given target.

    Returns:
        (timestamps, values, is_available, primary_provenance)
    """
    accessor = TARGET_ACCESSORS.get(target_name)
    if accessor is None:
        raise ValueError(f"Unknown target variable: '{target_name}'")

    timestamps: List[float] = []
    values: List[Optional[float]] = []
    available_count = 0
    sources: Dict[str, int] = {}

    for s in states:
        timestamps.append(s.timestamp)
        tv: TaggedValue = accessor(s)
        src = tv.source.value
        sources[src] = sources.get(src, 0) + 1

        if tv.source == DataSource.UNAVAILABLE or tv.value is None:
            values.append(None)
        else:
            values.append(float(tv.value))
            available_count += 1

    is_available = available_count > 0
    # Determine dominant provenance
    dominant_src = max(sources.items(), key=lambda x: x[1])[0] if sources else "unavailable"

    return timestamps, values, is_available, dominant_src


def generate_forecast_timestamps(
    t0: float,
    horizon_seconds: float,
    step_interval_seconds: Optional[float] = None,
) -> Tuple[List[float], float]:
    """
    Generate discrete future timestamps from t0 across horizon_seconds.

    Returns:
        (forecast_timestamps, step_interval_seconds)
    """
    if horizon_seconds <= 0:
        raise ValueError(f"horizon_seconds must be > 0, got {horizon_seconds}")

    dt = step_interval_seconds if step_interval_seconds is not None else min(0.5, horizon_seconds / 5.0)
    if dt <= 0:
        dt = 0.1

    num_steps = max(1, int(round(horizon_seconds / dt)))
    actual_dt = float(horizon_seconds / num_steps)

    timestamps = [t0 + (i + 1) * actual_dt for i in range(num_steps)]
    return timestamps, actual_dt


class Predictor(ABC):
    """
    Abstract base interface for all forecasting engines in the system.
    """

    def __init__(
        self,
        targets: Optional[Sequence[str]] = None,
        minimum_history_length: int = 3,
    ) -> None:
        self._targets = list(targets) if targets is not None else list(DEFAULT_TARGETS)
        self._min_history = max(1, minimum_history_length)

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable predictor identifier."""
        raise NotImplementedError

    @property
    def supported_targets(self) -> List[str]:
        return list(self._targets)

    @property
    def minimum_history_length(self) -> int:
        return self._min_history

    def predict(
        self,
        state_window: Sequence[RuntimeState],
        horizon_seconds: float = 2.0,
        step_interval_seconds: Optional[float] = None,
    ) -> PredictionResult:
        """
        Produce a PredictionResult for the requested horizon over all configured targets.
        """
        start_mono = time.monotonic()

        if not state_window:
            return PredictionResult(
                prediction_timestamp=0.0,
                horizon_seconds=horizon_seconds,
                step_interval_seconds=step_interval_seconds or 0.1,
                forecast_timestamps=[],
                targets={},
                predictor_name=self.name,
                input_window_length=0,
                is_valid=False,
                status_message="Empty state window provided",
                prediction_latency_ms=0.0,
            )

        t0 = state_window[-1].timestamp
        forecast_ts, dt = generate_forecast_timestamps(
            t0=t0,
            horizon_seconds=horizon_seconds,
            step_interval_seconds=step_interval_seconds,
        )

        n_obs = len(state_window)
        if n_obs < self._min_history:
            # Insufficient history: do not fabricate predictions
            target_forecasts: Dict[str, TargetForecast] = {}
            for tgt in self._targets:
                _, _, is_avail, prov = extract_target_series(state_window, tgt)
                if not is_avail:
                    target_forecasts[tgt] = TargetForecast.unavailable(tgt, forecast_ts)
                else:
                    target_forecasts[tgt] = TargetForecast.insufficient_history(
                        target_name=tgt,
                        timestamps=forecast_ts,
                        required_length=self._min_history,
                        actual_length=n_obs,
                    )

            lat_ms = (time.monotonic() - start_mono) * 1000.0
            return PredictionResult(
                prediction_timestamp=t0,
                horizon_seconds=horizon_seconds,
                step_interval_seconds=dt,
                forecast_timestamps=forecast_ts,
                targets=target_forecasts,
                predictor_name=self.name,
                input_window_length=n_obs,
                is_valid=False,
                status_message=f"insufficient_history: required {self._min_history}, got {n_obs}",
                prediction_latency_ms=lat_ms,
            )

        # Sufficient history: delegate to forecast_target for each variable
        target_forecasts = {}
        for tgt in self._targets:
            timestamps, values, is_avail, provenance = extract_target_series(state_window, tgt)

            if not is_avail:
                target_forecasts[tgt] = TargetForecast.unavailable(tgt, forecast_ts)
                continue

            # Compute raw forecast and error estimate
            raw_future, error_est = self.forecast_series(
                timestamps=timestamps,
                values=values,
                forecast_timestamps=forecast_ts,
                target_name=tgt,
            )

            # Apply physical bounds
            constrained_future, constraint_applied = apply_physical_constraints(
                target_name=tgt,
                raw_values=raw_future,
            )

            target_forecasts[tgt] = TargetForecast(
                target_name=tgt,
                values=constrained_future,
                raw_values=raw_future,
                timestamps=forecast_ts,
                is_available=True,
                status="ok",
                error_estimate=error_est,
                provenance=provenance,
                constraint_applied=constraint_applied,
            )

        lat_ms = (time.monotonic() - start_mono) * 1000.0
        return PredictionResult(
            prediction_timestamp=t0,
            horizon_seconds=horizon_seconds,
            step_interval_seconds=dt,
            forecast_timestamps=forecast_ts,
            targets=target_forecasts,
            predictor_name=self.name,
            input_window_length=n_obs,
            is_valid=True,
            status_message="ok",
            prediction_latency_ms=lat_ms,
        )

    @abstractmethod
    def forecast_series(
        self,
        timestamps: Sequence[float],
        values: Sequence[Optional[float]],
        forecast_timestamps: Sequence[float],
        target_name: str,
    ) -> Tuple[List[Optional[float]], Optional[float]]:
        """
        Forecast values for a single target series.

        Returns:
            (raw_predicted_values, error_estimate)
        """
        raise NotImplementedError
