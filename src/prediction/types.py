"""
Data structures and result representations for the predictive forecasting engine.

Defines TargetForecast and PredictionResult with strict provenance preservation,
availability flags, physical constraint tracking, and error estimation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class TargetForecast:
    """
    Forecast trajectory for a single scalar system variable over future timestamps.

    Attributes:
        target_name: Variable name (e.g., 'bandwidth_mbps', 'vram_free_mb').
        values: Predicted values with physical constraints applied (None if unavailable).
        raw_values: Unconstrained predicted values before bounding/clipping.
        timestamps: Future timestamps corresponding to predicted steps.
        is_available: False if hardware metric is unavailable on the host.
        status: Status string ('ok', 'unavailable', 'insufficient_history').
        error_estimate: Expected error / residual magnitude (e.g. recent MAE).
        provenance: Provenance tag (e.g., 'estimated', 'unavailable', 'emulated').
        constraint_applied: True if any value required physical bounding/clipping.
    """
    target_name: str
    values: List[Optional[float]]
    raw_values: List[Optional[float]]
    timestamps: List[float]
    is_available: bool
    status: str
    error_estimate: Optional[float] = None
    provenance: str = "estimated"
    constraint_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_name": self.target_name,
            "values": self.values,
            "raw_values": self.raw_values,
            "timestamps": self.timestamps,
            "is_available": self.is_available,
            "status": self.status,
            "error_estimate": self.error_estimate,
            "provenance": self.provenance,
            "constraint_applied": self.constraint_applied,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TargetForecast:
        return cls(
            target_name=data["target_name"],
            values=data["values"],
            raw_values=data["raw_values"],
            timestamps=data["timestamps"],
            is_available=data["is_available"],
            status=data["status"],
            error_estimate=data.get("error_estimate"),
            provenance=data.get("provenance", "estimated"),
            constraint_applied=data.get("constraint_applied", False),
        )

    @classmethod
    def unavailable(
        cls,
        target_name: str,
        timestamps: Sequence[float],
        reason: str = "Hardware/metric unavailable",
    ) -> TargetForecast:
        """Convenience constructor for unmeasurable targets (e.g., VRAM on CPU-only machines)."""
        ts_list = list(timestamps)
        none_list: List[Optional[float]] = [None] * len(ts_list)
        return cls(
            target_name=target_name,
            values=none_list,
            raw_values=none_list,
            timestamps=ts_list,
            is_available=False,
            status="unavailable",
            error_estimate=None,
            provenance="unavailable",
            constraint_applied=False,
        )

    @classmethod
    def insufficient_history(
        cls,
        target_name: str,
        timestamps: Sequence[float],
        required_length: int,
        actual_length: int,
    ) -> TargetForecast:
        """Convenience constructor when historical observations are below minimum requirement."""
        ts_list = list(timestamps)
        none_list: List[Optional[float]] = [None] * len(ts_list)
        return cls(
            target_name=target_name,
            values=none_list,
            raw_values=none_list,
            timestamps=ts_list,
            is_available=True,
            status=f"insufficient_history (needed {required_length}, got {actual_length})",
            error_estimate=None,
            provenance="estimated",
            constraint_applied=False,
        )


@dataclass(frozen=True)
class PredictionResult:
    """
    Composite forecast result produced by a Predictor.

    Contains individual target forecasts, validation flags, execution latency,
    and metadata.
    """
    prediction_timestamp: float
    horizon_seconds: float
    step_interval_seconds: float
    forecast_timestamps: List[float]
    targets: Dict[str, TargetForecast]
    predictor_name: str
    input_window_length: int
    is_valid: bool
    status_message: str
    prediction_latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_target(self, name: str) -> Optional[TargetForecast]:
        """Retrieve forecast for a specific target variable."""
        return self.targets.get(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_timestamp": self.prediction_timestamp,
            "horizon_seconds": self.horizon_seconds,
            "step_interval_seconds": self.step_interval_seconds,
            "forecast_timestamps": self.forecast_timestamps,
            "targets": {k: v.to_dict() for k, v in self.targets.items()},
            "predictor_name": self.predictor_name,
            "input_window_length": self.input_window_length,
            "is_valid": self.is_valid,
            "status_message": self.status_message,
            "prediction_latency_ms": self.prediction_latency_ms,
            "metadata": self.metadata,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PredictionResult:
        targets_dict = {
            k: TargetForecast.from_dict(v) for k, v in data["targets"].items()
        }
        return cls(
            prediction_timestamp=data["prediction_timestamp"],
            horizon_seconds=data["horizon_seconds"],
            step_interval_seconds=data["step_interval_seconds"],
            forecast_timestamps=data["forecast_timestamps"],
            targets=targets_dict,
            predictor_name=data["predictor_name"],
            input_window_length=data["input_window_length"],
            is_valid=data["is_valid"],
            status_message=data["status_message"],
            prediction_latency_ms=data.get("prediction_latency_ms", 0.0),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> PredictionResult:
        return cls.from_dict(json.loads(json_str))
