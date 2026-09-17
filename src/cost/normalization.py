"""
Normalization layer for multi-objective cost evaluation.

Maps heterogeneous physical units (milliseconds, megabytes, pressure ratios, joules, disruption units)
into comparable, unitless cost scales using deterministic, configured reference denominators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class NormalizationConfig:
    """
    Scale reference denominators for normalizing raw cost metrics.

    All candidates within a single optimization step share the EXACT same
    normalization scales to ensure fair, mathematically sound comparability.
    """
    latency_scale_ms: float = 100.0
    communication_scale_mb: float = 10.0
    memory_pressure_scale: float = 1.0
    energy_scale_j: float = 10.0
    switching_scale: float = 1.0

    def __post_init__(self) -> None:
        for name, val in [
            ("latency_scale_ms", self.latency_scale_ms),
            ("communication_scale_mb", self.communication_scale_mb),
            ("memory_pressure_scale", self.memory_pressure_scale),
            ("energy_scale_j", self.energy_scale_j),
            ("switching_scale", self.switching_scale),
        ]:
            if val <= 0.0:
                raise ValueError(f"Normalization scale '{name}' must be positive: {val}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latency_scale_ms": self.latency_scale_ms,
            "communication_scale_mb": self.communication_scale_mb,
            "memory_pressure_scale": self.memory_pressure_scale,
            "energy_scale_j": self.energy_scale_j,
            "switching_scale": self.switching_scale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NormalizationConfig:
        return cls(
            latency_scale_ms=float(data.get("latency_scale_ms", 100.0)),
            communication_scale_mb=float(data.get("communication_scale_mb", 10.0)),
            memory_pressure_scale=float(data.get("memory_pressure_scale", 1.0)),
            energy_scale_j=float(data.get("energy_scale_j", 10.0)),
            switching_scale=float(data.get("switching_scale", 1.0)),
        )


class CostNormalizer:
    """
    Deterministic normalizer mapping raw metrics into non-negative unitless costs.
    """

    def __init__(self, config: Optional[NormalizationConfig] = None) -> None:
        self.config = config or NormalizationConfig()

    def _validate_and_scale(self, value: float, scale: float, metric_name: str) -> float:
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"Cannot normalize non-finite value {value} for {metric_name}")
        if value < 0.0:
            raise ValueError(f"Cannot normalize negative value {value} for {metric_name}")
        return float(value / scale)

    def normalize_latency(self, latency_ms: float) -> float:
        return self._validate_and_scale(latency_ms, self.config.latency_scale_ms, "latency_ms")

    def normalize_communication(self, comm_mb: float) -> float:
        return self._validate_and_scale(comm_mb, self.config.communication_scale_mb, "communication_mb")

    def normalize_memory_pressure(self, pressure_ratio: float) -> float:
        return self._validate_and_scale(pressure_ratio, self.config.memory_pressure_scale, "memory_pressure")

    def normalize_energy(self, energy_j: float) -> float:
        return self._validate_and_scale(energy_j, self.config.energy_scale_j, "energy_j")

    def normalize_switching(self, switching_units: float) -> float:
        return self._validate_and_scale(switching_units, self.config.switching_scale, "switching_units")
