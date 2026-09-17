"""
Configuration schemas for Closed-Loop Predictive Runtime Integration (Module 9).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.orchestration.types import ExecutionMode, OrchestrationMode


@dataclass
class OrchestrationConfig:
    """
    Complete configuration for an integrated closed-loop runtime experiment.
    """
    # Policy mode (STATIC, REACTIVE, PREDICTIVE)
    mode: OrchestrationMode = OrchestrationMode.PREDICTIVE

    # Execution regime (SIMULATION, EXECUTION, REPLAY)
    execution_mode: ExecutionMode = ExecutionMode.EXECUTION

    # Synchronization interval
    control_interval_tokens: int = 4
    control_interval_seconds: Optional[float] = None

    # Prediction configuration
    horizon: int = 4
    predictor_type: str = "linear_trend"

    # Termination bounds
    max_generated_tokens: int = 32
    max_runtime_seconds: float = 60.0
    stop_on_failure: bool = False

    # Reproducibility & Identification
    random_seed: int = 42
    scenario_name: str = "stable"
    experiment_id: Optional[str] = None
    model_name: str = "synthetic-gpt2"
    device: str = "cpu"

    # Migration & Verification settings
    migration_verification_mode: str = "FAST"
    rollback_enabled: bool = True
    emulated_bandwidth_delay: bool = False

    # SLO targets
    slo_itl_ms: float = 200.0

    # Cost weights (passed to CostModel)
    cost_weights: Dict[str, float] = field(default_factory=lambda: {
        "alpha": 0.35,   # latency
        "beta": 0.20,    # communication
        "gamma": 0.25,   # memory
        "delta": 0.10,   # energy
        "epsilon": 0.10, # switching penalty
    })

    # Metadata snapshot
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.control_interval_tokens < 1:
            raise ValueError(f"control_interval_tokens must be >= 1, got {self.control_interval_tokens}")
        if self.max_generated_tokens < 1:
            raise ValueError(f"max_generated_tokens must be >= 1, got {self.max_generated_tokens}")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.experiment_id is None:
            ts = int(time.time())
            self.experiment_id = f"exp_{self.mode.value}_{self.scenario_name}_{ts}_{self.random_seed}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "execution_mode": self.execution_mode.value,
            "control_interval_tokens": self.control_interval_tokens,
            "control_interval_seconds": self.control_interval_seconds,
            "horizon": self.horizon,
            "predictor_type": self.predictor_type,
            "max_generated_tokens": self.max_generated_tokens,
            "max_runtime_seconds": self.max_runtime_seconds,
            "stop_on_failure": self.stop_on_failure,
            "random_seed": self.random_seed,
            "scenario_name": self.scenario_name,
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "device": self.device,
            "migration_verification_mode": self.migration_verification_mode,
            "rollback_enabled": self.rollback_enabled,
            "emulated_bandwidth_delay": self.emulated_bandwidth_delay,
            "slo_itl_ms": self.slo_itl_ms,
            "cost_weights": dict(self.cost_weights),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OrchestrationConfig:
        data = dict(data)
        if "mode" in data:
            data["mode"] = OrchestrationMode(data["mode"])
        if "execution_mode" in data:
            data["execution_mode"] = ExecutionMode(data["execution_mode"])
        return cls(**data)
