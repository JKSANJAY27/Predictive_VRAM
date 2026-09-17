"""
Baseline policy definitions for Module 10 experimental evaluation.

Defines the five baselines B1-B5 as configured variants of the existing
ClosedLoopRuntime (Module 9). No new algorithm is introduced — only the
information available to the controller and cost model differs.

Fairness contract (must be identical across baselines):
  - model, dtype, prompt, generation target, random seed
  - initial partition, candidate space, cost normalization
  - hardware/emulation, environmental traces
  - sampling interval, migration execution mode

Only these differ:
  - controller information (network / memory / both / neither / + prediction)
  - prediction availability
  - effective cost term enablement (via zero weights for ablated terms)
"""

from __future__ import annotations

import enum
import dataclasses
from typing import Any, Dict, Tuple

from src.controller.types import ControllerConfig, ControllerMode
from src.orchestration.config import OrchestrationConfig
from src.orchestration.types import ExecutionMode, OrchestrationMode
from src.cost.types import CostWeights


class BaselinePolicy(str, enum.Enum):
    """
    The five research baselines B1-B5.

    B1 STATIC            — fixed initial partition, no adaptation
    B2 NETWORK_REACTIVE  — reacts to current network conditions only
    B3 MEMORY_REACTIVE   — reacts to current memory/KV conditions only
    B4 JOINT_REACTIVE    — reacts to joint network + memory (no prediction)
    B5 PREDICTIVE        — full closed-loop: predict + score + switching cost
    """
    STATIC           = "static"
    NETWORK_REACTIVE = "network_reactive"
    MEMORY_REACTIVE  = "memory_reactive"
    JOINT_REACTIVE   = "joint_reactive"
    PREDICTIVE       = "predictive"


# ---------------------------------------------------------------------------
# Precise policy definitions — DO NOT CHANGE between experiments
# ---------------------------------------------------------------------------

# Default cost weights shared by all baselines (fairness contract).
# B2/B3 zero-out specific terms to disable those signals.
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "alpha":   0.35,   # latency
    "beta":    0.20,   # communication / network
    "gamma":   0.25,   # memory pressure
    "delta":   0.10,   # energy
    "epsilon": 0.10,   # switching penalty
}

_B2_WEIGHTS: Dict[str, float] = {        # network signal ONLY
    "alpha":   0.35,
    "beta":    0.50,   # emphasise network
    "gamma":   0.00,   # no memory signal
    "delta":   0.05,
    "epsilon": 0.10,
}

_B3_WEIGHTS: Dict[str, float] = {        # memory signal ONLY
    "alpha":   0.35,
    "beta":    0.00,   # no network signal
    "gamma":   0.55,   # emphasise memory
    "delta":   0.05,
    "epsilon": 0.05,
}


@dataclasses.dataclass
class PolicySpec:
    """Complete specification for one baseline policy."""
    baseline: BaselinePolicy
    description: str
    orchestration_mode: OrchestrationMode
    controller_mode: ControllerMode
    prediction_enabled: bool
    network_signal: bool    # whether network features influence decisions
    vram_signal: bool       # whether memory/VRAM features influence decisions
    cost_weights: Dict[str, float]
    # Additional controller knobs (None = use experiment-level defaults)
    hysteresis_cycles: int = 2
    cooldown_seconds: float = 5.0
    switch_threshold: float = 0.25
    minimum_dwell_seconds: float = 5.0


_POLICY_SPECS: Dict[BaselinePolicy, PolicySpec] = {
    BaselinePolicy.STATIC: PolicySpec(
        baseline=BaselinePolicy.STATIC,
        description=(
            "B1 STATIC: Fixed initial partition. Never performs ordinary adaptive "
            "switching. Reference baseline."
        ),
        orchestration_mode=OrchestrationMode.STATIC,
        controller_mode=ControllerMode.STATIC,
        prediction_enabled=False,
        network_signal=False,
        vram_signal=False,
        cost_weights=_DEFAULT_WEIGHTS,
        hysteresis_cycles=2,
        cooldown_seconds=5.0,
        switch_threshold=0.25,
        minimum_dwell_seconds=5.0,
    ),
    BaselinePolicy.NETWORK_REACTIVE: PolicySpec(
        baseline=BaselinePolicy.NETWORK_REACTIVE,
        description=(
            "B2 NETWORK_AWARE: Reacts to current network conditions only. "
            "No memory/VRAM influence. No future predictions."
        ),
        orchestration_mode=OrchestrationMode.REACTIVE,
        controller_mode=ControllerMode.REACTIVE,
        prediction_enabled=False,
        network_signal=True,
        vram_signal=False,
        cost_weights=_B2_WEIGHTS,
        hysteresis_cycles=2,
        cooldown_seconds=5.0,
        switch_threshold=0.25,
        minimum_dwell_seconds=5.0,
    ),
    BaselinePolicy.MEMORY_REACTIVE: PolicySpec(
        baseline=BaselinePolicy.MEMORY_REACTIVE,
        description=(
            "B3 MEMORY_AWARE: Reacts to current memory/KV-cache conditions only. "
            "No network influence. No future predictions."
        ),
        orchestration_mode=OrchestrationMode.REACTIVE,
        controller_mode=ControllerMode.REACTIVE,
        prediction_enabled=False,
        network_signal=False,
        vram_signal=True,
        cost_weights=_B3_WEIGHTS,
        hysteresis_cycles=2,
        cooldown_seconds=5.0,
        switch_threshold=0.25,
        minimum_dwell_seconds=5.0,
    ),
    BaselinePolicy.JOINT_REACTIVE: PolicySpec(
        baseline=BaselinePolicy.JOINT_REACTIVE,
        description=(
            "B4 REACTIVE_JOINT: Reacts to current network + memory + compute state. "
            "Forecast disabled. Full joint signal."
        ),
        orchestration_mode=OrchestrationMode.REACTIVE,
        controller_mode=ControllerMode.REACTIVE,
        prediction_enabled=False,
        network_signal=True,
        vram_signal=True,
        cost_weights=_DEFAULT_WEIGHTS,
        hysteresis_cycles=2,
        cooldown_seconds=5.0,
        switch_threshold=0.25,
        minimum_dwell_seconds=5.0,
    ),
    BaselinePolicy.PREDICTIVE: PolicySpec(
        baseline=BaselinePolicy.PREDICTIVE,
        description=(
            "B5 PREDICTIVE_COST_AWARE: Full proposed system. Uses short-horizon "
            "forecast + candidate scoring + switching cost + stability policy."
        ),
        orchestration_mode=OrchestrationMode.PREDICTIVE,
        controller_mode=ControllerMode.PREDICTIVE,
        prediction_enabled=True,
        network_signal=True,
        vram_signal=True,
        cost_weights=_DEFAULT_WEIGHTS,
        hysteresis_cycles=2,
        cooldown_seconds=5.0,
        switch_threshold=0.25,
        minimum_dwell_seconds=5.0,
    ),
}


class BaselinePolicyFactory:
    """
    Translates a BaselinePolicy into concrete configuration objects.

    Returns (OrchestrationConfig, ControllerConfig, cost_weights_dict) —
    ready to instantiate ClosedLoopRuntime via the existing factory helpers.
    """

    @staticmethod
    def get_spec(policy: BaselinePolicy) -> PolicySpec:
        """Return the immutable PolicySpec for *policy*."""
        return _POLICY_SPECS[policy]

    @staticmethod
    def build_configs(
        policy: BaselinePolicy,
        *,
        scenario_name: str = "stable",
        seed: int = 42,
        max_new_tokens: int = 32,
        control_interval_tokens: int = 4,
        predictor_type: str = "linear_trend",
        horizon: int = 4,
        execution_mode: ExecutionMode = ExecutionMode.SIMULATION,
        experiment_id: str | None = None,
        # Override knobs (for sensitivity sweeps / ablations)
        switch_threshold: float | None = None,
        cooldown_seconds: float | None = None,
        hysteresis_cycles: int | None = None,
        cost_weights_override: Dict[str, float] | None = None,
        slo_itl_ms: float = 200.0,
    ) -> Tuple[OrchestrationConfig, ControllerConfig, Dict[str, float]]:
        """
        Build (OrchestrationConfig, ControllerConfig, cost_weights) for *policy*.

        All experiment-level constants are kept identical across policies;
        only the policy-specific knobs differ.
        """
        spec = _POLICY_SPECS[policy]

        # Apply overrides (for sensitivity / ablation sweeps)
        threshold  = switch_threshold  if switch_threshold  is not None else spec.switch_threshold
        cooldown   = cooldown_seconds  if cooldown_seconds  is not None else spec.cooldown_seconds
        hysteresis = hysteresis_cycles if hysteresis_cycles is not None else spec.hysteresis_cycles
        weights    = cost_weights_override if cost_weights_override is not None else spec.cost_weights

        orch_cfg = OrchestrationConfig(
            mode=spec.orchestration_mode,
            execution_mode=execution_mode,
            control_interval_tokens=control_interval_tokens,
            horizon=horizon,
            predictor_type=predictor_type if spec.prediction_enabled else "last_value",
            max_generated_tokens=max_new_tokens,
            random_seed=seed,
            scenario_name=scenario_name,
            experiment_id=experiment_id,
            cost_weights=dict(weights),
            slo_itl_ms=slo_itl_ms,
        )

        ctrl_cfg = ControllerConfig(
            mode=spec.controller_mode,
            switch_threshold=threshold,
            minimum_dwell_seconds=spec.minimum_dwell_seconds,
            cooldown_seconds=cooldown,
            hysteresis_cycles=hysteresis,
            emergency_override=True,
            allow_reactive_fallback=True,
        )

        return orch_cfg, ctrl_cfg, dict(weights)

    @staticmethod
    def all_policies() -> list[BaselinePolicy]:
        """Return all five baselines in canonical comparison order."""
        return [
            BaselinePolicy.STATIC,
            BaselinePolicy.NETWORK_REACTIVE,
            BaselinePolicy.MEMORY_REACTIVE,
            BaselinePolicy.JOINT_REACTIVE,
            BaselinePolicy.PREDICTIVE,
        ]

    @staticmethod
    def describe_all() -> Dict[str, str]:
        """Return {policy_id: description} for documentation."""
        return {p.value: _POLICY_SPECS[p].description for p in _POLICY_SPECS}
