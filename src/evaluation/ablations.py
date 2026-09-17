"""
Ablation suite for Module 10.

AblationSpec    — defines which single component changes and the exact delta
AblationSuite   — factory producing all required ablations (A1-A8)

Ablation fairness invariant:
  Each ablation changes EXACTLY ONE component versus the full PREDICTIVE system.
  Scenario, model, workload, candidate space, migration config, and seed are
  held constant.  Violations should be caught by tests.

Mandatory (A1-A4):
  A1  prediction_removed       — predictor disabled; PREDICTIVE → REACTIVE
  A2  switching_penalty_removed — epsilon = 0.0 in cost weights
  A3  network_signal_removed   — network cost term zeroed (beta = 0)
  A4  vram_signal_removed      — memory cost term zeroed (gamma = 0)

Optional / sensitivity (A5-A8):
  A5  predictor_comparison     — sweeps all predictor types
  A6  hysteresis_disabled      — hysteresis_cycles = 1
  A7  cooldown_disabled        — cooldown_seconds = 0.0
  A8  threshold_zero           — switch_threshold = 0.0
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AblationSpec:
    """
    Full description of a single ablation.

    changed_component names EXACTLY ONE module parameter that differs from
    the full PREDICTIVE baseline.
    """
    ablation_id: str
    name: str
    description: str
    changed_component: str         # e.g. "predictor", "epsilon", "beta", "gamma"
    # Exact config deltas applied on top of PREDICTIVE defaults
    cost_weights_override: Dict[str, float] = field(default_factory=dict)
    switch_threshold_override: Optional[float] = None
    cooldown_seconds_override: Optional[float] = None
    hysteresis_cycles_override: Optional[int]  = None
    predictor_type_override: Optional[str]     = None
    disable_prediction: bool = False

    is_mandatory: bool = True    # A1-A4 are mandatory; A5-A8 are optional
    sweep_values: List[Any] = field(default_factory=list)   # for A5 predictor sweep

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ablation_id":            self.ablation_id,
            "name":                   self.name,
            "description":            self.description,
            "changed_component":      self.changed_component,
            "cost_weights_override":  self.cost_weights_override,
            "switch_threshold_override": self.switch_threshold_override,
            "cooldown_seconds_override": self.cooldown_seconds_override,
            "hysteresis_cycles_override": self.hysteresis_cycles_override,
            "predictor_type_override": self.predictor_type_override,
            "disable_prediction":     self.disable_prediction,
            "is_mandatory":           self.is_mandatory,
            "sweep_values":           self.sweep_values,
        }


# ---------------------------------------------------------------------------
# Full PREDICTIVE baseline cost weights (reference)
# ---------------------------------------------------------------------------
_PREDICTIVE_WEIGHTS: Dict[str, float] = {
    "alpha":   0.35,
    "beta":    0.20,
    "gamma":   0.25,
    "delta":   0.10,
    "epsilon": 0.10,
}


class AblationSuite:
    """
    Factory for all required ablation specifications.

    Usage:
        suite   = AblationSuite()
        a1      = suite.get("prediction_removed")
        all_abl = suite.mandatory()
        full    = suite.all()
    """

    _SPECS: Dict[str, AblationSpec] = {

        # --- Mandatory A1-A4 ---------------------------------------------------

        "prediction_removed": AblationSpec(
            ablation_id="A1",
            name="prediction_removed",
            description=(
                "A1: Full PREDICTIVE system with predictor disabled. "
                "Controller falls back to REACTIVE mode. "
                "Changed: predictor availability (prediction → none)."
            ),
            changed_component="predictor",
            disable_prediction=True,
            is_mandatory=True,
        ),

        "switching_penalty_removed": AblationSpec(
            ablation_id="A2",
            name="switching_penalty_removed",
            description=(
                "A2: Full PREDICTIVE system with switching penalty epsilon = 0.0. "
                "Changed: cost weight epsilon (0.10 → 0.00). "
                "All other weights scaled proportionally to maintain sum."
            ),
            changed_component="epsilon",
            cost_weights_override={
                "alpha":   0.389,
                "beta":    0.222,
                "gamma":   0.278,
                "delta":   0.111,
                "epsilon": 0.000,
            },
            is_mandatory=True,
        ),

        "network_signal_removed": AblationSpec(
            ablation_id="A3",
            name="network_signal_removed",
            description=(
                "A3: Full PREDICTIVE system with network cost term zeroed. "
                "Changed: cost weight beta (0.20 → 0.00). "
                "Network telemetry is still collected but not used by CostModel."
            ),
            changed_component="beta",
            cost_weights_override={
                "alpha":   0.438,
                "beta":    0.000,
                "gamma":   0.312,
                "delta":   0.125,
                "epsilon": 0.125,
            },
            is_mandatory=True,
        ),

        "vram_signal_removed": AblationSpec(
            ablation_id="A4",
            name="vram_signal_removed",
            description=(
                "A4: Full PREDICTIVE system with memory/VRAM cost term zeroed. "
                "Changed: cost weight gamma (0.25 → 0.00). "
                "Memory telemetry is still collected but not used by CostModel."
            ),
            changed_component="gamma",
            cost_weights_override={
                "alpha":   0.467,
                "beta":    0.267,
                "gamma":   0.000,
                "delta":   0.133,
                "epsilon": 0.133,
            },
            is_mandatory=True,
        ),

        # --- Optional A5-A8 ---------------------------------------------------

        "predictor_comparison": AblationSpec(
            ablation_id="A5",
            name="predictor_comparison",
            description=(
                "A5: Compare prediction methods: last_value, linear_trend, "
                "moving_average, learned_ridge. "
                "Changed: predictor type (one per trial)."
            ),
            changed_component="predictor_type",
            sweep_values=["last_value", "linear_trend", "moving_average"],
            is_mandatory=False,
        ),

        "hysteresis_disabled": AblationSpec(
            ablation_id="A6",
            name="hysteresis_disabled",
            description=(
                "A6: Full PREDICTIVE system with hysteresis disabled. "
                "Changed: hysteresis_cycles (2 → 1 = effectively disabled)."
            ),
            changed_component="hysteresis_cycles",
            hysteresis_cycles_override=1,
            is_mandatory=False,
        ),

        "cooldown_disabled": AblationSpec(
            ablation_id="A7",
            name="cooldown_disabled",
            description=(
                "A7: Full PREDICTIVE system with cooldown disabled. "
                "Changed: cooldown_seconds (5.0 → 0.0)."
            ),
            changed_component="cooldown_seconds",
            cooldown_seconds_override=0.0,
            is_mandatory=False,
        ),

        "threshold_zero": AblationSpec(
            ablation_id="A8",
            name="threshold_zero",
            description=(
                "A8: Full PREDICTIVE system with switch threshold = 0.0. "
                "Changed: switch_threshold (0.25 → 0.0). "
                "Controller switches whenever any candidate is strictly better."
            ),
            changed_component="switch_threshold",
            switch_threshold_override=0.0,
            is_mandatory=False,
        ),
    }

    @classmethod
    def get(cls, ablation_id: str) -> AblationSpec:
        if ablation_id not in cls._SPECS:
            raise KeyError(
                f"Unknown ablation '{ablation_id}'. "
                f"Available: {cls.list_all()}"
            )
        return cls._SPECS[ablation_id]

    @classmethod
    def mandatory(cls) -> List[AblationSpec]:
        """Return only A1-A4 mandatory ablations."""
        return [s for s in cls._SPECS.values() if s.is_mandatory]

    @classmethod
    def optional(cls) -> List[AblationSpec]:
        """Return optional ablations A5-A8."""
        return [s for s in cls._SPECS.values() if not s.is_mandatory]

    @classmethod
    def all(cls) -> List[AblationSpec]:
        return list(cls._SPECS.values())

    @classmethod
    def list_all(cls) -> List[str]:
        return sorted(cls._SPECS.keys())

    def apply_to_configs(
        self,
        spec: AblationSpec,
        base_orch_cfg: Any,   # OrchestrationConfig
        base_ctrl_cfg: Any,   # ControllerConfig
    ) -> tuple:
        """
        Apply the ablation delta to a (OrchestrationConfig, ControllerConfig) pair.
        Returns a new (OrchestrationConfig, ControllerConfig) with ONE change applied.
        """
        import dataclasses as dc
        from src.orchestration.types import OrchestrationMode

        new_orch = dc.replace(base_orch_cfg)
        new_ctrl = dc.replace(base_ctrl_cfg)

        # A1: disable prediction
        if spec.disable_prediction:
            new_orch = dc.replace(new_orch,
                mode=OrchestrationMode.REACTIVE,
                predictor_type="last_value",
            )
            from src.controller.types import ControllerMode
            new_ctrl = dc.replace(new_ctrl, mode=ControllerMode.REACTIVE)

        # Cost weight overrides (A2/A3/A4)
        if spec.cost_weights_override:
            new_orch = dc.replace(new_orch, cost_weights=dict(spec.cost_weights_override))

        # Controller knob overrides (A6/A7/A8)
        if spec.switch_threshold_override is not None:
            new_ctrl = dc.replace(new_ctrl, switch_threshold=spec.switch_threshold_override)
        if spec.cooldown_seconds_override is not None:
            new_ctrl = dc.replace(new_ctrl, cooldown_seconds=spec.cooldown_seconds_override)
        if spec.hysteresis_cycles_override is not None:
            new_ctrl = dc.replace(new_ctrl, hysteresis_cycles=spec.hysteresis_cycles_override)

        # A5: predictor type
        if spec.predictor_type_override is not None:
            new_orch = dc.replace(new_orch, predictor_type=spec.predictor_type_override)

        return new_orch, new_ctrl
