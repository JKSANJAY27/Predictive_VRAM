"""
Research Ablation Study Framework (Module 12).

Executes hypothesis-driven ablation studies isolating individual algorithmic mechanisms
against the reference PREDICTIVE baseline (B5) under controlled, fair experimental conditions.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.evaluation.ablations import AblationSpec, AblationSuite
from src.research.statistics import PairedComparisonResult, StatisticalAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ResearchAblationHypothesis:
    """Pre-stated hypothesis and mechanical definition for an ablation study."""
    ablation_id: str
    name: str
    target_component: str
    pre_stated_hypothesis: str
    parameters_held_constant: str
    config_delta: Dict[str, Any]
    reference_condition: str = "B5 (PREDICTIVE)"


# Canonical research hypotheses for Module 12
RESEARCH_ABLATION_HYPOTHESES: Dict[str, ResearchAblationHypothesis] = {
    "prediction_removed": ResearchAblationHypothesis(
        ablation_id="A1",
        name="Prediction Disablement",
        target_component="Forecasting Predictor (Module 4)",
        pre_stated_hypothesis=(
            "Disabling proactive horizon forecasting will force reactive migrations "
            "during sudden network/memory drops, increasing tail latency (P95 ITL) and SLO violations."
        ),
        parameters_held_constant="Model architecture, cost weights, candidate partitions, trace inputs, seeds.",
        config_delta={"disable_prediction": True},
    ),
    "switching_penalty_removed": ResearchAblationHypothesis(
        ablation_id="A2",
        name="Zero Switching Penalty",
        target_component="Cost Model Switching Term epsilon (Module 6)",
        pre_stated_hypothesis=(
            "Setting switching penalty weight epsilon to 0.0 will cause partition thrashing / oscillation "
            "under fluctuating network conditions, drastically increasing cumulative migration overhead."
        ),
        parameters_held_constant="Model architecture, forecasting, candidate partitions, dwell times, seeds.",
        config_delta={"cost_weights_override": {"epsilon": 0.0}},
    ),
    "network_signal_removed": ResearchAblationHypothesis(
        ablation_id="A3",
        name="Network Signal Disablement",
        target_component="Network Communication Cost beta (Module 6)",
        pre_stated_hypothesis=(
            "Removing network telemetry from partition scoring (beta = 0.0) will cause the controller "
            "to retain multi-tier splits across bandwidth degradation, inducing communication bottlenecks."
        ),
        parameters_held_constant="Model architecture, latency/memory weights, switching penalty, seeds.",
        config_delta={"cost_weights_override": {"beta": 0.0}},
    ),
    "vram_signal_removed": ResearchAblationHypothesis(
        ablation_id="A4",
        name="Memory/VRAM Signal Disablement",
        target_component="Memory Pressure Cost gamma (Module 6)",
        pre_stated_hypothesis=(
            "Removing memory telemetry and KV-cache tracking (gamma = 0.0) will cause edge nodes to exceed "
            "memory safety margins during long context generation, leading to emergency rollbacks or OOMs."
        ),
        parameters_held_constant="Model architecture, latency/network weights, switching penalty, seeds.",
        config_delta={"cost_weights_override": {"gamma": 0.0}},
    ),
    "hysteresis_disabled": ResearchAblationHypothesis(
        ablation_id="A6",
        name="Hysteresis Disablement",
        target_component="Controller Hysteresis Filter (Module 7)",
        pre_stated_hypothesis=(
            "Setting hysteresis cycles to 1 will cause immediate partition switching on single-step "
            "transient spikes, increasing switch frequency without improving steady-state throughput."
        ),
        parameters_held_constant="Model architecture, forecasting, cost weights, cooldown seconds, seeds.",
        config_delta={"hysteresis_cycles_override": 1},
    ),
    "cooldown_disabled": ResearchAblationHypothesis(
        ablation_id="A7",
        name="Cooldown Disablement",
        target_component="Controller Cooldown Dwell Guard (Module 7)",
        pre_stated_hypothesis=(
            "Setting cooldown duration to 0.0s will permit consecutive back-to-back migrations, "
            "destabilizing split execution during continuous network volatility."
        ),
        parameters_held_constant="Model architecture, forecasting, cost weights, hysteresis cycles, seeds.",
        config_delta={"cooldown_seconds_override": 0.0},
    ),
}


@dataclass
class AblationAnalysisResult:
    """Statistical outcome of evaluating an ablation against reference PREDICTIVE baseline."""
    ablation_id: str
    ablation_name: str
    hypothesis: ResearchAblationHypothesis
    scenario_id: str
    comparisons: List[PairedComparisonResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ablation_id": self.ablation_id,
            "ablation_name": self.ablation_name,
            "hypothesis": asdict(self.hypothesis),
            "scenario_id": self.scenario_id,
            "comparisons": [c.to_dict() for c in self.comparisons],
        }


class ResearchAblationRunner:
    """
    Executes and analyzes ablation studies against reference baseline runs.
    """

    @classmethod
    def get_registered_hypotheses(cls) -> Dict[str, ResearchAblationHypothesis]:
        return RESEARCH_ABLATION_HYPOTHESES.copy()

    @classmethod
    def evaluate_ablation_effects(
        cls,
        ablation_key: str,
        scenario_id: str,
        trials_df: pd.DataFrame,
        reference_baseline: str = "predictive",
        target_metrics: Optional[Sequence[str]] = None,
        confidence_level: float = 0.95,
    ) -> Optional[AblationAnalysisResult]:
        """
        Compare ablation trials (with matching ablation_id) against reference baseline trials
        for a scenario under matched seeds.
        """
        hypo = RESEARCH_ABLATION_HYPOTHESES.get(ablation_key)
        if hypo is None:
            logger.warning(f"Unknown ablation key: {ablation_key}")
            return None

        metrics = list(target_metrics or [
            "mean_itl_ms", "p95_itl_ms", "tokens_per_second",
            "total_switches", "total_migration_time_ms", "total_control_overhead_ms"
        ])

        scen_df = trials_df[trials_df["scenario_id"] == scenario_id]

        # Filter ablation trials vs reference baseline trials
        abl_df = scen_df[scen_df["ablation_id"] == ablation_key]
        ref_df = scen_df[(scen_df["baseline_id"] == reference_baseline) & (scen_df["ablation_id"].isin(["", "none"]))]

        if len(abl_df) == 0 or len(ref_df) == 0:
            logger.warning(f"Insufficient trials for ablation {ablation_key} in scenario {scenario_id}")
            return None

        comparisons: List[PairedComparisonResult] = []
        for metric in metrics:
            if metric not in scen_df.columns:
                continue

            abl_series = abl_df.set_index("seed")[metric]
            ref_series = ref_df.set_index("seed")[metric]

            common_seeds = abl_series.index.intersection(ref_series.index)
            if len(common_seeds) == 0:
                continue

            vals_abl = abl_series.loc[common_seeds].tolist()
            vals_ref = ref_series.loc[common_seeds].tolist()

            # Compare Ablated (Policy A) vs Reference (Policy B)
            cmp_res = StatisticalAnalyzer.compare_paired(
                metric_name=metric,
                policy_a_name=f"Ablation:{ablation_key}",
                values_a=vals_abl,
                policy_b_name=reference_baseline,
                values_b=vals_ref,
                confidence_level=confidence_level,
            )
            comparisons.append(cmp_res)

        return AblationAnalysisResult(
            ablation_id=hypo.ablation_id,
            ablation_name=hypo.name,
            hypothesis=hypo,
            scenario_id=scenario_id,
            comparisons=comparisons,
        )
