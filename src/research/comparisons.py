"""
Fair Policy Comparison Engine (Module 12).

Executes rigorous pairwise and multi-baseline comparisons across the 5 canonical
policies (B1 STATIC, B2 NETWORK_REACTIVE, B3 MEMORY_REACTIVE, B4 JOINT_REACTIVE, B5 PREDICTIVE)
under matched repetition seeds, identical input traces, and multiple-testing corrections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.evaluation.baselines import BaselinePolicy
from src.research.statistics import PairedComparisonResult, StatisticalAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ScenarioComparisonReport:
    """Comprehensive comparison report across all policies for a single scenario."""
    scenario_id: str
    comparisons: List[PairedComparisonResult] = field(default_factory=list)
    reference_policy: str = "predictive"
    metric_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "reference_policy": self.reference_policy,
            "comparisons": [c.to_dict() for c in self.comparisons],
        }


class PolicyComparisonEngine:
    """
    Evaluates statistical differences between inference policies across trials.
    """

    CORE_COMPUTED_METRICS = [
        "mean_itl_ms",
        "median_itl_ms",
        "p95_itl_ms",
        "tokens_per_second",
        "total_switches",
        "switch_frequency_per_token",
        "total_migration_time_ms",
        "total_transferred_mb",
        "slo_violation_rate",
        "total_control_overhead_ms",
        "control_overhead_fraction",
    ]

    @classmethod
    def compare_scenario_trials(
        cls,
        scenario_id: str,
        trials_df: pd.DataFrame,
        reference_policy: str = "predictive",
        target_metrics: Optional[Sequence[str]] = None,
        confidence_level: float = 0.95,
        alpha_significance: float = 0.05,
    ) -> ScenarioComparisonReport:
        """
        Compute paired statistical comparisons between all baselines and a reference policy
        for a specific scenario, applying Holm-Bonferroni multiple testing corrections.
        """
        metrics = list(target_metrics or cls.CORE_COMPUTED_METRICS)
        scen_df = trials_df[trials_df["scenario_id"] == scenario_id]

        policies = sorted(scen_df["baseline_id"].unique())
        other_policies = [p for p in policies if p != reference_policy]

        comparisons: List[PairedComparisonResult] = []

        # Gather matched pairs by seed/repetition
        for baseline in other_policies:
            for metric in metrics:
                if metric not in scen_df.columns:
                    continue

                # Pivot on seed / repetition to match rows
                ref_series = scen_df[scen_df["baseline_id"] == reference_policy].set_index("seed")[metric]
                base_series = scen_df[scen_df["baseline_id"] == baseline].set_index("seed")[metric]

                common_seeds = ref_series.index.intersection(base_series.index)
                if len(common_seeds) == 0:
                    continue

                vals_ref = ref_series.loc[common_seeds].tolist()
                vals_base = base_series.loc[common_seeds].tolist()

                # Compare Reference (Policy A) vs Baseline (Policy B)
                cmp_res = StatisticalAnalyzer.compare_paired(
                    metric_name=metric,
                    policy_a_name=reference_policy,
                    values_a=vals_ref,
                    policy_b_name=baseline,
                    values_b=vals_base,
                    confidence_level=confidence_level,
                )
                comparisons.append(cmp_res)

        # Apply multiple testing corrections across all p-values in this scenario comparison
        valid_p_indices = [i for i, c in enumerate(comparisons) if c.p_value_t is not None]
        if valid_p_indices:
            raw_p_values = [comparisons[i].p_value_t for i in valid_p_indices]
            holm_adjusted = StatisticalAnalyzer.adjust_p_values_holm(raw_p_values)
            fdr_adjusted = StatisticalAnalyzer.adjust_p_values_fdr(raw_p_values)

            for idx, p_holm, p_fdr in zip(valid_p_indices, holm_adjusted, fdr_adjusted):
                comparisons[idx].p_value_adjusted_holm = p_holm
                comparisons[idx].p_value_adjusted_fdr = p_fdr
                # Significant if Holm adjusted p < alpha
                comparisons[idx].is_statistically_significant = p_holm < alpha_significance

        return ScenarioComparisonReport(
            scenario_id=scenario_id,
            comparisons=comparisons,
            reference_policy=reference_policy,
            metric_names=metrics,
        )

    @classmethod
    def comparisons_to_dataframe(cls, comparisons: Sequence[PairedComparisonResult]) -> pd.DataFrame:
        """Flatten a list of comparison results into a structured tabular DataFrame."""
        rows = []
        for c in comparisons:
            rows.append({
                "metric_name": c.metric_name,
                "policy_a": c.policy_a,
                "policy_b": c.policy_b,
                "n_pairs": c.n_pairs,
                "mean_a": c.mean_a,
                "mean_b": c.mean_b,
                "mean_diff_a_minus_b": c.mean_diff,
                "median_diff": c.median_diff,
                "std_diff": c.std_diff,
                "ci_lower": c.ci_diff_t.lower if c.ci_diff_t else None,
                "ci_upper": c.ci_diff_t.upper if c.ci_diff_t else None,
                "t_statistic": c.t_statistic,
                "p_value_raw": c.p_value_t,
                "p_value_holm": c.p_value_adjusted_holm,
                "p_value_fdr": c.p_value_adjusted_fdr,
                "p_value_wilcoxon": c.p_value_wilcoxon,
                "cohens_d": c.cohens_d,
                "hedges_g": c.hedges_g,
                "cliffs_delta": c.cliffs_delta,
                "is_significant": c.is_statistically_significant,
                "practical_significance": c.practical_significance,
                "warning": c.warning,
            })
        return pd.DataFrame(rows)
