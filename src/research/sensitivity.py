"""
Research Sensitivity & Robustness Analysis (Module 12).

Performs controlled multi-dimensional parameter sweeps across network dynamics,
controller thresholds, and forecasting horizons to map operational trade-offs,
degradation curves, and stability boundaries.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.research.statistics import MetricSummaryStats, StatisticalAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class SensitivitySweepPoint:
    """Statistical summary for one parameter setting along a sweep dimension."""
    param_value: Any
    n_trials: int
    mean_itl_stats: MetricSummaryStats
    throughput_stats: MetricSummaryStats
    switches_stats: MetricSummaryStats
    degradation_pct_vs_nominal: Optional[float] = None
    is_infeasible: bool = False
    infeasible_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["mean_itl_stats"] = self.mean_itl_stats.to_dict()
        d["throughput_stats"] = self.throughput_stats.to_dict()
        d["switches_stats"] = self.switches_stats.to_dict()
        return d


@dataclass
class SensitivityDimensionAnalysis:
    """Complete sensitivity study along one parameter dimension."""
    dimension_name: str
    unit: str
    nominal_value: Any
    sweep_values: List[Any]
    points: List[SensitivitySweepPoint] = field(default_factory=list)
    failure_boundaries: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["points"] = [p.to_dict() for p in self.points]
        return d


class ResearchSensitivityRunner:
    """
    Analyzes parameter sweep data and extracts sensitivity degradation curves.
    """

    @classmethod
    def analyze_parameter_sweep(
        cls,
        dimension_name: str,
        unit: str,
        trials_df: pd.DataFrame,
        nominal_value: Any,
        param_column: Optional[str] = None,
    ) -> SensitivityDimensionAnalysis:
        """
        Extract sensitivity response curves from a sweep DataFrame.
        """
        col = param_column or dimension_name
        if col not in trials_df.columns:
            logger.warning(f"Parameter column {col} not found in trials DataFrame.")
            return SensitivityDimensionAnalysis(
                dimension_name=dimension_name,
                unit=unit,
                nominal_value=nominal_value,
                sweep_values=[],
                points=[],
            )

        unique_vals = sorted(trials_df[col].dropna().unique())
        points: List[SensitivitySweepPoint] = []
        failure_boundaries: List[str] = []

        nominal_itl_mean: Optional[float] = None

        # First pass: find nominal mean ITL
        nominal_slice = trials_df[trials_df[col] == nominal_value]
        if len(nominal_slice) > 0 and "mean_itl_ms" in nominal_slice.columns:
            nom_stats = StatisticalAnalyzer.compute_metric_stats("mean_itl_ms", nominal_slice["mean_itl_ms"])
            nominal_itl_mean = nom_stats.mean

        for val in unique_vals:
            sub_df = trials_df[trials_df[col] == val]
            n_sub = len(sub_df)

            itl_stats = StatisticalAnalyzer.compute_metric_stats("mean_itl_ms", sub_df.get("mean_itl_ms", []))
            tps_stats = StatisticalAnalyzer.compute_metric_stats("tokens_per_second", sub_df.get("tokens_per_second", []))
            sw_stats = StatisticalAnalyzer.compute_metric_stats("total_switches", sub_df.get("total_switches", []))

            deg_pct = None
            if nominal_itl_mean and itl_stats.mean and nominal_itl_mean > 0:
                deg_pct = float(((itl_stats.mean - nominal_itl_mean) / nominal_itl_mean) * 100.0)

            # Check infeasibility / failure boundary (e.g., OOM count > 0 or excessive SLO violation)
            is_infeasible = False
            infeas_reason = None
            if "oom_count" in sub_df.columns and sub_df["oom_count"].sum() > 0:
                is_infeasible = True
                infeas_reason = "Out-Of-Memory events occurred at this setting."
                failure_boundaries.append(f"{dimension_name}={val}: Out-of-memory detected.")
            elif "slo_violation_rate" in sub_df.columns and sub_df["slo_violation_rate"].mean() > 0.8:
                is_infeasible = True
                infeas_reason = "Critical SLO violation (>80%) operating boundary crossed."
                failure_boundaries.append(f"{dimension_name}={val}: Severe SLO degradation.")

            points.append(
                SensitivitySweepPoint(
                    param_value=val,
                    n_trials=n_sub,
                    mean_itl_stats=itl_stats,
                    throughput_stats=tps_stats,
                    switches_stats=sw_stats,
                    degradation_pct_vs_nominal=deg_pct,
                    is_infeasible=is_infeasible,
                    infeasible_reason=infeas_reason,
                )
            )

        return SensitivityDimensionAnalysis(
            dimension_name=dimension_name,
            unit=unit,
            nominal_value=nominal_value,
            sweep_values=unique_vals,
            points=points,
            failure_boundaries=failure_boundaries,
        )

    @classmethod
    def sweep_to_dataframe(cls, sweep: SensitivityDimensionAnalysis) -> pd.DataFrame:
        """Flatten sensitivity sweep points into tabular DataFrame."""
        rows = []
        for p in sweep.points:
            rows.append({
                "dimension": sweep.dimension_name,
                "unit": sweep.unit,
                "param_value": p.param_value,
                "n_trials": p.n_trials,
                "mean_itl_ms": p.mean_itl_stats.mean,
                "mean_itl_ci_lower": p.mean_itl_stats.ci_t.lower if p.mean_itl_stats.ci_t else None,
                "mean_itl_ci_upper": p.mean_itl_stats.ci_t.upper if p.mean_itl_stats.ci_t else None,
                "throughput_tps": p.throughput_stats.mean,
                "total_switches": p.switches_stats.mean,
                "degradation_pct_vs_nominal": p.degradation_pct_vs_nominal,
                "is_infeasible": p.is_infeasible,
                "infeasible_reason": p.infeasible_reason,
            })
        return pd.DataFrame(rows)
