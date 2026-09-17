"""
Sensitivity analysis framework for Module 10.

Provides parameter sweeps and sensitivity evaluation across:
  - Switching threshold (theta): stability vs responsiveness trade-off
  - Prediction horizon (H): forecast window sensitivity
  - Control interval (k): sampling frequency and overhead trade-off
  - Predictor type: architecture comparison under identical control
  - Cost weights: alpha/beta/gamma/delta/epsilon weight sensitivity

Extracts sensitivity curves and trade-off metrics without ranking/winner claims.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.evaluation.experiment_config import TrialConfig
from src.evaluation.metrics import EvaluationMetrics


class SensitivityDimension(str, enum.Enum):
    """The parameter dimension being varied in a sensitivity experiment."""
    SWITCH_THRESHOLD   = "switch_threshold"
    HORIZON_STEPS      = "horizon_steps"
    CONTROL_INTERVAL   = "control_interval_tokens"
    PREDICTOR_TYPE     = "predictor_type"
    WEIGHT_ALPHA       = "weight_alpha"
    WEIGHT_BETA        = "weight_beta"
    WEIGHT_GAMMA       = "weight_gamma"
    WEIGHT_EPSILON     = "weight_epsilon"


@dataclass
class SweepPoint:
    """A single evaluation point along a sensitivity dimension."""
    param_value: Any
    metrics: EvaluationMetrics
    trial_id: str
    seed: int = 42


@dataclass
class SensitivityCurve:
    """
    Sensitivity curve along one parameter dimension for a specific metric.

    Contains raw (param_value, metric_value) points, summary statistics,
    and monotonicity / curvature characteristics.
    """
    dimension: SensitivityDimension
    metric_name: str
    param_values: List[Any] = field(default_factory=list)
    metric_values: List[float] = field(default_factory=list)
    metric_stds: List[float] = field(default_factory=list)

    @property
    def min_metric_point(self) -> Tuple[Any, float]:
        """Point with minimum metric value."""
        if not self.metric_values:
            return None, float("nan")
        idx = min(range(len(self.metric_values)), key=lambda i: self.metric_values[i])
        return self.param_values[idx], self.metric_values[idx]

    @property
    def max_metric_point(self) -> Tuple[Any, float]:
        """Point with maximum metric value."""
        if not self.metric_values:
            return None, float("nan")
        idx = max(range(len(self.metric_values)), key=lambda i: self.metric_values[i])
        return self.param_values[idx], self.metric_values[idx]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "metric_name": self.metric_name,
            "param_values": self.param_values,
            "metric_values": self.metric_values,
            "metric_stds": self.metric_stds,
            "min_point": list(self.min_metric_point),
            "max_point": list(self.max_metric_point),
        }


@dataclass
class SensitivitySweepResult:
    """Aggregated outcome of a multi-metric sensitivity sweep."""
    dimension: SensitivityDimension
    sweep_values: List[Any]
    curves: Dict[str, SensitivityCurve] = field(default_factory=dict)
    n_trials: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "sweep_values": self.sweep_values,
            "curves": {k: v.to_dict() for k, v in self.curves.items()},
            "n_trials": self.n_trials,
            "metadata": self.metadata,
        }


class SensitivityAnalyzer:
    """
    Analyzes sensitivity across trials.

    Extracts parameter response curves for key metrics:
      - mean_itl_ms
      - p95_itl_ms
      - slo_violation_count
      - total_switches (stability / churn proxy)
      - total_migration_time_ms
      - control_overhead_fraction
    """

    DEFAULT_METRICS = [
        "mean_itl_ms",
        "p95_itl_ms",
        "slo_violation_count",
        "total_switches",
        "proactive_switches",
        "total_migration_time_ms",
        "control_overhead_fraction",
    ]

    @classmethod
    def analyze_sweep(
        cls,
        dimension: SensitivityDimension,
        sweep_data: Sequence[Tuple[Any, EvaluationMetrics, str]],
        metrics_to_extract: Optional[List[str]] = None,
    ) -> SensitivitySweepResult:
        """
        Build sensitivity curves from list of (param_value, EvaluationMetrics, trial_id).
        """
        metrics = metrics_to_extract or cls.DEFAULT_METRICS

        # Group metrics by param_value
        grouped: Dict[Any, List[EvaluationMetrics]] = {}
        for val, m, _ in sweep_data:
            grouped.setdefault(val, []).append(m)

        sorted_vals = sorted(grouped.keys(), key=lambda x: (isinstance(x, str), x))
        curves: Dict[str, SensitivityCurve] = {}

        for m_name in metrics:
            val_list = []
            mean_list = []
            std_list = []
            for val in sorted_vals:
                metric_entries = grouped[val]
                vals = []
                for entry in metric_entries:
                    raw_val = getattr(entry, m_name, None)
                    if raw_val is not None and not (isinstance(raw_val, float) and (raw_val != raw_val)):
                        vals.append(float(raw_val))
                if vals:
                    mean = sum(vals) / len(vals)
                    variance = sum((x - mean) ** 2 for x in vals) / max(1, len(vals) - 1) if len(vals) > 1 else 0.0
                    std = variance ** 0.5
                else:
                    mean = float("nan")
                    std = 0.0

                val_list.append(val)
                mean_list.append(round(mean, 4))
                std_list.append(round(std, 4))

            curves[m_name] = SensitivityCurve(
                dimension=dimension,
                metric_name=m_name,
                param_values=val_list,
                metric_values=mean_list,
                metric_stds=std_list,
            )

        return SensitivitySweepResult(
            dimension=dimension,
            sweep_values=sorted_vals,
            curves=curves,
            n_trials=len(sweep_data),
        )

    @staticmethod
    def extract_tradeoff(
        result: SensitivitySweepResult,
        cost_metric: str = "total_switches",
        benefit_metric: str = "p95_itl_ms",
    ) -> List[Dict[str, Any]]:
        """
        Extract trade-off table pairing cost (churn/migrations) against benefit (latency/SLO).
        """
        if cost_metric not in result.curves or benefit_metric not in result.curves:
            return []

        cost_curve = result.curves[cost_metric]
        benefit_curve = result.curves[benefit_metric]

        table = []
        for val, c_val, b_val in zip(
            result.sweep_values,
            cost_curve.metric_values,
            benefit_curve.metric_values,
        ):
            table.append({
                "param_value": val,
                cost_metric: c_val,
                benefit_metric: b_val,
            })
        return table
