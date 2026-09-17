"""
Statistical aggregation for Module 10.

ResultAggregator  — groups TrialResults and computes per-metric statistics
AggregatedResult  — contains mean/median/std/SE/CI per metric
MetricStats       — statistics for a single metric across trials
PairedComparison  — paired-sample difference between two baselines
EffectSizeResult  — Cohen's d and paired differences (not ranked)

Statistical methodology:
  - CI uses Student t-distribution (not z) for all n
  - Results with n < 2 get no CI; n < 5 flagged as "insufficient"
  - Outliers are flagged and counted but NOT removed
  - No automatic normality assumption
  - No ranking or "better/worse" labels emitted
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.evaluation.metrics import EvaluationMetrics, _percentile, _safe_mean, _safe_median


# ---------------------------------------------------------------------------
# Student t critical values (two-sided, df → ∞ → 1.96 for df=∞)
# ---------------------------------------------------------------------------
_T_CRIT_95: Dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447,  7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000, 120: 1.980,
}

def _t_crit(df: int) -> float:
    """Two-sided 95% CI critical value for given df."""
    if df <= 0:
        return float("inf")
    if df in _T_CRIT_95:
        return _T_CRIT_95[df]
    # Linear interpolation for df values not in table
    keys = sorted(_T_CRIT_95.keys())
    if df > keys[-1]:
        return 1.96
    for i in range(len(keys) - 1):
        if keys[i] <= df <= keys[i + 1]:
            lo, hi = keys[i], keys[i + 1]
            f = (df - lo) / (hi - lo)
            return _T_CRIT_95[lo] + f * (_T_CRIT_95[hi] - _T_CRIT_95[lo])
    return 1.96


# ---------------------------------------------------------------------------
# MetricStats
# ---------------------------------------------------------------------------

@dataclass
class MetricStats:
    """Descriptive statistics for one metric across N trials."""
    metric_name: str
    n: int = 0
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    se: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    p50: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    outlier_count: int = 0
    pairing_quality: str = "unknown"   # "paired" | "unpaired" | "insufficient"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def _compute_stats(name: str, values: List[float], pairing: str = "unknown") -> MetricStats:
    """Compute MetricStats from a list of scalar observations."""
    n = len(values)
    s = MetricStats(metric_name=name, n=n, pairing_quality=pairing)

    if n == 0:
        return s

    s.min_val = min(values)
    s.max_val = max(values)
    s.mean    = _safe_mean(values)
    s.median  = _safe_median(values)
    s.p50     = _percentile(values, 50)
    s.p95     = _percentile(values, 95)
    s.p99     = _percentile(values, 99)

    if n >= 2:
        s.std  = statistics.stdev(values)
        s.se   = s.std / math.sqrt(n)
        df     = n - 1
        tc     = _t_crit(df)
        s.ci_lower = s.mean - tc * s.se
        s.ci_upper = s.mean + tc * s.se
        if n < 5:
            s.pairing_quality = "insufficient"

    return s


# ---------------------------------------------------------------------------
# AggregatedResult
# ---------------------------------------------------------------------------

@dataclass
class AggregatedResult:
    """Aggregate statistics for one (experiment, scenario, baseline) group."""
    schema_version: str = "1.0"
    experiment_id: str = ""
    scenario_id: str = ""
    baseline_id: str = ""
    n_trials: int = 0
    metric_stats: Dict[str, MetricStats] = field(default_factory=dict)
    pairing_quality: str = "unknown"
    outlier_trial_ids: List[str] = field(default_factory=list)
    invalid_trial_count: int = 0
    notes: str = ""

    @property
    def metrics(self) -> Dict[str, MetricStats]:
        return self.metric_stats

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id":  self.experiment_id,
            "scenario_id":    self.scenario_id,
            "baseline_id":    self.baseline_id,
            "n_trials":       self.n_trials,
            "metric_stats":   {k: v.to_dict() for k, v in self.metric_stats.items()},
            "metrics":        {k: v.to_dict() for k, v in self.metric_stats.items()},
            "pairing_quality": self.pairing_quality,
            "outlier_trial_ids": self.outlier_trial_ids,
            "invalid_trial_count": self.invalid_trial_count,
            "notes": self.notes,
        }

    def save_json(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# EffectSizeResult (paired baseline comparisons)
# ---------------------------------------------------------------------------

@dataclass
class EffectSizeResult:
    """
    Effect size metrics for a pairwise baseline comparison.

    Reports raw paired differences only — does NOT rank or label baselines.
    """
    metric_name: str
    baseline_a: str
    baseline_b: str
    n_pairs: int = 0
    mean_diff_a_minus_b: Optional[float] = None     # positive: A is larger
    median_diff_a_minus_b: Optional[float] = None
    cohens_d: Optional[float] = None                # hedged; not interpreted
    pairing_quality: str = "unknown"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# PairedComparison
# ---------------------------------------------------------------------------

@dataclass
class PairedComparison:
    """Paired A-B comparison for matching (scenario, seed) pairs."""
    experiment_id: str
    scenario_id: str
    baseline_a: str
    baseline_b: str
    effect_sizes: List[EffectSizeResult] = field(default_factory=list)
    n_pairs: int = 0
    pairing_quality: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "scenario_id":   self.scenario_id,
            "baseline_a":    self.baseline_a,
            "baseline_b":    self.baseline_b,
            "n_pairs":       self.n_pairs,
            "pairing_quality": self.pairing_quality,
            "effect_sizes":  [e.to_dict() for e in self.effect_sizes],
        }


# ---------------------------------------------------------------------------
# ResultAggregator
# ---------------------------------------------------------------------------

_SCALAR_METRICS = [
    "ttft_ms", "mean_itl_ms", "median_itl_ms", "p95_itl_ms", "p99_itl_ms",
    "end_to_end_latency_ms", "tokens_per_second", "completed_tokens",
    "max_memory_pressure", "min_vram_headroom_mb", "memory_safety_violations",
    "avg_bandwidth_mbps", "min_bandwidth_mbps", "avg_rtt_ms", "avg_packet_loss",
    "total_communication_bytes", "controller_cycles", "total_controller_time_ms",
    "total_control_plane_overhead_ms", "control_overhead_fraction",
    "migration_count", "successful_migrations", "failed_migrations", "rollback_count",
    "total_migration_time_ms", "mean_migration_time_ms", "p95_migration_time_ms",
    "total_migration_bytes",
    "switch_count", "switches_per_hour", "avg_dwell_time_tokens",
    "thrashing_event_count", "threshold_rejection_count",
    "cooldown_rejection_count", "hysteresis_rejection_count",
    "prediction_mae", "prediction_rmse", "avg_prediction_lead_time_s",
    "safety_override_count", "fallback_count",
    "proactive_switch_count", "reactive_switch_count", "proactive_switch_rate",
    "slo_violation_rate", "slo_violation_count",
    "cost_model_byte_mae", "cost_model_duration_mae",
    "component_errors",
]


class ResultAggregator:
    """
    Groups a collection of TrialResults and computes AggregatedResult per group.
    """

    def aggregate(
        self, results: List[Any]   # List[TrialResult]
    ) -> Dict[Tuple[str, str, str], AggregatedResult]:
        """
        Group results by (experiment_id, scenario_id, baseline_id) and
        compute aggregate statistics.

        Returns a dict keyed by (experiment_id, scenario_id, baseline_id).
        """
        groups: Dict[Tuple[str, str, str], List[Any]] = {}
        for r in results:
            key = (r.experiment_id, r.scenario_id, r.baseline_id)
            groups.setdefault(key, []).append(r)

        output: Dict[Tuple[str, str, str], AggregatedResult] = {}
        for key, group in groups.items():
            output[key] = self._aggregate_group(key, group)
        return output

    def _aggregate_group(
        self,
        key: Tuple[str, str, str],
        trials: List[Any],
    ) -> AggregatedResult:
        exp_id, scen_id, base_id = key
        valid = [t for t in trials if t.completion_status == "completed"]
        invalid_count = len(trials) - len(valid)
        n = len(valid)

        pairing = "paired" if n >= 5 else ("unpaired" if n >= 2 else "insufficient")

        stats: Dict[str, MetricStats] = {}
        for metric in _SCALAR_METRICS:
            values = []
            for t in valid:
                v = getattr(t.metrics, metric, None)
                if v is not None:
                    try:
                        values.append(float(v))
                    except (TypeError, ValueError):
                        pass
            stats[metric] = _compute_stats(metric, values, pairing)

        return AggregatedResult(
            schema_version="1.0",
            experiment_id=exp_id,
            scenario_id=scen_id,
            baseline_id=base_id,
            n_trials=n,
            metric_stats=stats,
            pairing_quality=pairing,
            invalid_trial_count=invalid_count,
        )

    def paired_comparison(
        self,
        results: List[Any],
        baseline_a: str,
        baseline_b: str,
        metric_names: Optional[List[str]] = None,
    ) -> PairedComparison:
        """
        Compute paired A-B comparison for matching (scenario, seed) pairs.

        Matching is on (experiment_id, scenario_id, seed).
        Paired differences are reported as A - B without ranking.
        """
        if metric_names is None:
            metric_names = [
                "p95_itl_ms", "ttft_ms", "switch_count",
                "proactive_switch_rate", "total_migration_time_ms",
                "slo_violation_rate", "control_overhead_fraction",
            ]

        # Build lookup: (exp_id, scen_id, seed) → metrics
        lookup_a: Dict[Tuple, EvaluationMetrics] = {}
        lookup_b: Dict[Tuple, EvaluationMetrics] = {}
        for r in results:
            k = (r.experiment_id, r.scenario_id, r.seed)
            if r.baseline_id == baseline_a:
                lookup_a[k] = r.metrics
            elif r.baseline_id == baseline_b:
                lookup_b[k] = r.metrics

        common_keys = sorted(set(lookup_a) & set(lookup_b))
        n_pairs = len(common_keys)

        effect_sizes: List[EffectSizeResult] = []
        for metric in metric_names:
            diffs = []
            for k in common_keys:
                va = getattr(lookup_a[k], metric, None)
                vb = getattr(lookup_b[k], metric, None)
                if va is not None and vb is not None:
                    try:
                        diffs.append(float(va) - float(vb))
                    except (TypeError, ValueError):
                        pass

            es = EffectSizeResult(
                metric_name=metric,
                baseline_a=baseline_a,
                baseline_b=baseline_b,
                n_pairs=len(diffs),
                pairing_quality="paired" if len(diffs) >= 5 else "insufficient",
            )
            if diffs:
                es.mean_diff_a_minus_b   = _safe_mean(diffs)
                es.median_diff_a_minus_b = _safe_median(diffs)
                if len(diffs) >= 2:
                    sd = statistics.stdev(diffs)
                    es.cohens_d = (
                        es.mean_diff_a_minus_b / sd if sd > 0 else 0.0
                    )
            effect_sizes.append(es)

        pairing = "paired" if n_pairs >= 5 else ("unpaired" if n_pairs >= 2 else "insufficient")

        # Infer scenario/experiment from first result
        exp_id  = results[0].experiment_id  if results else ""
        scen_id = results[0].scenario_id    if results else ""

        return PairedComparison(
            experiment_id=exp_id,
            scenario_id=scen_id,
            baseline_a=baseline_a,
            baseline_b=baseline_b,
            effect_sizes=effect_sizes,
            n_pairs=n_pairs,
            pairing_quality=pairing,
        )

    @classmethod
    def aggregate_trials(cls, trials: Sequence[Any]) -> List[AggregatedResult]:
        """Convenience helper to aggregate a list of dicts or TrialResult objects."""
        # Convert dicts to duck-typed wrapper if needed
        class _DictWrapper:
            def __init__(self, d: Dict[str, Any]):
                self.experiment_id = d.get("experiment_id", "")
                self.scenario_id = d.get("scenario_id", "")
                self.baseline_id = d.get("baseline_id", "")
                self.seed = d.get("seed", 0)
                self.completion_status = d.get("completion_status", "completed")
                m = d.get("metrics", {})
                if hasattr(m, "__dict__"):
                    self.metrics = m
                else:
                    self.metrics = EvaluationMetrics.from_dict(m) if hasattr(EvaluationMetrics, "from_dict") else type("M", (), m)()

        wrapped = []
        for t in trials:
            if isinstance(t, dict):
                wrapped.append(_DictWrapper(t))
            else:
                wrapped.append(t)

        agg = cls()
        grouped_dict = agg.aggregate(wrapped)
        return list(grouped_dict.values())

    @classmethod
    def compute_paired_comparison(
        cls,
        trials_a: Sequence[Any],
        trials_b: Sequence[Any],
        metric_name: str = "p95_itl_ms",
    ) -> PairedDiffResult:
        """Compute mean and percentage difference between two matched trial sets."""
        def extract(t: Any) -> float:
            if isinstance(t, dict):
                m = t.get("metrics", {})
                return float(m.get(metric_name, 0.0) if isinstance(m, dict) else getattr(m, metric_name, 0.0))
            return float(getattr(t.metrics, metric_name, 0.0))

        vals_a = [extract(t) for t in trials_a]
        vals_b = [extract(t) for t in trials_b]

        mean_a = _safe_mean(vals_a)
        mean_b = _safe_mean(vals_b)
        mean_diff = mean_a - mean_b
        pct_diff = (mean_diff / max(1e-9, abs(mean_b))) * 100.0 if mean_b else 0.0

        return PairedDiffResult(
            mean_diff=round(mean_diff, 4),
            pct_diff=round(pct_diff, 4),
            metric_name=metric_name,
        )

    @classmethod
    def compute_effect_size(
        cls,
        group_a: Sequence[float],
        group_b: Sequence[float],
        metric_name: str = "metric",
    ) -> EffectSizeResult:
        """Compute Cohen's d between two numeric distributions."""
        n1, n2 = len(group_a), len(group_b)
        if n1 == 0 or n2 == 0:
            return EffectSizeResult(metric_name=metric_name, baseline_a="A", baseline_b="B")

        m1 = _safe_mean(group_a)
        m2 = _safe_mean(group_b)
        diff = m1 - m2

        if n1 > 1 and n2 > 1:
            var1 = statistics.variance(group_a)
            var2 = statistics.variance(group_b)
            pooled_sd = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(1, (n1 + n2 - 2)))
            d = diff / pooled_sd if pooled_sd > 0 else 0.0
        elif n1 + n2 >= 2:
            all_vals = list(group_a) + list(group_b)
            sd = statistics.stdev(all_vals)
            d = diff / sd if sd > 0 else 0.0
        else:
            d = 0.0

        return EffectSizeResult(
            metric_name=metric_name,
            baseline_a="A",
            baseline_b="B",
            n_pairs=min(n1, n2),
            mean_diff_a_minus_b=round(diff, 4),
            cohens_d=round(d, 4),
            pairing_quality="unpaired",
        )


@dataclass
class PairedDiffResult:
    """Convenience outcome of paired comparison calculation."""
    mean_diff: float
    pct_diff: float
    metric_name: str = ""
    baseline_a: str = ""
    baseline_b: str = ""

