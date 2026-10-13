"""
Statistical Analysis & Hypothesis Testing Layer (Module 12).

Provides robust, auditable statistical analysis for experimental trials:
- Descriptive statistics (Mean, Std, Median, IQR, Min, Max)
- Student's t Confidence Intervals (using exact scipy.stats Student-t distribution)
- Non-parametric Bootstrap Confidence Intervals for skewed distributions
- Matched Paired Hypothesis Tests (Paired t-test, Wilcoxon signed-rank)
- Standardized Effect Sizes (Cohen's d, Hedges' g correction, Cliff's delta)
- Multiple Testing Corrections (Holm-Bonferroni, Benjamini-Hochberg FDR)
- Strict small-sample safeguards (N < 3, N < 5) and zero-fabrication protections
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.stats as stats


@dataclass
class ConfidenceInterval:
    """Confidence interval for an estimand."""
    lower: Optional[float] = None
    upper: Optional[float] = None
    confidence_level: float = 0.95
    method: str = "student_t"  # "student_t" | "bootstrap_percentile" | "bootstrap_bca"
    is_valid: bool = False
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricSummaryStats:
    """
    Complete statistical summary for one metric across N independent repetitions.
    """
    metric_name: str
    n_total: int = 0
    n_valid: int = 0
    n_missing: int = 0
    mean: Optional[float] = None
    std: Optional[float] = None
    se: Optional[float] = None
    median: Optional[float] = None
    iqr: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    p50: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None
    ci_t: ConfidenceInterval = field(default_factory=ConfidenceInterval)
    ci_bootstrap: ConfidenceInterval = field(default_factory=ConfidenceInterval)
    skewness: Optional[float] = None
    small_sample_warning: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ci_t"] = self.ci_t.to_dict()
        d["ci_bootstrap"] = self.ci_bootstrap.to_dict()
        return d


@dataclass
class PairedComparisonResult:
    """
    Statistical result for a paired comparison between Policy A and Policy B
    evaluated under identical scenarios and matched random seeds.
    """
    metric_name: str
    policy_a: str
    policy_b: str
    n_pairs: int = 0
    mean_a: Optional[float] = None
    mean_b: Optional[float] = None
    mean_diff: Optional[float] = None     # A - B
    median_diff: Optional[float] = None   # Median(A - B)
    std_diff: Optional[float] = None
    ci_diff_t: ConfidenceInterval = field(default_factory=ConfidenceInterval)
    t_statistic: Optional[float] = None
    p_value_t: Optional[float] = None
    wilcoxon_statistic: Optional[float] = None
    p_value_wilcoxon: Optional[float] = None
    cohens_d: Optional[float] = None
    hedges_g: Optional[float] = None
    cliffs_delta: Optional[float] = None
    p_value_adjusted_holm: Optional[float] = None
    p_value_adjusted_fdr: Optional[float] = None
    is_statistically_significant: Optional[bool] = None  # p < alpha after correction
    practical_significance: str = "unclassified"  # "negligible" | "small" | "medium" | "large"
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ci_diff_t"] = self.ci_diff_t.to_dict()
        return d


class StatisticalAnalyzer:
    """
    Production-grade statistical estimation and hypothesis testing engine.
    """

    @staticmethod
    def clean_observations(values: Sequence[Any]) -> List[float]:
        """
        Filter out None, NaN, and non-numeric entries without zero-filling.
        """
        cleaned: List[float] = []
        for v in values:
            if v is None:
                continue
            try:
                fv = float(v)
                if not (math.isnan(fv) or math.isinf(fv)):
                    cleaned.append(fv)
            except (ValueError, TypeError):
                continue
        return cleaned

    @classmethod
    def compute_metric_stats(
        cls,
        metric_name: str,
        observations: Sequence[Any],
        confidence_level: float = 0.95,
        n_bootstraps: int = 2000,
        random_seed: int = 42,
    ) -> MetricSummaryStats:
        """
        Compute descriptive and inferential statistics for a single metric.
        Enforces small-sample safeguards (N < 3, N < 5).
        """
        n_total = len(observations)
        valid = cls.clean_observations(observations)
        n_valid = len(valid)
        n_missing = n_total - n_valid

        stats_obj = MetricSummaryStats(
            metric_name=metric_name,
            n_total=n_total,
            n_valid=n_valid,
            n_missing=n_missing,
        )

        if n_valid == 0:
            stats_obj.notes.append("No valid observations.")
            return stats_obj

        # Basic summaries
        arr = np.array(valid, dtype=np.float64)
        stats_obj.min_val = float(np.min(arr))
        stats_obj.max_val = float(np.max(arr))
        stats_obj.mean = float(np.mean(arr))
        stats_obj.median = float(np.median(arr))

        q25, q75 = np.percentile(arr, [25.0, 75.0])
        stats_obj.q25 = float(q25)
        stats_obj.q75 = float(q75)
        stats_obj.iqr = float(q75 - q25)

        stats_obj.p50 = float(np.percentile(arr, 50.0))
        stats_obj.p95 = float(np.percentile(arr, 95.0))
        stats_obj.p99 = float(np.percentile(arr, 99.0))

        if n_valid == 1:
            stats_obj.notes.append("Single observation: variance and confidence intervals are undefined.")
            return stats_obj

        # Sample standard deviation & standard error
        std_val = float(np.std(arr, ddof=1))
        stats_obj.std = std_val
        stats_obj.se = std_val / math.sqrt(n_valid)

        if n_valid >= 3 and std_val > 1e-9:
            try:
                stats_obj.skewness = float(stats.skew(arr))
            except Exception:
                stats_obj.skewness = None

        # Small sample flag
        if n_valid < 5:
            stats_obj.small_sample_warning = True
            stats_obj.notes.append(f"Small sample warning (N={n_valid} < 5): inference exhibits wide bounds.")

        # Student's t Confidence Interval
        if n_valid >= 2 and stats_obj.se is not None:
            df = n_valid - 1
            alpha = 1.0 - confidence_level
            # Exact Student-t critical value
            t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
            margin = t_crit * stats_obj.se
            stats_obj.ci_t = ConfidenceInterval(
                lower=stats_obj.mean - margin,
                upper=stats_obj.mean + margin,
                confidence_level=confidence_level,
                method="student_t",
                is_valid=True,
                warning="Small sample (N < 5)" if n_valid < 5 else None,
            )

        # Bootstrap Confidence Interval (percentile bootstrap)
        if n_valid >= 3:
            ci_boot = cls.bootstrap_ci_mean(
                arr, confidence_level=confidence_level, n_bootstraps=n_bootstraps, seed=random_seed
            )
            stats_obj.ci_bootstrap = ci_boot

        return stats_obj

    @classmethod
    def bootstrap_ci_mean(
        cls,
        data: np.ndarray,
        confidence_level: float = 0.95,
        n_bootstraps: int = 2000,
        seed: int = 42,
    ) -> ConfidenceInterval:
        """
        Compute non-parametric bootstrap confidence interval for sample mean.
        """
        n = len(data)
        if n < 3:
            return ConfidenceInterval(
                confidence_level=confidence_level,
                method="bootstrap_percentile",
                is_valid=False,
                warning=f"Insufficient sample size for bootstrap (N={n} < 3).",
            )

        rng = np.random.default_rng(seed)
        indices = rng.integers(0, n, size=(n_bootstraps, n))
        bootstrap_means = np.mean(data[indices], axis=1)

        alpha = 1.0 - confidence_level
        lo_pct = (alpha / 2.0) * 100.0
        hi_pct = (1.0 - alpha / 2.0) * 100.0

        lo = float(np.percentile(bootstrap_means, lo_pct))
        hi = float(np.percentile(bootstrap_means, hi_pct))

        return ConfidenceInterval(
            lower=lo,
            upper=hi,
            confidence_level=confidence_level,
            method="bootstrap_percentile",
            is_valid=True,
            warning="Small sample size (N < 5) for bootstrap" if n < 5 else None,
        )

    @classmethod
    def compare_paired(
        cls,
        metric_name: str,
        policy_a_name: str,
        values_a: Sequence[float],
        policy_b_name: str,
        values_b: Sequence[float],
        confidence_level: float = 0.95,
    ) -> PairedComparisonResult:
        """
        Compute matched paired comparison statistics between two policies.
        Pairing assumption: len(values_a) == len(values_b) with matched scenarios/seeds.
        """
        res = PairedComparisonResult(
            metric_name=metric_name,
            policy_a=policy_a_name,
            policy_b=policy_b_name,
        )

        n_pairs = min(len(values_a), len(values_b))
        if n_pairs == 0:
            res.warning = "Zero observations available for paired comparison."
            return res

        # Filter to pairs where both values are valid numbers
        pairs: List[Tuple[float, float]] = []
        for va, vb in zip(values_a[:n_pairs], values_b[:n_pairs]):
            if va is None or vb is None:
                continue
            try:
                fa, fb = float(va), float(vb)
                if not (math.isnan(fa) or math.isinf(fa) or math.isnan(fb) or math.isinf(fb)):
                    pairs.append((fa, fb))
            except (ValueError, TypeError):
                continue

        n_clean_pairs = len(pairs)
        res.n_pairs = n_clean_pairs

        if n_clean_pairs == 0:
            res.warning = "Zero valid matched pairs."
            return res

        arr_a = np.array([p[0] for p in pairs], dtype=np.float64)
        arr_b = np.array([p[1] for p in pairs], dtype=np.float64)
        diffs = arr_a - arr_b

        res.mean_a = float(np.mean(arr_a))
        res.mean_b = float(np.mean(arr_b))
        res.mean_diff = float(np.mean(diffs))
        res.median_diff = float(np.median(diffs))

        # Effect sizes: Cliff's delta (non-parametric ordinal overlap)
        res.cliffs_delta = cls.calculate_cliffs_delta(arr_a, arr_b)

        if n_clean_pairs < 2:
            res.warning = "Insufficient pairs for variance, t-test, and effect sizes (N < 2)."
            return res

        std_d = float(np.std(diffs, ddof=1))
        res.std_diff = std_d

        if std_d > 0.0:
            # Paired Cohen's d: mean_diff / std_diff
            d_val = res.mean_diff / std_d
            res.cohens_d = float(d_val)

            # Hedges' g correction factor for small samples: J = 1 - 3 / (4 * df - 1)
            df = n_clean_pairs - 1
            j_correction = 1.0 - (3.0 / (4.0 * df - 1.0)) if df > 1 else 1.0
            res.hedges_g = float(d_val * j_correction)

            # Classify practical significance magnitude (|g|)
            abs_g = abs(res.hedges_g)
            if abs_g < 0.2:
                res.practical_significance = "negligible"
            elif abs_g < 0.5:
                res.practical_significance = "small"
            elif abs_g < 0.8:
                res.practical_significance = "medium"
            else:
                res.practical_significance = "large"
        else:
            res.cohens_d = 0.0
            res.hedges_g = 0.0
            res.practical_significance = "negligible"

        # Paired Student's t-test
        se_d = std_d / math.sqrt(n_clean_pairs)
        if se_d > 0.0:
            df = n_clean_pairs - 1
            t_stat = res.mean_diff / se_d
            res.t_statistic = float(t_stat)
            # Two-tailed p-value
            res.p_value_t = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df)))

            alpha = 1.0 - confidence_level
            t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
            res.ci_diff_t = ConfidenceInterval(
                lower=res.mean_diff - t_crit * se_d,
                upper=res.mean_diff + t_crit * se_d,
                confidence_level=confidence_level,
                method="paired_student_t",
                is_valid=True,
            )
        else:
            res.t_statistic = 0.0
            res.p_value_t = 1.0
            res.ci_diff_t = ConfidenceInterval(
                lower=res.mean_diff,
                upper=res.mean_diff,
                confidence_level=confidence_level,
                method="paired_student_t",
                is_valid=True,
            )

        # Wilcoxon signed-rank test (requires N >= 5 for meaningful non-parametric power)
        if n_clean_pairs >= 5 and np.any(diffs != 0.0):
            try:
                w_stat, w_p = stats.wilcoxon(diffs, zero_method="wilcox", correction=True)
                res.wilcoxon_statistic = float(w_stat)
                res.p_value_wilcoxon = float(w_p)
            except Exception as e:
                res.notes = [f"Wilcoxon test skipped: {e}"]

        if n_clean_pairs < 5:
            res.warning = f"Small paired sample size (N={n_clean_pairs} < 5): inference is exploratory."

        return res

    @staticmethod
    def calculate_cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate Cliff's delta non-parametric effect size:
        d = (#(a_i > b_j) - #(a_i < b_j)) / (n_a * n_b)
        """
        na, nb = len(a), len(b)
        if na == 0 or nb == 0:
            return 0.0

        # Efficient matrix calculation
        greater = np.sum(a[:, np.newaxis] > b[np.newaxis, :])
        lesser = np.sum(a[:, np.newaxis] < b[np.newaxis, :])
        return float((greater - lesser) / (na * nb))

    @staticmethod
    def adjust_p_values_holm(p_values: Sequence[float]) -> List[float]:
        """
        Adjust p-values using Holm-Bonferroni step-down multiple testing correction.
        Controls Family-Wise Error Rate (FWER) under arbitrary dependence.
        """
        m = len(p_values)
        if m <= 1:
            return [float(p) for p in p_values]

        # Sort indices by original p-value
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        adjusted_indexed: List[Tuple[int, float]] = []

        running_max = 0.0
        for rank, (orig_idx, p) in enumerate(indexed):
            # Holm multiplier: (m - rank)
            multiplier = m - rank
            p_adj = min(1.0, p * multiplier)
            # Enforce monotonicity: adjusted p_i >= adjusted p_{i-1}
            p_adj = max(running_max, p_adj)
            running_max = p_adj
            adjusted_indexed.append((orig_idx, float(p_adj)))

        # Restore original order
        adjusted_indexed.sort(key=lambda x: x[0])
        return [p for _, p in adjusted_indexed]

    @staticmethod
    def adjust_p_values_fdr(p_values: Sequence[float]) -> List[float]:
        """
        Adjust p-values using Benjamini-Hochberg False Discovery Rate (FDR) procedure.
        """
        m = len(p_values)
        if m <= 1:
            return [float(p) for p in p_values]

        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        adjusted_indexed: List[Tuple[int, float]] = []

        # Step-up adjustment: q_i = min_{k >= i} (p_k * m / rank_k)
        raw_adjusted = []
        for rank_zero, (orig_idx, p) in enumerate(indexed):
            rank = rank_zero + 1
            q = min(1.0, p * m / rank)
            raw_adjusted.append((orig_idx, q))

        # Enforce step-up monotonicity in reverse
        running_min = 1.0
        final_adjusted: List[Tuple[int, float]] = []
        for orig_idx, q in reversed(raw_adjusted):
            q_mono = min(running_min, q)
            running_min = q_mono
            final_adjusted.append((orig_idx, float(q_mono)))

        final_adjusted.sort(key=lambda x: x[0])
        return [q for _, q in final_adjusted]
