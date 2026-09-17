"""
Scientific visualization functions for Module 10 experimental evaluation.

Implements 12 publication-ready plots:
  1.  plot_itl_timeline                 — ITL over token generation steps
  2.  plot_bandwidth_and_memory_trace   — trace environmental dynamics
  3.  plot_baseline_comparison_bar      — metric comparison across B1-B5 with error bars
  4.  plot_p95_vs_overhead_scatter      — Pareto trade-off: P95 ITL vs overhead
  5.  plot_migration_time_cdf           — empirical CDF of migration durations
  6.  plot_ablation_waterfall           — component ablation impact (A1-A8)
  7.  plot_sensitivity_threshold_sweep  — switching threshold trade-off curve
  8.  plot_sensitivity_horizon_sweep    — forecast horizon sensitivity
  9.  plot_partition_state_strip        — partition timeline Gantt / strip chart
  10. plot_slo_violation_rate           — SLO violation rate across scenarios
  11. plot_forecast_vs_actual           — predicted vs actual telemetry points
  12. plot_multi_scenario_radar         — multi-scenario baseline footprint

All functions gracefully handle the absence of `matplotlib` by returning None
with an informative warning, ensuring full headless compatibility.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Check matplotlib availability
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive headless backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    ticker = None
    MATPLOTLIB_AVAILABLE = False


def _check_mpl() -> bool:
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib is not installed; skipping plot generation.")
        return False
    return True


# Distinct neutral color palette for scientific publication (no winner bias)
PALETTE = {
    "static": "#6c757d",           # Neutral Slate Gray
    "network_reactive": "#17a2b8",  # Cyan/Teal
    "memory_reactive": "#fd7e14",   # Amber/Orange
    "joint_reactive": "#6f42c1",    # Purple
    "predictive": "#007bff",        # Blue
    "bandwidth": "#28a745",        # Green
    "memory": "#e83e8c",           # Rose
    "slo": "#dc3545",              # Crimson
}


# ---------------------------------------------------------------------------
# 1. ITL Timeline
# ---------------------------------------------------------------------------

def plot_itl_timeline(
    trials: Sequence[Dict[str, Any]],
    output_path: Path | str,
    slo_threshold_ms: float = 200.0,
) -> Optional[Path]:
    """Plot Inter-Token Latency (ITL) across token decoding steps for each baseline."""
    if not _check_mpl():
        return None

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    for t in trials:
        b_id = t.get("baseline_id", "unknown")
        trace = t.get("runtime_trace", {})
        tokens = trace.get("token_records", [])
        if not tokens:
            continue
        xs = [tok.get("token_index", i) for i, tok in enumerate(tokens) if not tok.get("is_ttft", False)]
        ys = [tok.get("step_latency_ms", 0.0) for tok in tokens if not tok.get("is_ttft", False)]
        color = PALETTE.get(b_id, "#333333")
        ax.plot(xs, ys, label=b_id.replace("_", " ").title(), color=color, alpha=0.85, linewidth=1.8)

    ax.axhline(y=slo_threshold_ms, color=PALETTE["slo"], linestyle="--", linewidth=1.5, label=f"SLO Target ({slo_threshold_ms:.0f} ms)")
    ax.set_xlabel("Generated Token Index", fontsize=11)
    ax.set_ylabel("Inter-Token Latency (ms)", fontsize=11)
    ax.set_title("Inter-Token Latency Dynamics Across Generation Steps", fontsize=12, pad=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 2. Bandwidth & Memory Trace
# ---------------------------------------------------------------------------

def plot_bandwidth_and_memory_trace(
    trace_data: Dict[str, Any],
    output_path: Path | str,
) -> Optional[Path]:
    """Plot environmental dynamics (bandwidth and memory pressure) over time."""
    if not _check_mpl():
        return None

    steps = trace_data.get("steps", [])
    if not steps:
        return None

    xs = [s.get("step_idx", i) for i, s in enumerate(steps)]
    bw = [s.get("bandwidth_mbps", 0.0) for s in steps]
    mem = [s.get("memory_headroom_fraction", 1.0) for s in steps]

    fig, ax1 = plt.subplots(figsize=(9, 4.5), dpi=150)
    color1 = PALETTE["bandwidth"]
    ax1.set_xlabel("Trace Step Index", fontsize=11)
    ax1.set_ylabel("Bandwidth (Mbps)", color=color1, fontsize=11)
    ax1.plot(xs, bw, color=color1, linewidth=2.0, label="Bandwidth (Mbps)")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = ax1.twinx()
    color2 = PALETTE["memory"]
    ax2.set_ylabel("Memory Headroom Fraction", color=color2, fontsize=11)
    ax2.plot(xs, mem, color=color2, linewidth=2.0, linestyle="--", label="Memory Headroom")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0.0, 1.05)

    plt.title(f"Environment Trace Dynamics: {trace_data.get('scenario_id', '')}", fontsize=12, pad=10)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 3. Baseline Comparison Bar Chart
# ---------------------------------------------------------------------------

def plot_baseline_comparison_bar(
    aggregated_results: Sequence[Dict[str, Any]],
    metric_name: str,
    output_path: Path | str,
    ylabel: Optional[str] = None,
) -> Optional[Path]:
    """Bar chart comparing all baselines on a selected metric with standard deviation bars."""
    if not _check_mpl():
        return None

    labels = []
    means = []
    stds = []
    colors = []

    for agg in aggregated_results:
        b_id = agg.get("baseline_id", "")
        m_stat = agg.get("metrics", {}).get(metric_name, {})
        mean = m_stat.get("mean", 0.0)
        std = m_stat.get("std", 0.0)
        labels.append(b_id.replace("_", "\n"))
        means.append(mean)
        stds.append(std)
        colors.append(PALETTE.get(b_id, "#555555"))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="#333333", width=0.55)

    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.annotate(f"{mean:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    ax.set_ylabel(ylabel or metric_name.replace("_", " ").title(), fontsize=11)
    ax.set_title(f"Baseline Comparison: {metric_name.replace('_', ' ').title()}", fontsize=12, pad=10)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 4. P95 vs Overhead Scatter
# ---------------------------------------------------------------------------

def plot_p95_vs_overhead_scatter(
    aggregated_results: Sequence[Dict[str, Any]],
    output_path: Path | str,
) -> Optional[Path]:
    """Scatter plot showing P95 ITL vs Control Overhead Fraction trade-off."""
    if not _check_mpl():
        return None

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    for agg in aggregated_results:
        b_id = agg.get("baseline_id", "")
        p95 = agg.get("metrics", {}).get("p95_itl_ms", {}).get("mean", 0.0)
        oh = agg.get("metrics", {}).get("control_overhead_fraction", {}).get("mean", 0.0) * 100.0
        color = PALETTE.get(b_id, "#333333")
        ax.scatter(oh, p95, color=color, s=120, edgecolors="#222222", label=b_id.replace("_", " ").title(), zorder=5)
        ax.annotate(b_id, (oh + 0.05, p95 + 1.0), fontsize=9)

    ax.set_xlabel("Control Plane Overhead (%)", fontsize=11)
    ax.set_ylabel("P95 Inter-Token Latency (ms)", fontsize=11)
    ax.set_title("Pareto Trade-off: P95 Latency vs Control Overhead", fontsize=12, pad=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 5. Migration Time CDF
# ---------------------------------------------------------------------------

def plot_migration_time_cdf(
    trials: Sequence[Dict[str, Any]],
    output_path: Path | str,
) -> Optional[Path]:
    """Empirical CDF of physical partition migration durations across baselines."""
    if not _check_mpl():
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    for t in trials:
        b_id = t.get("baseline_id", "")
        migs = t.get("runtime_trace", {}).get("migration_records", [])
        durs = sorted([m.get("total_duration_ms", 0.0) for m in migs if not m.get("rolled_back", False)])
        if not durs:
            continue
        n = len(durs)
        ys = [(i + 1) / n for i in range(n)]
        ax.step(durs, ys, where="post", label=b_id.replace("_", " ").title(),
                color=PALETTE.get(b_id, "#333333"), linewidth=2.0)

    ax.set_xlabel("Migration Duration (ms)", fontsize=11)
    ax.set_ylabel("Empirical Cumulative Probability", fontsize=11)
    ax.set_title("Empirical Distribution of Migration Times", fontsize=12, pad=10)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 6. Ablation Waterfall Chart
# ---------------------------------------------------------------------------

def plot_ablation_waterfall(
    ablation_deltas: Dict[str, float],
    output_path: Path | str,
    baseline_value: float = 150.0,
    metric_label: str = "P95 ITL Delta (ms)",
) -> Optional[Path]:
    """Waterfall plot displaying the degradation/delta from removing components A1-A8."""
    if not _check_mpl():
        return None

    names = list(ablation_deltas.keys())
    deltas = list(ablation_deltas.values())

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    colors = ["#dc3545" if d > 0 else "#28a745" for d in deltas]

    ax.barh([n.replace("_", " ").title() for n in names], deltas, color=colors, alpha=0.85, edgecolor="#333333")
    ax.axvline(x=0.0, color="#444444", linewidth=1.2)
    ax.set_xlabel(metric_label, fontsize=11)
    ax.set_title("Component Ablation Sensitivity Analysis (Impact vs Baseline)", fontsize=12, pad=10)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 7. Sensitivity Threshold Sweep
# ---------------------------------------------------------------------------

def plot_sensitivity_threshold_sweep(
    thresholds: Sequence[float],
    p95_latencies: Sequence[float],
    switch_counts: Sequence[int],
    output_path: Path | str,
) -> Optional[Path]:
    """Switching threshold (theta) sensitivity trade-off: responsiveness vs stability."""
    if not _check_mpl():
        return None

    fig, ax1 = plt.subplots(figsize=(8.5, 4.5), dpi=150)

    color1 = "#007bff"
    ax1.set_xlabel("Switching Cost Threshold (theta)", fontsize=11)
    ax1.set_ylabel("P95 ITL (ms)", color=color1, fontsize=11)
    ax1.plot(thresholds, p95_latencies, marker="o", color=color1, linewidth=2.0, label="P95 Latency")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    color2 = "#e83e8c"
    ax2.set_ylabel("Total Partition Switches", color=color2, fontsize=11)
    ax2.plot(thresholds, switch_counts, marker="s", linestyle="--", color=color2, linewidth=2.0, label="Switches")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Hysteresis & Threshold Sensitivity: Latency vs Switching Frequency", fontsize=12, pad=10)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 8. Sensitivity Horizon Sweep
# ---------------------------------------------------------------------------

def plot_sensitivity_horizon_sweep(
    horizons: Sequence[int],
    p95_latencies: Sequence[float],
    proactive_switches: Sequence[int],
    output_path: Path | str,
) -> Optional[Path]:
    """Forecast horizon (H) sensitivity curve."""
    if not _check_mpl():
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(horizons, p95_latencies, marker="^", color="#28a745", linewidth=2.0, label="P95 ITL (ms)")
    ax.set_xlabel("Forecast Horizon Steps (H)", fontsize=11)
    ax.set_ylabel("P95 Inter-Token Latency (ms)", fontsize=11)
    ax.set_title("Forecast Horizon Sensitivity Analysis", fontsize=12, pad=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 9. Partition State Timeline Strip
# ---------------------------------------------------------------------------

def plot_partition_state_strip(
    trial_data: Dict[str, Any],
    output_path: Path | str,
) -> Optional[Path]:
    """Strip chart illustrating active partition plan across token generation steps."""
    if not _check_mpl():
        return None

    tokens = trial_data.get("runtime_trace", {}).get("token_records", [])
    if not tokens:
        return None

    steps = [tok.get("token_index", i) for i, tok in enumerate(tokens)]
    plans = [str(tok.get("active_plan_id", "plan_0")) for tok in tokens]
    unique_plans = sorted(list(set(plans)))
    plan_to_y = {p: i for i, p in enumerate(unique_plans)}
    ys = [plan_to_y[p] for p in plans]

    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=150)
    ax.step(steps, ys, where="post", color="#007bff", linewidth=2.5)
    ax.scatter(steps, ys, color="#0056b3", s=25, zorder=5)

    ax.set_yticks(range(len(unique_plans)))
    ax.set_yticklabels([f"P{i}: {p[:18]}" for i, p in enumerate(unique_plans)], fontsize=9)
    ax.set_xlabel("Token Decoding Step", fontsize=11)
    ax.set_ylabel("Active Partition", fontsize=11)
    ax.set_title(f"Partition Adaptation Trajectory: {trial_data.get('baseline_id', '')}", fontsize=12, pad=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 10. SLO Violation Rate Bar Chart
# ---------------------------------------------------------------------------

def plot_slo_violation_rate(
    aggregated_results: Sequence[Dict[str, Any]],
    output_path: Path | str,
) -> Optional[Path]:
    """Compare SLO violation rates across baselines."""
    if not _check_mpl():
        return None

    labels = []
    rates = []
    colors = []

    for agg in aggregated_results:
        b_id = agg.get("baseline_id", "")
        rate_stat = agg.get("metrics", {}).get("slo_violation_rate", {})
        rate = rate_stat.get("mean", 0.0) * 100.0
        labels.append(b_id.replace("_", "\n"))
        rates.append(rate)
        colors.append(PALETTE.get(b_id, "#555555"))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    bars = ax.bar(labels, rates, color=colors, alpha=0.85, edgecolor="#333333", width=0.55)

    for bar, r in zip(bars, rates):
        ax.annotate(f"{r:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("SLO Violation Rate (%)", fontsize=11)
    ax.set_title("SLO Violation Rate Across Control Policies", fontsize=12, pad=10)
    ax.set_ylim(0.0, max(rates + [10.0]) * 1.15)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 11. Forecast vs Actual Telemetry
# ---------------------------------------------------------------------------

def plot_forecast_vs_actual(
    actuals: Sequence[float],
    forecasts: Sequence[float],
    output_path: Path | str,
    feature_name: str = "Bandwidth (Mbps)",
) -> Optional[Path]:
    """Plot realized observation trajectory against multi-step forecasts."""
    if not _check_mpl():
        return None

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    steps = range(len(actuals))
    ax.plot(steps, actuals, label="Actual Realized", color="#333333", linewidth=2.0)
    ax.plot(steps, forecasts, label="Predictor Forecast", color="#007bff", linestyle="--", linewidth=1.8)

    ax.set_xlabel("Observation Step", fontsize=11)
    ax.set_ylabel(feature_name, fontsize=11)
    ax.set_title(f"Predictive Forecasting Accuracy: {feature_name}", fontsize=12, pad=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 12. Multi-Scenario Footprint
# ---------------------------------------------------------------------------

def plot_multi_scenario_radar(
    scenario_metrics: Dict[str, Dict[str, float]],
    output_path: Path | str,
) -> Optional[Path]:
    """Compare multi-scenario performance as a grouped bar chart."""
    if not _check_mpl():
        return None

    scenarios = list(scenario_metrics.keys())
    if not scenarios:
        return None

    baselines = sorted(list(next(iter(scenario_metrics.values())).keys()))
    x = range(len(scenarios))
    width = 0.8 / max(1, len(baselines))

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for i, b in enumerate(baselines):
        vals = [scenario_metrics[sc].get(b, 0.0) for sc in scenarios]
        ax.bar([pos + i * width for pos in x], vals, width=width,
               label=b.replace("_", " ").title(), color=PALETTE.get(b, "#777777"), alpha=0.85)

    ax.set_xticks([pos + width * (len(baselines) - 1) / 2 for pos in x])
    ax.set_xticklabels([sc.replace("_", "\n") for sc in scenarios], fontsize=9)
    ax.set_ylabel("P95 ITL (ms)", fontsize=11)
    ax.set_title("Cross-Scenario Policy Robustness", fontsize=12, pad=10)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
