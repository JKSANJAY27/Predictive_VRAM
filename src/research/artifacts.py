"""
Publication Artifacts Generator (Module 12).

Produces publication-ready scientific figures and structured tables suitable
for technical papers and research reports from raw experimental trial data.
Adheres strictly to academic visualization standards:
- Non-truncated axes with explicit zero baselines where appropriate
- Fully labeled axes with physical measurement units
- 95% confidence intervals and dispersion bars
- High-contrast, colorblind-friendly neutral scientific palette
- Machine-readable CSV and Markdown tables with sample sizes and provenance
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.research.comparisons import PolicyComparisonEngine
from src.research.manifest import CANONICAL_METRIC_DEFINITIONS
from src.research.statistics import MetricSummaryStats, StatisticalAnalyzer

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    ticker = None
    MATPLOTLIB_AVAILABLE = False


# Colorblind-safe publication palette
COLOR_PALETTE = {
    "static": "#5c636a",           # Neutral Charcoal Gray
    "network_reactive": "#0d6efd",  # Royal Blue
    "memory_reactive": "#fd7e14",   # Orange
    "joint_reactive": "#6f42c1",    # Purple
    "predictive": "#198754",        # Forest Green
    "accent_red": "#dc3545",        # Crimson Red
    "accent_teal": "#0dcaf0",       # Teal
}


class PublicationArtifactGenerator:
    """
    Generates publication figures and summary tables from saved raw trial datasets.
    """

    @classmethod
    def generate_all_artifacts(
        cls,
        trials_df: pd.DataFrame,
        output_dir: Path | str,
        comparisons_df: Optional[pd.DataFrame] = None,
        ablation_df: Optional[pd.DataFrame] = None,
        sensitivity_df: Optional[pd.DataFrame] = None,
        is_cpu_only: bool = True,
    ) -> Dict[str, Path]:
        """
        Produce all figures, tables, and provenance datasets into the output directory.
        """
        out_path = Path(output_dir)
        figures_dir = out_path / "figures"
        tables_dir = out_path / "tables"
        figures_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        generated: Dict[str, Path] = {}

        # 1. Summary Metrics Table (CSV & Markdown)
        t1_csv = tables_dir / "table1_policy_summary.csv"
        t1_md = tables_dir / "table1_policy_summary.md"
        cls.generate_policy_summary_table(trials_df, t1_csv, t1_md, is_cpu_only=is_cpu_only)
        generated["tables/table1_policy_summary.csv"] = t1_csv
        generated["tables/table1_policy_summary.md"] = t1_md

        # 2. Paired Comparisons Table
        if comparisons_df is not None and len(comparisons_df) > 0:
            t2_csv = tables_dir / "table2_paired_comparisons.csv"
            t2_md = tables_dir / "table2_paired_comparisons.md"
            cls.generate_comparisons_table(comparisons_df, t2_csv, t2_md)
            generated["tables/table2_paired_comparisons.csv"] = t2_csv
            generated["tables/table2_paired_comparisons.md"] = t2_md

        # 3. Ablation Table
        if ablation_df is not None and len(ablation_df) > 0:
            t3_csv = tables_dir / "table3_ablation_summary.csv"
            t3_md = tables_dir / "table3_ablation_summary.md"
            cls.generate_ablation_table(ablation_df, t3_csv, t3_md)
            generated["tables/table3_ablation_summary.csv"] = t3_csv
            generated["tables/table3_ablation_summary.md"] = t3_md

        # 4. Provenance Matrix Table
        t4_csv = tables_dir / "table4_provenance_matrix.csv"
        t4_md = tables_dir / "table4_provenance_matrix.md"
        cls.generate_provenance_matrix(t4_csv, t4_md, is_cpu_only=is_cpu_only)
        generated["tables/table4_provenance_matrix.csv"] = t4_csv
        generated["tables/table4_provenance_matrix.md"] = t4_md

        # 5. Scientific Figures (if matplotlib available)
        if MATPLOTLIB_AVAILABLE:
            fig1 = figures_dir / "fig1_latency_distribution.png"
            if cls.plot_latency_distribution(trials_df, fig1):
                generated["figures/fig1_latency_distribution.png"] = fig1

            fig2 = figures_dir / "fig2_throughput_comparison.png"
            if cls.plot_throughput_comparison(trials_df, fig2):
                generated["figures/fig2_throughput_comparison.png"] = fig2

            fig3 = figures_dir / "fig3_switching_and_overhead.png"
            if cls.plot_switching_and_overhead(trials_df, fig3):
                generated["figures/fig3_switching_and_overhead.png"] = fig3

            if ablation_df is not None and len(ablation_df) > 0:
                fig6 = figures_dir / "fig6_ablation_waterfall.png"
                if cls.plot_ablation_waterfall(ablation_df, fig6):
                    generated["figures/fig6_ablation_waterfall.png"] = fig6

            if sensitivity_df is not None and len(sensitivity_df) > 0:
                fig7 = figures_dir / "fig7_sensitivity_sweeps.png"
                if cls.plot_sensitivity_sweeps(sensitivity_df, fig7):
                    generated["figures/fig7_sensitivity_sweeps.png"] = fig7

        return generated

    # --- TABLE GENERATION ---

    @classmethod
    def generate_policy_summary_table(
        cls,
        df: pd.DataFrame,
        csv_path: Path,
        md_path: Path,
        is_cpu_only: bool = True,
    ) -> None:
        """
        Aggregate metric summaries across policies with mean, 95% CI, median, IQR, and N.
        """
        target_metrics = [
            "mean_itl_ms", "median_itl_ms", "p95_itl_ms", "tokens_per_second",
            "total_switches", "total_migration_time_ms", "total_transferred_mb",
            "total_control_overhead_ms", "peak_memory_pressure", "peak_vram_mb", "host_ram_mb",
        ]

        rows = []
        for baseline in sorted(df["baseline_id"].unique()):
            b_df = df[df["baseline_id"] == baseline]
            n_reps = len(b_df)

            for m in target_metrics:
                if m not in b_df.columns:
                    continue

                stats_m = StatisticalAnalyzer.compute_metric_stats(m, b_df[m])
                def_m = CANONICAL_METRIC_DEFINITIONS.get(m, {})
                unit = def_m.get("unit", "units")

                # Format formatted string: Mean (95% CI)
                if stats_m.ci_t.is_valid and stats_m.ci_t.lower is not None and stats_m.ci_t.upper is not None:
                    ci_str = f"[{stats_m.ci_t.lower:.2f}, {stats_m.ci_t.upper:.2f}]"
                else:
                    ci_str = "N/A"

                iqr_str = f"[{stats_m.q25:.2f}, {stats_m.q75:.2f}]" if stats_m.q25 is not None else "N/A"

                rows.append({
                    "Policy": baseline,
                    "Metric": m,
                    "Unit": unit,
                    "N": stats_m.n_valid,
                    "Mean": round(stats_m.mean, 2) if stats_m.mean is not None else None,
                    "95% CI": ci_str,
                    "Median": round(stats_m.median, 2) if stats_m.median is not None else None,
                    "IQR [Q25, Q75]": iqr_str,
                    "Std Dev": round(stats_m.std, 2) if stats_m.std is not None else None,
                    "Min": round(stats_m.min_val, 2) if stats_m.min_val is not None else None,
                    "Max": round(stats_m.max_val, 2) if stats_m.max_val is not None else None,
                })

        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(csv_path, index=False)

        # Markdown format with notes
        md_content = ["# Table 1: Comprehensive Inference Policy Summary\n",
                      "| Policy | Metric | Unit | N | Mean | 95% CI | Median | IQR | Std Dev |",
                      "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"]
        for _, r in summary_df.iterrows():
            mean_val = f"{r['Mean']:.2f}" if r["Mean"] is not None else "UNAVAILABLE"
            med_val = f"{r['Median']:.2f}" if r["Median"] is not None else "UNAVAILABLE"
            std_val = f"{r['Std Dev']:.2f}" if r["Std Dev"] is not None else "N/A"
            md_content.append(
                f"| {r['Policy']} | `{r['Metric']}` | {r['Unit']} | {r['N']} | {mean_val} | {r['95% CI']} | {med_val} | {r['IQR [Q25, Q75]']} | {std_val} |"
            )
        if is_cpu_only:
            md_content.append("\n*Note: Physical GPU VRAM is explicitly UNAVAILABLE (value=None) in CPU environments.*")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content) + "\n")

    @classmethod
    def generate_comparisons_table(cls, cmp_df: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
        """Export pairwise policy comparisons table."""
        cmp_df.to_csv(csv_path, index=False)

        md_content = ["# Table 2: Paired Hypothesis Testing & Effect Sizes\n",
                      "| Policy Pair (A vs B) | Metric | N Pairs | Mean Diff (A-B) | 95% CI | Raw p | Holm p | Hedges' g | Significance |",
                      "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
        for _, r in cmp_df.iterrows():
            pair_name = f"{r['policy_a']} vs {r['policy_b']}"
            ci_s = f"[{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]" if pd.notnull(r["ci_lower"]) else "N/A"
            diff_s = f"{r['mean_diff_a_minus_b']:.2f}" if pd.notnull(r["mean_diff_a_minus_b"]) else "N/A"
            raw_p = f"{r['p_value_raw']:.4f}" if pd.notnull(r["p_value_raw"]) else "N/A"
            holm_p = f"{r['p_value_holm']:.4f}" if pd.notnull(r["p_value_holm"]) else "N/A"
            g_s = f"{r['hedges_g']:.2f} ({r['practical_significance']})" if pd.notnull(r["hedges_g"]) else "N/A"
            sig_s = "**Significant**" if r.get("is_significant") else "Not Significant"

            md_content.append(
                f"| {pair_name} | `{r['metric_name']}` | {r['n_pairs']} | {diff_s} | {ci_s} | {raw_p} | {holm_p} | {g_s} | {sig_s} |"
            )

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content) + "\n")

    @classmethod
    def generate_ablation_table(cls, abl_df: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
        """Export ablation impact summary table."""
        abl_df.to_csv(csv_path, index=False)

        md_content = ["# Table 3: Mechanism Ablation Impact Analysis\n",
                      "| Ablation ID | Mechanism Target | Metric | Delta vs Reference | 95% CI | Hedges' g | Statistical Conclusion |",
                      "| :--- | :--- | :--- | :---: | :---: | :---: | :--- |"]
        for _, r in abl_df.iterrows():
            ci_s = f"[{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]" if pd.notnull(r.get("ci_lower")) else "N/A"
            diff_s = f"{r['mean_diff']:.2f}" if pd.notnull(r.get("mean_diff")) else "N/A"
            g_s = f"{r['hedges_g']:.2f}" if pd.notnull(r.get("hedges_g")) else "N/A"
            concl = r.get("conclusion", "Observed variation within margin of error.")

            md_content.append(
                f"| {r.get('ablation_id', '')} | {r.get('mechanism', '')} | `{r.get('metric', '')}` | {diff_s} | {ci_s} | {g_s} | {concl} |"
            )

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content) + "\n")

    @classmethod
    def generate_provenance_matrix(cls, csv_path: Path, md_path: Path, is_cpu_only: bool = True) -> None:
        """Export explicit measurement provenance classifications."""
        matrix = [
            {"Observation Category": "Inference Latency (ITL, TTFT)", "Platform Classification": "MEASURED", "Method": "Local high-resolution monotonic timer on executor runtime."},
            {"Observation Category": "Host RAM Utilization", "Platform Classification": "MEASURED", "Method": "Host OS memory query via psutil.virtual_memory()."},
            {"Observation Category": "Physical GPU VRAM", "Platform Classification": "UNAVAILABLE" if is_cpu_only else "MEASURED", "Method": "PyTorch CUDA mem_get_info() where GPU exists; None on CPU."},
            {"Observation Category": "6G Sub-THz / High Mobility Profiles", "Platform Classification": "EMULATED", "Method": "Discrete profile parameter replay with deterministic synthetic link dynamics."},
            {"Observation Category": "Loopback TCP Socket Transfer", "Platform Classification": "MEASURED (Loopback)", "Method": "Physical OS socket transfer on 127.0.0.1; does not reflect cellular WAN."},
            {"Observation Category": "Modeled Energy Consumption", "Platform Classification": "ESTIMATED", "Method": "Proxy power coefficients per tier multiplied by execution duration."},
        ]

        p_df = pd.DataFrame(matrix)
        p_df.to_csv(csv_path, index=False)

        md_content = ["# Table 4: Measurement Provenance & Evidence Quality Matrix\n",
                      "| Observation Category | Provenance Status | Measurement Methodology |",
                      "| :--- | :---: | :--- |"]
        for _, r in p_df.iterrows():
            md_content.append(f"| {r['Observation Category']} | `{r['Platform Classification']}` | {r['Method']} |")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content) + "\n")

    # --- SCIENTIFIC PLOTS ---

    @classmethod
    def plot_latency_distribution(cls, df: pd.DataFrame, out_path: Path) -> bool:
        """Plot ITL distributions across policies with error bars and boxplots."""
        if not MATPLOTLIB_AVAILABLE or "mean_itl_ms" not in df.columns:
            return False

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        policies = sorted(df["baseline_id"].unique())
        data = [df[df["baseline_id"] == p]["mean_itl_ms"].dropna().tolist() for p in policies]

        # Boxplot with mean markers
        bp = ax.boxplot(
            data,
            tick_labels=policies,
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "D", "markeredgecolor": "black", "markerfacecolor": "gold"},
        )

        for patch, p in zip(bp["boxes"], policies):
            patch.set_facecolor(COLOR_PALETTE.get(p, "#cccccc"))
            patch.set_alpha(0.7)

        ax.set_ylabel("Mean Inter-Token Latency (ms)", fontsize=11, fontweight="bold")
        ax.set_title("Figure 1: Inter-Token Latency Distribution Across Policies", fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_ylim(bottom=0.0)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return True

    @classmethod
    def plot_throughput_comparison(cls, df: pd.DataFrame, out_path: Path) -> bool:
        """Plot effective generation throughput (tokens/sec) with 95% CIs."""
        if not MATPLOTLIB_AVAILABLE or "tokens_per_second" not in df.columns:
            return False

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
        policies = sorted(df["baseline_id"].unique())

        means = []
        errors = []
        colors = []

        for p in policies:
            sub = df[df["baseline_id"] == p]["tokens_per_second"].dropna()
            st = StatisticalAnalyzer.compute_metric_stats("tokens_per_second", sub)
            means.append(st.mean or 0.0)
            if st.ci_t.is_valid and st.ci_t.upper is not None and st.mean is not None:
                errors.append(st.ci_t.upper - st.mean)
            else:
                errors.append(0.0)
            colors.append(COLOR_PALETTE.get(p, "#0d6efd"))

        bars = ax.bar(policies, means, yerr=errors, capsize=5, color=colors, alpha=0.85, edgecolor="black")
        ax.set_ylabel("Throughput (Tokens / Second)", fontsize=11, fontweight="bold")
        ax.set_title("Figure 2: Effective Generation Throughput with 95% CIs", fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_ylim(bottom=0.0)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return True

    @classmethod
    def plot_switching_and_overhead(cls, df: pd.DataFrame, out_path: Path) -> bool:
        """Scatter/bar plot showing partition switches versus cumulative migration overhead."""
        if not MATPLOTLIB_AVAILABLE or "total_switches" not in df.columns or "total_control_overhead_ms" not in df.columns:
            return False

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
        policies = sorted(df["baseline_id"].unique())

        for p in policies:
            sub = df[df["baseline_id"] == p]
            ax.scatter(
                sub["total_switches"],
                sub["total_control_overhead_ms"],
                label=p,
                color=COLOR_PALETTE.get(p, "#6c757d"),
                s=80,
                alpha=0.75,
                edgecolors="black",
            )

        ax.set_xlabel("Partition Switch Count", fontsize=11, fontweight="bold")
        ax.set_ylabel("Control Overhead (ms)", fontsize=11, fontweight="bold")
        ax.set_title("Figure 3: Partition Transitions vs Controller Overhead", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(title="Policy", frameon=True)
        ax.set_xlim(left=-0.5)
        ax.set_ylim(bottom=0.0)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return True

    @classmethod
    def plot_ablation_waterfall(cls, abl_df: pd.DataFrame, out_path: Path) -> bool:
        """Waterfall bar chart of metric deltas across ablations relative to reference baseline."""
        if not MATPLOTLIB_AVAILABLE or "mean_diff" not in abl_df.columns:
            return False

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        labels = abl_df.get("ablation_id", [f"A{i}" for i in range(len(abl_df))])
        deltas = abl_df["mean_diff"]

        colors = ["#dc3545" if d > 0 else "#198754" for d in deltas]
        ax.bar(labels, deltas, color=colors, alpha=0.85, edgecolor="black")

        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_ylabel("Delta in Mean ITL vs Reference (ms)", fontsize=11, fontweight="bold")
        ax.set_title("Figure 6: Impact of Individual Mechanism Ablations on Latency", fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return True

    @classmethod
    def plot_sensitivity_sweeps(cls, sweep_df: pd.DataFrame, out_path: Path) -> bool:
        """Plot sensitivity response curves across parameter sweep settings."""
        if not MATPLOTLIB_AVAILABLE or "param_value" not in sweep_df.columns or "mean_itl_ms" not in sweep_df.columns:
            return False

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
        dim_name = sweep_df.get("dimension", ["Parameter"])[0] if len(sweep_df) > 0 else "Parameter"
        unit = sweep_df.get("unit", [""])[0] if len(sweep_df) > 0 else ""

        x = sweep_df["param_value"]
        y = sweep_df["mean_itl_ms"]
        y_lo = sweep_df.get("mean_itl_ci_lower")
        y_hi = sweep_df.get("mean_itl_ci_upper")

        ax.plot(x, y, marker="o", linewidth=2.0, color="#0d6efd", label="Mean ITL")
        if y_lo is not None and y_hi is not None and not y_lo.isna().all():
            ax.fill_between(x, y_lo, y_hi, color="#0d6efd", alpha=0.2, label="95% CI")

        ax.set_xlabel(f"{dim_name} ({unit})", fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Inter-Token Latency (ms)", fontsize=11, fontweight="bold")
        ax.set_title(f"Figure 7: Sensitivity Curve along {dim_name}", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True)
        ax.set_ylim(bottom=0.0)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return True
