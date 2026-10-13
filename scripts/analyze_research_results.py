"""
CLI Script: Analyze Research Results (Module 12).

Performs statistical analysis, paired hypothesis testing, and publication artifact
generation independently from raw saved observations without rerunning expensive inference.

Usage:
    uv run python scripts/analyze_research_results.py --input-dir results/research/<experiment_id>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.research.artifacts import PublicationArtifactGenerator
from src.research.comparisons import PolicyComparisonEngine
from src.research.manifest import ResearchManifest
from src.research.report import ResearchReportGenerator

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("analyze_research_results")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze research experiment results independently of execution.")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to experiment directory.")
    parser.add_argument("--reference-policy", type=str, default="predictive", help="Reference baseline policy.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold alpha.")

    args = parser.parse_args()
    input_path = Path(args.input_dir)
    manifest_path = input_path / "manifest.json"
    raw_csv_path = input_path / "raw_observations.csv"

    if not manifest_path.is_file() or not raw_csv_path.is_file():
        logger.error(f"Directory {input_path} does not contain manifest.json and raw_observations.csv.")
        sys.exit(1)

    manifest = ResearchManifest.load(manifest_path)
    trials_df = pd.read_csv(raw_csv_path)

    print("=" * 80)
    print(" RESEARCH DATASET STATISTICAL ANALYZER (MODULE 12)")
    print("=" * 80)
    print(f"Experiment ID    : {manifest.experiment_id}")
    print(f"Total Trials     : {len(trials_df)}")
    print(f"Reference Policy : {args.reference_policy}")
    print(f"Alpha Threshold  : {args.alpha}")
    print("=" * 80)

    # 1. Run pairwise comparisons across scenarios
    scenarios = trials_df["scenario_id"].unique()
    all_cmp_rows = []
    for scen_id in scenarios:
        rep = PolicyComparisonEngine.compare_scenario_trials(
            scenario_id=scen_id,
            trials_df=trials_df,
            reference_policy=args.reference_policy,
            alpha_significance=args.alpha,
        )
        cmp_df = PolicyComparisonEngine.comparisons_to_dataframe(rep.comparisons)
        if not cmp_df.empty:
            cmp_df["scenario_id"] = scen_id
            all_cmp_rows.append(cmp_df)

    comparisons_df = pd.concat(all_cmp_rows, ignore_index=True) if all_cmp_rows else pd.DataFrame()
    if not comparisons_df.empty:
        cmp_csv_path = input_path / "policy_comparisons.csv"
        comparisons_df.to_csv(cmp_csv_path, index=False)
        manifest.record_artifact_checksum("policy_comparisons.csv", cmp_csv_path)
        print(f"[OK] Generated {len(comparisons_df)} paired comparison tests in policy_comparisons.csv")

    # 2. Generate publication figures and tables
    is_cpu = not manifest.environment.cuda_available
    artifacts = PublicationArtifactGenerator.generate_all_artifacts(
        trials_df=trials_df,
        output_dir=input_path,
        comparisons_df=comparisons_df if not comparisons_df.empty else None,
        is_cpu_only=is_cpu,
    )
    for rel_k, abs_v in artifacts.items():
        manifest.record_artifact_checksum(rel_k, abs_v)
    print(f"[OK] Generated {len(artifacts)} publication figures and tables.")

    # 3. Regenerate research report
    report_path = input_path / "report.md"
    ResearchReportGenerator.generate_report(
        manifest=manifest,
        trials_df=trials_df,
        comparisons_df=comparisons_df if not comparisons_df.empty else None,
        output_path=report_path,
    )
    manifest.record_artifact_checksum("report.md", report_path)
    print(f"[OK] Regenerated comprehensive report at {report_path}")

    # 4. Save updated manifest
    manifest.save(manifest_path)
    print(f"[OK] Updated manifest checksums in {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
