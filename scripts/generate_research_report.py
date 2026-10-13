"""
CLI Script: Generate Research Report (Module 12).

Compiles a 15-section scientific report in Markdown from saved experimental
observations and statistical comparison tables.

Usage:
    uv run python scripts/generate_research_report.py --input-dir results/research/<experiment_id>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.research.manifest import ResearchManifest
from src.research.report import ResearchReportGenerator

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("generate_research_report")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 15-section research report from saved experiment artifacts.")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to experiment directory.")
    parser.add_argument("--output-file", type=str, default=None, help="Optional output path for report Markdown.")

    args = parser.parse_args()
    input_path = Path(args.input_dir)
    manifest_path = input_path / "manifest.json"
    raw_csv_path = input_path / "raw_observations.csv"

    if not manifest_path.is_file() or not raw_csv_path.is_file():
        logger.error(f"Directory {input_path} missing manifest.json or raw_observations.csv.")
        sys.exit(1)

    manifest = ResearchManifest.load(manifest_path)
    trials_df = pd.read_csv(raw_csv_path)

    cmp_csv = input_path / "policy_comparisons.csv"
    comparisons_df = pd.read_csv(cmp_csv) if cmp_csv.is_file() else None

    abl_csv = input_path / "tables" / "table3_ablation_summary.csv"
    ablation_df = pd.read_csv(abl_csv) if abl_csv.is_file() else None

    out_file = Path(args.output_file) if args.output_file else (input_path / "report.md")

    print("=" * 80)
    print(" SCIENTIFIC RESEARCH REPORT COMPILER (MODULE 12)")
    print("=" * 80)
    print(f"Input Directory  : {input_path}")
    print(f"Target Output    : {out_file}")
    print("=" * 80)

    res_path = ResearchReportGenerator.generate_report(
        manifest=manifest,
        trials_df=trials_df,
        comparisons_df=comparisons_df,
        ablation_df=ablation_df,
        output_path=out_file,
    )

    # Update manifest checksum
    manifest.record_artifact_checksum(out_file.name, res_path)
    manifest.save(manifest_path)

    print(f"[SUCCESS] Research report successfully compiled and saved to {res_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
