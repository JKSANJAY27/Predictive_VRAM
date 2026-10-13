"""
CLI Script: Validate Research Artifacts & Integrity (Module 12).

Audits an experiment output directory, verifying manifest integrity,
artifact checksum matching, and scientific invariance rules.

Usage:
    uv run python scripts/validate_research_artifacts.py --input-dir results/research/<experiment_id>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.research.validation import ResearchValidator

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("validate_research_artifacts")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and validate research experiment artifacts.")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to experiment directory.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to write JSON audit report.")

    args = parser.parse_args()
    input_path = Path(args.input_dir)
    if not input_path.is_dir():
        logger.error(f"Directory not found: {input_path}")
        sys.exit(1)

    print("=" * 80)
    print(" RESEARCH ARTIFACT VALIDATOR & INTEGRITY AUDITOR (MODULE 12)")
    print("=" * 80)
    print(f"Target Directory : {input_path}")
    print("=" * 80)

    report = ResearchValidator.validate_experiment_artifacts(input_path)

    print(f"Total Invariant Checks : {report.total_checks}")
    print(f"Passed Checks          : {report.passed_checks}")
    print(f"Error Count            : {report.error_count}")
    print(f"Warning Count          : {report.warning_count}")
    print("-" * 80)

    if report.issues:
        print("Detected Issues:")
        for issue in report.issues:
            print(f"  [{issue.level}] {issue.code}: {issue.message}")
            if issue.details:
                print(f"         Details: {issue.details}")
        print("-" * 80)

    if report.is_valid:
        print("[SUCCESS] All experiment artifacts pass verification and provenance integrity checks.")
    else:
        print("[FAILURE] Experiment artifacts failed integrity verification.")

    if args.output_json:
        out_json_path = Path(args.output_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"[OK] Audit report written to {out_json_path}")

    print("=" * 80)
    if not report.is_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
