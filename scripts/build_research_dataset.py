#!/usr/bin/env python
"""
scripts/build_research_dataset.py

Module 10 -- Research Dataset Builder
Consolidates raw trial JSON records into clean tabular CSV/JSON files with full
metadata and provenance schemas for open science and archival.

Usage:
  python scripts/build_research_dataset.py --raw-dir results/raw --output-dir results/export
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.export import ResearchDatasetBuilder


def parse_args():
    parser = argparse.ArgumentParser(description="Build consolidated research dataset")
    parser.add_argument("--raw-dir", type=str, default="results/raw", help="Directory of raw trial JSONs")
    parser.add_argument("--output-dir", type=str, default="results/export", help="Output directory")
    parser.add_argument("--experiment-id", type=str, default=None, help="Optional filter by experiment_id")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 78)
    print("  MODULE 10 RESEARCH DATASET BUILDER")
    print("=" * 78)

    builder = ResearchDatasetBuilder(raw_results_dir=args.raw_dir)
    paths = builder.build_dataset(output_dir=args.output_dir, experiment_id=args.experiment_id)

    print(f"Dataset generated successfully in: {args.output_dir}")
    print(f"  CSV Table:     {paths['csv']}")
    print(f"  JSON Archive:  {paths['json']}")
    print(f"  Metadata:      {paths['metadata']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
