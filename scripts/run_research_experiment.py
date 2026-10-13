"""
CLI Script: Run Research Experiment Campaign (Module 12).

Executes reproducible research experiments across policies and scenarios,
persisting raw observations, trial records, and manifest metadata.

Usage:
    uv run python scripts/run_research_experiment.py --config configs/research/default.yaml
    uv run python scripts/run_research_experiment.py --config configs/research/default.yaml --repetitions 3 --resume
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.research.runner import ResearchRunner

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("run_research_experiment")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible research experiment campaign.")
    parser.add_argument("--config", type=str, default="configs/research/default.yaml", help="Path to YAML config.")
    parser.add_argument("--experiment-id", type=str, default=None, help="Override experiment ID.")
    parser.add_argument("--seed", type=int, default=None, help="Override root random seed.")
    parser.add_argument("--repetitions", type=int, default=None, help="Override repetition count.")
    parser.add_argument("--scenarios", nargs="+", default=None, help="List of scenario IDs to run.")
    parser.add_argument("--policies", nargs="+", default=None, help="List of baseline policies to run.")
    parser.add_argument("--output-dir", type=str, default="results/research", help="Base output directory.")
    parser.add_argument("--resume", action="store_true", help="Resume incomplete campaign without overwriting.")

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if args.experiment_id:
        cfg["experiment_id"] = args.experiment_id
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.repetitions is not None:
        cfg["repetitions"] = args.repetitions
    if args.scenarios:
        cfg["scenarios"] = args.scenarios
    if args.policies:
        cfg["policies"] = args.policies

    print("=" * 80)
    print(" RESEARCH EXPERIMENT CAMPAIGN RUNNER (MODULE 12)")
    print("=" * 80)
    print(f"Config File     : {config_path}")
    print(f"Experiment ID   : {cfg.get('experiment_id', 'auto-generated')}")
    print(f"Root Seed       : {cfg.get('seed', 42)}")
    print(f"Repetitions     : {cfg.get('repetitions', 1)}")
    print(f"Scenarios ({len(cfg.get('scenarios', []))}): {cfg.get('scenarios', [])}")
    print(f"Policies  ({len(cfg.get('policies', []))}): {cfg.get('policies', [])}")
    print(f"Resume Mode     : {args.resume}")
    print("=" * 80)

    runner = ResearchRunner(
        config=cfg,
        experiment_id=cfg.get("experiment_id"),
        base_output_dir=args.output_dir,
        resume=args.resume,
    )

    result = runner.execute_campaign()

    print("\n" + "=" * 80)
    print(" CAMPAIGN EXECUTION COMPLETE")
    print("=" * 80)
    print(f"Experiment ID   : {result['experiment_id']}")
    print(f"Output Directory: {result['output_dir']}")
    print(f"Manifest Path   : {result['manifest_path']}")
    print(f"Total Trials    : {result['total_trials']}")
    print(f"Completed Trials: {result['completed']}")
    print(f"Failed Trials   : {result['failed']}")
    print(f"Research Report : {result['report_path']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
