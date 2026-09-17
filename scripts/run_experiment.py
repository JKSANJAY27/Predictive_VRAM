#!/usr/bin/env python
"""
scripts/run_experiment.py

Module 10 -- Experimental Harness
Run a complete experiment from configuration (JSON/YAML) or canonical preset.

Usage:
  python scripts/run_experiment.py --scenario combined_degradation --seed 42
  python scripts/run_experiment.py --config path/to/experiment_config.json
  python scripts/run_experiment.py --canonical
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.experiment_config import (
    ExperimentConfig,
    NetworkConfig,
    WorkloadConfig,
    make_canonical_experiment,
)
from src.evaluation.manifest import DuplicateRunGuard, ExperimentManifest
from src.evaluation.runner import ExperimentRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Run experimental trials for Module 10 evaluation")
    parser.add_argument("--config", type=str, default=None, help="Path to experiment JSON/YAML config")
    parser.add_argument("--scenario", type=str, default="combined_degradation", help="Scenario ID (default: combined_degradation)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--tokens", type=int, default=32, help="Max generated tokens per trial (default: 32)")
    parser.add_argument("--canonical", action="store_true", help="Run the canonical research experiment")
    parser.add_argument("--force", action="store_true", help="Force rerun of completed trials")
    parser.add_argument("--raw-dir", type=str, default="results/raw", help="Raw results output directory")
    parser.add_argument("--manifest-path", type=str, default="results/manifest.json", help="Manifest path")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.suffix in (".yaml", ".yml"):
            cfg = ExperimentConfig.load_yaml(cfg_path)
        else:
            cfg = ExperimentConfig.load_json(cfg_path)
    elif args.canonical:
        cfg = make_canonical_experiment()
    else:
        cfg = ExperimentConfig(
            name=f"experiment_{args.scenario}_seed{args.seed}",
            seed=args.seed,
            workload=WorkloadConfig(max_new_tokens=args.tokens),
            network=NetworkConfig(scenario=args.scenario, n_steps=64),
            baselines=["static", "network_reactive", "memory_reactive", "joint_reactive", "predictive"],
        )

    print("=" * 78)
    print(f"  MODULE 10 EXPERIMENT RUNNER: {cfg.name}")
    print("=" * 78)
    print(f"Scenario:     {cfg.network.scenario}")
    print(f"Baselines:    {', '.join(cfg.baselines)}")
    print(f"Seed:         {cfg.seed}")
    print(f"Max Tokens:   {cfg.workload.max_new_tokens}")
    print(f"Output Dir:   {args.raw_dir}")
    print("-" * 78)

    manifest = ExperimentManifest.create_from_config(cfg)
    runner = ExperimentRunner(raw_results_dir=args.raw_dir, force_rerun=args.force)

    start_t = time.time()
    results = runner.run_experiment(cfg, include_ablations=bool(cfg.ablations))
    elapsed = time.time() - start_t

    print(f"\nCompleted {len(results)} trials in {elapsed:.2f}s.")
    print("\nSummary of Executed Trials:")
    print(f"{'Trial ID':<26} {'Baseline':<18} {'Mean ITL':<12} {'P95 ITL':<12} {'Switches':<10}")
    print("-" * 78)

    for r in results:
        m = r.metrics
        print(f"{r.trial_id:<26} {r.baseline_id:<18} {m.mean_itl_ms:>8.1f} ms {m.p95_itl_ms:>8.1f} ms {m.total_switches:>8d}")

    # Update and save manifest
    for r in results:
        manifest.register_trial_completed(r.trial_id, str(Path(args.raw_dir) / f"{r.trial_id}.json"))
    manifest.save_json(args.manifest_path)
    print(f"\nManifest saved to {args.manifest_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
