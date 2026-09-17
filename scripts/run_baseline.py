"""
Baseline runner script for Distributed Split Inference.

Demonstrates:
1. Loading the configured model.
2. Executing monolithic (fully local) inference.
3. Executing 3-tier split inference across User Device, Edge A, and Edge B.
4. Comparing semantic and token consistency between local and split execution.
5. Printing structured execution metrics and transfer telemetry.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure workspace root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config.settings import get_default_config_path, load_yaml
from src.runtime.executor import DistributedInferenceExecutor
from src.runtime.model import LayeredTransformer
from src.runtime.partition import PartitionPlan
from src.runtime.tier import Tier, TierId
from src.runtime.transfer import TransferManager
from src.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline split inference.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(get_default_config_path("baseline.yaml")),
        help="Path to baseline YAML configuration.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt override for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["three_tier", "two_tier", "monolithic", "compare_all"],
        default="compare_all",
        help="Partition mode to execute.",
    )
    return parser.parse_args()


def print_banner(title: str) -> None:
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def run() -> None:
    logger = setup_logger("run_baseline")
    args = parse_args()

    config = load_yaml(args.config)
    model_cfg = config.get("model", {})
    gen_cfg = config.get("generation", {})
    partitions_cfg = config.get("partitions", {})

    prompt = args.prompt or gen_cfg.get(
        "prompt",
        "The future of distributed edge computing and predictive resource allocation",
    )
    max_new_tokens = args.max_new_tokens or gen_cfg.get("max_new_tokens", 16)
    temperature = gen_cfg.get("temperature", 0.0)
    seed = gen_cfg.get("seed", 42)

    torch.manual_seed(seed)

    print_banner("MODULE 1: DISTRIBUTED SPLIT-INFERENCE BASELINE")
    print(f"Model: {model_cfg.get('identifier', 'gpt2')} (device: {model_cfg.get('device', 'cpu')})")
    print(f"Prompt: {prompt!r}")
    print(f"Max New Tokens: {max_new_tokens} (temperature: {temperature}, seed: {seed})")

    logger.info("Loading model and tokenizer...")
    t0 = time.time()
    model = LayeredTransformer.from_pretrained(
        model_name_or_path=model_cfg.get("identifier", "gpt2"),
        device=model_cfg.get("device", "cpu"),
    )
    logger.info(f"Model loaded in {round(time.time() - t0, 2)}s with {model.num_layers} layers.")

    # Initialize Tiers
    tiers = {
        TierId.USER_DEVICE: Tier(
            tier_id=TierId.USER_DEVICE,
            name="User Device",
            device=config.get("tiers", {}).get("user_device", {}).get("device", "cpu"),
        ),
        TierId.EDGE_A: Tier(
            tier_id=TierId.EDGE_A,
            name="Edge Node A",
            device=config.get("tiers", {}).get("edge_a", {}).get("device", "cpu"),
        ),
        TierId.EDGE_B: Tier(
            tier_id=TierId.EDGE_B,
            name="Edge Node B",
            device=config.get("tiers", {}).get("edge_b", {}).get("device", "cpu"),
        ),
    }

    transfer_mgr = TransferManager()
    executor = DistributedInferenceExecutor(
        model=model,
        tiers=tiers,
        transfer_manager=transfer_mgr,
    )

    total_layers = model.num_layers

    # Construct Partition Plans
    plan_monolithic = PartitionPlan.monolithic(total_layers=total_layers)
    plan_two_tier = PartitionPlan.two_tier(total_layers=total_layers, cut_layer=5)
    plan_three_tier = PartitionPlan.three_tier(total_layers=total_layers, cut1=3, cut2=7)

    plans_to_run = []
    if args.split_mode == "monolithic":
        plans_to_run.append(("Monolithic (All User Device)", plan_monolithic))
    elif args.split_mode == "two_tier":
        plans_to_run.append(("Two-Tier Split (UserDevice + EdgeA)", plan_two_tier))
    elif args.split_mode == "three_tier":
        plans_to_run.append(("Three-Tier Split (UserDevice + EdgeA + EdgeB)", plan_three_tier))
    else:  # compare_all
        plans_to_run.append(("Monolithic (Fully Local)", plan_monolithic))
        plans_to_run.append(("Two-Tier Split", plan_two_tier))
        plans_to_run.append(("Three-Tier Split", plan_three_tier))

    results = []

    for name, plan in plans_to_run:
        print_banner(f"EXECUTING: {name}")
        print(f"Partition Plan Configuration:")
        for tier_id, rng in plan.get_active_tiers():
            print(f"  - {tiers[tier_id].name:<18} [{tier_id.value}]: Layers {rng[0]}..{rng[1]} ({rng[1] - rng[0] + 1} layers)")
        boundaries = plan.get_transfer_boundaries()
        if boundaries:
            print(f"Inter-tier Transfer Boundaries: {len(boundaries)}")
            for b in boundaries:
                print(f"  * Cut after layer {b.cut_layer}: {b.source_tier.value} -> {b.destination_tier.value}")
        else:
            print("No inter-tier transfers (Monolithic execution).")

        res = executor.generate(
            prompt=prompt,
            partition_plan=plan,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
        )
        results.append((name, res))

        print("\nExecution Telemetry:")
        print(f"  Generated Text:            {res.generated_text!r}")
        print(f"  Total Execution Time:      {res.execution_time_seconds:.3f} s")
        print(f"  Time To First Token (TTFT):{res.time_to_first_token_seconds:.3f} s")
        print(f"  Avg Inter-Token Latency:   {res.average_token_latency_seconds:.3f} s")
        print(f"  Tokens / Second:           {res.tokens_per_second:.2f}")
        print(f"  Tokens Generated:          {res.generated_token_count}")
        print(f"  Total Boundary Transfers:  {res.total_transfers}")
        print(f"  Total Transferred Bytes:   {res.total_transfer_bytes:,} bytes ({res.total_transfer_bytes / 1024:.2f} KB)")
        print(f"  KV Cache Memory (est):     {res.kv_cache_summary['total_kv_mb']:.3f} MB across {res.kv_cache_summary['cached_layers_count']} layers")

    # Correctness verification
    if len(results) > 1:
        print_banner("CORRECTNESS VERIFICATION: LOCAL vs SPLIT INFERENCE")
        baseline_name, baseline_res = results[0]
        all_passed = True
        for comp_name, comp_res in results[1:]:
            tokens_match = baseline_res.generated_tokens == comp_res.generated_tokens
            text_match = baseline_res.generated_text == comp_res.generated_text
            status = "PASSED (Deterministic Exact Match)" if tokens_match and text_match else "FAILED"
            print(f"Comparison: {baseline_name} vs {comp_name}:")
            print(f"  - Token Match:      {tokens_match}")
            print(f"  - Text Match:       {text_match}")
            print(f"  - Status:           {status}")
            if not (tokens_match and text_match):
                all_passed = False
                print(f"    Expected: {baseline_res.generated_tokens}")
                print(f"    Received: {comp_res.generated_tokens}")

        if all_passed:
            print("\n[SUCCESS] Baseline split-inference verification completed with zero discrepancies!")
        else:
            print("\n[WARNING] Discrepancies detected between execution modes.")


if __name__ == "__main__":
    run()
