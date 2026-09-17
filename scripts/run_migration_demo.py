"""
Module 8 Migration Demo: Physical Migration and Runtime State Transition.

Demonstrates a complete, instrumented end-to-end migration cycle:
 1. Generate tokens on source partition plan (2+2 split)
 2. Execute transactional migration to target plan (1+3 split)
 3. Generate tokens on target partition plan
 4. Verify token parity (continuity invariant)
 5. Print detailed timing, metrics, and event log

Usage:
    python scripts/run_migration_demo.py

No GPU required. Runs entirely on CPU with the synthetic model.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.controller.types import MigrationRequest
from src.migration import (
    EmulatedNetworkTransfer,
    MigrationConfig,
    MigrationManager,
    MigrationMode,
    RuntimeAdapter,
    VerificationMode,
)
from src.migration.types import MigrationStatus
from src.runtime.executor import DistributedInferenceExecutor
from src.runtime.model import LayeredTransformer
from src.runtime.partition import PartitionPlan
from src.runtime.tier import Tier, TierId


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def sep(char: str = "-", width: int = 70) -> str:
    return char * width


def header(title: str) -> None:
    print()
    print(sep("="))
    print(f"  {title}")
    print(sep("="))


def section(title: str) -> None:
    print()
    print(sep("-"))
    print(f"  {title}")
    print(sep("-"))


def kv(key: str, value: object, indent: int = 2) -> None:
    pad = " " * indent
    print(f"{pad}{key:<40} {value}")


def bool_icon(flag: bool) -> str:
    return "YES" if flag else "NO "


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    header("Module 8: Physical Migration and Runtime State Transition Demo")

    torch.manual_seed(42)

    # -----------------------------------------------------------------------
    # 1. Build synthetic model and runtime
    # -----------------------------------------------------------------------
    section("1. Initialising Synthetic Model and Executor")

    NUM_LAYERS = 4
    model = LayeredTransformer.create_synthetic(
        num_layers=NUM_LAYERS,
        hidden_size=64,
        num_heads=2,
        vocab_size=500,
        device="cpu",
    )
    tiers = {
        TierId.USER_DEVICE: Tier(tier_id=TierId.USER_DEVICE, name="User Device", device="cpu"),
        TierId.EDGE_A:      Tier(tier_id=TierId.EDGE_A,      name="Edge Node A", device="cpu"),
        TierId.EDGE_B:      Tier(tier_id=TierId.EDGE_B,      name="Edge Node B", device="cpu"),
    }
    executor = DistributedInferenceExecutor(model=model, tiers=tiers)

    kv("Model layers:", NUM_LAYERS)
    kv("Hidden size:", 64)
    kv("Tiers:", "user_device=cpu | edge_a=cpu | edge_b=cpu")

    # -----------------------------------------------------------------------
    # 2. Define partition plans
    # -----------------------------------------------------------------------
    section("2. Partition Plans")

    src_plan = PartitionPlan(
        total_layers=NUM_LAYERS,
        user_device=(0, 1),
        edge_a=(2, NUM_LAYERS - 1),
    )
    tgt_plan = PartitionPlan(
        total_layers=NUM_LAYERS,
        user_device=(0, 0),
        edge_a=(1, NUM_LAYERS - 1),
    )

    kv("Source plan:", f"USER_DEVICE [0-1]  EDGE_A [2-{NUM_LAYERS-1}]")
    kv("Target plan:", f"USER_DEVICE [0-0]  EDGE_A [1-{NUM_LAYERS-1}]")
    kv("Layers changing tier:", "[1]  (UD -> EA)")

    # -----------------------------------------------------------------------
    # 3. Establish RuntimeAdapter and initialise migration manager
    # -----------------------------------------------------------------------
    section("3. Configuring RuntimeAdapter and MigrationManager")

    executor._setup_tiers_for_plan(src_plan)
    executor.active_plan = src_plan
    adapter = RuntimeAdapter(executor)

    # Use emulated network transfer with moderate bandwidth
    provider = EmulatedNetworkTransfer(
        bandwidth_mbps=100.0,
        latency_ms=5.0,
        fast_mode=True,  # Analytical simulation — no real sleep
    )
    config = MigrationConfig(
        execution_mode=MigrationMode.EXECUTE,
        verification_mode=VerificationMode.FAST,
        rollback_enabled=True,
        single_flight=True,
        emulated_bandwidth_delay=True,
        fast_mode=True,
        emulated_bandwidth_mbps=100.0,
        emulated_latency_ms=5.0,
    )
    manager = MigrationManager(
        runtime_adapter=adapter,
        config=config,
        transfer_provider=provider,
    )

    kv("Transfer provider:", "EmulatedNetworkTransfer (100 Mbps, 5ms RTT, analytical)")
    kv("Verification mode:", "FAST")
    kv("Rollback enabled:", bool_icon(config.rollback_enabled))
    kv("Single-flight guard:", bool_icon(config.single_flight))

    # -----------------------------------------------------------------------
    # 4. Generate tokens BEFORE migration
    # -----------------------------------------------------------------------
    section("4. Pre-Migration Inference")

    PROMPT = "The quick brown fox"
    MAX_TOKENS = 10

    print(f"  Prompt: '{PROMPT}'")
    print(f"  Max new tokens: {MAX_TOKENS}")
    t0 = time.perf_counter()
    result_before = executor.generate(
        prompt=PROMPT,
        partition_plan=src_plan,
        max_new_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    elapsed_before = (time.perf_counter() - t0) * 1000.0

    kv("Tokens generated:", result_before.generated_token_count)
    kv("Generated token IDs:", result_before.generated_tokens)
    kv("Inference time:", f"{elapsed_before:.2f} ms")

    # -----------------------------------------------------------------------
    # 5. Build MigrationRequest and execute
    # -----------------------------------------------------------------------
    section("5. Executing Migration")

    request = MigrationRequest(
        source_plan=src_plan,
        target_plan=tgt_plan,
        source_plan_id="plan-v1",
        target_plan_id="plan-v2",
        expected_gain=0.18,
        switching_cost=0.04,
        controller_cycle=7,
        timestamp=time.time(),
        changed_layers=[1],
        affected_tiers=["user_device", "edge_a"],
        estimated_kv_transfer_bytes=2048,
        reason="predictive_switch_approved: latency forecast improved by 18%",
    )

    print(f"  Request ID generated by manager.")
    print(f"  Expected gain: {request.expected_gain:.2%}")
    print(f"  Switching cost: {request.switching_cost:.2%}")

    t_mig = time.perf_counter()
    mig_result = manager.execute(request)
    elapsed_mig = (time.perf_counter() - t_mig) * 1000.0

    # -----------------------------------------------------------------------
    # 6. Print migration outcome
    # -----------------------------------------------------------------------
    section("6. Migration Outcome")

    status_icon = "OK " if mig_result.success else "FAIL"
    print(f"  [{status_icon}] Status: {mig_result.status.value}")
    kv("Success:", bool_icon(mig_result.success))
    kv("Rollback performed:", bool_icon(mig_result.rollback_performed))
    if mig_result.error_message:
        kv("Error:", mig_result.error_message)

    print()
    print("  Phase Timings:")
    t = mig_result.timings
    kv("  Validation:", f"{t.validation_ms:.3f} ms")
    kv("  Preparation:", f"{t.preparation_ms:.3f} ms")
    kv("  Model transfer:", f"{t.model_transfer_ms:.3f} ms")
    kv("  KV-cache transfer:", f"{t.kv_transfer_ms:.3f} ms")
    kv("  Verification:", f"{t.verification_ms:.3f} ms")
    kv("  Commit:", f"{t.commit_ms:.3f} ms")
    kv("  Total:", f"{t.total_duration_ms:.3f} ms")

    print()
    print("  Transfer Metrics:")
    m = mig_result.metrics
    kv("  Model bytes transferred:", f"{m.model_bytes_transferred:,} bytes")
    kv("  KV bytes transferred:", f"{m.kv_cache_bytes_transferred:,} bytes")
    kv("  Total bytes transferred:", f"{m.total_bytes_transferred:,} bytes")
    kv("  Affected layers:", m.affected_layers)

    print()
    print("  Cost Comparison (Predicted vs Actual):")
    c = mig_result.comparison
    kv("  Predicted switching cost:", f"{c.predicted_switching_cost:.4f}")
    kv("  Actual migration duration:", f"{c.actual_migration_duration_ms:.3f} ms")
    kv("  Predicted KV bytes:", f"{c.predicted_kv_bytes:,}")
    kv("  Actual KV bytes:", f"{c.actual_kv_bytes:,}")

    print()
    print("  Event Log:")
    for ev in mig_result.events:
        print(f"    > {ev}")

    # -----------------------------------------------------------------------
    # 7. Verify active plan updated
    # -----------------------------------------------------------------------
    section("7. Post-Migration Runtime State")

    active = adapter.get_current_plan()
    plan_match = active == tgt_plan
    kv("Active plan matches target:", bool_icon(plan_match))
    kv("Active plan user_device:", active.user_device)
    kv("Active plan edge_a:", active.edge_a)

    # Verify device placement
    all_on_cpu = all(
        str(next(model.blocks[i].parameters()).device) == "cpu"
        for i in range(NUM_LAYERS)
    )
    kv("All layers on CPU:", bool_icon(all_on_cpu))

    # -----------------------------------------------------------------------
    # 8. Post-migration inference and token parity check
    # -----------------------------------------------------------------------
    section("8. Post-Migration Inference and Token Parity")

    t1 = time.perf_counter()
    result_after = executor.generate(
        prompt=PROMPT,
        partition_plan=tgt_plan,
        max_new_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    elapsed_after = (time.perf_counter() - t1) * 1000.0

    kv("Tokens generated:", result_after.generated_token_count)
    kv("Generated token IDs:", result_after.generated_tokens)
    kv("Inference time:", f"{elapsed_after:.2f} ms")

    tokens_match = result_before.generated_tokens == result_after.generated_tokens
    print()
    icon = "PASS" if tokens_match else "FAIL"
    print(f"  [{icon}] Token Parity: {'VERIFIED' if tokens_match else 'BROKEN'}")

    if not tokens_match:
        print("  CRITICAL: Output tokens changed after migration!")
        print(f"    Before: {result_before.generated_tokens}")
        print(f"    After:  {result_after.generated_tokens}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 9. Summary
    # -----------------------------------------------------------------------
    header("Demo Summary")

    kv("Migration status:", mig_result.status.value)
    kv("Migration duration:", f"{mig_result.timings.total_duration_ms:.3f} ms")
    kv("Bytes transferred:", f"{mig_result.metrics.total_bytes_transferred:,}")
    kv("Rollback performed:", bool_icon(mig_result.rollback_performed))
    kv("Token parity:", "PASS" if tokens_match else "FAIL")

    print()
    print("  Module 8 demo completed successfully.")
    print(sep("="))
    print()


if __name__ == "__main__":
    main()
