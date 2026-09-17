"""
Module 8 Migration Failure Demo: Failure Injection and Rollback Verification.

Demonstrates the transactional safety of Module 8 by deliberately triggering
failures at every stage of the migration pipeline and verifying automatic
rollback restores the system to its pre-migration state.

Failure scenarios demonstrated:
 A. Preparation stage failure
 B. Model layer transfer failure (specific layer)
 C. KV-cache transfer failure (with live KV cache)
 D. Post-migration verification failure
 E. Stale migration request rejection
 F. Idempotent (already-at-target) no-op handling

Usage:
    python scripts/run_migration_failure_demo.py

No GPU required. Runs entirely on CPU with the synthetic model.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.controller.types import MigrationRequest
from src.migration import (
    MigrationConfig,
    MigrationManager,
    MigrationMode,
    MigrationStatus,
    RuntimeAdapter,
    VerificationMode,
)
from src.migration.planner import MigrationPlanner
from src.runtime.executor import DistributedInferenceExecutor
from src.runtime.model import LayeredTransformer
from src.runtime.partition import PartitionPlan
from src.runtime.tier import Tier, TierId


# ---------------------------------------------------------------------------
# Minimal KV-cache stub for failure injection testing
# ---------------------------------------------------------------------------

class _FakeLayer:
    def __init__(self):
        self.keys = torch.randn(1, 8, 16)
        self.values = torch.randn(1, 8, 16)


class FakeKVCache:
    """Minimal KV-cache stand-in that satisfies the LayerCache duck-type."""
    def __init__(self, num_layers: int = 4):
        self.layers = [_FakeLayer() for _ in range(num_layers)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sep(char: str = "-", width: int = 72) -> str:
    return char * width


def header(title: str) -> None:
    print()
    print(sep("="))
    print(f"  {title}")
    print(sep("="))


def scenario(label: str) -> None:
    print()
    print(sep())
    print(f"  SCENARIO: {label}")
    print(sep())


def result_line(success: bool, status: MigrationStatus, rollback: bool, plan_restored: bool) -> None:
    ok = lambda b: "OK " if b else "FAIL"
    print(f"  Success        : {ok(not success)} (expected FAIL)")
    print(f"  Status         : {status.value}")
    print(f"  Rollback       : {ok(rollback)} (expected YES)")
    print(f"  Plan restored  : {ok(plan_restored)} (expected YES)")


def make_runtime(num_layers: int = 4) -> tuple:
    """Return (executor, adapter, src_plan, tgt_plan)."""
    torch.manual_seed(42)
    model = LayeredTransformer.create_synthetic(
        num_layers=num_layers, hidden_size=64, num_heads=2, vocab_size=500, device="cpu"
    )
    tiers = {
        TierId.USER_DEVICE: Tier(tier_id=TierId.USER_DEVICE, name="User Device", device="cpu"),
        TierId.EDGE_A:      Tier(tier_id=TierId.EDGE_A,      name="Edge Node A", device="cpu"),
        TierId.EDGE_B:      Tier(tier_id=TierId.EDGE_B,      name="Edge Node B", device="cpu"),
    }
    executor = DistributedInferenceExecutor(model=model, tiers=tiers)
    src_plan = PartitionPlan(total_layers=num_layers, user_device=(0, 1), edge_a=(2, num_layers - 1))
    executor._setup_tiers_for_plan(src_plan)
    executor.active_plan = src_plan
    adapter = RuntimeAdapter(executor)
    tgt_plan = PartitionPlan(total_layers=num_layers, user_device=(0, 0), edge_a=(1, num_layers - 1))
    return executor, adapter, src_plan, tgt_plan


def base_config() -> MigrationConfig:
    return MigrationConfig(
        emulated_bandwidth_delay=False,
        fast_mode=True,
        rollback_enabled=True,
        single_flight=True,
    )


def make_request(src: PartitionPlan, tgt: PartitionPlan) -> MigrationRequest:
    return MigrationRequest(
        source_plan=src,
        target_plan=tgt,
        source_plan_id="plan-v1",
        target_plan_id="plan-v2",
        expected_gain=0.2,
        switching_cost=0.05,
        controller_cycle=10,
        timestamp=time.time(),
        changed_layers=[1],
        affected_tiers=["user_device", "edge_a"],
        estimated_kv_transfer_bytes=2048,
        reason="test failure injection",
    )


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    header("Module 8: Migration Failure Injection and Rollback Demo")

    failures_verified = 0
    failures_failed = 0

    # =======================================================================
    # Scenario A: Preparation stage failure
    # =======================================================================
    scenario("A -- Preparation Stage Failure")
    print("  Injecting failure at stage: PREPARATION")

    executor, adapter, src, tgt = make_runtime()
    mgr = MigrationManager(runtime_adapter=adapter, config=base_config())
    mgr.fail_on_stage = "preparation"

    req = make_request(src, tgt)
    res = mgr.execute(req)

    plan_after = adapter.get_current_plan()
    plan_restored = plan_after == src

    result_line(res.success, res.status, res.rollback_performed, plan_restored)
    print(f"  Events: {res.events}")

    ok_a = (not res.success) and res.rollback_performed and plan_restored
    print(f"  -> {'PASS' if ok_a else 'FAIL'}")
    if ok_a:
        failures_verified += 1
    else:
        failures_failed += 1

    # =======================================================================
    # Scenario B: Model layer transfer failure (specific layer)
    # =======================================================================
    scenario("B -- Model Layer Transfer Failure (Layer-Level Injection)")
    print("  Injecting failure at stage: MODEL TRANSFER (layer 1)")

    executor, adapter, src, tgt = make_runtime()
    mgr = MigrationManager(runtime_adapter=adapter, config=base_config())
    mp = MigrationPlanner.plan(src, tgt)
    mgr.fail_on_layer = mp.affected_layers[0]  # Inject on first affected layer

    req = make_request(src, tgt)
    res = mgr.execute(req)
    plan_after = adapter.get_current_plan()
    plan_restored = plan_after == src

    result_line(res.success, res.status, res.rollback_performed, plan_restored)
    print(f"  Injected on layer: {mgr.fail_on_layer}")

    ok_b = (not res.success) and res.rollback_performed and plan_restored
    print(f"  -> {'PASS' if ok_b else 'FAIL'}")
    if ok_b:
        failures_verified += 1
    else:
        failures_failed += 1

    # =======================================================================
    # Scenario C: KV-cache transfer failure
    # =======================================================================
    scenario("C -- KV-Cache Transfer Failure")
    print("  Injecting failure at stage: KV TRANSFER")

    executor, adapter, src, tgt = make_runtime()
    mgr = MigrationManager(runtime_adapter=adapter, config=base_config())
    mgr.fail_kv = True

    # Must supply a real (fake) KV cache so the KV transfer stage is entered
    kv = FakeKVCache(num_layers=4)
    req = make_request(src, tgt)
    res = mgr.execute(req, kv_cache=kv)
    plan_after = adapter.get_current_plan()
    plan_restored = plan_after == src

    result_line(res.success, res.status, res.rollback_performed, plan_restored)

    ok_c = (not res.success) and res.rollback_performed and plan_restored
    print(f"  -> {'PASS' if ok_c else 'FAIL'}")
    if ok_c:
        failures_verified += 1
    else:
        failures_failed += 1

    # =======================================================================
    # Scenario D: Verification failure
    # =======================================================================
    scenario("D -- Post-Migration Verification Failure")
    print("  Injecting failure at stage: VERIFICATION")

    executor, adapter, src, tgt = make_runtime()
    mgr = MigrationManager(runtime_adapter=adapter, config=base_config())
    mgr.fail_verification = True

    req = make_request(src, tgt)
    res = mgr.execute(req)
    plan_after = adapter.get_current_plan()
    plan_restored = plan_after == src

    result_line(res.success, res.status, res.rollback_performed, plan_restored)

    ok_d = (not res.success) and res.rollback_performed and plan_restored
    print(f"  -> {'PASS' if ok_d else 'FAIL'}")
    if ok_d:
        failures_verified += 1
    else:
        failures_failed += 1

    # =======================================================================
    # Scenario E: Stale request rejection
    # =======================================================================
    scenario("E -- Stale Migration Request Rejection")
    print("  Submitting request whose source_plan doesn't match active plan.")

    executor, adapter, src, tgt = make_runtime()
    mgr = MigrationManager(runtime_adapter=adapter, config=base_config())

    wrong_src = PartitionPlan(total_layers=4, user_device=(0, 0), edge_a=(1, 3))
    req = make_request(wrong_src, tgt)
    res = mgr.execute(req)

    print(f"  Success        : {'NO ' if not res.success else 'YES'} (expected NO)")
    print(f"  Status         : {res.status.value}")
    print(f"  Error          : {res.error_message[:80] if res.error_message else 'None'}")

    ok_e = not res.success and res.status == MigrationStatus.FAILED
    print(f"  -> {'PASS' if ok_e else 'FAIL'}")
    if ok_e:
        failures_verified += 1
    else:
        failures_failed += 1

    # =======================================================================
    # Scenario F: Idempotent (already at target) no-op
    # =======================================================================
    scenario("F -- Idempotent No-Op (Already at Target)")
    print("  Submitting request where source == target == current active plan.")

    executor, adapter, src, tgt = make_runtime()
    mgr = MigrationManager(runtime_adapter=adapter, config=base_config())

    req_idem = make_request(src, src)  # Same plan
    res = mgr.execute(req_idem)

    print(f"  Success        : {'YES' if res.success else 'NO '} (expected YES)")
    print(f"  Status         : {res.status.value}  (expected: already_at_target)")
    print(f"  Rollback       : {'NO ' if not res.rollback_performed else 'YES'} (expected NO)")

    ok_f = res.success and res.status == MigrationStatus.ALREADY_AT_TARGET and not res.rollback_performed
    print(f"  -> {'PASS' if ok_f else 'FAIL'}")
    if ok_f:
        failures_verified += 1
    else:
        failures_failed += 1

    # =======================================================================
    # Final summary
    # =======================================================================
    header("Failure Injection Summary")

    total = failures_verified + failures_failed
    print(f"  Scenarios passed : {failures_verified}/{total}")
    print(f"  Scenarios failed : {failures_failed}/{total}")
    print()

    if failures_failed == 0:
        print("  ALL FAILURE SCENARIOS CORRECTLY HANDLED.")
        print("  Module 8 transactional safety VERIFIED.")
    else:
        print("  WARNING: Some failure scenarios did not behave as expected!")

    print(sep("="))
    print()

    if failures_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
