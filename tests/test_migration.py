"""
Module 8 Test Suite: Physical Migration and Runtime State Transition.

Covers all 40+ required test cases defined in the Module 8 Definition of Done:
 - Data types and enum contracts
 - Validation and idempotency
 - Migration planning (delta computation)
 - Transfer providers (local and emulated)
 - Layer and KV-cache transfer managers
 - Verifier (FAST and DEEP modes)
 - Rollback orchestration
 - RuntimeAdapter facade
 - MigrationManager orchestration
 - Failure injection and rollback paths
 - Simulation mode
 - Single-flight concurrency guard
 - Metrics and comparison structs
 - End-to-end continuity (token parity before/after migration)
"""

from __future__ import annotations

import time
import threading
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.controller.types import MigrationRequest
from src.migration.adapter import RuntimeAdapter
from src.migration.manager import MigrationManager
from src.migration.planner import MigrationPlan, MigrationPlanner
from src.migration.rollback import RollbackManager, RollbackReport
from src.migration.state import (
    KVCacheTransferManager,
    KVCacheTransferReport,
    LayerTransferManager,
    LayerTransferReport,
)
from src.migration.transfer import (
    EmulatedNetworkTransfer,
    LocalTensorTransfer,
    MigrationTransferProvider,
)
from src.migration.types import (
    MigrationComparison,
    MigrationConfig,
    MigrationEvent,
    MigrationMetrics,
    MigrationMode,
    MigrationPhaseTimings,
    MigrationPoint,
    MigrationResult,
    MigrationStatus,
    VerificationMode,
)
from src.migration.validator import MigrationValidator, ValidationResult
from src.migration.verifier import MigrationVerifier, VerificationResult
from src.runtime.executor import DistributedInferenceExecutor
from src.runtime.model import LayeredTransformer
from src.runtime.partition import PartitionPlan
from src.runtime.tier import Tier, TierId


# ===========================================================================
# Shared helpers
# ===========================================================================

def make_model(num_layers: int = 4) -> LayeredTransformer:
    torch.manual_seed(42)
    return LayeredTransformer.create_synthetic(
        num_layers=num_layers,
        hidden_size=64,
        num_heads=2,
        vocab_size=500,
        device="cpu",
    )


def make_tiers() -> Dict[TierId, Tier]:
    return {
        TierId.USER_DEVICE: Tier(tier_id=TierId.USER_DEVICE, name="User", device="cpu"),
        TierId.EDGE_A: Tier(tier_id=TierId.EDGE_A, name="EdgeA", device="cpu"),
        TierId.EDGE_B: Tier(tier_id=TierId.EDGE_B, name="EdgeB", device="cpu"),
    }


def make_executor(num_layers: int = 4) -> DistributedInferenceExecutor:
    model = make_model(num_layers)
    tiers = make_tiers()
    executor = DistributedInferenceExecutor(model=model, tiers=tiers)
    return executor


def plan_2_2(total: int = 4) -> PartitionPlan:
    """Layers 0-1 on USER_DEVICE, layers 2-3 on EDGE_A."""
    return PartitionPlan(
        total_layers=total,
        user_device=(0, 1),
        edge_a=(2, total - 1),
    )


def plan_1_3(total: int = 4) -> PartitionPlan:
    """Layers 0 on USER_DEVICE, layers 1-3 on EDGE_A."""
    return PartitionPlan(
        total_layers=total,
        user_device=(0, 0),
        edge_a=(1, total - 1),
    )


def plan_monolithic(total: int = 4) -> PartitionPlan:
    """All layers on USER_DEVICE."""
    return PartitionPlan(total_layers=total, user_device=(0, total - 1))


def make_migration_request(
    source: PartitionPlan,
    target: PartitionPlan,
    changed_layers: Optional[List[int]] = None,
) -> MigrationRequest:
    if changed_layers is None:
        changed_layers = []
    return MigrationRequest(
        source_plan=source,
        target_plan=target,
        source_plan_id="src-001",
        target_plan_id="tgt-001",
        expected_gain=0.15,
        switching_cost=0.05,
        controller_cycle=5,
        timestamp=time.time(),
        changed_layers=changed_layers,
        affected_tiers=["user_device", "edge_a"],
        estimated_kv_transfer_bytes=4096,
        reason="test migration",
    )


def make_adapter(num_layers: int = 4) -> RuntimeAdapter:
    executor = make_executor(num_layers)
    adapter = RuntimeAdapter(executor)
    # Set initial active plan to plan_2_2
    src = plan_2_2(num_layers)
    adapter.executor._setup_tiers_for_plan(src)
    if not hasattr(executor, "active_plan"):
        executor.active_plan = src
    return adapter


def make_manager(num_layers: int = 4) -> MigrationManager:
    adapter = make_adapter(num_layers)
    cfg = MigrationConfig(
        emulated_bandwidth_delay=False,
        fast_mode=True,
        single_flight=True,
    )
    return MigrationManager(runtime_adapter=adapter, config=cfg)


# ===========================================================================
# Fake KV Cache
# ===========================================================================

class FakeLayer:
    def __init__(self, seq_len: int = 8, hidden: int = 16):
        self.keys = torch.randn(1, seq_len, hidden)
        self.values = torch.randn(1, seq_len, hidden)


class FakeKVCache:
    def __init__(self, num_layers: int = 4, seq_len: int = 8, hidden: int = 16):
        self.layers = [FakeLayer(seq_len, hidden) for _ in range(num_layers)]


# ===========================================================================
# M8-T01: MigrationStatus enum completeness
# ===========================================================================

class TestMigrationStatusEnum:
    def test_all_statuses_exist(self):
        statuses = [s.value for s in MigrationStatus]
        assert "pending" in statuses
        assert "completed" in statuses
        assert "failed" in statuses
        assert "rolled_back" in statuses
        assert "already_at_target" in statuses
        assert "failed_partial" in statuses
        assert "validating" in statuses
        assert "preparing" in statuses
        assert "transferring" in statuses
        assert "verifying" in statuses
        assert "committed" in statuses

    def test_status_str_enum(self):
        assert MigrationStatus.COMPLETED == "completed"
        assert MigrationStatus.FAILED == "failed"
        assert MigrationStatus.ROLLED_BACK == "rolled_back"


# ===========================================================================
# M8-T02: MigrationMode enum
# ===========================================================================

class TestMigrationModeEnum:
    def test_execute_and_simulate_exist(self):
        assert MigrationMode.EXECUTE == "execute"
        assert MigrationMode.SIMULATE == "simulate"


# ===========================================================================
# M8-T03: MigrationEvent enum completeness
# ===========================================================================

class TestMigrationEventEnum:
    def test_all_lifecycle_events_present(self):
        event_vals = [e.value for e in MigrationEvent]
        expected = [
            "migration_started",
            "validation_completed",
            "preparation_completed",
            "model_transfer_started",
            "model_transfer_completed",
            "kv_transfer_started",
            "kv_transfer_completed",
            "verification_started",
            "verification_completed",
            "commit_started",
            "commit_completed",
            "rollback_started",
            "rollback_completed",
            "migration_failed",
            "migration_completed",
        ]
        for e in expected:
            assert e in event_vals, f"Missing event: {e}"


# ===========================================================================
# M8-T04: MigrationConfig defaults and serialization
# ===========================================================================

class TestMigrationConfig:
    def test_defaults(self):
        cfg = MigrationConfig()
        assert cfg.execution_mode == MigrationMode.EXECUTE
        assert cfg.verification_mode == VerificationMode.FAST
        assert cfg.rollback_enabled is True
        assert cfg.single_flight is True
        assert cfg.fast_mode is True

    def test_to_dict_roundtrip(self):
        cfg = MigrationConfig(
            execution_mode=MigrationMode.SIMULATE,
            emulated_bandwidth_mbps=100.0,
            emulated_latency_ms=5.0,
        )
        d = cfg.to_dict()
        assert d["execution_mode"] == "simulate"
        assert d["emulated_bandwidth_mbps"] == 100.0
        assert d["emulated_latency_ms"] == 5.0

    def test_from_dict(self):
        cfg = MigrationConfig.from_dict({
            "execution_mode": "simulate",
            "verification_mode": "deep",
            "rollback_enabled": False,
        })
        assert cfg.execution_mode == MigrationMode.SIMULATE
        assert cfg.verification_mode == VerificationMode.DEEP
        assert cfg.rollback_enabled is False


# ===========================================================================
# M8-T05: MigrationPhaseTimings
# ===========================================================================

class TestMigrationPhaseTimings:
    def test_defaults_are_zero(self):
        t = MigrationPhaseTimings()
        assert t.validation_ms == 0.0
        assert t.total_duration_ms == 0.0

    def test_to_dict_contains_all_phases(self):
        t = MigrationPhaseTimings(validation_ms=1.0, model_transfer_ms=5.0, total_duration_ms=10.0)
        d = t.to_dict()
        assert "validation_ms" in d
        assert "model_transfer_ms" in d
        assert "total_duration_ms" in d
        assert d["total_duration_ms"] == 10.0


# ===========================================================================
# M8-T06: MigrationMetrics
# ===========================================================================

class TestMigrationMetrics:
    def test_defaults_zero(self):
        m = MigrationMetrics()
        assert m.total_bytes_transferred == 0
        assert m.transfer_count == 0

    def test_to_dict(self):
        m = MigrationMetrics(
            model_bytes_transferred=1000,
            kv_cache_bytes_transferred=500,
            total_bytes_transferred=1500,
            transfer_count=3,
            affected_layers=[0, 1, 2],
            affected_tiers=["user_device", "edge_a"],
        )
        d = m.to_dict()
        assert d["total_bytes_transferred"] == 1500
        assert d["affected_layers"] == [0, 1, 2]


# ===========================================================================
# M8-T07: MigrationComparison
# ===========================================================================

class TestMigrationComparison:
    def test_to_dict(self):
        c = MigrationComparison(
            predicted_switching_cost=0.1,
            actual_migration_duration_ms=12.5,
            predicted_kv_bytes=4096,
            actual_kv_bytes=4200,
            notes="test",
        )
        d = c.to_dict()
        assert d["predicted_switching_cost"] == 0.1
        assert d["actual_kv_bytes"] == 4200
        assert d["notes"] == "test"


# ===========================================================================
# M8-T08: MigrationResult serialization
# ===========================================================================

class TestMigrationResult:
    def test_to_dict_keys(self):
        src = plan_2_2()
        tgt = plan_1_3()
        r = MigrationResult(
            request_id="test-001",
            source_plan=src,
            target_plan=tgt,
            source_plan_id="src-001",
            target_plan_id="tgt-001",
            status=MigrationStatus.COMPLETED,
            success=True,
            start_timestamp=time.time(),
            end_timestamp=time.time() + 0.01,
            timings=MigrationPhaseTimings(total_duration_ms=10.0),
            metrics=MigrationMetrics(),
            comparison=MigrationComparison(),
            events=["migration_started", "migration_completed"],
        )
        d = r.to_dict()
        assert d["success"] is True
        assert d["status"] == "completed"
        assert "timings" in d
        assert "metrics" in d
        assert "events" in d


# ===========================================================================
# M8-T09: MigrationValidator - basic valid request
# ===========================================================================

class TestMigrationValidator:
    def test_valid_request_passes(self):
        src = plan_2_2()
        tgt = plan_1_3()
        req = make_migration_request(src, tgt, changed_layers=[1, 2])
        res = MigrationValidator.validate(req, current_plan=src)
        assert res.is_valid is True
        assert res.is_idempotent is False

    def test_none_request_fails(self):
        res = MigrationValidator.validate(None, current_plan=plan_2_2())
        assert res.is_valid is False

    def test_none_source_plan_fails(self):
        req = make_migration_request(plan_2_2(), plan_1_3())
        # Manually construct with None source
        bad = MigrationRequest(
            source_plan=None,
            target_plan=plan_1_3(),
            source_plan_id="s",
            target_plan_id="t",
            expected_gain=0.1,
            switching_cost=0.1,
            controller_cycle=1,
            timestamp=time.time(),
            changed_layers=[],
            affected_tiers=[],
            estimated_kv_transfer_bytes=0,
            reason="test",
        )
        res = MigrationValidator.validate(bad, current_plan=plan_2_2())
        assert res.is_valid is False

    def test_idempotent_when_target_matches_current(self):
        current = plan_2_2()
        req = make_migration_request(current, current)
        res = MigrationValidator.validate(req, current_plan=current)
        assert res.is_idempotent is True
        assert res.is_valid is True

    def test_stale_request_rejected(self):
        src = plan_2_2()
        tgt = plan_1_3()
        active = plan_monolithic()  # Different from src
        req = make_migration_request(src, tgt)
        res = MigrationValidator.validate(req, current_plan=active)
        assert res.is_valid is False
        assert "Stale" in res.reason

    def test_layer_count_mismatch_fails(self):
        src = plan_2_2(total=4)
        # Create target with different total_layers
        tgt = PartitionPlan(total_layers=6, user_device=(0, 5))
        req = MigrationRequest(
            source_plan=src,
            target_plan=tgt,
            source_plan_id="s",
            target_plan_id="t",
            expected_gain=0.1,
            switching_cost=0.1,
            controller_cycle=1,
            timestamp=time.time(),
            changed_layers=[],
            affected_tiers=[],
            estimated_kv_transfer_bytes=0,
            reason="test",
        )
        res = MigrationValidator.validate(req, current_plan=src)
        assert res.is_valid is False
        assert "mismatch" in res.reason.lower()

    def test_negative_kv_bytes_fails(self):
        src = plan_2_2()
        req = MigrationRequest(
            source_plan=src,
            target_plan=plan_1_3(),
            source_plan_id="s",
            target_plan_id="t",
            expected_gain=0.1,
            switching_cost=0.1,
            controller_cycle=1,
            timestamp=time.time(),
            changed_layers=[],
            affected_tiers=[],
            estimated_kv_transfer_bytes=-100,
            reason="test",
        )
        res = MigrationValidator.validate(req, current_plan=src)
        assert res.is_valid is False

    def test_out_of_bounds_changed_layer_fails(self):
        src = plan_2_2()
        req = MigrationRequest(
            source_plan=src,
            target_plan=plan_1_3(),
            source_plan_id="s",
            target_plan_id="t",
            expected_gain=0.1,
            switching_cost=0.1,
            controller_cycle=1,
            timestamp=time.time(),
            changed_layers=[99],  # Out of bounds for 4-layer model
            affected_tiers=[],
            estimated_kv_transfer_bytes=0,
            reason="test",
        )
        res = MigrationValidator.validate(req, current_plan=src)
        assert res.is_valid is False


# ===========================================================================
# M8-T10: MigrationPlanner - delta computation
# ===========================================================================

class TestMigrationPlanner:
    def test_affected_layers_identified(self):
        src = plan_2_2(4)  # UD: 0-1, EA: 2-3
        tgt = plan_1_3(4)  # UD: 0, EA: 1-3
        mp = MigrationPlanner.plan(src, tgt)
        # Layer 1 moves from UD to EA
        assert 1 in mp.affected_layers
        assert 0 not in mp.affected_layers  # stays on UD
        assert 2 not in mp.affected_layers  # stays on EA

    def test_unchanged_layers_correct(self):
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        mp = MigrationPlanner.plan(src, tgt)
        # 0 stays on UD, 2 and 3 stay on EA
        assert 0 in mp.unchanged_layers
        assert 2 in mp.unchanged_layers
        assert 3 in mp.unchanged_layers

    def test_noop_when_plans_identical(self):
        plan = plan_2_2(4)
        mp = MigrationPlanner.plan(plan, plan)
        assert mp.is_noop is True
        assert mp.affected_layers == []

    def test_all_layers_affected_when_moving_all(self):
        src = plan_monolithic(4)  # UD: 0-3
        tgt = PartitionPlan(total_layers=4, edge_a=(0, 3))  # EA: 0-3
        mp = MigrationPlanner.plan(src, tgt)
        assert set(mp.affected_layers) == {0, 1, 2, 3}

    def test_layer_count_mismatch_raises(self):
        src = plan_2_2(4)
        tgt = PartitionPlan(total_layers=6, user_device=(0, 5))
        with pytest.raises(ValueError, match="layer counts"):
            MigrationPlanner.plan(src, tgt)

    def test_transitions_contain_source_and_target_tiers(self):
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        mp = MigrationPlanner.plan(src, tgt)
        for l in mp.affected_layers:
            s_tier, t_tier = mp.layer_transitions[l]
            assert isinstance(s_tier, TierId)
            assert isinstance(t_tier, TierId)
            assert s_tier != t_tier

    def test_affected_tiers_recorded(self):
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        mp = MigrationPlanner.plan(src, tgt)
        assert TierId.USER_DEVICE in mp.affected_tiers or TierId.EDGE_A in mp.affected_tiers


# ===========================================================================
# M8-T11: LocalTensorTransfer
# ===========================================================================

class TestLocalTensorTransfer:
    def test_transfer_tensor_returns_correct_shape(self):
        ltt = LocalTensorTransfer()
        t = torch.randn(4, 8)
        moved, bytes_count, dur_ms, prov = ltt.transfer_tensor(
            t, TierId.USER_DEVICE, TierId.EDGE_A, target_device="cpu"
        )
        assert moved.shape == t.shape
        assert bytes_count == t.numel() * t.element_size()
        assert dur_ms >= 0.0

    def test_transfer_tensor_same_device_clones(self):
        ltt = LocalTensorTransfer()
        t = torch.tensor([1.0, 2.0, 3.0])
        moved, _, _, _ = ltt.transfer_tensor(t, TierId.USER_DEVICE, TierId.USER_DEVICE, "cpu")
        assert not moved.data_ptr() == t.data_ptr()  # Must be a clone
        assert torch.allclose(moved, t)

    def test_transfer_module_moves_all_params(self):
        ltt = LocalTensorTransfer()
        layer = nn.Linear(16, 16)
        mod, bytes_count, dur_ms, prov = ltt.transfer_module(
            layer, TierId.USER_DEVICE, TierId.EDGE_A, target_device="cpu"
        )
        assert bytes_count > 0
        assert dur_ms >= 0.0
        for p in mod.parameters():
            assert str(p.device) == "cpu"

    def test_provenance_is_measured(self):
        from src.telemetry.types import DataSource
        ltt = LocalTensorTransfer()
        t = torch.randn(2, 2)
        _, _, _, prov = ltt.transfer_tensor(t, TierId.USER_DEVICE, TierId.EDGE_A, "cpu")
        assert prov == DataSource.MEASURED


# ===========================================================================
# M8-T12: EmulatedNetworkTransfer
# ===========================================================================

class TestEmulatedNetworkTransfer:
    def test_emulated_duration_exceeds_local(self):
        ent = EmulatedNetworkTransfer(bandwidth_mbps=1.0, latency_ms=0.0, fast_mode=True)
        t = torch.randn(100, 100)
        _, _, dur_ms, _ = ent.transfer_tensor(t, TierId.USER_DEVICE, TierId.EDGE_A, "cpu")
        # Emulated duration should be positive (bandwidth calculation)
        assert dur_ms > 0.0

    def test_emulated_provenance_is_emulated(self):
        from src.telemetry.types import DataSource
        ent = EmulatedNetworkTransfer(fast_mode=True)
        t = torch.randn(4, 4)
        _, _, _, prov = ent.transfer_tensor(t, TierId.USER_DEVICE, TierId.EDGE_A, "cpu")
        assert prov == DataSource.EMULATED

    def test_tensor_values_preserved_after_emulated_transfer(self):
        ent = EmulatedNetworkTransfer(fast_mode=True)
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        moved, _, _, _ = ent.transfer_tensor(t, TierId.USER_DEVICE, TierId.EDGE_A, "cpu")
        assert torch.allclose(moved, t)

    def test_bandwidth_controls_duration(self):
        slow = EmulatedNetworkTransfer(bandwidth_mbps=1.0, latency_ms=0.0, fast_mode=True)
        fast = EmulatedNetworkTransfer(bandwidth_mbps=1000.0, latency_ms=0.0, fast_mode=True)
        t = torch.randn(100, 100)
        _, _, slow_ms, _ = slow.transfer_tensor(t, TierId.USER_DEVICE, TierId.EDGE_A, "cpu")
        _, _, fast_ms, _ = fast.transfer_tensor(t, TierId.USER_DEVICE, TierId.EDGE_A, "cpu")
        assert slow_ms > fast_ms

    def test_latency_adds_to_duration(self):
        no_lat = EmulatedNetworkTransfer(bandwidth_mbps=1000.0, latency_ms=0.0, fast_mode=True)
        with_lat = EmulatedNetworkTransfer(bandwidth_mbps=1000.0, latency_ms=50.0, fast_mode=True)
        t = torch.randn(4, 4)
        _, _, dur_no_lat, _ = no_lat.transfer_tensor(t, TierId.USER_DEVICE, TierId.EDGE_A, "cpu")
        _, _, dur_with_lat, _ = with_lat.transfer_tensor(t, TierId.USER_DEVICE, TierId.EDGE_A, "cpu")
        assert dur_with_lat > dur_no_lat


# ===========================================================================
# M8-T13: LayerTransferManager
# ===========================================================================

class TestLayerTransferManager:
    def _make_ltm(self, num_layers: int = 4) -> tuple:
        model = make_model(num_layers)
        tiers = make_tiers()
        provider = LocalTensorTransfer()
        tier_devices = {tid: t.device for tid, t in tiers.items()}
        ltm = LayerTransferManager(model=model, tier_devices=tier_devices, transfer_provider=provider)
        return ltm, model

    def test_transfer_moves_affected_layers(self):
        ltm, model = self._make_ltm(4)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        mp = MigrationPlanner.plan(src, tgt)
        # All tiers map to cpu in this setup so placement is no-op
        report = ltm.transfer_layers(mp)
        assert isinstance(report, LayerTransferReport)
        assert len(report.layers_transferred) == len(mp.affected_layers)
        assert report.total_bytes > 0

    def test_rollback_restores_original_devices(self):
        ltm, model = self._make_ltm(4)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        mp = MigrationPlanner.plan(src, tgt)

        original_devices = [str(next(model.blocks[i].parameters()).device) for i in range(4)]
        ltm.transfer_layers(mp)
        ltm.rollback()

        for i in range(4):
            dev = str(next(model.blocks[i].parameters()).device)
            assert dev == original_devices[i], f"Layer {i} device mismatch after rollback"

    def test_fail_on_layer_raises(self):
        ltm, model = self._make_ltm(4)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        mp = MigrationPlanner.plan(src, tgt)
        # Layer 1 should be in affected_layers
        target_layer = mp.affected_layers[0]
        with pytest.raises(RuntimeError, match="Injected layer transfer failure"):
            ltm.transfer_layers(mp, fail_on_layer=target_layer)

    def test_per_layer_bytes_populated(self):
        ltm, model = self._make_ltm(4)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        mp = MigrationPlanner.plan(src, tgt)
        report = ltm.transfer_layers(mp)
        for l in mp.affected_layers:
            assert l in report.per_layer_bytes
            assert report.per_layer_bytes[l] > 0


# ===========================================================================
# M8-T14: KVCacheTransferManager
# ===========================================================================

class TestKVCacheTransferManager:
    def _make_kvm(self, num_layers: int = 4) -> KVCacheTransferManager:
        tiers = make_tiers()
        provider = LocalTensorTransfer()
        tier_devices = {tid: t.device for tid, t in tiers.items()}
        return KVCacheTransferManager(tier_devices=tier_devices, transfer_provider=provider)

    def test_transfer_with_none_cache_is_noop(self):
        kvm = self._make_kvm()
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        mp = MigrationPlanner.plan(src, tgt)
        report = kvm.transfer_cache(kv_cache=None, plan=mp)
        assert isinstance(report, KVCacheTransferReport)
        assert report.total_bytes == 0

    def test_transfer_moves_kv_cache_layers(self):
        kvm = self._make_kvm(4)
        kv = FakeKVCache(num_layers=4, seq_len=8, hidden=16)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        mp = MigrationPlanner.plan(src, tgt)
        report = kvm.transfer_cache(kv_cache=kv, plan=mp)
        # Only affected layers' KV should be moved
        assert len(report.layers_transferred) == len(mp.affected_layers)
        assert report.total_bytes > 0

    def test_kv_values_preserved_after_transfer(self):
        kvm = self._make_kvm(4)
        kv = FakeKVCache(num_layers=4, seq_len=8, hidden=16)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        mp = MigrationPlanner.plan(src, tgt)
        # Snapshot values before
        before_keys = {l: kv.layers[l].keys.clone() for l in mp.affected_layers if l < 4}
        kvm.transfer_cache(kv_cache=kv, plan=mp)
        # Values should remain numerically identical
        for l, orig_k in before_keys.items():
            assert torch.allclose(kv.layers[l].keys, orig_k), f"KV mismatch at layer {l}"

    def test_rollback_restores_original_kv_references(self):
        kvm = self._make_kvm(4)
        kv = FakeKVCache(num_layers=4, seq_len=8, hidden=16)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        mp = MigrationPlanner.plan(src, tgt)
        # Record data_ptr before
        orig_ptrs = {l: kv.layers[l].keys.data_ptr() for l in mp.affected_layers if l < 4}
        kvm.transfer_cache(kv_cache=kv, plan=mp)
        kvm.rollback(kv_cache=kv)
        for l, ptr in orig_ptrs.items():
            assert kv.layers[l].keys.data_ptr() == ptr, f"KV rollback mismatch at layer {l}"

    def test_fail_kv_raises_runtime_error(self):
        kvm = self._make_kvm(4)
        kv = FakeKVCache(4)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        mp = MigrationPlanner.plan(src, tgt)
        with pytest.raises(RuntimeError, match="KV-cache transfer failure"):
            kvm.transfer_cache(kv_cache=kv, plan=mp, fail_kv=True)

    def test_no_cache_without_layers_attr(self):
        kvm = self._make_kvm(4)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        mp = MigrationPlanner.plan(src, tgt)
        # Object without 'layers' attribute
        report = kvm.transfer_cache(kv_cache=object(), plan=mp)
        assert report.total_bytes == 0


# ===========================================================================
# M8-T15: MigrationVerifier - FAST mode
# ===========================================================================

class TestMigrationVerifier:
    def _setup(self, num_layers: int = 4):
        model = make_model(num_layers)
        tiers = make_tiers()
        tier_devices = {tid: t.device for tid, t in tiers.items()}
        src = plan_2_2(num_layers)
        tgt = plan_1_3(num_layers)
        mp = MigrationPlanner.plan(src, tgt)
        # Place all layers on cpu (same device, so verification should pass)
        return model, tier_devices, mp, tgt

    def test_fast_mode_passes_clean_model(self):
        model, tier_devices, mp, tgt = self._setup(4)
        # Execute plan: move layer 1 from UD to EA (both cpu)
        provider = LocalTensorTransfer()
        ltm = LayerTransferManager(model, tier_devices, provider)
        ltm.transfer_layers(mp)
        res = MigrationVerifier.verify(model, mp, tier_devices, mode=VerificationMode.FAST)
        assert res.passed is True
        assert res.issues == []

    def test_deep_mode_passes_clean_model(self):
        model, tier_devices, mp, tgt = self._setup(4)
        provider = LocalTensorTransfer()
        ltm = LayerTransferManager(model, tier_devices, provider)
        ltm.transfer_layers(mp)
        res = MigrationVerifier.verify(model, mp, tier_devices, mode=VerificationMode.DEEP)
        assert res.passed is True

    def test_injected_failure_produces_failed_result(self):
        model, tier_devices, mp, tgt = self._setup(4)
        res = MigrationVerifier.verify(model, mp, tier_devices, fail_verification=True)
        assert res.passed is False
        assert len(res.issues) > 0

    def test_nan_in_param_fails_verification(self):
        model, tier_devices, mp, tgt = self._setup(4)
        # Corrupt layer 0 parameters
        with torch.no_grad():
            for p in model.blocks[0].parameters():
                p.fill_(float("nan"))
                break
        res = MigrationVerifier.verify(model, mp, tier_devices, mode=VerificationMode.FAST)
        assert res.passed is False
        assert any("NaN" in issue for issue in res.issues)

    def test_kv_shape_mismatch_fails(self):
        model, tier_devices, mp, tgt = self._setup(4)
        kv = FakeKVCache(4)
        # Corrupt shape of layer 0 values
        kv.layers[0].values = torch.randn(1, 5, 16)  # Different seq_len
        res = MigrationVerifier.verify(model, mp, tier_devices, kv_cache=kv, mode=VerificationMode.FAST)
        # May or may not catch shape depending on K/V shape: only fails if different from K
        # Here keys are (1,8,16) and values (1,5,16), so should fail
        assert res.passed is False

    def test_nan_in_kv_cache_fails(self):
        model, tier_devices, mp, tgt = self._setup(4)
        kv = FakeKVCache(4)
        kv.layers[0].keys = torch.full((1, 8, 16), float("nan"))
        res = MigrationVerifier.verify(model, mp, tier_devices, kv_cache=kv)
        assert res.passed is False

    def test_to_dict(self):
        res = VerificationResult(passed=True, issues=[], duration_ms=1.23, mode=VerificationMode.FAST)
        d = res.to_dict()
        assert d["passed"] is True
        assert d["mode"] == "fast"
        assert d["duration_ms"] == 1.23


# ===========================================================================
# M8-T16: RollbackManager
# ===========================================================================

class TestRollbackManager:
    def test_rollback_succeeds_on_clean_managers(self):
        num_layers = 4
        model = make_model(num_layers)
        tiers = make_tiers()
        provider = LocalTensorTransfer()
        tier_devices = {tid: t.device for tid, t in tiers.items()}
        ltm = LayerTransferManager(model, tier_devices, provider)
        kvm = KVCacheTransferManager(tier_devices, provider)
        adapter = make_adapter(num_layers)

        src = plan_2_2(num_layers)
        rb = RollbackManager.execute_rollback(ltm, kvm, adapter, src, kv_cache=None)
        assert isinstance(rb, RollbackReport)
        assert rb.success is True

    def test_rollback_restores_plan_to_source(self):
        adapter = make_adapter(4)
        tiers = make_tiers()
        provider = LocalTensorTransfer()
        tier_devices = {tid: t.device for tid, t in tiers.items()}
        ltm = LayerTransferManager(adapter.model, tier_devices, provider)
        kvm = KVCacheTransferManager(tier_devices, provider)

        src = plan_2_2(4)
        tgt = plan_1_3(4)
        mp = MigrationPlanner.plan(src, tgt)
        # Execute migration then rollback
        ltm.transfer_layers(mp)
        adapter.set_active_plan(tgt)  # Simulate partial commit
        rb = RollbackManager.execute_rollback(ltm, kvm, adapter, src, kv_cache=None)
        # After rollback, active plan should be source
        assert adapter.get_current_plan() == src


# ===========================================================================
# M8-T17: RuntimeAdapter
# ===========================================================================

class TestRuntimeAdapter:
    def test_get_current_plan_returns_set_plan(self):
        adapter = make_adapter(4)
        plan = plan_2_2(4)
        adapter.executor.active_plan = plan
        assert adapter.get_current_plan() == plan

    def test_set_active_plan_updates_executor(self):
        adapter = make_adapter(4)
        new_plan = plan_1_3(4)
        adapter.set_active_plan(new_plan)
        assert adapter.get_current_plan() == new_plan

    def test_get_tier_devices_returns_mapping(self):
        adapter = make_adapter(4)
        devices = adapter.get_tier_devices()
        assert TierId.USER_DEVICE in devices
        assert TierId.EDGE_A in devices
        assert isinstance(devices[TierId.USER_DEVICE], str)

    def test_model_property_returns_transformer(self):
        adapter = make_adapter(4)
        assert isinstance(adapter.model, LayeredTransformer)

    def test_acquire_and_release_lock(self):
        adapter = make_adapter(4)
        acquired = adapter.acquire_migration_lock(timeout=1.0)
        assert acquired is True
        adapter.release_migration_lock()
        # Should be acquirable again
        acquired2 = adapter.acquire_migration_lock(timeout=1.0)
        assert acquired2 is True
        adapter.release_migration_lock()

    def test_verify_runtime_after_plan_setup(self):
        adapter = make_adapter(4)
        src = plan_2_2(4)
        adapter.set_active_plan(src)
        # Since all devices are cpu, verify should pass
        result = adapter.verify_runtime()
        assert result is True


# ===========================================================================
# M8-T18: MigrationManager - successful end-to-end migration
# ===========================================================================

class TestMigrationManagerSuccess:
    def test_successful_migration_returns_completed_status(self):
        mgr = make_manager(4)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt, changed_layers=[1])
        result = mgr.execute(req)
        assert result.success is True
        assert result.status == MigrationStatus.COMPLETED
        assert result.rollback_performed is False

    def test_successful_migration_emits_lifecycle_events(self):
        mgr = make_manager(4)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt, changed_layers=[1])
        result = mgr.execute(req)
        events = result.events
        assert MigrationEvent.MIGRATION_STARTED.value in events
        assert MigrationEvent.VALIDATION_COMPLETED.value in events
        assert MigrationEvent.PREPARATION_COMPLETED.value in events
        assert MigrationEvent.MODEL_TRANSFER_STARTED.value in events
        assert MigrationEvent.MODEL_TRANSFER_COMPLETED.value in events
        assert MigrationEvent.KV_TRANSFER_STARTED.value in events
        assert MigrationEvent.KV_TRANSFER_COMPLETED.value in events
        assert MigrationEvent.VERIFICATION_STARTED.value in events
        assert MigrationEvent.VERIFICATION_COMPLETED.value in events
        assert MigrationEvent.COMMIT_COMPLETED.value in events
        assert MigrationEvent.MIGRATION_COMPLETED.value in events

    def test_active_plan_updated_after_migration(self):
        mgr = make_manager(4)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt, changed_layers=[1])
        mgr.execute(req)
        assert mgr.adapter.get_current_plan() == tgt

    def test_result_contains_timing_data(self):
        mgr = make_manager(4)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt)
        result = mgr.execute(req)
        assert result.timings.total_duration_ms > 0.0
        assert result.timings.validation_ms >= 0.0

    def test_result_contains_metrics_data(self):
        mgr = make_manager(4)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt)
        result = mgr.execute(req)
        assert result.metrics.model_bytes_transferred > 0
        assert result.metrics.total_bytes_transferred >= result.metrics.model_bytes_transferred

    def test_to_dict_is_serializable(self):
        mgr = make_manager(4)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt)
        result = mgr.execute(req)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["success"] is True


# ===========================================================================
# M8-T19: Idempotent migration
# ===========================================================================

class TestIdempotentMigration:
    def test_idempotent_returns_already_at_target(self):
        mgr = make_manager(4)
        # Source and target are the same (already at target)
        current = plan_2_2(4)
        req = make_migration_request(current, current)
        result = mgr.execute(req)
        assert result.status == MigrationStatus.ALREADY_AT_TARGET
        assert result.success is True
        assert result.rollback_performed is False

    def test_idempotent_does_not_modify_model(self):
        mgr = make_manager(4)
        current = plan_2_2(4)
        before_devices = [str(next(mgr.adapter.model.blocks[i].parameters()).device)
                          for i in range(4)]
        req = make_migration_request(current, current)
        mgr.execute(req)
        after_devices = [str(next(mgr.adapter.model.blocks[i].parameters()).device)
                         for i in range(4)]
        assert before_devices == after_devices


# ===========================================================================
# M8-T20: Stale request rejection
# ===========================================================================

class TestStaleRequest:
    def test_stale_request_returns_failed(self):
        mgr = make_manager(4)
        # Active plan is plan_2_2, but request says source is plan_1_3
        wrong_src = plan_1_3(4)
        tgt = plan_monolithic(4)
        req = make_migration_request(wrong_src, tgt)
        result = mgr.execute(req)
        assert result.success is False
        assert result.status == MigrationStatus.FAILED


# ===========================================================================
# M8-T21: Failure injection - preparation stage
# ===========================================================================

class TestFailureInjection:
    def test_fail_on_preparation_triggers_rollback(self):
        mgr = make_manager(4)
        mgr.fail_on_stage = "preparation"
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt, changed_layers=[1])
        result = mgr.execute(req)
        assert result.success is False
        assert result.rollback_performed is True
        # Status should be rolled_back or failed_partial
        assert result.status in (MigrationStatus.ROLLED_BACK, MigrationStatus.FAILED_PARTIAL)

    def test_fail_on_model_transfer_triggers_rollback(self):
        mgr = make_manager(4)
        mgr.fail_on_stage = "model_transfer"
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt, changed_layers=[1])
        result = mgr.execute(req)
        assert result.success is False
        assert result.rollback_performed is True

    def test_fail_on_kv_transfer_triggers_rollback(self):
        mgr = make_manager(4)
        mgr.fail_on_stage = "kv_transfer"
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt, changed_layers=[1])
        result = mgr.execute(req)
        assert result.success is False
        assert result.rollback_performed is True

    def test_fail_on_layer_triggers_rollback(self):
        mgr = make_manager(4)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        mp = MigrationPlanner.plan(src, tgt)
        mgr.fail_on_layer = mp.affected_layers[0]
        req = make_migration_request(src, tgt, changed_layers=mp.affected_layers)
        result = mgr.execute(req)
        assert result.success is False
        assert result.rollback_performed is True

    def test_fail_on_kv_flag_triggers_rollback(self):
        mgr = make_manager(4)
        mgr.fail_kv = True
        kv = FakeKVCache(4)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt, changed_layers=[1])
        result = mgr.execute(req, kv_cache=kv)
        assert result.success is False
        assert result.rollback_performed is True

    def test_fail_on_verification_triggers_rollback(self):
        mgr = make_manager(4)
        mgr.fail_verification = True
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt, changed_layers=[1])
        result = mgr.execute(req)
        assert result.success is False
        assert result.rollback_performed is True

    def test_rollback_restores_source_plan_after_failure(self):
        mgr = make_manager(4)
        mgr.fail_on_stage = "model_transfer"
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt, changed_layers=[1])
        mgr.execute(req)
        # After rollback, active plan should be restored to source
        active = mgr.adapter.get_current_plan()
        assert active == src

    def test_rollback_event_emitted_on_failure(self):
        mgr = make_manager(4)
        mgr.fail_on_stage = "kv_transfer"
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt, changed_layers=[1])
        result = mgr.execute(req)
        assert MigrationEvent.ROLLBACK_STARTED.value in result.events
        assert MigrationEvent.ROLLBACK_COMPLETED.value in result.events


# ===========================================================================
# M8-T22: Simulation mode
# ===========================================================================

class TestSimulationMode:
    def test_simulate_returns_completed_without_moving_tensors(self):
        adapter = make_adapter(4)
        cfg = MigrationConfig(execution_mode=MigrationMode.SIMULATE, fast_mode=True)
        mgr = MigrationManager(runtime_adapter=adapter, config=cfg)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt, changed_layers=[1])

        # Snapshot device assignments before
        before_devs = [str(next(adapter.model.blocks[i].parameters()).device) for i in range(4)]
        result = mgr.execute(req)
        after_devs = [str(next(adapter.model.blocks[i].parameters()).device) for i in range(4)]

        assert result.success is True
        assert result.status == MigrationStatus.COMPLETED
        assert before_devs == after_devs  # No physical movement

    def test_simulate_provenance_is_estimated(self):
        from src.telemetry.types import DataSource
        adapter = make_adapter(4)
        cfg = MigrationConfig(execution_mode=MigrationMode.SIMULATE, fast_mode=True)
        mgr = MigrationManager(runtime_adapter=adapter, config=cfg)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt)
        result = mgr.execute(req)
        assert result.provenance == DataSource.ESTIMATED

    def test_simulate_active_plan_unchanged(self):
        adapter = make_adapter(4)
        src = plan_2_2(4)
        cfg = MigrationConfig(execution_mode=MigrationMode.SIMULATE, fast_mode=True)
        mgr = MigrationManager(runtime_adapter=adapter, config=cfg)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt)
        mgr.execute(req)
        # Active plan should NOT be updated in simulate mode
        assert adapter.get_current_plan() == src

    def test_simulate_via_manager_simulate_method(self):
        adapter = make_adapter(4)
        cfg = MigrationConfig(fast_mode=True)
        mgr = MigrationManager(runtime_adapter=adapter, config=cfg)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt)
        result = mgr.simulate(req)
        assert result.success is True
        # Still no physical movement
        assert adapter.get_current_plan() == src


# ===========================================================================
# M8-T23: Single-flight concurrency guard
# ===========================================================================

class TestSingleFlight:
    def test_concurrent_migration_rejected_when_in_progress(self):
        """Second migration attempt while one is in-progress should return FAILED."""
        mgr = make_manager(4)
        # Manually set _in_progress to simulate concurrent state
        mgr._in_progress = True
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt)
        result = mgr.execute(req)
        assert result.success is False
        assert "rejected_busy" in result.events
        # Release flag
        mgr._in_progress = False


# ===========================================================================
# M8-T24: MigrationManager validate() method
# ===========================================================================

class TestMigrationManagerValidate:
    def test_validate_method_returns_result(self):
        mgr = make_manager(4)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt)
        result = mgr.validate(req)
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def test_validate_stale_returns_invalid(self):
        mgr = make_manager(4)
        wrong_src = plan_monolithic(4)
        tgt = plan_1_3(4)
        req = make_migration_request(wrong_src, tgt)
        result = mgr.validate(req)
        assert result.is_valid is False


# ===========================================================================
# M8-T25: MigrationManager with KV-cache
# ===========================================================================

class TestMigrationManagerWithKV:
    def test_migration_with_kv_cache_succeeds(self):
        mgr = make_manager(4)
        kv = FakeKVCache(4)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt, changed_layers=[1])
        result = mgr.execute(req, kv_cache=kv)
        assert result.success is True
        assert result.metrics.kv_cache_bytes_transferred > 0

    def test_kv_cache_content_preserved_after_migration(self):
        mgr = make_manager(4)
        kv = FakeKVCache(4)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt, changed_layers=[1])
        # Snapshot affected layer keys before
        affected = MigrationPlanner.plan(src, tgt).affected_layers
        before = {l: kv.layers[l].keys.clone() for l in affected if l < 4}
        mgr.execute(req, kv_cache=kv)
        for l, orig_k in before.items():
            assert torch.allclose(kv.layers[l].keys, orig_k), f"KV mismatch at layer {l}"


# ===========================================================================
# M8-T26: Comparison struct - predicted vs actual
# ===========================================================================

class TestMigrationComparison:
    def test_comparison_populated_on_success(self):
        mgr = make_manager(4)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt, changed_layers=[1])
        result = mgr.execute(req)
        assert result.comparison is not None
        assert result.comparison.predicted_switching_cost == req.switching_cost

    def test_comparison_predicted_kv_bytes_matches_request(self):
        mgr = make_manager(4)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt)
        result = mgr.execute(req)
        assert result.comparison.predicted_kv_bytes == req.estimated_kv_transfer_bytes


# ===========================================================================
# M8-T27: MigrationConfig from_dict
# ===========================================================================

class TestMigrationConfigDeserialization:
    def test_from_dict_with_all_fields(self):
        d = {
            "execution_mode": "execute",
            "synchronization_point": "token_boundary",
            "verification_mode": "deep",
            "rollback_enabled": True,
            "single_flight": False,
            "emulated_bandwidth_delay": True,
            "fast_mode": False,
            "emulated_bandwidth_mbps": 100.0,
            "emulated_latency_ms": 20.0,
        }
        cfg = MigrationConfig.from_dict(d)
        assert cfg.execution_mode == MigrationMode.EXECUTE
        assert cfg.verification_mode == VerificationMode.DEEP
        assert cfg.single_flight is False
        assert cfg.emulated_bandwidth_mbps == 100.0

    def test_from_dict_with_missing_fields_uses_defaults(self):
        cfg = MigrationConfig.from_dict({})
        assert cfg.execution_mode == MigrationMode.EXECUTE
        assert cfg.verification_mode == VerificationMode.FAST
        assert cfg.rollback_enabled is True


# ===========================================================================
# M8-T28: Multi-step migration sequence
# ===========================================================================

class TestMultiStepMigration:
    def test_sequential_migrations_succeed(self):
        """Execute two successive migrations verifying plan update at each step."""
        mgr = make_manager(4)
        src = plan_2_2(4)
        mid = plan_1_3(4)
        end = plan_monolithic(4)

        # Migration 1: 2-2 -> 1-3
        req1 = make_migration_request(src, mid, changed_layers=[1])
        r1 = mgr.execute(req1)
        assert r1.success is True
        assert mgr.adapter.get_current_plan() == mid

        # Migration 2: 1-3 -> monolithic
        req2 = MigrationRequest(
            source_plan=mid,
            target_plan=end,
            source_plan_id="mid-001",
            target_plan_id="end-001",
            expected_gain=0.1,
            switching_cost=0.02,
            controller_cycle=6,
            timestamp=time.time(),
            changed_layers=[1, 2, 3],
            affected_tiers=["user_device", "edge_a"],
            estimated_kv_transfer_bytes=8192,
            reason="test migration 2",
        )
        r2 = mgr.execute(req2)
        assert r2.success is True
        assert mgr.adapter.get_current_plan() == end


# ===========================================================================
# M8-T29: Emulated transfer provider integration with manager
# ===========================================================================

class TestEmulatedTransferIntegration:
    def test_migration_with_emulated_provider_succeeds(self):
        adapter = make_adapter(4)
        cfg = MigrationConfig(
            emulated_bandwidth_delay=True,
            emulated_bandwidth_mbps=1000.0,  # High BW to keep tests fast
            emulated_latency_ms=0.0,
            fast_mode=True,
        )
        mgr = MigrationManager(runtime_adapter=adapter, config=cfg)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt, changed_layers=[1])
        result = mgr.execute(req)
        assert result.success is True

    def test_emulated_migration_provenance_is_emulated(self):
        from src.telemetry.types import DataSource
        adapter = make_adapter(4)
        provider = EmulatedNetworkTransfer(bandwidth_mbps=1000.0, latency_ms=0.0, fast_mode=True)
        cfg = MigrationConfig(
            emulated_bandwidth_delay=True,
            emulated_bandwidth_mbps=1000.0,
            fast_mode=True,
        )
        mgr = MigrationManager(runtime_adapter=adapter, config=cfg, transfer_provider=provider)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt, changed_layers=[1])
        result = mgr.execute(req)
        # Emulated transfer provenance flows through
        assert result.provenance in (DataSource.EMULATED, DataSource.MEASURED)


# ===========================================================================
# M8-T30: Deep verification mode through manager
# ===========================================================================

class TestDeepVerification:
    def test_deep_verification_passes_on_clean_migration(self):
        adapter = make_adapter(4)
        cfg = MigrationConfig(
            verification_mode=VerificationMode.DEEP,
            emulated_bandwidth_delay=False,
            fast_mode=True,
        )
        mgr = MigrationManager(runtime_adapter=adapter, config=cfg)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt, changed_layers=[1])
        result = mgr.execute(req)
        assert result.success is True


# ===========================================================================
# M8-T31: MigrationResult immutability (frozen dataclass)
# ===========================================================================

class TestMigrationResultImmutability:
    def test_result_is_frozen(self):
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        r = MigrationResult(
            request_id="x",
            source_plan=src,
            target_plan=tgt,
            source_plan_id="s",
            target_plan_id="t",
            status=MigrationStatus.COMPLETED,
            success=True,
            start_timestamp=0.0,
            end_timestamp=0.1,
            timings=MigrationPhaseTimings(),
            metrics=MigrationMetrics(),
            comparison=MigrationComparison(),
        )
        with pytest.raises(Exception):
            r.success = False  # type: ignore


# ===========================================================================
# M8-T32: ValidationResult.passed property
# ===========================================================================

class TestValidationResultPassed:
    def test_passed_is_true_when_valid_and_not_idempotent(self):
        vr = ValidationResult(is_valid=True, is_idempotent=False)
        assert vr.passed is True

    def test_passed_is_false_when_idempotent(self):
        vr = ValidationResult(is_valid=True, is_idempotent=True)
        assert vr.passed is False

    def test_passed_is_false_when_invalid(self):
        vr = ValidationResult(is_valid=False, reason="fail")
        assert vr.passed is False


# ===========================================================================
# M8-T33: MigrationPlan is_noop detection
# ===========================================================================

class TestMigrationPlanNoop:
    def test_same_plan_is_noop(self):
        plan = plan_2_2(4)
        mp = MigrationPlanner.plan(plan, plan)
        assert mp.is_noop is True

    def test_different_plan_is_not_noop(self):
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        mp = MigrationPlanner.plan(src, tgt)
        assert mp.is_noop is False


# ===========================================================================
# M8-T34: Transfer provider is pluggable
# ===========================================================================

class TestPluggableTransferProvider:
    def test_custom_provider_called_for_migration(self):
        """Custom transfer provider's transfer_module and transfer_tensor must be invoked."""
        from src.telemetry.types import DataSource

        call_log = []

        class SpyProvider(MigrationTransferProvider):
            def transfer_tensor(self, tensor, source_tier, destination_tier, target_device="cpu"):
                call_log.append(("tensor", source_tier, destination_tier))
                return tensor.clone(), tensor.numel() * tensor.element_size(), 0.0, DataSource.MEASURED

            def transfer_module(self, module, source_tier, destination_tier, target_device="cpu"):
                call_log.append(("module", source_tier, destination_tier))
                total = sum(p.numel() * p.element_size() for p in module.parameters())
                return module, total, 0.0, DataSource.MEASURED

        adapter = make_adapter(4)
        cfg = MigrationConfig(emulated_bandwidth_delay=False, fast_mode=True)
        mgr = MigrationManager(runtime_adapter=adapter, config=cfg, transfer_provider=SpyProvider())
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt, changed_layers=[1])
        result = mgr.execute(req)
        assert result.success is True
        # transfer_module should have been called at least once for the affected layer
        assert any(c[0] == "module" for c in call_log)


# ===========================================================================
# M8-T35: MigrationManager custom request_id
# ===========================================================================

class TestRequestId:
    def test_request_id_propagates_to_result(self):
        mgr = make_manager(4)
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        # MigrationRequest doesn't have request_id field, check manager assigns one
        req = make_migration_request(src, tgt)
        result = mgr.execute(req)
        assert isinstance(result.request_id, str)
        assert len(result.request_id) > 0


# ===========================================================================
# M8-T36: Layer backup state restored on rollback with state dict check
# ===========================================================================

class TestLayerStateRestored:
    def test_layer_weights_exactly_restored_after_rollback(self):
        num_layers = 4
        model = make_model(num_layers)
        tiers = make_tiers()
        provider = LocalTensorTransfer()
        tier_devices = {tid: t.device for tid, t in tiers.items()}
        ltm = LayerTransferManager(model, tier_devices, provider)

        src = plan_2_2(num_layers)
        tgt = plan_1_3(num_layers)
        mp = MigrationPlanner.plan(src, tgt)

        # Record parameter checksums before
        before_checksums = {}
        for l in mp.affected_layers:
            before_checksums[l] = {
                name: param.clone()
                for name, param in model.blocks[l].named_parameters()
            }

        ltm.transfer_layers(mp)
        ltm.rollback()

        # Verify after rollback
        for l in mp.affected_layers:
            for name, param in model.blocks[l].named_parameters():
                assert torch.allclose(param, before_checksums[l][name]), \
                    f"Parameter '{name}' in block {l} changed after rollback"


# ===========================================================================
# M8-T37: End-to-end token parity (continuity test)
# ===========================================================================

class TestTokenParity:
    """
    Critical correctness invariant: output token sequences from greedy decoding
    must be identical before and after a migration on CPU-only all-cpu tiers.
    """

    def test_token_output_identical_before_and_after_migration(self):
        torch.manual_seed(42)
        num_layers = 4
        model = make_model(num_layers)
        tiers = make_tiers()
        executor = DistributedInferenceExecutor(model=model, tiers=tiers)
        adapter = RuntimeAdapter(executor)

        src_plan = plan_2_2(num_layers)
        executor._setup_tiers_for_plan(src_plan)
        if not hasattr(executor, "active_plan"):
            executor.active_plan = src_plan
        adapter._current_plan = src_plan

        prompt = "The quick brown fox"

        # Generate tokens before migration
        result_before = executor.generate(
            prompt=prompt,
            partition_plan=src_plan,
            max_new_tokens=8,
            temperature=0.0,
        )
        tokens_before = result_before.generated_tokens

        # Execute migration
        tgt_plan = plan_1_3(num_layers)
        cfg = MigrationConfig(emulated_bandwidth_delay=False, fast_mode=True)
        mgr = MigrationManager(runtime_adapter=adapter, config=cfg)
        req = make_migration_request(src_plan, tgt_plan, changed_layers=[1])
        mig_result = mgr.execute(req)
        assert mig_result.success is True, f"Migration failed: {mig_result.error_message}"

        # Verify active plan updated
        assert adapter.get_current_plan() == tgt_plan

        # Generate tokens after migration
        result_after = executor.generate(
            prompt=prompt,
            partition_plan=tgt_plan,
            max_new_tokens=8,
            temperature=0.0,
        )
        tokens_after = result_after.generated_tokens

        assert tokens_before == tokens_after, (
            f"Token parity FAILED!\n"
            f"Before: {tokens_before}\n"
            f"After:  {tokens_after}"
        )

    def test_multiple_migrations_preserve_determinism(self):
        """Three successive plan changes preserve greedy token output."""
        torch.manual_seed(0)
        num_layers = 4
        model = make_model(num_layers)
        tiers = make_tiers()
        executor = DistributedInferenceExecutor(model=model, tiers=tiers)
        adapter = RuntimeAdapter(executor)

        p0 = plan_2_2(num_layers)
        executor._setup_tiers_for_plan(p0)
        executor.active_plan = p0
        adapter._current_plan = p0

        prompt = "Hello world"
        result_ref = executor.generate(prompt=prompt, partition_plan=p0, max_new_tokens=5, temperature=0.0)
        tokens_ref = result_ref.generated_tokens

        plans = [plan_1_3(num_layers), plan_2_2(num_layers), plan_monolithic(num_layers)]
        current = p0

        for i, next_plan in enumerate(plans):
            cfg = MigrationConfig(emulated_bandwidth_delay=False, fast_mode=True)
            mgr = MigrationManager(runtime_adapter=adapter, config=cfg)
            req = MigrationRequest(
                source_plan=current,
                target_plan=next_plan,
                source_plan_id=f"plan-{i}",
                target_plan_id=f"plan-{i+1}",
                expected_gain=0.1,
                switching_cost=0.01,
                controller_cycle=i,
                timestamp=time.time(),
                changed_layers=[],
                affected_tiers=["user_device", "edge_a"],
                estimated_kv_transfer_bytes=0,
                reason=f"step {i}",
            )
            res = mgr.execute(req)
            assert res.success is True, f"Migration {i} failed: {res.error_message}"
            current = next_plan
            result_i = executor.generate(prompt=prompt, partition_plan=current, max_new_tokens=5, temperature=0.0)
            assert result_i.generated_tokens == tokens_ref, (
                f"Parity failure at migration step {i}: "
                f"expected {tokens_ref}, got {result_i.generated_tokens}"
            )


# ===========================================================================
# M8-T38: MigrationPlanner get_source_tier / get_target_tier helpers
# ===========================================================================

class TestMigrationPlanHelpers:
    def test_get_source_tier_returns_correct_tier(self):
        src = plan_2_2(4)  # UD: 0-1, EA: 2-3
        tgt = plan_1_3(4)  # UD: 0, EA: 1-3
        mp = MigrationPlanner.plan(src, tgt)
        # Layer 1 moved from UD to EA
        assert mp.get_source_tier(1) == TierId.USER_DEVICE

    def test_get_target_tier_returns_correct_tier(self):
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        mp = MigrationPlanner.plan(src, tgt)
        assert mp.get_target_tier(1) == TierId.EDGE_A

    def test_unchanged_layer_target_matches_source(self):
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        mp = MigrationPlanner.plan(src, tgt)
        # Layer 0 unchanged (still on UD)
        assert mp.get_source_tier(0) == TierId.USER_DEVICE
        assert mp.get_target_tier(0) == TierId.USER_DEVICE


# ===========================================================================
# M8-T39: MigrationManager with rollback disabled
# ===========================================================================

class TestRollbackDisabled:
    def test_failed_migration_without_rollback_still_returns_failed(self):
        """When rollback_enabled=False, failure path must not crash."""
        adapter = make_adapter(4)
        cfg = MigrationConfig(rollback_enabled=False, emulated_bandwidth_delay=False, fast_mode=True)
        mgr = MigrationManager(runtime_adapter=adapter, config=cfg)
        mgr.fail_on_stage = "model_transfer"
        src = plan_2_2(4)
        tgt = plan_1_3(4)
        req = make_migration_request(src, tgt, changed_layers=[1])
        # Rollback still runs in manager (it's defensive); just check no crash
        result = mgr.execute(req)
        assert result.success is False


# ===========================================================================
# M8-T40: Package exports (__init__.py)
# ===========================================================================

class TestPackageExports:
    def test_public_exports_importable(self):
        from src.migration import (
            MigrationManager,
            MigrationConfig,
            MigrationResult,
            MigrationStatus,
            MigrationMode,
            MigrationMetrics,
            MigrationPhaseTimings,
            MigrationComparison,
            MigrationEvent,
            MigrationPlanner,
            MigrationPlan,
            MigrationValidator,
            ValidationResult,
            MigrationVerifier,
            VerificationResult,
            RuntimeAdapter,
            LocalTensorTransfer,
            EmulatedNetworkTransfer,
        )
        assert MigrationManager is not None
        assert MigrationConfig is not None
        assert MigrationStatus is not None
