"""
Module 2 — Runtime Telemetry Layer: comprehensive test suite.

Covers:
  - DataSource enum
  - TaggedValue constructors and to_dict
  - MemoryCollector: MEASURED RAM + UNAVAILABLE VRAM
  - CPUCollector: MEASURED fields, plausible values
  - ActivationCollector: MEASURED bytes + EMULATED latency
  - KVCacheCollector: MEASURED layer counts and bytes
  - NetworkConditionCollector: all EMULATED, scenario switching
  - TelemetryBuffer: capacity enforcement, ordering, thread safety
  - NetworkEmulationState: presets, update validation
  - Instrumented executor: buffer grows 1 per step, backward compatibility
  - TelemetrySnapshot.to_dict round-trip
  - KVCacheCollector growth delta across multiple steps
"""

from __future__ import annotations

import threading
import time
from typing import List

import pytest
import torch

from src.telemetry.types import (
    DataSource,
    TaggedValue,
    MemorySnapshot,
    CPUSnapshot,
    ActivationSnapshot,
    KVCacheSnapshot,
    NetworkConditionSnapshot,
    TelemetrySnapshot,
)
from src.telemetry.collectors import (
    MemoryCollector,
    CPUCollector,
    ActivationCollector,
    KVCacheCollector,
    NetworkConditionCollector,
)
from src.telemetry.buffer import TelemetryBuffer
from src.telemetry.network_emulation import NetworkEmulationState


# ===========================================================================
# Helpers
# ===========================================================================

def _make_snapshot(step: int = 0) -> TelemetrySnapshot:
    """Create a minimal valid TelemetrySnapshot for buffer testing."""
    return TelemetrySnapshot(step=step, timestamp=time.monotonic())


# ===========================================================================
# DataSource enum
# ===========================================================================

class TestDataSource:
    def test_all_values_exist(self):
        assert DataSource.MEASURED.value == "measured"
        assert DataSource.ESTIMATED.value == "estimated"
        assert DataSource.UNAVAILABLE.value == "unavailable"
        assert DataSource.EMULATED.value == "emulated"

    def test_str_returns_value(self):
        assert str(DataSource.MEASURED) == "measured"
        assert str(DataSource.EMULATED) == "emulated"


# ===========================================================================
# TaggedValue
# ===========================================================================

class TestTaggedValue:
    def test_measured_constructor(self):
        tv = TaggedValue.measured(42.5, "MB")
        assert tv.value == pytest.approx(42.5)
        assert tv.source == DataSource.MEASURED
        assert tv.unit == "MB"

    def test_estimated_constructor(self):
        tv = TaggedValue.estimated(3.14, "ms")
        assert tv.source == DataSource.ESTIMATED

    def test_emulated_constructor(self):
        tv = TaggedValue.emulated(100.0, "Mbps")
        assert tv.source == DataSource.EMULATED

    def test_unavailable_constructor(self):
        tv = TaggedValue.unavailable("MB")
        assert tv.value is None
        assert tv.source == DataSource.UNAVAILABLE

    def test_to_dict_keys(self):
        tv = TaggedValue.measured(5.0, "%")
        d = tv.to_dict()
        assert set(d.keys()) == {"value", "source", "unit"}
        assert d["source"] == "measured"

    def test_frozen(self):
        tv = TaggedValue.measured(1.0)
        with pytest.raises((AttributeError, TypeError)):
            tv.value = 99.0  # type: ignore[misc]


# ===========================================================================
# MemoryCollector
# ===========================================================================

class TestMemoryCollector:
    def test_returns_memory_snapshot(self):
        snap = MemoryCollector().collect()
        assert isinstance(snap, MemorySnapshot)

    def test_vram_always_unavailable(self):
        """Core research integrity: never fabricate GPU metrics."""
        snap = MemoryCollector().collect()
        assert snap.vram_total_mb.source == DataSource.UNAVAILABLE
        assert snap.vram_used_mb.source == DataSource.UNAVAILABLE
        assert snap.vram_free_mb.source == DataSource.UNAVAILABLE
        assert snap.vram_total_mb.value is None
        assert snap.vram_used_mb.value is None
        assert snap.vram_free_mb.value is None

    def test_ram_fields_measured_or_unavailable(self):
        """RAM fields must be either MEASURED (psutil present) or UNAVAILABLE."""
        snap = MemoryCollector().collect()
        for field in (snap.ram_total_mb, snap.ram_used_mb, snap.ram_available_mb, snap.ram_percent):
            assert field.source in (DataSource.MEASURED, DataSource.UNAVAILABLE), (
                f"Expected MEASURED or UNAVAILABLE, got {field.source}"
            )

    def test_ram_values_plausible_when_measured(self):
        """When psutil is present, RAM values should be positive."""
        snap = MemoryCollector().collect()
        if snap.ram_total_mb.source == DataSource.MEASURED:
            assert snap.ram_total_mb.value > 0
            assert snap.ram_available_mb.value >= 0
            assert 0.0 <= snap.ram_percent.value <= 100.0

    def test_process_ram_is_tagged(self):
        snap = MemoryCollector().collect()
        assert snap.process_ram_mb.source in (DataSource.MEASURED, DataSource.UNAVAILABLE)

    def test_snapshot_has_monotonic_timestamp(self):
        t_before = time.monotonic()
        snap = MemoryCollector().collect()
        t_after = time.monotonic()
        assert t_before <= snap.timestamp <= t_after

    def test_to_dict_has_expected_keys(self):
        snap = MemoryCollector().collect()
        d = snap.to_dict()
        assert "ram_used_mb" in d
        assert "vram_used_mb" in d
        assert d["vram_used_mb"]["source"] == "unavailable"


# ===========================================================================
# CPUCollector
# ===========================================================================

class TestCPUCollector:
    def test_returns_cpu_snapshot(self):
        snap = CPUCollector().collect()
        assert isinstance(snap, CPUSnapshot)

    def test_all_fields_tagged(self):
        snap = CPUCollector().collect()
        for field in (
            snap.cpu_percent_overall,
            snap.cpu_count_logical,
            snap.cpu_count_physical,
            snap.cpu_freq_mhz,
        ):
            assert isinstance(field, TaggedValue)
            assert field.source in (
                DataSource.MEASURED, DataSource.UNAVAILABLE
            ), f"Expected MEASURED or UNAVAILABLE, got {field.source}"

    def test_cpu_percent_plausible(self):
        snap = CPUCollector().collect()
        if snap.cpu_percent_overall.source == DataSource.MEASURED:
            assert 0.0 <= snap.cpu_percent_overall.value <= 100.0

    def test_core_count_positive(self):
        snap = CPUCollector().collect()
        if snap.cpu_count_logical.source == DataSource.MEASURED:
            assert snap.cpu_count_logical.value >= 1


# ===========================================================================
# ActivationCollector
# ===========================================================================

class TestActivationCollector:
    def test_returns_activation_snapshot(self):
        tensor = torch.zeros(1, 8, 64)
        snap = ActivationCollector().collect(
            tensor, source_tier="user_device", destination_tier="edge_a", step=0
        )
        assert isinstance(snap, ActivationSnapshot)

    def test_byte_size_measured_correctly(self):
        """1 * 8 * 64 elements * 4 bytes (float32) = 2048 bytes."""
        tensor = torch.zeros(1, 8, 64, dtype=torch.float32)
        snap = ActivationCollector().collect(
            tensor, source_tier="user_device", destination_tier="edge_a"
        )
        assert snap.byte_size.source == DataSource.MEASURED
        assert snap.byte_size.value == pytest.approx(1 * 8 * 64 * 4)

    def test_num_elements_measured(self):
        tensor = torch.zeros(2, 4, 16)
        snap = ActivationCollector().collect(
            tensor, source_tier="user_device", destination_tier="edge_a"
        )
        assert snap.num_elements.source == DataSource.MEASURED
        assert snap.num_elements.value == pytest.approx(2 * 4 * 16)

    def test_estimated_transfer_latency_emulated(self):
        """Transfer latency must be EMULATED, never MEASURED or fabricated."""
        tensor = torch.zeros(1, 8, 64)
        snap = ActivationCollector().collect(
            tensor, source_tier="user_device", destination_tier="edge_a"
        )
        assert snap.estimated_transfer_ms.source == DataSource.EMULATED
        assert snap.estimated_transfer_ms.value > 0

    def test_tensor_shape_preserved(self):
        tensor = torch.zeros(1, 5, 32)
        snap = ActivationCollector().collect(
            tensor, source_tier="user_device", destination_tier="edge_a"
        )
        assert snap.tensor_shape == (1, 5, 32)

    def test_rejects_non_tensor(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            ActivationCollector().collect(
                "not a tensor",  # type: ignore[arg-type]
                source_tier="user_device",
                destination_tier="edge_a",
            )

    def test_step_propagated(self):
        tensor = torch.zeros(1, 4, 64)
        snap = ActivationCollector().collect(
            tensor, source_tier="user_device", destination_tier="edge_a", step=7
        )
        assert snap.step == 7

    def test_network_state_affects_latency(self):
        """Higher bandwidth should produce lower estimated latency."""
        tensor = torch.zeros(1, 8, 64)
        state_fast = NetworkEmulationState(bandwidth_mbps=1000.0, rtt_ms=1.0)
        state_slow = NetworkEmulationState(bandwidth_mbps=1.0, rtt_ms=200.0)
        snap_fast = ActivationCollector(network_state=state_fast).collect(
            tensor, source_tier="user_device", destination_tier="edge_a"
        )
        snap_slow = ActivationCollector(network_state=state_slow).collect(
            tensor, source_tier="user_device", destination_tier="edge_a"
        )
        assert snap_fast.estimated_transfer_ms.value < snap_slow.estimated_transfer_ms.value


# ===========================================================================
# KVCacheCollector
# ===========================================================================

class TestKVCacheCollector:
    def _make_kv_cache(self, num_layers: int = 4, seq_len: int = 5):
        """Build a real DynamicCache from a synthetic model forward pass."""
        from transformers import DynamicCache
        from src.runtime.model import LayeredTransformer
        torch.manual_seed(0)
        model = LayeredTransformer.create_synthetic(
            num_layers=num_layers, hidden_size=64, num_heads=2, vocab_size=500
        )
        kv = DynamicCache()
        tokens = torch.randint(0, 500, (1, seq_len))
        hidden = model.embed(tokens)
        model.forward_layer_range(hidden, 0, num_layers - 1, past_key_values=kv)
        return kv

    def test_returns_kv_snapshot(self):
        kv = self._make_kv_cache()
        snap = KVCacheCollector().collect(kv, step=0)
        assert isinstance(snap, KVCacheSnapshot)

    def test_layer_count_matches(self):
        kv = self._make_kv_cache(num_layers=4)
        snap = KVCacheCollector().collect(kv, step=0)
        assert snap.num_cached_layers.source == DataSource.MEASURED
        assert snap.num_cached_layers.value == pytest.approx(4)

    def test_total_bytes_positive(self):
        kv = self._make_kv_cache()
        snap = KVCacheCollector().collect(kv, step=0)
        assert snap.total_kv_bytes.source == DataSource.MEASURED
        assert snap.total_kv_bytes.value > 0

    def test_growth_is_estimated(self):
        kv = self._make_kv_cache()
        snap = KVCacheCollector().collect(kv, step=0)
        assert snap.kv_growth_bytes_since_last.source == DataSource.ESTIMATED

    def test_growth_delta_positive_on_first_call(self):
        """First call: previous total is 0, so growth = total_bytes."""
        kv = self._make_kv_cache()
        collector = KVCacheCollector()
        snap = collector.collect(kv, step=0)
        assert snap.kv_growth_bytes_since_last.value == pytest.approx(
            snap.total_kv_bytes.value
        )

    def test_empty_cache_returns_zero_bytes(self):
        from transformers import DynamicCache
        empty_kv = DynamicCache()
        snap = KVCacheCollector().collect(empty_kv, step=0)
        assert snap.total_kv_bytes.value == pytest.approx(0)
        assert snap.num_cached_layers.value == pytest.approx(0)

    def test_per_layer_bytes_list_length(self):
        kv = self._make_kv_cache(num_layers=4)
        snap = KVCacheCollector().collect(kv, step=0)
        assert len(snap.per_layer_bytes) == 4


# ===========================================================================
# NetworkConditionCollector
# ===========================================================================

class TestNetworkConditionCollector:
    def test_returns_network_snapshot(self):
        snap = NetworkConditionCollector().collect()
        assert isinstance(snap, NetworkConditionSnapshot)

    def test_all_numeric_fields_emulated(self):
        """Core requirement: ALL network fields must be EMULATED."""
        snap = NetworkConditionCollector().collect()
        assert snap.rtt_ms.source == DataSource.EMULATED
        assert snap.bandwidth_mbps.source == DataSource.EMULATED
        assert snap.packet_loss_rate.source == DataSource.EMULATED
        assert snap.jitter_ms.source == DataSource.EMULATED

    def test_default_values_match_baseline(self):
        state = NetworkEmulationState()
        snap = NetworkConditionCollector(state=state).collect()
        assert snap.rtt_ms.value == pytest.approx(20.0)
        assert snap.bandwidth_mbps.value == pytest.approx(100.0)
        assert snap.packet_loss_rate.value == pytest.approx(0.0)
        assert snap.scenario_label == "baseline"

    def test_state_update_reflected_in_next_snapshot(self):
        state = NetworkEmulationState()
        collector = NetworkConditionCollector(state=state)
        state.update(rtt_ms=80.0, bandwidth_mbps=10.0, scenario_label="degraded")
        snap = collector.collect()
        assert snap.rtt_ms.value == pytest.approx(80.0)
        assert snap.bandwidth_mbps.value == pytest.approx(10.0)
        assert snap.scenario_label == "degraded"

    def test_to_dict_has_expected_keys(self):
        snap = NetworkConditionCollector().collect()
        d = snap.to_dict()
        assert "rtt_ms" in d
        assert "bandwidth_mbps" in d
        assert "packet_loss_rate" in d
        assert "scenario_label" in d


# ===========================================================================
# NetworkEmulationState
# ===========================================================================

class TestNetworkEmulationState:
    def test_default_values(self):
        s = NetworkEmulationState()
        assert s.rtt_ms == pytest.approx(20.0)
        assert s.bandwidth_mbps == pytest.approx(100.0)
        assert s.packet_loss_rate == pytest.approx(0.0)
        assert s.jitter_ms == pytest.approx(2.0)
        assert s.scenario_label == "baseline"

    def test_update_valid_fields(self):
        s = NetworkEmulationState()
        s.update(rtt_ms=50.0, bandwidth_mbps=25.0)
        assert s.rtt_ms == pytest.approx(50.0)
        assert s.bandwidth_mbps == pytest.approx(25.0)

    def test_update_rejects_unknown_field(self):
        s = NetworkEmulationState()
        with pytest.raises(ValueError, match="Unknown"):
            s.update(latency_us=100)

    def test_update_rejects_negative_rtt(self):
        s = NetworkEmulationState()
        with pytest.raises(ValueError):
            s.update(rtt_ms=-1.0)

    def test_update_rejects_zero_bandwidth(self):
        s = NetworkEmulationState()
        with pytest.raises(ValueError):
            s.update(bandwidth_mbps=0.0)

    def test_update_rejects_out_of_range_loss(self):
        s = NetworkEmulationState()
        with pytest.raises(ValueError):
            s.update(packet_loss_rate=1.5)

    def test_all_presets_valid(self):
        presets = NetworkEmulationState.preset_scenarios()
        expected = {"baseline", "degraded", "lossy", "weak_wifi", "mobile_4g", "ideal"}
        assert set(presets.keys()) == expected
        for name, state in presets.items():
            assert isinstance(state, NetworkEmulationState), f"Preset '{name}' invalid"
            assert state.bandwidth_mbps > 0
            assert 0.0 <= state.packet_loss_rate <= 1.0
            assert state.rtt_ms >= 0
            assert state.scenario_label == name

    def test_estimate_transfer_ms_positive(self):
        s = NetworkEmulationState(rtt_ms=20.0, bandwidth_mbps=100.0)
        ms = s.estimate_transfer_ms(byte_size=1024 * 1024)  # 1 MB
        assert ms > 0

    def test_faster_bandwidth_lower_latency(self):
        s_fast = NetworkEmulationState(rtt_ms=10.0, bandwidth_mbps=1000.0)
        s_slow = NetworkEmulationState(rtt_ms=10.0, bandwidth_mbps=1.0)
        assert s_fast.estimate_transfer_ms(1_000_000) < s_slow.estimate_transfer_ms(1_000_000)

    def test_to_dict_keys(self):
        s = NetworkEmulationState()
        d = s.to_dict()
        assert set(d.keys()) == {
            "rtt_ms", "bandwidth_mbps", "packet_loss_rate", "jitter_ms", "scenario_label"
        }


# ===========================================================================
# TelemetryBuffer
# ===========================================================================

class TestTelemetryBuffer:
    def test_capacity_enforced(self):
        buf = TelemetryBuffer(capacity=5)
        for i in range(10):
            buf.push(_make_snapshot(step=i))
        assert len(buf) == 5

    def test_oldest_evicted_when_full(self):
        buf = TelemetryBuffer(capacity=3)
        for i in range(5):
            buf.push(_make_snapshot(step=i))
        snaps = buf.get_all()
        # Only steps 2, 3, 4 should remain
        assert [s.step for s in snaps] == [2, 3, 4]

    def test_get_recent_returns_n(self):
        buf = TelemetryBuffer(capacity=10)
        for i in range(8):
            buf.push(_make_snapshot(step=i))
        recent = buf.get_recent(3)
        assert len(recent) == 3
        assert recent[-1].step == 7

    def test_get_recent_zero(self):
        buf = TelemetryBuffer(capacity=10)
        buf.push(_make_snapshot(step=0))
        assert buf.get_recent(0) == []

    def test_get_all_chronological_order(self):
        buf = TelemetryBuffer(capacity=10)
        for i in range(5):
            buf.push(_make_snapshot(step=i))
        snaps = buf.get_all()
        steps = [s.step for s in snaps]
        assert steps == sorted(steps)

    def test_clear_empties_buffer(self):
        buf = TelemetryBuffer(capacity=10)
        for i in range(5):
            buf.push(_make_snapshot(step=i))
        buf.clear()
        assert len(buf) == 0

    def test_rejects_invalid_capacity(self):
        with pytest.raises(ValueError):
            TelemetryBuffer(capacity=0)

    def test_rejects_non_snapshot(self):
        buf = TelemetryBuffer()
        with pytest.raises(TypeError):
            buf.push("not a snapshot")  # type: ignore[arg-type]

    def test_get_recent_returns_fewer_when_insufficient(self):
        buf = TelemetryBuffer(capacity=10)
        buf.push(_make_snapshot(step=0))
        recent = buf.get_recent(5)
        assert len(recent) == 1

    def test_summary_dict_empty(self):
        buf = TelemetryBuffer(capacity=10)
        summary = buf.summary_dict()
        assert summary["count"] == 0
        assert summary["step_range"] is None

    def test_summary_dict_populated(self):
        buf = TelemetryBuffer(capacity=10)
        for i in range(3):
            buf.push(_make_snapshot(step=i))
        summary = buf.summary_dict()
        assert summary["count"] == 3
        assert summary["step_range"] == (0, 2)

    def test_thread_safety_concurrent_push(self):
        """Multiple threads pushing concurrently must not corrupt state."""
        buf = TelemetryBuffer(capacity=1000)
        errors: List[Exception] = []

        def push_many(start: int, count: int) -> None:
            try:
                for i in range(count):
                    buf.push(_make_snapshot(step=start + i))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=push_many, args=(i * 100, 100))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert len(buf) == 500  # 5 threads × 100 pushes

    def test_repr_informative(self):
        buf = TelemetryBuffer(capacity=64)
        buf.push(_make_snapshot())
        r = repr(buf)
        assert "64" in r
        assert "1" in r


# ===========================================================================
# TelemetrySnapshot
# ===========================================================================

class TestTelemetrySnapshot:
    def test_minimal_snapshot(self):
        snap = TelemetrySnapshot(step=0, timestamp=time.monotonic())
        assert snap.step == 0
        assert snap.memory is None
        assert snap.activations == []

    def test_to_dict_round_trip(self):
        snap = TelemetrySnapshot(
            step=3,
            timestamp=1234.567,
            metadata={"foo": "bar"},
        )
        d = snap.to_dict()
        assert d["step"] == 3
        assert d["timestamp"] == pytest.approx(1234.567)
        assert d["memory"] is None
        assert d["activations"] == []
        assert d["metadata"]["foo"] == "bar"

    def test_frozen(self):
        snap = TelemetrySnapshot(step=0, timestamp=time.monotonic())
        with pytest.raises((AttributeError, TypeError)):
            snap.step = 99  # type: ignore[misc]


# ===========================================================================
# Instrumented Executor (Module 2 integration)
# ===========================================================================

class TestInstrumentedExecutor:
    """Integration tests: executor + telemetry_buffer together."""

    @pytest.fixture
    def executor_and_model(self, synthetic_model, standard_tiers):
        from src.runtime.executor import DistributedInferenceExecutor
        from src.runtime.transfer import TransferManager
        executor = DistributedInferenceExecutor(
            model=synthetic_model,
            tiers=standard_tiers,
            transfer_manager=TransferManager(),
        )
        return executor

    def test_buffer_grows_one_per_token_monolithic(self, executor_and_model):
        from src.runtime.partition import PartitionPlan
        buf = TelemetryBuffer(capacity=64)
        max_tokens = 5
        res = executor_and_model.generate(
            prompt="Test",
            partition_plan=PartitionPlan.monolithic(total_layers=4),
            max_new_tokens=max_tokens,
            temperature=0.0,
            telemetry_buffer=buf,
        )
        assert len(buf) == max_tokens
        assert res.telemetry_snapshot_count == max_tokens

    def test_buffer_grows_one_per_token_two_tier(self, executor_and_model):
        from src.runtime.partition import PartitionPlan
        buf = TelemetryBuffer(capacity=64)
        max_tokens = 4
        executor_and_model.generate(
            prompt="Split test",
            partition_plan=PartitionPlan.two_tier(total_layers=4, cut_layer=1),
            max_new_tokens=max_tokens,
            temperature=0.0,
            telemetry_buffer=buf,
        )
        assert len(buf) == max_tokens

    def test_two_tier_snapshots_have_activations(self, executor_and_model):
        """In a 2-tier split, each step should record exactly one activation."""
        from src.runtime.partition import PartitionPlan
        buf = TelemetryBuffer(capacity=64)
        executor_and_model.generate(
            prompt="Activation test",
            partition_plan=PartitionPlan.two_tier(total_layers=4, cut_layer=1),
            max_new_tokens=3,
            temperature=0.0,
            telemetry_buffer=buf,
        )
        for snap in buf.get_all():
            assert len(snap.activations) == 1, (
                f"Expected 1 activation per step in 2-tier, got {len(snap.activations)}"
            )

    def test_three_tier_snapshots_have_two_activations(self, executor_and_model):
        """In a 3-tier split, each step should record exactly 2 activations."""
        from src.runtime.partition import PartitionPlan
        buf = TelemetryBuffer(capacity=64)
        executor_and_model.generate(
            prompt="Three tier test",
            partition_plan=PartitionPlan.three_tier(total_layers=4, cut1=0, cut2=2),
            max_new_tokens=3,
            temperature=0.0,
            telemetry_buffer=buf,
        )
        for snap in buf.get_all():
            assert len(snap.activations) == 2

    def test_snapshots_have_memory_and_cpu(self, executor_and_model):
        """Every snapshot must carry both memory and CPU sub-snapshots."""
        from src.runtime.partition import PartitionPlan
        buf = TelemetryBuffer(capacity=64)
        executor_and_model.generate(
            prompt="Memory test",
            partition_plan=PartitionPlan.monolithic(total_layers=4),
            max_new_tokens=3,
            temperature=0.0,
            telemetry_buffer=buf,
        )
        for snap in buf.get_all():
            assert snap.memory is not None, "Expected memory snapshot in each step"
            assert snap.cpu is not None, "Expected CPU snapshot in each step"

    def test_snapshots_have_kv_cache(self, executor_and_model):
        from src.runtime.partition import PartitionPlan
        buf = TelemetryBuffer(capacity=64)
        executor_and_model.generate(
            prompt="KV cache test",
            partition_plan=PartitionPlan.monolithic(total_layers=4),
            max_new_tokens=3,
            temperature=0.0,
            telemetry_buffer=buf,
        )
        for snap in buf.get_all():
            assert snap.kv_cache is not None

    def test_snapshots_have_network(self, executor_and_model):
        from src.runtime.partition import PartitionPlan
        buf = TelemetryBuffer(capacity=64)
        executor_and_model.generate(
            prompt="Network test",
            partition_plan=PartitionPlan.monolithic(total_layers=4),
            max_new_tokens=3,
            temperature=0.0,
            telemetry_buffer=buf,
        )
        for snap in buf.get_all():
            assert snap.network is not None
            assert snap.network.rtt_ms.source == DataSource.EMULATED

    def test_backward_compatibility_no_buffer(self, executor_and_model):
        """Without telemetry_buffer, generate() must behave exactly as Module 1."""
        from src.runtime.partition import PartitionPlan
        res = executor_and_model.generate(
            prompt="Compat test",
            partition_plan=PartitionPlan.monolithic(total_layers=4),
            max_new_tokens=4,
            temperature=0.0,
            # No telemetry_buffer argument
        )
        assert res.telemetry_snapshot_count == 0
        assert res.telemetry_buffer is None
        assert res.generated_token_count == 4

    def test_result_references_buffer(self, executor_and_model):
        from src.runtime.partition import PartitionPlan
        buf = TelemetryBuffer(capacity=32)
        res = executor_and_model.generate(
            prompt="Ref test",
            partition_plan=PartitionPlan.monolithic(total_layers=4),
            max_new_tokens=3,
            temperature=0.0,
            telemetry_buffer=buf,
        )
        assert res.telemetry_buffer is buf

    def test_step_indices_sequential(self, executor_and_model):
        """Snapshot step indices must run 0, 1, 2, ..."""
        from src.runtime.partition import PartitionPlan
        buf = TelemetryBuffer(capacity=64)
        executor_and_model.generate(
            prompt="Step test",
            partition_plan=PartitionPlan.monolithic(total_layers=4),
            max_new_tokens=5,
            temperature=0.0,
            telemetry_buffer=buf,
        )
        steps = [s.step for s in buf.get_all()]
        assert steps == list(range(5))

    def test_vram_unavailable_in_captured_snapshots(self, executor_and_model):
        """Even after capture, VRAM must remain UNAVAILABLE (no GPU)."""
        from src.runtime.partition import PartitionPlan
        buf = TelemetryBuffer(capacity=16)
        executor_and_model.generate(
            prompt="VRAM integrity test",
            partition_plan=PartitionPlan.monolithic(total_layers=4),
            max_new_tokens=2,
            temperature=0.0,
            telemetry_buffer=buf,
        )
        for snap in buf.get_all():
            assert snap.memory.vram_used_mb.source == DataSource.UNAVAILABLE
            assert snap.memory.vram_used_mb.value is None
