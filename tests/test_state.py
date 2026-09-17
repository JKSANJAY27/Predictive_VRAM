"""
Unit and integration tests for Module 3: Unified Runtime State and Rolling StateBuffer.

Validates:
- RuntimeState immutability, field integrity, and DataSource provenance
- Physical domain validation and error handling
- Conversion from TelemetrySnapshot (full and partial)
- JSON serialization, deserialization, and trace replay
- StateBuffer ring-buffer behavior, FIFO eviction, and thread safety
- Strict chronological ordering enforcement (rejection of out-of-order states)
- FeatureExtractor column determinism, availability masking, and derived rates
- Deterministic min-max and z-score normalization
"""

import json
import threading
import time
from pathlib import Path
import pytest

from src.state.buffer import StateBuffer
from src.state.features import (
    BASE_FEATURE_NAMES,
    DERIVED_FEATURE_NAMES,
    FEATURE_NAMES,
    FeatureExtractor,
    FeatureVector,
)
from src.state.types import (
    ActivationState,
    ComputeState,
    EnergyState,
    InferenceState,
    MemoryState,
    NetworkState,
    RuntimeState,
)
from src.telemetry.types import (
    ActivationSnapshot,
    CPUSnapshot,
    DataSource,
    KVCacheSnapshot,
    MemorySnapshot,
    NetworkConditionSnapshot,
    TaggedValue,
    TelemetrySnapshot,
)


# ---------------------------------------------------------------------------
# Helpers for Synthetic Fixtures
# ---------------------------------------------------------------------------

def make_valid_runtime_state(
    timestamp: float = 100.0,
    step_index: int = 0,
    bandwidth_mbps: float = 50.0,
    latency_ms: float = 20.0,
    packet_loss: float = 0.01,
    jitter_ms: float = 2.0,
    ram_used_mb: float = 2048.0,
    ram_available_mb: float = 4096.0,
    cpu_percent: float = 35.0,
    kv_cache_bytes: float = 102400.0,
    generated_tokens: int = 5,
    request_state: str = "decode",
) -> RuntimeState:
    """Construct a clean, fully populated synthetic RuntimeState for tests."""
    return RuntimeState(
        timestamp=timestamp,
        wall_clock=1700000000.0 + timestamp,
        step_index=step_index,
        network=NetworkState(
            bandwidth_mbps=TaggedValue.emulated(bandwidth_mbps, "Mbps"),
            latency_ms=TaggedValue.emulated(latency_ms, "ms"),
            packet_loss=TaggedValue.emulated(packet_loss, "ratio"),
            jitter_ms=TaggedValue.emulated(jitter_ms, "ms"),
        ),
        memory=MemoryState(
            vram_allocated_mb=TaggedValue.unavailable("MB"),
            vram_free_mb=TaggedValue.unavailable("MB"),
            ram_used_mb=TaggedValue.measured(ram_used_mb, "MB"),
            ram_available_mb=TaggedValue.measured(ram_available_mb, "MB"),
        ),
        compute=ComputeState(
            gpu_utilization=TaggedValue.unavailable("%"),
            cpu_utilization=TaggedValue.measured(cpu_percent, "%"),
        ),
        inference=InferenceState(
            kv_cache_bytes=TaggedValue.measured(kv_cache_bytes, "bytes"),
            kv_cache_growth_bytes=TaggedValue.estimated(20480.0, "bytes"),
            generated_tokens=generated_tokens,
            context_length=generated_tokens + 10,
            generation_rate=15.0,
            request_state=request_state,
            request_id="req-test-01",
        ),
        activation=ActivationState(
            latest_activation_bytes=TaggedValue.measured(8192.0, "bytes"),
            latest_transfer_latency_ms=TaggedValue.estimated(4.2, "ms"),
        ),
        energy=EnergyState(
            energy_proxy=TaggedValue.estimated(cpu_percent * 0.05, "proxy_units"),
        ),
        metadata={"scenario": "test_synthetic"},
    )


# ---------------------------------------------------------------------------
# Test Suite: RuntimeState & Component Dataclasses
# ---------------------------------------------------------------------------

class TestRuntimeStateTypes:
    def test_runtime_state_instantiation_and_immutability(self):
        state = make_valid_runtime_state()
        assert state.timestamp == 100.0
        assert state.step_index == 0
        assert state.network.bandwidth_mbps.value == 50.0

        with pytest.raises(Exception):
            state.timestamp = 200.0  # type: ignore

    def test_provenance_preservation(self):
        state = make_valid_runtime_state()
        assert state.network.bandwidth_mbps.source == DataSource.EMULATED
        assert state.memory.vram_allocated_mb.source == DataSource.UNAVAILABLE
        assert state.memory.vram_allocated_mb.value is None
        assert state.memory.ram_used_mb.source == DataSource.MEASURED
        assert state.inference.kv_cache_growth_bytes.source == DataSource.ESTIMATED

    def test_validation_rejects_negative_bandwidth(self):
        with pytest.raises(ValueError, match="bandwidth_mbps cannot be negative"):
            NetworkState(
                bandwidth_mbps=TaggedValue.measured(-10.0, "Mbps"),
                latency_ms=TaggedValue.measured(5.0, "ms"),
                packet_loss=TaggedValue.measured(0.0, "ratio"),
                jitter_ms=TaggedValue.measured(1.0, "ms"),
            )

    def test_validation_rejects_negative_latency(self):
        with pytest.raises(ValueError, match="latency_ms cannot be negative"):
            NetworkState(
                bandwidth_mbps=TaggedValue.measured(10.0, "Mbps"),
                latency_ms=TaggedValue.measured(-1.0, "ms"),
                packet_loss=TaggedValue.measured(0.0, "ratio"),
                jitter_ms=TaggedValue.measured(1.0, "ms"),
            )

    def test_validation_rejects_out_of_range_packet_loss(self):
        with pytest.raises(ValueError, match="packet_loss must be in"):
            NetworkState(
                bandwidth_mbps=TaggedValue.measured(10.0, "Mbps"),
                latency_ms=TaggedValue.measured(5.0, "ms"),
                packet_loss=TaggedValue.measured(1.5, "ratio"),
                jitter_ms=TaggedValue.measured(1.0, "ms"),
            )

    def test_validation_rejects_negative_jitter(self):
        with pytest.raises(ValueError, match="jitter_ms cannot be negative"):
            NetworkState(
                bandwidth_mbps=TaggedValue.measured(10.0, "Mbps"),
                latency_ms=TaggedValue.measured(5.0, "ms"),
                packet_loss=TaggedValue.measured(0.0, "ratio"),
                jitter_ms=TaggedValue.measured(-0.5, "ms"),
            )

    def test_validation_rejects_negative_memory(self):
        with pytest.raises(ValueError, match="ram_used_mb cannot be negative"):
            MemoryState(
                vram_allocated_mb=TaggedValue.unavailable("MB"),
                vram_free_mb=TaggedValue.unavailable("MB"),
                ram_used_mb=TaggedValue.measured(-100.0, "MB"),
                ram_available_mb=TaggedValue.measured(100.0, "MB"),
            )

    def test_validation_rejects_out_of_range_cpu_utilization(self):
        with pytest.raises(ValueError, match="cpu_utilization must be in"):
            ComputeState(
                gpu_utilization=TaggedValue.unavailable("%"),
                cpu_utilization=TaggedValue.measured(120.0, "%"),
            )

    def test_validation_rejects_negative_kv_cache(self):
        with pytest.raises(ValueError, match="kv_cache_bytes cannot be negative"):
            InferenceState(
                kv_cache_bytes=TaggedValue.measured(-5.0, "bytes"),
                kv_cache_growth_bytes=TaggedValue.estimated(0.0, "bytes"),
                generated_tokens=1,
                context_length=1,
            )

    def test_validation_skips_unavailable_fields(self):
        # Should NOT raise any validation error even if value is None
        mem = MemoryState(
            vram_allocated_mb=TaggedValue.unavailable("MB"),
            vram_free_mb=TaggedValue.unavailable("MB"),
            ram_used_mb=TaggedValue.unavailable("MB"),
            ram_available_mb=TaggedValue.unavailable("MB"),
        )
        assert mem.ram_used_mb.source == DataSource.UNAVAILABLE
        assert mem.ram_used_mb.value is None


# ---------------------------------------------------------------------------
# Test Suite: TelemetrySnapshot Conversion
# ---------------------------------------------------------------------------

class TestTelemetryConversion:
    def test_from_telemetry_full_snapshot(self):
        mem_snap = MemorySnapshot(
            ram_total_mb=TaggedValue.measured(8000.0, "MB"),
            ram_used_mb=TaggedValue.measured(2500.0, "MB"),
            ram_available_mb=TaggedValue.measured(5500.0, "MB"),
            ram_percent=TaggedValue.measured(31.25, "%"),
            swap_used_mb=TaggedValue.measured(0.0, "MB"),
            swap_total_mb=TaggedValue.measured(1000.0, "MB"),
            vram_total_mb=TaggedValue.unavailable("MB"),
            vram_used_mb=TaggedValue.unavailable("MB"),
            vram_free_mb=TaggedValue.unavailable("MB"),
        )
        cpu_snap = CPUSnapshot(
            cpu_percent_overall=TaggedValue.measured(42.0, "%"),
            cpu_count_logical=TaggedValue.measured(8.0, "cores"),
            cpu_count_physical=TaggedValue.measured(4.0, "cores"),
            cpu_freq_mhz=TaggedValue.measured(2400.0, "MHz"),
        )
        net_snap = NetworkConditionSnapshot(
            rtt_ms=TaggedValue.emulated(15.0, "ms"),
            bandwidth_mbps=TaggedValue.emulated(100.0, "Mbps"),
            packet_loss_rate=TaggedValue.emulated(0.005, "ratio"),
            jitter_ms=TaggedValue.emulated(1.2, "ms"),
        )
        kv_snap = KVCacheSnapshot(
            num_cached_layers=TaggedValue.measured(4.0, "layers"),
            total_kv_bytes=TaggedValue.measured(65536.0, "bytes"),
            total_kv_mb=TaggedValue.measured(0.0625, "MB"),
            kv_growth_bytes_since_last=TaggedValue.estimated(16384.0, "bytes"),
            per_layer_bytes=[16384, 16384, 16384, 16384],
            sample_layer_shape=(1, 12, 1, 64),
        )
        act_snap = ActivationSnapshot(
            tensor_shape=(1, 1, 768),
            dtype="torch.float32",
            byte_size=TaggedValue.measured(3072.0, "bytes"),
            num_elements=TaggedValue.measured(768.0, "elements"),
            estimated_transfer_ms=TaggedValue.emulated(0.8, "ms"),
            source_tier="user_device",
            destination_tier="edge_a",
        )

        telemetry = TelemetrySnapshot(
            step=1,
            timestamp=10.5,
            memory=mem_snap,
            cpu=cpu_snap,
            kv_cache=kv_snap,
            network=net_snap,
            activations=[act_snap],
            metadata={"test_key": "val"},
        )

        state = RuntimeState.from_telemetry(telemetry)

        assert state.step_index == 1
        assert state.timestamp == 10.5
        assert state.network.bandwidth_mbps.value == 100.0
        assert state.memory.vram_allocated_mb.source == DataSource.UNAVAILABLE
        assert state.memory.ram_used_mb.value == 2500.0
        assert state.compute.cpu_utilization.value == 42.0
        assert state.inference.kv_cache_bytes.value == 65536.0
        assert state.activation.latest_activation_bytes is not None
        assert state.activation.latest_activation_bytes.value == 3072.0
        assert state.energy.energy_proxy.source == DataSource.ESTIMATED
        assert state.metadata["test_key"] == "val"

    def test_from_telemetry_partial_snapshot(self):
        # Empty sub-snapshots should yield TaggedValue.unavailable, not zeros
        telemetry = TelemetrySnapshot(
            step=0,
            timestamp=5.0,
            memory=None,
            cpu=None,
            kv_cache=None,
            network=None,
            activations=[],
        )

        state = RuntimeState.from_telemetry(telemetry)

        assert state.network.bandwidth_mbps.source == DataSource.UNAVAILABLE
        assert state.network.bandwidth_mbps.value is None
        assert state.memory.ram_used_mb.source == DataSource.UNAVAILABLE
        assert state.compute.cpu_utilization.source == DataSource.UNAVAILABLE
        assert state.inference.kv_cache_bytes.source == DataSource.UNAVAILABLE
        assert state.activation.latest_activation_bytes is None
        assert state.energy.energy_proxy.source == DataSource.UNAVAILABLE

    def test_from_telemetry_successive_rates(self):
        s1 = TelemetrySnapshot(
            step=0,
            timestamp=1.0,
            cpu=CPUSnapshot(
                cpu_percent_overall=TaggedValue.measured(20.0, "%"),
                cpu_count_logical=TaggedValue.measured(8.0, "cores"),
                cpu_count_physical=TaggedValue.measured(4.0, "cores"),
                cpu_freq_mhz=TaggedValue.measured(2000.0, "MHz"),
            ),
        )
        state1 = RuntimeState.from_telemetry(s1)

        s2 = TelemetrySnapshot(
            step=5,
            timestamp=2.0,
            cpu=CPUSnapshot(
                cpu_percent_overall=TaggedValue.measured(30.0, "%"),
                cpu_count_logical=TaggedValue.measured(8.0, "cores"),
                cpu_count_physical=TaggedValue.measured(4.0, "cores"),
                cpu_freq_mhz=TaggedValue.measured(2000.0, "MHz"),
            ),
        )
        state2 = RuntimeState.from_telemetry(s2, prev_state=state1)

        # dt = 1.0s, delta_tokens = 5 -> rate = 5.0 tokens/s
        assert state2.inference.generation_rate == 5.0
        assert state2.energy.energy_proxy.value is not None


# ---------------------------------------------------------------------------
# Test Suite: Serialization & Round-Trip
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_dict_round_trip(self):
        state = make_valid_runtime_state()
        data = state.to_dict()
        reconstructed = RuntimeState.from_dict(data)

        assert reconstructed.timestamp == state.timestamp
        assert reconstructed.step_index == state.step_index
        assert reconstructed.network.bandwidth_mbps == state.network.bandwidth_mbps
        assert reconstructed.memory.ram_used_mb == state.memory.ram_used_mb
        assert reconstructed.compute.cpu_utilization == state.compute.cpu_utilization
        assert reconstructed.inference.kv_cache_bytes == state.inference.kv_cache_bytes

    def test_json_round_trip(self):
        state = make_valid_runtime_state()
        json_str = state.to_json()
        reconstructed = RuntimeState.from_json(json_str)

        assert reconstructed.timestamp == state.timestamp
        assert reconstructed.inference.generated_tokens == state.inference.generated_tokens


# ---------------------------------------------------------------------------
# Test Suite: StateBuffer Ring Buffer & Replay
# ---------------------------------------------------------------------------

class TestStateBuffer:
    def test_buffer_capacity_and_fifo(self):
        buf = StateBuffer(capacity=3)
        assert buf.capacity == 3
        assert len(buf) == 0

        s1 = make_valid_runtime_state(timestamp=1.0, step_index=1)
        s2 = make_valid_runtime_state(timestamp=2.0, step_index=2)
        s3 = make_valid_runtime_state(timestamp=3.0, step_index=3)
        s4 = make_valid_runtime_state(timestamp=4.0, step_index=4)

        buf.append(s1)
        buf.append(s2)
        buf.append(s3)
        assert len(buf) == 3

        # FIFO eviction
        buf.append(s4)
        assert len(buf) == 3
        all_states = buf.get_all()
        assert [s.step_index for s in all_states] == [2, 3, 4]

    def test_latest_and_recent(self):
        buf = StateBuffer(capacity=5)
        assert buf.latest() is None
        assert buf.get_recent(2) == []

        buf.append(make_valid_runtime_state(timestamp=1.0, step_index=1))
        buf.append(make_valid_runtime_state(timestamp=2.0, step_index=2))
        buf.append(make_valid_runtime_state(timestamp=3.0, step_index=3))

        assert buf.latest().step_index == 3
        recent = buf.get_recent(2)
        assert len(recent) == 2
        assert [s.step_index for s in recent] == [2, 3]

    def test_get_window_metadata(self):
        buf = StateBuffer(capacity=10)
        buf.append(make_valid_runtime_state(timestamp=1.0, step_index=1))
        buf.append(make_valid_runtime_state(timestamp=2.0, step_index=2))

        win = buf.get_window(5)
        assert win["requested_length"] == 5
        assert win["actual_length"] == 2
        assert win["is_partial"] is True
        assert len(win["states"]) == 2

    def test_out_of_order_rejection_timestamp(self):
        buf = StateBuffer(capacity=5)
        buf.append(make_valid_runtime_state(timestamp=10.0, step_index=1))

        # Attempt to append earlier timestamp -> must raise ValueError
        with pytest.raises(ValueError, match="Out-of-order state rejected"):
            buf.append(make_valid_runtime_state(timestamp=9.0, step_index=2))

    def test_out_of_order_rejection_step_index(self):
        buf = StateBuffer(capacity=5)
        buf.append(make_valid_runtime_state(timestamp=10.0, step_index=2))

        # Attempt same timestamp with lower step_index -> must raise ValueError
        with pytest.raises(ValueError, match="Out-of-order state rejected"):
            buf.append(make_valid_runtime_state(timestamp=10.0, step_index=1))

    def test_clear_empties_buffer(self):
        buf = StateBuffer(capacity=5)
        buf.append(make_valid_runtime_state(timestamp=1.0))
        assert len(buf) == 1
        buf.clear()
        assert len(buf) == 0
        assert buf.latest() is None

    def test_summary_dict(self):
        buf = StateBuffer(capacity=10)
        s_empty = buf.summary_dict()
        assert s_empty["count"] == 0
        assert s_empty["time_span_s"] == 0.0

        buf.append(make_valid_runtime_state(timestamp=1.0, step_index=1))
        buf.append(make_valid_runtime_state(timestamp=4.5, step_index=4))
        s_pop = buf.summary_dict()
        assert s_pop["count"] == 2
        assert pytest.approx(s_pop["time_span_s"], 0.001) == 3.5
        assert s_pop["earliest_step"] == 1
        assert s_pop["latest_step"] == 4

    def test_thread_safety(self):
        buf = StateBuffer(capacity=200)
        barrier = threading.Barrier(4)

        # Thread-safe concurrent access test
        def worker(thread_idx: int):
            barrier.wait()
            for i in range(25):
                # Use monotonic timestamp with thread-level guarantee
                t = 1000.0 + thread_idx * 100 + i
                buf.append(make_valid_runtime_state(timestamp=t, step_index=i))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        # Run sequentially or concurrent with distinct disjoint timestamps
        # Since strict chronological order rejects out-of-order, test thread-safety with a single chronological writer and concurrent readers
        writer_buf = StateBuffer(capacity=200)
        stop_flag = threading.Event()

        def reader():
            while not stop_flag.is_set():
                _ = writer_buf.get_recent(5)
                _ = writer_buf.summary_dict()
                time.sleep(0.001)

        r_thread = threading.Thread(target=reader)
        r_thread.start()

        for step in range(50):
            writer_buf.append(make_valid_runtime_state(timestamp=float(step), step_index=step))
            time.sleep(0.001)

        stop_flag.set()
        r_thread.join()

        assert len(writer_buf) == 50

    def test_json_save_and_load_replay(self, tmp_path: Path):
        buf = StateBuffer(capacity=5)
        for i in range(3):
            buf.append(make_valid_runtime_state(timestamp=float(i), step_index=i))

        save_path = tmp_path / "trace.json"
        buf.save_json(save_path)
        assert save_path.exists()

        replayed = StateBuffer.from_json_file(save_path, capacity=10)
        assert len(replayed) == 3
        assert replayed.capacity == 10
        assert [s.step_index for s in replayed.get_all()] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Test Suite: FeatureExtractor & FeatureVector
# ---------------------------------------------------------------------------

class TestFeatureExtractor:
    def test_feature_names_order_and_count(self):
        assert len(FEATURE_NAMES) == 22
        assert len(BASE_FEATURE_NAMES) == 18
        assert len(DERIVED_FEATURE_NAMES) == 4
        assert FEATURE_NAMES[0] == "bandwidth_mbps"
        assert FEATURE_NAMES[4] == "vram_allocated_mb"
        assert FEATURE_NAMES[6] == "ram_used_mb"
        assert FEATURE_NAMES[18] == "kv_cache_growth_rate"
        assert FEATURE_NAMES[21] == "elapsed_time"

    def test_feature_vector_masks(self):
        state = make_valid_runtime_state()
        vec = FeatureExtractor.to_vector(state)

        assert len(vec.values) == len(FEATURE_NAMES)
        assert len(vec.availability_mask) == len(FEATURE_NAMES)
        assert len(vec.source_mask) == len(FEATURE_NAMES)

        # VRAM is UNAVAILABLE
        vram_idx = FEATURE_NAMES.index("vram_allocated_mb")
        assert vec.availability_mask[vram_idx] == 0
        assert vec.source_mask[vram_idx] == DataSource.UNAVAILABLE.value
        assert vec.values[vram_idx] is None
        assert not vec.is_available("vram_allocated_mb")

        # RAM is MEASURED
        ram_idx = FEATURE_NAMES.index("ram_used_mb")
        assert vec.availability_mask[ram_idx] == 1
        assert vec.source_mask[ram_idx] == DataSource.MEASURED.value
        assert vec.values[ram_idx] == 2048.0
        assert vec.is_available("ram_used_mb")

    def test_to_dense_fill_value(self):
        state = make_valid_runtime_state()
        vec = FeatureExtractor.to_vector(state)

        dense_zero = vec.to_dense(fill_value=0.0)
        dense_neg = vec.to_dense(fill_value=-999.0)

        vram_idx = FEATURE_NAMES.index("vram_allocated_mb")
        assert dense_zero[vram_idx] == 0.0
        assert dense_neg[vram_idx] == -999.0

    def test_derived_features_sequence(self):
        s1 = make_valid_runtime_state(
            timestamp=10.0,
            kv_cache_bytes=1000.0,
            ram_used_mb=2000.0,
            generated_tokens=10,
        )
        s2 = make_valid_runtime_state(
            timestamp=12.0,
            kv_cache_bytes=1600.0,
            ram_used_mb=2100.0,
            generated_tokens=14,
        )

        vectors = FeatureExtractor.from_states([s1, s2])
        assert len(vectors) == 2

        # First sample: elapsed = 0, rates = 0.0 (initial reference)
        vec1 = vectors[0]
        assert vec1.get("elapsed_time") == 0.0
        assert vec1.get("kv_cache_growth_rate") == 0.0
        assert vec1.get("memory_growth_rate") == 0.0

        # Second sample: dt = 2.0s
        # kv_cache growth = (1600 - 1000) / 2 = 300.0 B/s
        # mem growth = (2100 - 2000) / 2 = 50.0 MB/s
        # token growth = (14 - 10) / 2 = 2.0 tokens/s
        # elapsed_time = 12.0 - 10.0 = 2.0s
        vec2 = vectors[1]
        assert vec2.get("kv_cache_growth_rate") == 300.0
        assert vec2.get("memory_growth_rate") == 50.0
        assert vec2.get("token_growth_rate") == 2.0
        assert vec2.get("elapsed_time") == 2.0

    def test_derived_features_zero_delta_time_guarded(self):
        s1 = make_valid_runtime_state(timestamp=10.0)
        s2 = make_valid_runtime_state(timestamp=10.0)  # dt = 0

        vec = FeatureExtractor.to_vector(s2, prev_state=s1)
        # dt == 0 must return None (guarded against division by zero)
        assert vec.get("kv_cache_growth_rate") is None
        assert vec.get("memory_growth_rate") is None

    def test_to_matrix_extraction(self):
        s1 = make_valid_runtime_state(timestamp=1.0)
        s2 = make_valid_runtime_state(timestamp=2.0)

        matrix = FeatureExtractor.to_matrix([s1, s2], fill_value=0.0)
        assert len(matrix) == 2
        assert len(matrix[0]) == len(FEATURE_NAMES)
        assert isinstance(matrix[0][0], float)

    def test_min_max_normalization(self):
        values = [10.0, 50.0, None, 100.0]
        min_vals = [0.0, 0.0, 0.0, 100.0]
        max_vals = [100.0, 100.0, 100.0, 100.0]  # last has span 0

        norm = FeatureExtractor.normalize_min_max(values, min_vals, max_vals)
        assert norm[0] == 0.1
        assert norm[1] == 0.5
        assert norm[2] is None  # preserved
        assert norm[3] == 0.0   # span 0 returns 0.0

    def test_z_score_normalization(self):
        values = [15.0, 20.0, None, 5.0]
        means = [10.0, 20.0, 10.0, 5.0]
        stds = [5.0, 2.0, 5.0, 0.0]  # last has std 0

        z = FeatureExtractor.normalize_z_score(values, means, stds)
        assert z[0] == 1.0
        assert z[1] == 0.0
        assert z[2] is None  # preserved
        assert z[3] == 0.0   # std 0 returns 0.0
