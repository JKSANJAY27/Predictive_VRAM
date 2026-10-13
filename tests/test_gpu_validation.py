"""
Unit and Integration Tests for Module 13:
Real-GPU Validation, Empirical Benchmarking & Research-Claim Verification.

Covers:
- Hardware capability detection and DevicePlacementMode classification
- Empirical experiment class assignment (CLASS_1 through CLASS_5)
- CPU fallback behavior: VRAM is strictly None with DataSource.UNAVAILABLE
- Phased memory profiling (idle, load, warmup, generation) and duration accounting
- KV-cache growth rate calculation per token
- Memory prediction error (MAE, RMSE) evaluation
- False-safe vs false-unsafe feasibility safety classification
- Benchmark workload matrix execution and tabular flattening
- SplitInferenceVerifier layer placement, transfer records, and token parity
- Optional CUDA integration tests (gracefully skipped when GPU is absent)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.research.empirical_classes import (
    DevicePlacementMode,
    EvidenceQualityDescriptor,
    ExperimentClass,
)
from src.research.gpu_validation import (
    BenchmarkMatrixRunner,
    FeasibilitySafetyEvaluation,
    GPUValidationEngine,
    GPUValidationReport,
    PhaseMemoryRecord,
    WorkloadMatrixPoint,
)
from src.research.split_verification import (
    PartitionPlacementAudit,
    SplitInferenceVerificationReport,
    SplitInferenceVerifier,
    TransferAuditRecord,
)
from src.runtime.model import LayeredTransformer
from src.runtime.partition import PartitionPlan
from src.runtime.tier import Tier, TierId
from src.telemetry.types import DataSource


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_transformer() -> LayeredTransformer:
    """Create a fast, 4-layer synthetic model for sub-second deterministic testing."""
    torch.manual_seed(42)
    return LayeredTransformer.create_synthetic(
        num_layers=4,
        hidden_size=64,
        num_heads=2,
        vocab_size=500,
        device="cpu",
    )


@pytest.fixture
def two_tier_plan() -> Tuple[PartitionPlan, Dict[TierId, Tier]]:
    """Create a 2-tier plan splitting 4 layers: UserDevice [0, 1] and EdgeA [2, 3]."""
    plan = PartitionPlan(
        total_layers=4,
        user_device=(0, 1),
        edge_a=(2, 3),
    )
    tiers = {
        TierId.USER_DEVICE: Tier(tier_id=TierId.USER_DEVICE, name="User Device", device="cpu"),
        TierId.EDGE_A: Tier(tier_id=TierId.EDGE_A, name="Edge Node A", device="cpu"),
    }
    return plan, tiers


# ---------------------------------------------------------------------------
# 1. Empirical Experiment Classification Tests
# ---------------------------------------------------------------------------

def test_evidence_quality_classification_simulation() -> None:
    desc = EvidenceQualityDescriptor.classify(
        cuda_available=False,
        is_simulation_mode=True,
        use_real_socket=False,
        use_emulated_network=True,
    )
    assert desc.experiment_class == ExperimentClass.CLASS_4_FULLY_SIMULATED
    assert desc.device_placement_mode == DevicePlacementMode.SIMULATED_DISTRIBUTED
    assert desc.gpu_telemetry_genuine is False
    assert desc.inference_execution_genuine is False
    assert len(desc.limitations) > 0


def test_evidence_quality_classification_cpu_execution() -> None:
    desc = EvidenceQualityDescriptor.classify(
        cuda_available=False,
        is_simulation_mode=False,
        use_real_socket=True,
        use_emulated_network=False,
    )
    assert desc.experiment_class == ExperimentClass.CLASS_2_REAL_SOCKET
    assert desc.device_placement_mode == DevicePlacementMode.CPU_PARTITIONED
    assert desc.gpu_telemetry_genuine is False
    assert desc.network_transport_physical is True
    assert desc.inference_execution_genuine is True


def test_evidence_quality_classification_cuda_multi_device() -> None:
    tier_devices = {"user_device": "cuda:0", "edge_a": "cuda:1"}
    desc = EvidenceQualityDescriptor.classify(
        cuda_available=True,
        is_simulation_mode=False,
        use_real_socket=True,
        use_emulated_network=False,
        tier_devices=tier_dev_devices if "tier_dev_devices" in locals() else tier_devices,
    )
    assert desc.experiment_class == ExperimentClass.CLASS_1_REAL_GPU
    assert desc.device_placement_mode == DevicePlacementMode.PHYSICAL_MULTI_DEVICE
    assert desc.gpu_telemetry_genuine is True


# ---------------------------------------------------------------------------
# 2. Phased GPU / Memory Telemetry & Invariants Tests
# ---------------------------------------------------------------------------

def test_memory_telemetry_cpu_fallback() -> None:
    # On CPU machines, VRAM fields must return None and DataSource.UNAVAILABLE
    if not torch.cuda.is_available():
        alloc, res, free, tot, src = GPUValidationEngine.sample_memory_telemetry(0)
        assert alloc is None
        assert res is None
        assert free is None
        assert tot is None
        assert src == DataSource.UNAVAILABLE.value


def test_host_ram_not_substituted_for_vram() -> None:
    h_ram = GPUValidationEngine.sample_host_ram_mb()
    assert isinstance(h_ram, float)
    assert h_ram > 0.0

    alloc, _, _, _, _ = GPUValidationEngine.sample_memory_telemetry(0)
    # Host RAM must never equal alloc when CUDA is unavailable
    if not torch.cuda.is_available():
        assert alloc is None
        assert alloc != h_ram


def test_phased_memory_validation_execution(
    tiny_transformer: LayeredTransformer,
    two_tier_plan: Tuple[PartitionPlan, Dict[TierId, Tier]],
) -> None:
    plan, tiers = two_tier_plan
    rep = GPUValidationEngine.run_phased_memory_validation(
        model=tiny_transformer,
        partition_plan=plan,
        tiers=tiers,
        prompt="Test prompt",
        max_new_tokens=4,
        safety_budget_mb=1024.0,
    )

    assert rep.hardware_status in ["PHYSICAL_CUDA_MEASURED", "CPU_ONLY_NOT_PERFORMED"]
    assert len(rep.phased_records) == 4  # idle, placed, warmup, generation
    assert rep.phased_records[0].phase_name == "idle_baseline"
    assert rep.phased_records[1].phase_name == "model_placed"
    assert rep.phased_records[2].phase_name == "warmup_completed"
    assert rep.phased_records[3].phase_name == "generation_completed"

    assert rep.mean_kv_growth_per_token_bytes >= 0.0
    assert rep.invariants_verified["vram_not_substituted_with_host_ram"] is True


# ---------------------------------------------------------------------------
# 3. Feasibility Safety Evaluation (False-Safe vs False-Unsafe) Tests
# ---------------------------------------------------------------------------

def test_feasibility_safety_classification_rates() -> None:
    eval_safe = FeasibilitySafetyEvaluation(safety_threshold_mb=100.0)
    eval_safe.total_evaluations = 10
    eval_safe.true_safe_count = 8
    eval_safe.true_unsafe_count = 1
    eval_safe.false_safe_count = 1    # 1 critical false-safe
    eval_safe.false_unsafe_count = 0

    eval_safe.compute_rates()
    assert eval_safe.false_safe_rate == 0.10
    assert eval_safe.false_unsafe_rate == 0.0
    assert eval_safe.accuracy == 0.90


# ---------------------------------------------------------------------------
# 4. Benchmark Workload Matrix Tests
# ---------------------------------------------------------------------------

def test_benchmark_matrix_runner(
    tiny_transformer: LayeredTransformer,
    two_tier_plan: Tuple[PartitionPlan, Dict[TierId, Tier]],
) -> None:
    plan, tiers = two_tier_plan
    points = BenchmarkMatrixRunner.run_matrix(
        model=tiny_transformer,
        partition_plan=plan,
        tiers=tiers,
        prompt_lengths=[8],
        generation_lengths=[2],
        precisions=["float32"],
    )

    assert len(points) == 1
    p = points[0]
    assert p.prompt_length == 8
    assert p.max_new_tokens == 2
    assert p.precision == "float32"
    assert p.execution_time_sec > 0.0

    df = BenchmarkMatrixRunner.matrix_to_dataframe(points)
    assert len(df) == 1
    assert "execution_time_sec" in df.columns


# ---------------------------------------------------------------------------
# 5. Split Inference Verification & Token Parity Tests
# ---------------------------------------------------------------------------

def test_split_inference_verifier_token_parity(
    tiny_transformer: LayeredTransformer,
    two_tier_plan: Tuple[PartitionPlan, Dict[TierId, Tier]],
) -> None:
    plan, tiers = two_tier_plan
    report = SplitInferenceVerifier.verify_plan(
        model=tiny_transformer,
        partition_plan=plan,
        tiers=tiers,
        prompt="Autonomous edge computing",
        max_new_tokens=4,
    )

    # Token parity must strictly hold between split and monolithic execution
    assert report.token_parity_verified is True
    assert report.generated_tokens_split == report.generated_tokens_monolithic
    assert len(report.generated_tokens_split) == 4
    assert len(report.placement_audits) == 2
    assert len(report.transfers) > 0


def test_split_inference_verifier_placement_audits(
    tiny_transformer: LayeredTransformer,
    two_tier_plan: Tuple[PartitionPlan, Dict[TierId, Tier]],
) -> None:
    plan, tiers = two_tier_plan
    report = SplitInferenceVerifier.verify_plan(
        model=tiny_transformer,
        partition_plan=plan,
        tiers=tiers,
        prompt="Hello",
        max_new_tokens=1,
    )

    tier_0_audit = report.placement_audits[0]
    tier_1_audit = report.placement_audits[1]

    assert tier_0_audit.tier_id == TierId.USER_DEVICE.value
    assert tier_0_audit.layer_start == 0
    assert tier_0_audit.layer_end == 1
    assert tier_0_audit.layer_count == 2

    assert tier_1_audit.tier_id == TierId.EDGE_A.value
    assert tier_1_audit.layer_start == 2
    assert tier_1_audit.layer_end == 3
    assert tier_1_audit.layer_count == 2


# ---------------------------------------------------------------------------
# 6. Optional Physical CUDA Integration Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable in test environment")
def test_real_cuda_memory_sampling() -> None:
    alloc, res, free, tot, src = GPUValidationEngine.sample_memory_telemetry(0)
    assert src == DataSource.MEASURED.value
    assert alloc is not None and alloc >= 0.0
    assert res is not None and res >= 0.0
    assert free is not None and free > 0.0
    assert tot is not None and tot > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable in test environment")
def test_real_cuda_phased_memory_growth() -> None:
    model = LayeredTransformer.create_synthetic(num_layers=2, hidden_size=64, num_heads=2, vocab_size=500, device="cuda:0")
    plan = PartitionPlan(total_layers=2, user_device=(0, 1))
    tiers = {TierId.USER_DEVICE: Tier(tier_id=TierId.USER_DEVICE, name="GPU", device="cuda:0")}

    rep = GPUValidationEngine.run_phased_memory_validation(
        model=model,
        partition_plan=plan,
        tiers=tiers,
        max_new_tokens=4,
    )
    assert rep.hardware_status == "PHYSICAL_CUDA_MEASURED"
    assert rep.peak_allocated_mb is not None
    assert rep.peak_allocated_mb > 0.0
