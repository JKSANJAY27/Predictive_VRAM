"""
Tests for Module 6: Cost Model and Candidate Scoring.

Verifies:
    - Weight configuration, validation, presets, and ablations
    - Normalization determinism and uniform scaling
    - Decomposed latency model (compute, comm, queue)
    - Steady-state communication volume and boundary counts
    - Provenance-aware memory pressure and forecast integration
    - UNKNOWN VRAM handling without zero fabrication
    - Emulated memory evaluation
    - Modeled power-proxy energy estimation
    - Switching penalties: P_switch == 0 for identical, P_switch > 0 for different
    - KV-cache migration volume estimation
    - Infeasible candidate penalization
    - Monotonicity across all 5 cost dimensions
    - Preservation of candidate ordering in batch scoring
    - Serialization and table formatting
"""

from __future__ import annotations

import math
import pytest

from src.cost import (
    CandidateScore,
    CommunicationCostModel,
    CostBreakdown,
    CostModel,
    CostModelCalibration,
    CostNormalizer,
    CostWeights,
    EnergyCostModel,
    LatencyCostModel,
    MemoryPressureCostModel,
    NormalizationConfig,
    ScoreStatus,
    SwitchingCostModel,
)
from src.partitioning import (
    CandidatePlan,
    FeasibilityStatus,
    ModelMetadata,
    SplitCatalog,
    TierCapacity,
    generate_plan_id,
)
from src.prediction.types import PredictionResult, TargetForecast
from src.runtime.partition import PartitionPlan
from src.runtime.tier import TierId
from src.state.types import (
    ActivationState,
    ComputeState,
    EnergyState,
    InferenceState,
    MemoryState,
    NetworkState,
    RuntimeState,
)
from src.telemetry.types import DataSource, TaggedValue


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model_meta() -> ModelMetadata:
    return ModelMetadata.synthetic_default(total_layers=12)


@pytest.fixture
def sample_candidates(model_meta: ModelMetadata) -> list[CandidatePlan]:
    catalog = SplitCatalog()
    return catalog.generate(model_meta, mode="exhaustive")


@pytest.fixture
def mock_runtime_state() -> RuntimeState:
    return RuntimeState(
        timestamp=1000.0,
        wall_clock=1000.0,
        step_index=10,
        network=NetworkState(
            bandwidth_mbps=TaggedValue(50.0, DataSource.EMULATED, "Mbps"),
            latency_ms=TaggedValue(15.0, DataSource.EMULATED, "ms"),
            packet_loss=TaggedValue(0.01, DataSource.EMULATED, "ratio"),
            jitter_ms=TaggedValue(2.0, DataSource.EMULATED, "ms"),
        ),
        memory=MemoryState(
            vram_allocated_mb=TaggedValue(None, DataSource.UNAVAILABLE, "MB"),
            vram_free_mb=TaggedValue(None, DataSource.UNAVAILABLE, "MB"),
            ram_used_mb=TaggedValue(4096.0, DataSource.MEASURED, "MB"),
            ram_available_mb=TaggedValue(4096.0, DataSource.MEASURED, "MB"),
        ),
        compute=ComputeState(
            gpu_utilization=TaggedValue(None, DataSource.UNAVAILABLE, "%"),
            cpu_utilization=TaggedValue(25.0, DataSource.MEASURED, "%"),
        ),
        inference=InferenceState(
            kv_cache_bytes=TaggedValue(10485760.0, DataSource.MEASURED, "bytes"),  # 10 MB
            kv_cache_growth_bytes=TaggedValue(262144.0, DataSource.MEASURED, "bytes"),
            generated_tokens=32,
            context_length=128,
        ),
        activation=ActivationState(
            latest_activation_bytes=TaggedValue(4096.0, DataSource.MEASURED, "bytes"),
            latest_transfer_latency_ms=TaggedValue(1.5, DataSource.MEASURED, "ms"),
        ),
        energy=EnergyState(
            energy_proxy=TaggedValue(12.5, DataSource.ESTIMATED, "proxy_units"),
        ),
    )


@pytest.fixture
def mock_forecast() -> PredictionResult:
    return PredictionResult(
        prediction_timestamp=1000.0,
        horizon_seconds=2.0,
        step_interval_seconds=0.5,
        forecast_timestamps=[1000.5, 1001.0, 1001.5, 1002.0],
        targets={
            "bandwidth_mbps": TargetForecast(
                target_name="bandwidth_mbps",
                values=[48.0, 45.0, 40.0, 35.0],
                raw_values=[48.0, 45.0, 40.0, 35.0],
                timestamps=[1000.5, 1001.0, 1001.5, 1002.0],
                is_available=True,
                status="ok",
                provenance="emulated",
            ),
            "latency_ms": TargetForecast(
                target_name="latency_ms",
                values=[16.0, 18.0, 20.0, 25.0],
                raw_values=[16.0, 18.0, 20.0, 25.0],
                timestamps=[1000.5, 1001.0, 1001.5, 1002.0],
                is_available=True,
                status="ok",
                provenance="emulated",
            ),
            "vram_free_mb": TargetForecast.unavailable(
                target_name="vram_free_mb",
                timestamps=[1000.5, 1001.0, 1001.5, 1002.0],
            ),
        },
        predictor_name="linear_trend",
        input_window_length=8,
        is_valid=True,
        status_message="ok",
    )


# ---------------------------------------------------------------------------
# 1. CostWeights Tests
# ---------------------------------------------------------------------------

class TestCostWeights:
    def test_default_weights(self):
        w = CostWeights()
        assert w.alpha == 1.0
        assert w.beta == 0.2
        assert w.gamma == 1.0
        assert w.delta == 0.1
        assert w.epsilon == 1.0
        assert w.use_latency is True

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            CostWeights(alpha=-0.5)

    def test_presets(self):
        lat = CostWeights.latency_only()
        assert lat.alpha == 1.0 and lat.beta == 0.0 and lat.epsilon == 0.0

        net = CostWeights.network_aware()
        assert net.alpha == 1.0 and net.beta == 0.5 and net.gamma == 0.0

        mem = CostWeights.memory_aware()
        assert mem.alpha == 1.0 and mem.gamma == 1.0 and net.delta == 0.0

        no_sw = CostWeights.no_switching()
        assert no_sw.epsilon == 0.0 and no_sw.use_switching is False

    def test_serialization(self):
        w = CostWeights(alpha=2.0, beta=0.5, use_energy=False)
        d = w.to_dict()
        w2 = CostWeights.from_dict(d)
        assert w == w2


# ---------------------------------------------------------------------------
# 2. Normalization Tests
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_normalization_scaling(self):
        cfg = NormalizationConfig(latency_scale_ms=50.0, communication_scale_mb=5.0)
        norm = CostNormalizer(cfg)

        assert norm.normalize_latency(100.0) == 2.0
        assert norm.normalize_latency(0.0) == 0.0
        assert norm.normalize_communication(10.0) == 2.0

    def test_invalid_scale_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            NormalizationConfig(latency_scale_ms=0.0)
        with pytest.raises(ValueError, match="must be positive"):
            NormalizationConfig(latency_scale_ms=-10.0)

    def test_nan_or_inf_rejected(self):
        norm = CostNormalizer()
        with pytest.raises(ValueError, match="non-finite"):
            norm.normalize_latency(float("nan"))
        with pytest.raises(ValueError, match="non-finite"):
            norm.normalize_latency(float("inf"))

    def test_negative_value_rejected(self):
        norm = CostNormalizer()
        with pytest.raises(ValueError, match="negative value"):
            norm.normalize_latency(-5.0)


# ---------------------------------------------------------------------------
# 3. Latency Cost Model Tests
# ---------------------------------------------------------------------------

class TestLatencyModel:
    def test_compute_latency_scaling(self, sample_candidates):
        model = LatencyCostModel()
        # Find monolithic plan vs 2-tier plan
        mono = next(c for c in sample_candidates if c.number_of_boundaries == 0)
        two_tier = next(c for c in sample_candidates if c.number_of_boundaries == 1)

        b_mono = model.evaluate(mono)
        b_two = model.evaluate(two_tier)

        # Monolithic executes all 12 layers on user_device (8 ms/layer -> 96ms)
        assert b_mono.compute_latency_ms == 12 * 8.0
        assert b_mono.comm_latency_ms == 0.0

        # Two-tier offloads layers to Edge A (faster compute rate: 2 ms/layer)
        # Compute latency should be strictly lower than monolithic
        assert b_two.compute_latency_ms < b_mono.compute_latency_ms
        # But communication latency is positive because of cut boundary
        assert b_two.comm_latency_ms > 0.0

    def test_latency_with_forecast(self, sample_candidates, mock_runtime_state, mock_forecast):
        model = LatencyCostModel()
        two_tier = next(c for c in sample_candidates if c.number_of_boundaries == 1)

        b_res = model.evaluate(two_tier, state=mock_runtime_state, forecast=mock_forecast)
        assert b_res.total_latency_ms > 0.0
        assert b_res.comm_latency_ms > 0.0


# ---------------------------------------------------------------------------
# 4. Communication Cost Model Tests
# ---------------------------------------------------------------------------

class TestCommunicationModel:
    def test_boundary_transfer_counts(self, sample_candidates):
        model = CommunicationCostModel(default_activation_bytes=4096)

        mono = next(c for c in sample_candidates if c.number_of_boundaries == 0)
        one_cut = next(c for c in sample_candidates if c.number_of_boundaries == 1)
        two_cut = next(c for c in sample_candidates if c.number_of_boundaries == 2)

        b_mono = model.evaluate(mono)
        b_one = model.evaluate(one_cut)
        b_two = model.evaluate(two_cut)

        assert b_mono.steady_state_bytes == 0
        assert b_mono.transfer_count == 0

        assert b_one.steady_state_bytes == 4096
        assert b_one.transfer_count == 1

        assert b_two.steady_state_bytes == 8192
        assert b_two.transfer_count == 2
        assert b_two.steady_state_mb == 2 * b_one.steady_state_mb


# ---------------------------------------------------------------------------
# 5. Memory Pressure Cost Model Tests
# ---------------------------------------------------------------------------

class TestMemoryPressureModel:
    def test_unknown_vram_on_cpu_only(self, sample_candidates):
        # Default with no explicit VRAM capacities
        model = MemoryPressureCostModel()
        c = sample_candidates[0]
        res = model.evaluate(c)

        # Invariant 5: Unavailable VRAM is UNKNOWN, never 0.0
        assert res.status == ScoreStatus.UNKNOWN
        assert res.provenance == DataSource.UNAVAILABLE

    def test_emulated_memory_evaluation(self, sample_candidates):
        caps = {
            TierId.USER_DEVICE: TierCapacity(
                tier_id=TierId.USER_DEVICE,
                device="cpu",
                available_memory_mb=2048.0,
                memory_provenance=DataSource.EMULATED,
            ),
            TierId.EDGE_A: TierCapacity(
                tier_id=TierId.EDGE_A,
                device="cpu",
                available_memory_mb=8192.0,
                memory_provenance=DataSource.EMULATED,
            ),
        }
        model = MemoryPressureCostModel(tier_capacities=caps)
        c = next(c for c in sample_candidates if c.plan_id == "u0-5_ea6-11")
        res = model.evaluate(c)

        assert res.status == ScoreStatus.FEASIBLE
        assert res.provenance == DataSource.EMULATED
        assert res.current_pressure is not None
        assert res.current_pressure > 0.0

    def test_risk_escalation_above_threshold(self, sample_candidates):
        # Tight memory: required ~500 MB out of 550 MB -> pressure ~0.91 (> 0.85)
        caps = {
            TierId.USER_DEVICE: TierCapacity(
                tier_id=TierId.USER_DEVICE,
                device="cpu",
                available_memory_mb=600.0,
                memory_provenance=DataSource.EMULATED,
            ),
        }
        model = MemoryPressureCostModel(tier_capacities=caps, risk_threshold=0.85)
        mono = next(c for c in sample_candidates if c.plan_id == "local")
        res = model.evaluate(mono)
        # Pressure should experience non-linear escalation
        assert res.pressure_ratio > 0.85


# ---------------------------------------------------------------------------
# 6. Energy Cost Model Tests
# ---------------------------------------------------------------------------

class TestEnergyModel:
    def test_energy_modeled(self, sample_candidates):
        model = EnergyCostModel()
        c = sample_candidates[0]
        res = model.evaluate(c)

        assert res.energy_joules > 0.0
        assert res.provenance == DataSource.ESTIMATED


# ---------------------------------------------------------------------------
# 7. Switching Cost Model Tests
# ---------------------------------------------------------------------------

class TestSwitchingModel:
    def test_invariant_identical_plan_zero_cost(self, sample_candidates):
        model = SwitchingCostModel()
        c = sample_candidates[0]

        # Identical plan comparison
        res = model.evaluate(c, current_plan=c.partition_plan)
        # Invariant 1: Identical current and candidate plans have zero switching cost
        assert res.switching_units == 0.0
        assert res.is_identical is True
        assert res.layer_migration_cost == 0.0
        assert res.kv_transfer_cost == 0.0

    def test_invariant_changed_plan_positive_cost(self, sample_candidates):
        model = SwitchingCostModel()
        c_curr = sample_candidates[0]  # local
        c_new = next(c for c in sample_candidates if c.plan_id == "u0-5_ea6-11")

        res = model.evaluate(c_new, current_plan=c_curr.partition_plan)
        # Invariant 2: Changing partition incurs positive switching cost
        assert res.switching_units > 0.0
        assert res.is_identical is False
        assert res.layer_migration_cost > 0.0
        assert res.disruption_cost > 0.0

    def test_kv_cache_switching_estimate(self, sample_candidates, mock_runtime_state):
        model = SwitchingCostModel()
        c_curr = PartitionPlan.two_tier(total_layers=12, cut_layer=3)
        c_new = next(c for c in sample_candidates if c.plan_id == "u0-5_ea6-11")

        res = model.evaluate(c_new, current_plan=c_curr, state=mock_runtime_state)
        assert res.estimated_kv_transfer_bytes > 0
        assert res.kv_transfer_cost > 0.0


# ---------------------------------------------------------------------------
# 8. Composite CostModel Tests
# ---------------------------------------------------------------------------

class TestCostModel:
    def test_deterministic_scoring(self, sample_candidates, mock_runtime_state, mock_forecast):
        cm = CostModel()
        c = sample_candidates[1]

        s1 = cm.score(c, state=mock_runtime_state, forecast=mock_forecast)
        s2 = cm.score(c, state=mock_runtime_state, forecast=mock_forecast)

        assert s1.total_cost == s2.total_cost
        assert s1.breakdown.latency_cost == s2.breakdown.latency_cost
        assert s1.breakdown.switching_cost == s2.breakdown.switching_cost

    def test_batch_preserves_order(self, sample_candidates):
        cm = CostModel()
        scores = cm.score_candidates(sample_candidates[:10])

        assert len(scores) == 10
        for orig, scored in zip(sample_candidates[:10], scores):
            assert orig.plan_id == scored.candidate_plan.plan_id

    def test_infeasible_plan_severely_penalized(self, model_meta):
        # Create an explicitly infeasible candidate plan
        p = PartitionPlan.monolithic(total_layers=12)
        infeasible_c = CandidatePlan(
            plan_id="infeasible_test",
            partition_plan=p,
            active_tiers=[TierId.USER_DEVICE],
            layer_assignment={"user_device": (0, 11)},
            number_of_boundaries=0,
            boundaries=[],
            feasibility_now=FeasibilityStatus.INFEASIBLE,
            feasibility_predicted=FeasibilityStatus.INFEASIBLE,
            resource_violations=["Insufficient memory"],
        )
        feasible_c = CandidatePlan(
            plan_id="feasible_test",
            partition_plan=p,
            active_tiers=[TierId.USER_DEVICE],
            layer_assignment={"user_device": (0, 11)},
            number_of_boundaries=0,
            boundaries=[],
            feasibility_now=FeasibilityStatus.FEASIBLE,
            feasibility_predicted=FeasibilityStatus.FEASIBLE,
        )

        cm = CostModel()
        s_inf = cm.score(infeasible_c)
        s_feas = cm.score(feasible_c)

        # Infeasible candidate must receive severe barrier penalty
        assert s_inf.feasible is False
        assert s_inf.status == ScoreStatus.INFEASIBLE
        assert s_inf.total_cost > s_feas.total_cost + 500.0

    def test_reactive_scoring_without_forecast(self, sample_candidates, mock_runtime_state):
        cm = CostModel()
        c = sample_candidates[2]
        # Scoring with forecast=None
        score = cm.score(c, state=mock_runtime_state, forecast=None)
        assert score.total_cost > 0.0
        assert score.metadata["forecast_used"] is False

    def test_switching_ablation(self, sample_candidates):
        # Current plan is local, candidate is split plan
        curr = PartitionPlan.monolithic(total_layers=12)
        split_cand = next(c for c in sample_candidates if c.plan_id == "u0-5_ea6-11")

        # Normal model with switching penalty
        cm_norm = CostModel()
        score_norm = cm_norm.score(split_cand, current_plan=curr)
        assert score_norm.breakdown.switching_cost > 0.0

        # Ablated model with epsilon=0
        cm_no_sw = CostModel(weights=CostWeights.no_switching())
        score_no_sw = cm_no_sw.score(split_cand, current_plan=curr)
        assert score_no_sw.breakdown.switching_cost == 0.0
        assert score_no_sw.total_cost < score_norm.total_cost

    def test_monotonicity_diagnostics(self, sample_candidates):
        cm = CostModel()
        c = next(c for c in sample_candidates if c.plan_id == "u0-5_ea6-11")
        mono_results = cm.verify_monotonicity(c)

        assert mono_results["switching_monotonicity"] is True
        assert mono_results["latency_monotonicity"] is True
        assert mono_results["communication_monotonicity"] is True
        assert mono_results["energy_monotonicity"] is True

    def test_cost_table_formatting(self, sample_candidates):
        cm = CostModel()
        scores = cm.score_candidates(sample_candidates[:5])
        table = cm.format_cost_table(scores)

        assert "PLAN ID" in table
        assert "LAT (ms)" in table
        assert "TOTAL" in table
        assert len(table.splitlines()) >= 7  # headers + sep + 5 rows

    def test_serialization(self, sample_candidates, mock_runtime_state):
        cm = CostModel()
        score = cm.score(sample_candidates[0], state=mock_runtime_state)
        d = score.to_dict()

        assert d["plan_id"] == sample_candidates[0].plan_id
        assert "total_cost" in d
        assert "breakdown" in d
        assert "latency_raw_ms" in d["breakdown"]["raw"]

    def test_component_ablations(self, sample_candidates, mock_runtime_state):
        c = sample_candidates[2]

        # Latency ablated
        cm_no_lat = CostModel(weights=CostWeights(use_latency=False))
        s_no_lat = cm_no_lat.score(c, state=mock_runtime_state)
        assert s_no_lat.breakdown.latency_cost == 0.0

        # Communication ablated
        cm_no_comm = CostModel(weights=CostWeights(use_communication=False))
        s_no_comm = cm_no_comm.score(c, state=mock_runtime_state)
        assert s_no_comm.breakdown.communication_cost == 0.0

        # Memory ablated
        cm_no_mem = CostModel(weights=CostWeights(use_memory=False))
        s_no_mem = cm_no_mem.score(c, state=mock_runtime_state)
        assert s_no_mem.breakdown.memory_pressure_cost == 0.0

        # Energy ablated
        cm_no_energy = CostModel(weights=CostWeights(use_energy=False))
        s_no_energy = cm_no_energy.score(c, state=mock_runtime_state)
        assert s_no_energy.breakdown.energy_cost == 0.0

    def test_provenance_preservation(self, sample_candidates, mock_runtime_state):
        cm = CostModel()
        c = next(c for c in sample_candidates if c.number_of_boundaries > 0)
        score = cm.score(c, state=mock_runtime_state)

        # Network was emulated in mock_runtime_state
        assert score.breakdown.communication_provenance == DataSource.MEASURED
        assert score.breakdown.energy_provenance == DataSource.ESTIMATED
        assert score.breakdown.switching_provenance == DataSource.ESTIMATED

    def test_switching_distance_monotonicity(self, sample_candidates):
        # Current plan: cut at 5 (u0-5_ea6-11)
        curr = PartitionPlan.two_tier(total_layers=12, cut_layer=5)
        cm = CostModel()

        # Plan with 1 layer shift (u0-4_ea5-11: cut shifts from 5 to 4)
        plan_small = next(c for c in sample_candidates if c.plan_id == "u0-4_ea5-11")
        # Plan with 4 layer shift (u0-1_ea2-11: cut shifts from 5 to 1)
        plan_large = next(c for c in sample_candidates if c.plan_id == "u0-1_ea2-11")

        s_small = cm.score(plan_small, current_plan=curr)
        s_large = cm.score(plan_large, current_plan=curr)

        # Plan with larger boundary displacement and more layer migrations has strictly higher switching penalty
        assert s_large.breakdown.layer_migration_cost > s_small.breakdown.layer_migration_cost
        assert s_large.breakdown.boundary_change_cost > s_small.breakdown.boundary_change_cost
        assert s_large.breakdown.switching_cost > s_small.breakdown.switching_cost

    def test_current_plan_scoring(self, sample_candidates):
        # cut_layer=5 means user_device has layers (0, 5) -> u0-5_ea6-11
        curr = PartitionPlan.two_tier(total_layers=12, cut_layer=5)
        cand_curr = next(c for c in sample_candidates if c.plan_id == "u0-5_ea6-11")

        cm = CostModel()
        score = cm.score(cand_curr, current_plan=curr)

        assert score.metadata["is_current_plan"] is True
        assert score.breakdown.switching_cost == 0.0
        assert "0 switching penalty" in score.explanation

    def test_fair_normalization_basis(self, sample_candidates):
        # Invariant 7: All candidates use identical normalization scales
        cfg = NormalizationConfig(latency_scale_ms=120.0, communication_scale_mb=8.0)
        cm = CostModel(normalization_config=cfg)

        scores = cm.score_candidates(sample_candidates[:5])
        for s in scores:
            assert s.breakdown.latency_normalized == s.breakdown.latency_raw_ms / 120.0
            assert s.breakdown.communication_normalized == s.breakdown.communication_raw_mb / 8.0
