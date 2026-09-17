"""
Module 5 Test Suite: Split Catalog and Feasible Candidate Plan Generation

Covers:
  - ModelMetadata construction, validation, and synthetic defaults
  - TierCapacity construction and validation
  - generate_plan_id determinism and format
  - enumerate_structural_plans exhaustiveness and deduplication
  - estimate_plan_memory correctness (parameter bytes, KV-cache, safety margin)
  - evaluate_plan_feasibility (FEASIBLE, INFEASIBLE, UNKNOWN paths)
  - CandidatePlan dataclass properties, serialization round-trip
  - compare_plans / PlanDifference delta analysis
  - format_candidate_table rendering
  - SplitCatalog.generate() exhaustive and filtered modes
  - SplitCatalog with disabled tiers
  - CPU-only provenance — VRAM never fabricated
"""

from __future__ import annotations

import pytest

from src.partitioning.candidate import CandidatePlan
from src.partitioning.catalog import SplitCatalog
from src.partitioning.comparison import PlanDifference, compare_plans
from src.partitioning.feasibility import FeasibilityStatus, evaluate_plan_feasibility
from src.partitioning.formatting import format_candidate_table
from src.partitioning.memory_estimator import TierMemoryRequirement, estimate_plan_memory
from src.partitioning.metadata import ModelMetadata, TierCapacity
from src.partitioning.plan_id import generate_plan_id
from src.partitioning.structural import enumerate_structural_plans
from src.runtime.partition import PartitionPlan
from src.runtime.tier import TierId
from src.telemetry.types import DataSource


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def meta_12() -> ModelMetadata:
    """Standard 12-layer GPT-2-like metadata."""
    return ModelMetadata.synthetic_default(total_layers=12)


@pytest.fixture
def meta_4() -> ModelMetadata:
    """Small 4-layer model for fast enumeration tests."""
    return ModelMetadata.synthetic_default(total_layers=4)


@pytest.fixture
def tier_caps_cpu():
    """All-CPU tiers, no memory limit configured (UNAVAILABLE provenance)."""
    return {
        TierId.USER_DEVICE: TierCapacity(tier_id=TierId.USER_DEVICE, device="cpu", is_enabled=True),
        TierId.EDGE_A: TierCapacity(tier_id=TierId.EDGE_A, device="cpu", is_enabled=True),
        TierId.EDGE_B: TierCapacity(tier_id=TierId.EDGE_B, device="cpu", is_enabled=True),
    }


@pytest.fixture
def tier_caps_abundant():
    """Tiers with large configured available_memory_mb so all plans are FEASIBLE."""
    return {
        TierId.USER_DEVICE: TierCapacity(
            tier_id=TierId.USER_DEVICE, device="cpu",
            available_memory_mb=32768.0,
            memory_provenance=DataSource.EMULATED,
        ),
        TierId.EDGE_A: TierCapacity(
            tier_id=TierId.EDGE_A, device="cpu",
            available_memory_mb=32768.0,
            memory_provenance=DataSource.EMULATED,
        ),
        TierId.EDGE_B: TierCapacity(
            tier_id=TierId.EDGE_B, device="cpu",
            available_memory_mb=32768.0,
            memory_provenance=DataSource.EMULATED,
        ),
    }


@pytest.fixture
def tier_caps_starved():
    """Tiers with tiny available_memory_mb — forces INFEASIBLE."""
    return {
        TierId.USER_DEVICE: TierCapacity(
            tier_id=TierId.USER_DEVICE, device="cpu",
            available_memory_mb=1.0,
            memory_provenance=DataSource.EMULATED,
        ),
        TierId.EDGE_A: TierCapacity(
            tier_id=TierId.EDGE_A, device="cpu",
            available_memory_mb=1.0,
            memory_provenance=DataSource.EMULATED,
        ),
        TierId.EDGE_B: TierCapacity(
            tier_id=TierId.EDGE_B, device="cpu",
            available_memory_mb=1.0,
            memory_provenance=DataSource.EMULATED,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. ModelMetadata
# ─────────────────────────────────────────────────────────────────────────────

class TestModelMetadata:

    def test_synthetic_default_12(self, meta_12):
        assert meta_12.total_layers == 12
        assert meta_12.hidden_size == 768
        assert meta_12.num_heads == 12
        assert meta_12.dtype == "float32"
        assert meta_12.dtype_bytes == 4

    def test_head_dim_property(self, meta_12):
        assert meta_12.head_dim == 64  # 768 / 12

    def test_per_layer_params_length(self, meta_12):
        assert len(meta_12.per_layer_params) == 12

    def test_synthetic_default_custom_layers(self):
        m = ModelMetadata.synthetic_default(total_layers=6)
        assert m.total_layers == 6
        assert len(m.per_layer_params) == 6

    def test_invalid_total_layers(self):
        with pytest.raises(ValueError, match="total_layers must be positive"):
            ModelMetadata(total_layers=0, hidden_size=768, num_heads=12)

    def test_invalid_hidden_size(self):
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            ModelMetadata(total_layers=4, hidden_size=-1, num_heads=4)

    def test_invalid_num_heads(self):
        with pytest.raises(ValueError, match="num_heads must be positive"):
            ModelMetadata(total_layers=4, hidden_size=256, num_heads=0)

    def test_round_trip_dict(self, meta_12):
        d = meta_12.to_dict()
        recovered = ModelMetadata.from_dict(d)
        assert recovered.total_layers == meta_12.total_layers
        assert recovered.hidden_size == meta_12.hidden_size
        assert recovered.num_heads == meta_12.num_heads
        assert recovered.per_layer_params == meta_12.per_layer_params


# ─────────────────────────────────────────────────────────────────────────────
# 2. TierCapacity
# ─────────────────────────────────────────────────────────────────────────────

class TestTierCapacity:

    def test_default_memory_provenance_unavailable(self):
        cap = TierCapacity(tier_id=TierId.USER_DEVICE)
        assert cap.memory_provenance == DataSource.UNAVAILABLE

    def test_negative_total_memory_rejected(self):
        with pytest.raises(ValueError, match="total_memory_mb cannot be negative"):
            TierCapacity(tier_id=TierId.USER_DEVICE, total_memory_mb=-10.0)

    def test_negative_available_memory_rejected(self):
        with pytest.raises(ValueError, match="available_memory_mb cannot be negative"):
            TierCapacity(tier_id=TierId.USER_DEVICE, available_memory_mb=-5.0)

    def test_zero_compute_capacity_rejected(self):
        with pytest.raises(ValueError, match="compute_capacity_score must be > 0"):
            TierCapacity(tier_id=TierId.USER_DEVICE, compute_capacity_score=0.0)

    def test_negative_max_layers_rejected(self):
        with pytest.raises(ValueError, match="max_layers cannot be negative"):
            TierCapacity(tier_id=TierId.USER_DEVICE, max_layers=-1)

    def test_round_trip_dict(self):
        cap = TierCapacity(
            tier_id=TierId.EDGE_A,
            device="cpu",
            total_memory_mb=8192.0,
            available_memory_mb=4096.0,
            memory_provenance=DataSource.MEASURED,
        )
        d = cap.to_dict()
        recovered = TierCapacity.from_dict(d)
        assert recovered.tier_id == cap.tier_id
        assert recovered.total_memory_mb == cap.total_memory_mb
        assert recovered.memory_provenance == cap.memory_provenance

    def test_is_enabled_default_true(self):
        cap = TierCapacity(tier_id=TierId.EDGE_B)
        assert cap.is_enabled is True


# ─────────────────────────────────────────────────────────────────────────────
# 3. generate_plan_id
# ─────────────────────────────────────────────────────────────────────────────

class TestGeneratePlanId:

    def test_monolithic_returns_local(self):
        plan = PartitionPlan.monolithic(total_layers=12)
        assert generate_plan_id(plan) == "local"

    def test_two_tier_user_edgea(self):
        plan = PartitionPlan.two_tier(total_layers=12, cut_layer=5)
        pid = generate_plan_id(plan)
        assert pid == "u0-5_ea6-11"

    def test_three_tier(self):
        plan = PartitionPlan.three_tier(total_layers=12, cut1=3, cut2=7)
        pid = generate_plan_id(plan)
        assert pid == "u0-3_ea4-7_eb8-11"

    def test_two_tier_edgeb_only(self):
        plan = PartitionPlan(total_layers=12, user_device=(0, 5), edge_b=(6, 11))
        pid = generate_plan_id(plan)
        assert pid == "u0-5_eb6-11"

    def test_determinism(self):
        plan = PartitionPlan.three_tier(total_layers=12, cut1=3, cut2=7)
        assert generate_plan_id(plan) == generate_plan_id(plan)


# ─────────────────────────────────────────────────────────────────────────────
# 4. enumerate_structural_plans
# ─────────────────────────────────────────────────────────────────────────────

class TestEnumerateStructuralPlans:

    def test_monolithic_included_when_all_allowed(self, meta_4):
        plans = enumerate_structural_plans(total_layers=4)
        ids = [generate_plan_id(p) for p in plans]
        assert "local" in ids

    def test_no_duplicates(self):
        plans = enumerate_structural_plans(total_layers=6)
        ids = [generate_plan_id(p) for p in plans]
        assert len(ids) == len(set(ids)), "Duplicate plan IDs detected"

    def test_all_plans_are_valid_partition_plans(self):
        plans = enumerate_structural_plans(total_layers=6)
        for p in plans:
            assert isinstance(p, PartitionPlan)

    def test_monolithic_disabled(self):
        plans = enumerate_structural_plans(total_layers=4, allow_monolithic=False)
        ids = [generate_plan_id(p) for p in plans]
        assert "local" not in ids

    def test_three_tier_disabled(self):
        plans = enumerate_structural_plans(total_layers=6, allow_three_tier=False)
        for p in plans:
            assert p.edge_b is None or p.edge_a is None, (
                "Three-tier plan present when three_tier disabled"
            )

    def test_two_tier_disabled(self):
        plans = enumerate_structural_plans(total_layers=6, allow_two_tier=False, allow_three_tier=False)
        # Only monolithic should remain
        assert all(generate_plan_id(p) == "local" for p in plans)

    def test_single_enabled_tier_only_monolithic(self):
        plans = enumerate_structural_plans(
            total_layers=4,
            enabled_tiers=[TierId.USER_DEVICE],
        )
        assert len(plans) == 1
        assert generate_plan_id(plans[0]) == "local"

    def test_invalid_total_layers_raises(self):
        with pytest.raises(ValueError, match="total_layers must be positive"):
            enumerate_structural_plans(total_layers=0)

    def test_two_tier_count_for_n_layers(self):
        n = 6
        plans = enumerate_structural_plans(
            total_layers=n,
            allow_monolithic=False,
            allow_three_tier=False,
        )
        # UserDevice->EdgeA: n-1 cuts, UserDevice->EdgeB: n-1 cuts
        assert len(plans) == 2 * (n - 1)

    def test_all_plans_cover_all_layers(self):
        """Every layer 0..total_layers-1 must be assigned in every plan."""
        plans = enumerate_structural_plans(total_layers=6)
        for p in plans:
            for l in range(p.total_layers):
                assert p.get_tier_for_layer(l) is not None


# ─────────────────────────────────────────────────────────────────────────────
# 5. estimate_plan_memory
# ─────────────────────────────────────────────────────────────────────────────

class TestEstimatePlanMemory:

    def test_monolithic_single_tier_requirement(self, meta_12):
        plan = PartitionPlan.monolithic(total_layers=12)
        reqs = estimate_plan_memory(plan, meta_12, context_length=128)
        assert TierId.USER_DEVICE in reqs
        assert len(reqs) == 1

    def test_two_tier_two_requirements(self, meta_12):
        plan = PartitionPlan.two_tier(total_layers=12, cut_layer=5)
        reqs = estimate_plan_memory(plan, meta_12, context_length=128)
        assert TierId.USER_DEVICE in reqs
        assert TierId.EDGE_A in reqs

    def test_three_tier_three_requirements(self, meta_12):
        plan = PartitionPlan.three_tier(total_layers=12, cut1=3, cut2=7)
        reqs = estimate_plan_memory(plan, meta_12, context_length=128)
        assert len(reqs) == 3

    def test_provenance_is_estimated(self, meta_12):
        plan = PartitionPlan.monolithic(total_layers=12)
        reqs = estimate_plan_memory(plan, meta_12)
        for req in reqs.values():
            assert req.provenance == DataSource.ESTIMATED

    def test_total_required_includes_safety_margin(self, meta_12):
        plan = PartitionPlan.monolithic(total_layers=12)
        margin = 512.0
        reqs = estimate_plan_memory(plan, meta_12, safety_margin_mb=margin)
        req = reqs[TierId.USER_DEVICE]
        assert abs(req.total_required_mb - (req.param_memory_mb + req.kv_cache_memory_mb + margin)) < 0.1

    def test_kv_cache_positive(self, meta_12):
        plan = PartitionPlan.monolithic(total_layers=12)
        reqs = estimate_plan_memory(plan, meta_12, context_length=256)
        assert reqs[TierId.USER_DEVICE].kv_cache_memory_mb > 0

    def test_larger_context_increases_kv_cache(self, meta_12):
        plan = PartitionPlan.monolithic(total_layers=12)
        r_small = estimate_plan_memory(plan, meta_12, context_length=64)
        r_large = estimate_plan_memory(plan, meta_12, context_length=1024)
        assert r_large[TierId.USER_DEVICE].kv_cache_memory_mb > r_small[TierId.USER_DEVICE].kv_cache_memory_mb

    def test_more_layers_more_memory(self, meta_12):
        """A tier holding more layers requires more param memory."""
        plan_small = PartitionPlan.two_tier(total_layers=12, cut_layer=2)  # user: 3 layers
        plan_large = PartitionPlan.two_tier(total_layers=12, cut_layer=8)  # user: 9 layers
        r_small = estimate_plan_memory(plan_small, meta_12)
        r_large = estimate_plan_memory(plan_large, meta_12)
        assert r_large[TierId.USER_DEVICE].param_memory_mb > r_small[TierId.USER_DEVICE].param_memory_mb

    def test_embeddings_on_first_tier(self, meta_12):
        """First tier (UserDevice) should carry embedding params."""
        plan = PartitionPlan.two_tier(total_layers=12, cut_layer=5)
        reqs = estimate_plan_memory(plan, meta_12)
        # UserDevice = first tier, should have embeddings_params added
        # EdgeA = second tier, should have head_params added
        user_req = reqs[TierId.USER_DEVICE]
        ea_req = reqs[TierId.EDGE_A]
        assert user_req.param_memory_mb > 0
        assert ea_req.param_memory_mb > 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. evaluate_plan_feasibility
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluatePlanFeasibility:

    def test_feasible_with_abundant_resources(self, meta_12, tier_caps_abundant):
        plan = PartitionPlan.monolithic(total_layers=12)
        reqs = estimate_plan_memory(plan, meta_12)
        s_now, s_pred, reasons, violations = evaluate_plan_feasibility(
            plan=plan,
            tier_capacities=tier_caps_abundant,
            memory_requirements=reqs,
        )
        assert s_now == FeasibilityStatus.FEASIBLE
        assert s_pred == FeasibilityStatus.FEASIBLE
        assert len(violations) == 0

    def test_infeasible_with_starved_resources(self, meta_12, tier_caps_starved):
        plan = PartitionPlan.monolithic(total_layers=12)
        reqs = estimate_plan_memory(plan, meta_12)
        s_now, s_pred, reasons, violations = evaluate_plan_feasibility(
            plan=plan,
            tier_capacities=tier_caps_starved,
            memory_requirements=reqs,
        )
        assert s_now == FeasibilityStatus.INFEASIBLE
        assert s_pred == FeasibilityStatus.INFEASIBLE
        assert len(violations) > 0

    def test_unknown_when_no_memory_limit_cpu(self, meta_12, tier_caps_cpu):
        """CPU-only tiers without configured available_memory_mb → UNKNOWN."""
        plan = PartitionPlan.monolithic(total_layers=12)
        reqs = estimate_plan_memory(plan, meta_12)
        s_now, s_pred, reasons, violations = evaluate_plan_feasibility(
            plan=plan,
            tier_capacities=tier_caps_cpu,
            memory_requirements=reqs,
        )
        assert s_now == FeasibilityStatus.UNKNOWN
        # UNKNOWN does NOT appear in violations list — it is a provenance state
        assert len(violations) == 0

    def test_disabled_tier_returns_infeasible(self, meta_12):
        plan = PartitionPlan.monolithic(total_layers=12)
        disabled_caps = {
            TierId.USER_DEVICE: TierCapacity(
                tier_id=TierId.USER_DEVICE, is_enabled=False
            )
        }
        reqs = estimate_plan_memory(plan, meta_12)
        s_now, s_pred, reasons, violations = evaluate_plan_feasibility(
            plan=plan,
            tier_capacities=disabled_caps,
            memory_requirements=reqs,
        )
        assert s_now == FeasibilityStatus.INFEASIBLE
        assert s_pred == FeasibilityStatus.INFEASIBLE

    def test_max_layers_exceeded_returns_infeasible(self, meta_12):
        plan = PartitionPlan.monolithic(total_layers=12)
        tight_caps = {
            TierId.USER_DEVICE: TierCapacity(
                tier_id=TierId.USER_DEVICE,
                device="cpu",
                available_memory_mb=32768.0,
                memory_provenance=DataSource.EMULATED,
                max_layers=3,  # only allows 3 layers, but plan assigns 12
            )
        }
        reqs = estimate_plan_memory(plan, meta_12)
        s_now, s_pred, reasons, violations = evaluate_plan_feasibility(
            plan=plan,
            tier_capacities=tight_caps,
            memory_requirements=reqs,
        )
        assert s_now == FeasibilityStatus.INFEASIBLE
        assert any("max_layers" in v for v in violations)

    def test_no_forecast_predicted_mirrors_current(self, meta_12, tier_caps_abundant):
        plan = PartitionPlan.monolithic(total_layers=12)
        reqs = estimate_plan_memory(plan, meta_12)
        s_now, s_pred, _, _ = evaluate_plan_feasibility(
            plan=plan,
            tier_capacities=tier_caps_abundant,
            memory_requirements=reqs,
            forecast=None,
        )
        assert s_now == s_pred

    def test_reasons_list_is_non_empty(self, meta_12, tier_caps_abundant):
        plan = PartitionPlan.monolithic(total_layers=12)
        reqs = estimate_plan_memory(plan, meta_12)
        _, _, reasons, _ = evaluate_plan_feasibility(
            plan=plan,
            tier_capacities=tier_caps_abundant,
            memory_requirements=reqs,
        )
        assert len(reasons) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. CandidatePlan
# ─────────────────────────────────────────────────────────────────────────────

class TestCandidatePlan:

    def _make_candidate(self, feasibility_now, feasibility_predicted) -> CandidatePlan:
        plan = PartitionPlan.monolithic(total_layers=4)
        return CandidatePlan(
            plan_id="local",
            partition_plan=plan,
            active_tiers=[TierId.USER_DEVICE],
            layer_assignment={"user_device": (0, 3)},
            number_of_boundaries=0,
            boundaries=[],
            feasibility_now=feasibility_now,
            feasibility_predicted=feasibility_predicted,
            feasibility_reasons=["test reason"],
        )

    def test_is_feasible_true_when_both_feasible(self):
        c = self._make_candidate(FeasibilityStatus.FEASIBLE, FeasibilityStatus.FEASIBLE)
        assert c.is_feasible is True

    def test_is_feasible_false_when_now_infeasible(self):
        c = self._make_candidate(FeasibilityStatus.INFEASIBLE, FeasibilityStatus.FEASIBLE)
        assert c.is_feasible is False

    def test_is_feasible_false_when_predicted_infeasible(self):
        c = self._make_candidate(FeasibilityStatus.FEASIBLE, FeasibilityStatus.INFEASIBLE)
        assert c.is_feasible is False

    def test_is_feasible_false_when_unknown(self):
        c = self._make_candidate(FeasibilityStatus.UNKNOWN, FeasibilityStatus.UNKNOWN)
        assert c.is_feasible is False

    def test_to_dict_roundtrip(self):
        c = self._make_candidate(FeasibilityStatus.FEASIBLE, FeasibilityStatus.FEASIBLE)
        d = c.to_dict()
        recovered = CandidatePlan.from_dict(d)
        assert recovered.plan_id == c.plan_id
        assert recovered.feasibility_now == c.feasibility_now
        assert recovered.feasibility_predicted == c.feasibility_predicted
        assert recovered.active_tiers == c.active_tiers

    def test_to_dict_contains_required_keys(self):
        c = self._make_candidate(FeasibilityStatus.FEASIBLE, FeasibilityStatus.FEASIBLE)
        d = c.to_dict()
        for key in [
            "plan_id", "partition_plan", "active_tiers", "layer_assignment",
            "number_of_boundaries", "boundaries", "feasibility_now",
            "feasibility_predicted", "is_feasible", "feasibility_reasons",
            "estimated_memory_requirements", "resource_violations", "metadata",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_immutable_frozen_dataclass(self):
        c = self._make_candidate(FeasibilityStatus.FEASIBLE, FeasibilityStatus.FEASIBLE)
        with pytest.raises((AttributeError, TypeError)):
            c.plan_id = "modified"  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# 8. compare_plans / PlanDifference
# ─────────────────────────────────────────────────────────────────────────────

class TestComparePlans:

    def test_identical_plans_is_identical(self):
        p1 = PartitionPlan.monolithic(total_layers=12)
        p2 = PartitionPlan.monolithic(total_layers=12)
        diff = compare_plans(p1, p2)
        assert diff.is_identical is True
        assert diff.changed_layer_count == 0
        assert diff.boundary_count_delta == 0

    def test_monolithic_to_two_tier_diff(self):
        p1 = PartitionPlan.monolithic(total_layers=12)
        p2 = PartitionPlan.two_tier(total_layers=12, cut_layer=5)
        diff = compare_plans(p1, p2)
        assert diff.is_identical is False
        assert diff.changed_layer_count > 0
        assert diff.boundary_count_delta == 1

    def test_two_tier_to_three_tier_diff(self):
        p1 = PartitionPlan.two_tier(total_layers=12, cut_layer=5)
        p2 = PartitionPlan.three_tier(total_layers=12, cut1=3, cut2=7)
        diff = compare_plans(p1, p2)
        assert diff.boundary_count_delta == 1

    def test_different_total_layers_raises(self):
        p1 = PartitionPlan.monolithic(total_layers=12)
        p2 = PartitionPlan.monolithic(total_layers=6)
        with pytest.raises(ValueError, match="Cannot compare plans"):
            compare_plans(p1, p2)

    def test_shift_cut_left_gives_positive_shift_distance(self):
        p1 = PartitionPlan.two_tier(total_layers=12, cut_layer=7)
        p2 = PartitionPlan.two_tier(total_layers=12, cut_layer=3)
        diff = compare_plans(p1, p2)
        assert diff.boundary_shift_distance == 4

    def test_affected_tiers_correct(self):
        p1 = PartitionPlan.monolithic(total_layers=12)
        p2 = PartitionPlan.two_tier(total_layers=12, cut_layer=5)
        diff = compare_plans(p1, p2)
        assert TierId.USER_DEVICE in diff.affected_tiers
        assert TierId.EDGE_A in diff.affected_tiers

    def test_to_dict_contains_required_keys(self):
        p1 = PartitionPlan.monolithic(total_layers=12)
        p2 = PartitionPlan.two_tier(total_layers=12, cut_layer=5)
        diff = compare_plans(p1, p2)
        d = diff.to_dict()
        for key in [
            "is_identical", "changed_layer_count", "affected_tiers",
            "boundary_shift_distance", "added_boundaries", "removed_boundaries",
            "boundary_count_delta",
        ]:
            assert key in d, f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# 9. format_candidate_table
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatCandidateTable:

    def _make_candidate(self, plan, feasibility=FeasibilityStatus.UNKNOWN) -> CandidatePlan:
        active = [t for t, _ in plan.get_active_tiers()]
        la = {t.value: rng for t, rng in plan.get_active_tiers()}
        return CandidatePlan(
            plan_id=generate_plan_id(plan),
            partition_plan=plan,
            active_tiers=active,
            layer_assignment=la,
            number_of_boundaries=len(plan.get_transfer_boundaries()),
            boundaries=plan.get_transfer_boundaries(),
            feasibility_now=feasibility,
            feasibility_predicted=feasibility,
            feasibility_reasons=["Test reason"],
        )

    def test_empty_candidates_returns_message(self):
        result = format_candidate_table([])
        assert "No candidate" in result

    def test_table_contains_plan_id(self):
        plan = PartitionPlan.monolithic(total_layers=4)
        c = self._make_candidate(plan)
        result = format_candidate_table([c])
        assert "local" in result

    def test_table_contains_header(self):
        plan = PartitionPlan.monolithic(total_layers=4)
        c = self._make_candidate(plan)
        result = format_candidate_table([c])
        assert "PLAN ID" in result
        assert "TIERS" in result

    def test_table_multiple_candidates(self):
        p1 = PartitionPlan.monolithic(total_layers=4)
        p2 = PartitionPlan.two_tier(total_layers=4, cut_layer=1)
        c1 = self._make_candidate(p1, FeasibilityStatus.FEASIBLE)
        c2 = self._make_candidate(p2, FeasibilityStatus.INFEASIBLE)
        result = format_candidate_table([c1, c2])
        assert "local" in result
        assert "u0-1_ea2-3" in result

    def test_table_is_string(self):
        plan = PartitionPlan.monolithic(total_layers=4)
        c = self._make_candidate(plan)
        result = format_candidate_table([c])
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# 10. SplitCatalog
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitCatalog:

    def test_generate_returns_list_of_candidates(self, meta_4):
        catalog = SplitCatalog()
        candidates = catalog.generate(meta_4)
        assert isinstance(candidates, list)
        assert len(candidates) > 0
        for c in candidates:
            assert isinstance(c, CandidatePlan)

    def test_all_candidates_have_unique_plan_ids(self, meta_4):
        catalog = SplitCatalog()
        candidates = catalog.generate(meta_4)
        ids = [c.plan_id for c in candidates]
        assert len(ids) == len(set(ids)), "Non-unique plan IDs in catalog"

    def test_exhaustive_mode_includes_all_structural_plans(self, meta_4):
        catalog = SplitCatalog()
        candidates = catalog.generate(meta_4, mode="exhaustive")
        structural = enumerate_structural_plans(total_layers=4)
        assert len(candidates) == len(structural)

    def test_filtered_mode_excludes_infeasible(self, meta_4, tier_caps_starved):
        catalog = SplitCatalog(tier_capacities=tier_caps_starved)
        candidates = catalog.generate(meta_4, mode="filtered")
        for c in candidates:
            assert c.feasibility_now != FeasibilityStatus.INFEASIBLE
            assert c.feasibility_predicted != FeasibilityStatus.INFEASIBLE

    def test_all_feasible_with_abundant_memory(self, meta_4, tier_caps_abundant):
        catalog = SplitCatalog(tier_capacities=tier_caps_abundant)
        candidates = catalog.generate(meta_4, mode="exhaustive")
        for c in candidates:
            assert c.feasibility_now == FeasibilityStatus.FEASIBLE, (
                f"Expected FEASIBLE for {c.plan_id}, got {c.feasibility_now}: {c.resource_violations}"
            )

    def test_cpu_only_tiers_yield_unknown_not_fabricated(self, meta_4):
        """CPU-only tiers without configured memory → UNKNOWN, never INFEASIBLE due to VRAM."""
        catalog = SplitCatalog()  # Default CPU caps without available_memory_mb
        candidates = catalog.generate(meta_4, mode="exhaustive")
        for c in candidates:
            # Must be UNKNOWN or FEASIBLE — never INFEASIBLE from fabricated VRAM
            assert c.feasibility_now in (
                FeasibilityStatus.UNKNOWN, FeasibilityStatus.FEASIBLE
            ), f"Unexpected {c.feasibility_now} for {c.plan_id}"

    def test_disabled_tier_excludes_plans_using_that_tier(self, meta_4):
        caps = {
            TierId.USER_DEVICE: TierCapacity(tier_id=TierId.USER_DEVICE, device="cpu", is_enabled=True),
            TierId.EDGE_A: TierCapacity(tier_id=TierId.EDGE_A, device="cpu", is_enabled=False),
            TierId.EDGE_B: TierCapacity(tier_id=TierId.EDGE_B, device="cpu", is_enabled=True),
        }
        catalog = SplitCatalog(tier_capacities=caps)
        candidates = catalog.generate(meta_4, mode="exhaustive")
        # No plan should reference EdgeA as active (its structural plans won't be generated)
        for c in candidates:
            assert TierId.EDGE_A not in c.active_tiers, (
                f"Disabled EdgeA tier appears in candidate {c.plan_id}"
            )

    def test_monolithic_only_catalog(self, meta_4):
        catalog = SplitCatalog(allow_two_tier=False, allow_three_tier=False)
        candidates = catalog.generate(meta_4)
        assert len(candidates) == 1
        assert candidates[0].plan_id == "local"

    def test_current_plan_metadata_injected(self, meta_4, tier_caps_abundant):
        current = PartitionPlan.monolithic(total_layers=4)
        catalog = SplitCatalog(tier_capacities=tier_caps_abundant)
        candidates = catalog.generate(meta_4, current_plan=current)
        # Find the 'local' candidate — it should be identical to current
        local = next(c for c in candidates if c.plan_id == "local")
        assert local.metadata.get("is_current_plan") is True
        assert local.metadata.get("changed_layers_vs_current") == 0

    def test_context_length_affects_memory_estimates(self, meta_4, tier_caps_abundant):
        catalog_small = SplitCatalog(tier_capacities=tier_caps_abundant, default_context_length=64)
        catalog_large = SplitCatalog(tier_capacities=tier_caps_abundant, default_context_length=1024)
        c_small = catalog_small.generate(meta_4)
        c_large = catalog_large.generate(meta_4)
        # The local plan memory estimate should be larger for longer context
        small_mem = c_small[0].estimated_memory_requirements.get("user_device", 0)
        large_mem = c_large[0].estimated_memory_requirements.get("user_device", 0)
        assert large_mem > small_mem

    def test_filter_feasible_method(self, meta_4, tier_caps_abundant):
        catalog = SplitCatalog(tier_capacities=tier_caps_abundant)
        candidates = catalog.generate(meta_4)
        feasible = catalog.filter_feasible(candidates)
        for c in feasible:
            assert c.is_feasible is True

    def test_compare_plans_method(self):
        catalog = SplitCatalog()
        p1 = PartitionPlan.monolithic(total_layers=6)
        p2 = PartitionPlan.two_tier(total_layers=6, cut_layer=2)
        diff = catalog.compare_plans(p1, p2)
        assert isinstance(diff, PlanDifference)
        assert diff.is_identical is False

    def test_format_table_method(self, meta_4):
        catalog = SplitCatalog()
        candidates = catalog.generate(meta_4)
        table = catalog.format_table(candidates)
        assert isinstance(table, str)
        assert "PLAN ID" in table

    def test_safety_margin_applied(self, meta_4, tier_caps_abundant):
        catalog_hi = SplitCatalog(tier_capacities=tier_caps_abundant, safety_margin_mb=1000.0)
        catalog_lo = SplitCatalog(tier_capacities=tier_caps_abundant, safety_margin_mb=10.0)
        c_hi = catalog_hi.generate(meta_4)
        c_lo = catalog_lo.generate(meta_4)
        mem_hi = c_hi[0].estimated_memory_requirements.get("user_device", 0)
        mem_lo = c_lo[0].estimated_memory_requirements.get("user_device", 0)
        assert mem_hi > mem_lo


# ─────────────────────────────────────────────────────────────────────────────
# 11. Public API Exports (__init__)
# ─────────────────────────────────────────────────────────────────────────────

class TestPublicApi:

    def test_all_symbols_importable(self):
        from src.partitioning import (
            CandidatePlan,
            FeasibilityStatus,
            ModelMetadata,
            PlanDifference,
            SplitCatalog,
            TierCapacity,
            TierMemoryRequirement,
            compare_plans,
            enumerate_structural_plans,
            estimate_plan_memory,
            evaluate_plan_feasibility,
            format_candidate_table,
            generate_plan_id,
        )
        # Just asserting they imported without error
        assert SplitCatalog is not None
        assert FeasibilityStatus is not None


# ─────────────────────────────────────────────────────────────────────────────
# 12. Regression: No Module 5 code touches execution or cost scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestSeparationOfConcerns:

    def test_split_catalog_has_no_score_method(self):
        catalog = SplitCatalog()
        assert not hasattr(catalog, "score"), "SplitCatalog must not implement scoring"
        assert not hasattr(catalog, "select"), "SplitCatalog must not implement selection"

    def test_candidate_plan_has_no_cost_field(self):
        plan = PartitionPlan.monolithic(total_layers=4)
        c = CandidatePlan(
            plan_id="local",
            partition_plan=plan,
            active_tiers=[TierId.USER_DEVICE],
            layer_assignment={"user_device": (0, 3)},
            number_of_boundaries=0,
            boundaries=[],
            feasibility_now=FeasibilityStatus.UNKNOWN,
            feasibility_predicted=FeasibilityStatus.UNKNOWN,
        )
        assert not hasattr(c, "cost"), "CandidatePlan must not carry a cost field"
        assert not hasattr(c, "utility"), "CandidatePlan must not carry a utility field"
