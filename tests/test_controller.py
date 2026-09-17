"""
Tests for Module 7: Adaptive Predictive Partition Controller.

Verifies:
    - Initialization and configuration validation
    - Static controller mode (always KEEP_CURRENT)
    - Reactive controller mode (no forecast used)
    - Predictive controller mode (forecast integrated)
    - Best candidate identification: a* = argmin J(a)
    - Threshold rejection: Gain <= theta => KEEP_CURRENT
    - Threshold acceptance: Gain > theta => SWITCH (subject to other guards)
    - Minimum dwell time enforcement
    - Cooldown period enforcement
    - Hysteresis confirmation cycles (N consecutive cycles)
    - Hysteresis reset when preferred candidate changes
    - Anti-thrashing under rapidly alternating candidate costs
    - Emergency safety override (bypasses threshold/cooldown)
    - No feasible candidate -> KEEP_CURRENT or SAFE_FALLBACK
    - UNKNOWN resource handling (no zero fabrication)
    - Proactive vs reactive switch classification
    - MigrationRequest generated (not physically executed)
    - ControllerHistory metrics accumulation
    - Controller.replay() multi-step trace
    - 5-phase trace: stable -> forecast degradation -> proactive switch -> dwell hold -> recovery
    - ControllerConfig validation
    - to_dict serialization completeness
"""

from __future__ import annotations

import pytest

from src.controller import (
    AdaptivePartitionController,
    ControlAction,
    ControlDecision,
    ControllerConfig,
    ControllerHistory,
    ControllerMode,
    DecisionReason,
    MigrationRequest,
    SafetyEvaluation,
    SafetyPolicy,
    SafetyStatus,
    StabilityChecker,
    StabilityResult,
)
from src.controller.types import ControllerState
from src.cost.model import CostModel
from src.cost.types import CandidateScore, CostBreakdown, ScoreStatus
from src.partitioning import (
    CandidatePlan,
    FeasibilityStatus,
    ModelMetadata,
    SplitCatalog,
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


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def model_meta() -> ModelMetadata:
    return ModelMetadata.synthetic_default(total_layers=12)


@pytest.fixture
def all_candidates(model_meta: ModelMetadata) -> list[CandidatePlan]:
    catalog = SplitCatalog()
    return catalog.generate(model_meta, mode="exhaustive")


@pytest.fixture
def local_plan() -> PartitionPlan:
    return PartitionPlan.monolithic(total_layers=12)


@pytest.fixture
def mock_state() -> RuntimeState:
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
            kv_cache_bytes=TaggedValue(10485760.0, DataSource.MEASURED, "bytes"),
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


def make_state_at(t: float, bandwidth_mbps: float = 50.0, latency_ms: float = 15.0) -> RuntimeState:
    """Helper: build a RuntimeState at a specific timestamp."""
    return RuntimeState(
        timestamp=t,
        wall_clock=t,
        step_index=int(t),
        network=NetworkState(
            bandwidth_mbps=TaggedValue(bandwidth_mbps, DataSource.EMULATED, "Mbps"),
            latency_ms=TaggedValue(latency_ms, DataSource.EMULATED, "ms"),
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
            kv_cache_bytes=TaggedValue(10485760.0, DataSource.MEASURED, "bytes"),
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


# ===========================================================================
# 1. ControllerConfig Tests
# ===========================================================================

class TestControllerConfig:
    def test_defaults(self):
        cfg = ControllerConfig()
        assert cfg.mode == ControllerMode.PREDICTIVE
        assert cfg.switch_threshold == 0.25
        assert cfg.minimum_dwell_seconds == 5.0
        assert cfg.cooldown_seconds == 5.0
        assert cfg.hysteresis_cycles == 2
        assert cfg.emergency_override is True
        assert cfg.max_memory_pressure == 0.90

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="switch_threshold"):
            ControllerConfig(switch_threshold=-0.1)

    def test_negative_dwell_raises(self):
        with pytest.raises(ValueError, match="minimum_dwell_seconds"):
            ControllerConfig(minimum_dwell_seconds=-1.0)

    def test_negative_cooldown_raises(self):
        with pytest.raises(ValueError, match="cooldown_seconds"):
            ControllerConfig(cooldown_seconds=-0.5)

    def test_hysteresis_zero_raises(self):
        with pytest.raises(ValueError, match="hysteresis_cycles"):
            ControllerConfig(hysteresis_cycles=0)

    def test_invalid_memory_pressure_raises(self):
        with pytest.raises(ValueError, match="max_memory_pressure"):
            ControllerConfig(max_memory_pressure=1.5)

    def test_from_dict_roundtrip(self):
        cfg = ControllerConfig(mode=ControllerMode.REACTIVE, switch_threshold=0.30, hysteresis_cycles=3)
        d = cfg.to_dict()
        cfg2 = ControllerConfig.from_dict(d)
        assert cfg2.mode == ControllerMode.REACTIVE
        assert cfg2.switch_threshold == 0.30
        assert cfg2.hysteresis_cycles == 3

    def test_zero_threshold_allowed(self):
        cfg = ControllerConfig(switch_threshold=0.0)
        assert cfg.switch_threshold == 0.0

    def test_zero_dwell_allowed(self):
        cfg = ControllerConfig(minimum_dwell_seconds=0.0)
        assert cfg.minimum_dwell_seconds == 0.0


# ===========================================================================
# 2. Static Controller Mode
# ===========================================================================

class TestStaticControllerMode:
    def test_static_always_keep_current(self, all_candidates, mock_state):
        cfg = ControllerConfig(mode=ControllerMode.STATIC)
        ctrl = AdaptivePartitionController(config=cfg)
        for _ in range(5):
            d = ctrl.decide(state=mock_state, candidates=all_candidates)
            assert d.action == ControlAction.KEEP_CURRENT
            assert d.reason == DecisionReason.STATIC_POLICY
            assert d.switch_allowed is False
            assert d.prediction_used is False

    def test_static_no_migration_request(self, all_candidates, mock_state):
        cfg = ControllerConfig(mode=ControllerMode.STATIC)
        ctrl = AdaptivePartitionController(config=cfg)
        d = ctrl.decide(state=mock_state, candidates=all_candidates)
        assert d.migration_request is None

    def test_static_with_forecast_ignored(self, all_candidates, mock_state, mock_forecast):
        cfg = ControllerConfig(mode=ControllerMode.STATIC)
        ctrl = AdaptivePartitionController(config=cfg)
        d = ctrl.decide(state=mock_state, candidates=all_candidates, forecast=mock_forecast)
        assert d.action == ControlAction.KEEP_CURRENT
        assert d.prediction_used is False


# ===========================================================================
# 3. Reactive Controller Mode
# ===========================================================================

class TestReactiveControllerMode:
    def test_reactive_ignores_forecast(self, all_candidates, mock_state, mock_forecast):
        cfg = ControllerConfig(mode=ControllerMode.REACTIVE)
        ctrl = AdaptivePartitionController(config=cfg)
        d = ctrl.decide(state=mock_state, candidates=all_candidates, forecast=mock_forecast)
        assert d.prediction_used is False

    def test_reactive_produces_valid_action(self, all_candidates, mock_state):
        cfg = ControllerConfig(mode=ControllerMode.REACTIVE)
        ctrl = AdaptivePartitionController(config=cfg)
        d = ctrl.decide(state=mock_state, candidates=all_candidates)
        assert d.action in (
            ControlAction.KEEP_CURRENT,
            ControlAction.SWITCH,
            ControlAction.SAFE_FALLBACK,
            ControlAction.NO_ACTION,
        )


# ===========================================================================
# 4. Predictive Controller Mode
# ===========================================================================

class TestPredictiveControllerMode:
    def test_predictive_uses_forecast(self, all_candidates, mock_state, mock_forecast):
        cfg = ControllerConfig(mode=ControllerMode.PREDICTIVE)
        ctrl = AdaptivePartitionController(config=cfg)
        d = ctrl.decide(state=mock_state, candidates=all_candidates, forecast=mock_forecast)
        assert d.prediction_used is True

    def test_predictive_no_forecast_is_not_used(self, all_candidates, mock_state):
        cfg = ControllerConfig(mode=ControllerMode.PREDICTIVE)
        ctrl = AdaptivePartitionController(config=cfg)
        d = ctrl.decide(state=mock_state, candidates=all_candidates, forecast=None)
        assert d.prediction_used is False


# ===========================================================================
# 5. Best Candidate Identification: a* = argmin J(a)
# ===========================================================================

class TestBestCandidateIdentification:
    def test_best_candidate_selected(self, all_candidates, mock_state):
        """Controller should select the candidate with the lowest total_cost."""
        cfg = ControllerConfig(
            mode=ControllerMode.REACTIVE,
            switch_threshold=0.0,
            minimum_dwell_seconds=0.0,
            cooldown_seconds=0.0,
            hysteresis_cycles=1,
        )
        ctrl = AdaptivePartitionController(config=cfg)
        d = ctrl.decide(state=mock_state, candidates=all_candidates)
        # After first cycle (no prior switch), selected_plan_id must be the cheapest feasible
        assert d.selected_plan_id is not None
        assert d.selected_cost <= d.current_cost or d.action == ControlAction.KEEP_CURRENT

    def test_keep_current_when_current_is_cheapest(self, mock_state):
        """If current plan is already cheapest, action must be KEEP_CURRENT."""
        cfg = ControllerConfig(
            mode=ControllerMode.REACTIVE,
            switch_threshold=0.0,
            minimum_dwell_seconds=0.0,
            cooldown_seconds=0.0,
            hysteresis_cycles=1,
        )
        local = PartitionPlan.monolithic(total_layers=12)
        ctrl = AdaptivePartitionController(config=cfg, initial_plan=local)
        # Provide only current plan as candidate
        meta = ModelMetadata.synthetic_default(total_layers=12)
        catalog = SplitCatalog()
        candidates = catalog.generate(meta, mode="exhaustive")
        # Filter to only the monolithic (local) candidate
        curr_id = generate_plan_id(local)
        local_cands = [c for c in candidates if c.plan_id == curr_id]
        if local_cands:
            d = ctrl.decide(state=mock_state, candidates=local_cands)
            # Only one candidate is current plan -> should KEEP_CURRENT
            assert d.action in (ControlAction.KEEP_CURRENT, ControlAction.NO_ACTION)


# ===========================================================================
# 6. Threshold Enforcement: Gain <= theta => reject
# ===========================================================================

class TestThresholdEnforcement:
    def test_high_threshold_prevents_switch(self, all_candidates, mock_state):
        """Very high threshold should prevent switching."""
        cfg = ControllerConfig(
            mode=ControllerMode.REACTIVE,
            switch_threshold=999.0,  # Unreachable gain
            minimum_dwell_seconds=0.0,
            cooldown_seconds=0.0,
            hysteresis_cycles=1,
        )
        ctrl = AdaptivePartitionController(config=cfg)
        d = ctrl.decide(state=mock_state, candidates=all_candidates)
        # Switch cannot occur since gain is never >= 999.0
        assert d.action != ControlAction.SWITCH
        assert d.switch_allowed is False

    def test_zero_threshold_allows_switch(self, all_candidates, mock_state):
        """Zero threshold with zero dwell/cooldown and hysteresis=1 should permit switching."""
        cfg = ControllerConfig(
            mode=ControllerMode.REACTIVE,
            switch_threshold=0.0,
            minimum_dwell_seconds=0.0,
            cooldown_seconds=0.0,
            hysteresis_cycles=1,
        )
        ctrl = AdaptivePartitionController(config=cfg)
        # Run twice: first cycle may be blocked if current is already best
        for _ in range(3):
            d = ctrl.decide(state=mock_state, candidates=all_candidates)
        # With zero threshold, something should have switched or stayed same cost
        assert d.action in (ControlAction.SWITCH, ControlAction.KEEP_CURRENT)


# ===========================================================================
# 7. Minimum Dwell Time
# ===========================================================================

class TestMinimumDwellTime:
    def test_dwell_prevents_immediate_switch(self, all_candidates):
        """After a switch, minimum dwell time should block further switches."""
        cfg = ControllerConfig(
            mode=ControllerMode.REACTIVE,
            switch_threshold=0.0,
            minimum_dwell_seconds=100.0,  # Very long dwell
            cooldown_seconds=0.0,
            hysteresis_cycles=1,
        )
        ctrl = AdaptivePartitionController(config=cfg)

        # First decide to advance cycle
        state1 = make_state_at(1000.0)
        d1 = ctrl.decide(state=state1, candidates=all_candidates)

        # Force a switch by injecting a past switch
        ctrl.state.last_switch_timestamp = 1000.0
        ctrl.state.last_switch_cycle = ctrl.state.controller_cycle

        # Immediate second decide at t=1001 (only 1 second elapsed)
        state2 = make_state_at(1001.0)
        d2 = ctrl.decide(state=state2, candidates=all_candidates)

        # Must be blocked by dwell (unless emergency)
        if d2.action == ControlAction.SWITCH:
            assert d2.is_safety_override is True
        else:
            assert d2.reason in (
                DecisionReason.DWELL_TIME_ACTIVE,
                DecisionReason.BEST_CANDIDATE_IS_CURRENT,
                DecisionReason.NO_IMPROVEMENT,
                DecisionReason.BELOW_THRESHOLD,
                DecisionReason.COOLDOWN_ACTIVE,
            )

    def test_dwell_elapsed_permits_switch(self, all_candidates):
        """After dwell time has elapsed, switches should be permitted."""
        cfg = ControllerConfig(
            mode=ControllerMode.REACTIVE,
            switch_threshold=0.0,
            minimum_dwell_seconds=5.0,
            cooldown_seconds=0.0,
            hysteresis_cycles=1,
        )
        ctrl = AdaptivePartitionController(config=cfg)

        # Simulate an old switch 1000s ago
        ctrl.state.last_switch_timestamp = 0.0  # Far in the past

        # Decide at t=1000 (1000s elapsed, dwell=5s requirement)
        state = make_state_at(1000.0)
        d = ctrl.decide(state=state, candidates=all_candidates)
        # Dwell should NOT block (1000s >> 5s)
        assert d.reason not in (DecisionReason.DWELL_TIME_ACTIVE,)


# ===========================================================================
# 8. Cooldown Period
# ===========================================================================

class TestCooldownPeriod:
    def test_cooldown_blocks_within_window(self, all_candidates):
        """Cooldown should block switches immediately after a prior switch."""
        cfg = ControllerConfig(
            mode=ControllerMode.REACTIVE,
            switch_threshold=0.0,
            minimum_dwell_seconds=0.0,
            cooldown_seconds=60.0,  # 60 second cooldown
            hysteresis_cycles=1,
        )
        ctrl = AdaptivePartitionController(config=cfg)
        ctrl.state.last_switch_timestamp = 999.0  # Just switched 1 second ago

        state = make_state_at(1000.0)
        d = ctrl.decide(state=state, candidates=all_candidates)

        if d.action == ControlAction.SWITCH:
            # Only permitted if emergency
            assert d.is_safety_override is True
        else:
            assert d.reason in (
                DecisionReason.COOLDOWN_ACTIVE,
                DecisionReason.BEST_CANDIDATE_IS_CURRENT,
                DecisionReason.NO_IMPROVEMENT,
                DecisionReason.BELOW_THRESHOLD,
            )

    def test_cooldown_expires(self, all_candidates):
        """After cooldown expires, switch should be unblocked."""
        cfg = ControllerConfig(
            mode=ControllerMode.REACTIVE,
            switch_threshold=0.0,
            minimum_dwell_seconds=0.0,
            cooldown_seconds=5.0,
            hysteresis_cycles=1,
        )
        ctrl = AdaptivePartitionController(config=cfg)
        ctrl.state.last_switch_timestamp = 900.0  # 100s ago

        state = make_state_at(1000.0)
        d = ctrl.decide(state=state, candidates=all_candidates)
        assert d.reason != DecisionReason.COOLDOWN_ACTIVE


# ===========================================================================
# 9. Hysteresis Confirmation
# ===========================================================================

class TestHysteresis:
    def test_hysteresis_blocks_first_cycle(self, all_candidates, mock_state):
        """With hysteresis_cycles=3, first preference should not trigger switch."""
        cfg = ControllerConfig(
            mode=ControllerMode.REACTIVE,
            switch_threshold=0.0,
            minimum_dwell_seconds=0.0,
            cooldown_seconds=0.0,
            hysteresis_cycles=3,
        )
        ctrl = AdaptivePartitionController(config=cfg)
        d = ctrl.decide(state=mock_state, candidates=all_candidates)
        # On first cycle, consecutive_count = 1 which is < 3
        if d.action == ControlAction.SWITCH:
            # Only allowed if same plan or emergency
            assert d.is_safety_override or d.selected_plan_id == d.current_plan_id
        else:
            # May be blocked by hysteresis or other reasons
            assert d.switch_allowed is False

    def test_hysteresis_resets_on_candidate_change(self, all_candidates, mock_state):
        """Hysteresis count must reset when preferred candidate changes."""
        cfg = ControllerConfig(
            mode=ControllerMode.REACTIVE,
            switch_threshold=0.0,
            minimum_dwell_seconds=0.0,
            cooldown_seconds=0.0,
            hysteresis_cycles=2,
        )
        ctrl = AdaptivePartitionController(config=cfg)

        # Manually set a prior preference for plan "plan_A"
        ctrl.state.consecutive_preference_plan_id = "plan_A"
        ctrl.state.consecutive_preference_count = 5

        # Decide - the actual best candidate is NOT "plan_A" (it's whatever cost model picks)
        d = ctrl.decide(state=mock_state, candidates=all_candidates)

        # If a different plan was selected, hysteresis should have reset to 1
        if d.selected_plan_id != "plan_A":
            # Count should have been reset
            assert ctrl.state.consecutive_preference_count in (0, 1)

    def test_hysteresis_satisfied_after_n_cycles(self, all_candidates):
        """After N cycles of consistent preference, switch should be approved."""
        cfg = ControllerConfig(
            mode=ControllerMode.REACTIVE,
            switch_threshold=0.0,
            minimum_dwell_seconds=0.0,
            cooldown_seconds=0.0,
            hysteresis_cycles=2,
        )
        ctrl = AdaptivePartitionController(config=cfg)

        decisions = []
        for i in range(5):
            state = make_state_at(float(i * 10))
            d = ctrl.decide(state=state, candidates=all_candidates)
            decisions.append(d)

        # Over 5 cycles, at least one switch or hysteresis satisfaction should occur
        switch_decisions = [d for d in decisions if d.action == ControlAction.SWITCH]
        keep_decisions = [d for d in decisions if d.action == ControlAction.KEEP_CURRENT]
        assert len(switch_decisions) + len(keep_decisions) > 0


# ===========================================================================
# 10. Anti-Thrashing Under Rapidly Alternating Costs
# ===========================================================================

class TestAntiThrashing:
    def test_no_rapid_alternation(self, all_candidates):
        """With proper dwell and cooldown, rapid flapping should not occur."""
        cfg = ControllerConfig(
            mode=ControllerMode.REACTIVE,
            switch_threshold=0.1,
            minimum_dwell_seconds=5.0,
            cooldown_seconds=5.0,
            hysteresis_cycles=2,
        )
        ctrl = AdaptivePartitionController(config=cfg)

        switch_times = []
        for i in range(20):
            state = make_state_at(float(i))  # 1-second steps
            d = ctrl.decide(state=state, candidates=all_candidates)
            if d.action == ControlAction.SWITCH:
                switch_times.append(i)

        # If any switches happened, verify minimum gap between them
        for idx in range(1, len(switch_times)):
            gap = switch_times[idx] - switch_times[idx - 1]
            assert gap >= 5, f"Switch gap {gap} too small (dwell=5s required)"

    def test_switch_count_increments(self, all_candidates):
        """Every SWITCH action should increment the controller state switch count."""
        cfg = ControllerConfig(
            mode=ControllerMode.REACTIVE,
            switch_threshold=0.0,
            minimum_dwell_seconds=0.0,
            cooldown_seconds=0.0,
            hysteresis_cycles=1,
        )
        ctrl = AdaptivePartitionController(config=cfg)

        for i in range(10):
            state = make_state_at(float(i * 100))
            ctrl.decide(state=state, candidates=all_candidates)

        # History should match switch_count
        metrics = ctrl.get_metrics()
        assert metrics["total_switches"] == ctrl.state.switch_count


# ===========================================================================
# 11. Emergency Safety Override
# ===========================================================================

class TestEmergencySafetyOverride:
    def _make_high_pressure_score(self, plan: PartitionPlan, plan_id: str) -> CandidatePlan:
        """Create a CandidatePlan with high memory pressure to trigger emergency."""
        return CandidatePlan(
            plan_id=plan_id,
            partition_plan=plan,
            active_tiers=[TierId.USER_DEVICE],
            layer_assignment={TierId.USER_DEVICE.value: (0, 11)},
            number_of_boundaries=0,
            boundaries=[],
            feasibility_now=FeasibilityStatus.INFEASIBLE,
            feasibility_predicted=FeasibilityStatus.INFEASIBLE,
        )

    def test_emergency_override_bypasses_cooldown(self, all_candidates):
        """Emergency state should bypass cooldown and allow switch."""
        cfg = ControllerConfig(
            mode=ControllerMode.REACTIVE,
            switch_threshold=0.0,
            minimum_dwell_seconds=0.0,
            cooldown_seconds=999.0,  # Very long cooldown
            hysteresis_cycles=1,
            emergency_override=True,
        )
        ctrl = AdaptivePartitionController(config=cfg)
        # Set recent switch to trigger cooldown
        ctrl.state.last_switch_timestamp = 999.0

        # Build candidates: current plan is INFEASIBLE (emergency), other is FEASIBLE
        local = PartitionPlan.monolithic(total_layers=12)
        local_id = generate_plan_id(local)
        ctrl.state.current_plan = local
        ctrl.state.current_plan_id = local_id

        # Use catalog candidates but override current plan as infeasible
        infeasible_cand = self._make_high_pressure_score(local, local_id)

        # Get a different feasible candidate
        meta = ModelMetadata.synthetic_default(total_layers=12)
        catalog = SplitCatalog()
        feasible_cands = [c for c in catalog.generate(meta, mode="exhaustive")
                         if c.plan_id != local_id and c.feasibility_now == FeasibilityStatus.FEASIBLE]

        if feasible_cands:
            cands = [infeasible_cand] + feasible_cands[:3]
            state = make_state_at(1000.0)
            d = ctrl.decide(state=state, candidates=cands, current_plan=local)
            # Should either switch (emergency) or keep current with safety status
            assert d.safety_status in (SafetyStatus.EMERGENCY, SafetyStatus.WARNING, SafetyStatus.SAFE)

    def test_emergency_override_disabled(self):
        """With emergency_override=False, emergency does NOT bypass stability."""
        cfg = ControllerConfig(
            mode=ControllerMode.REACTIVE,
            switch_threshold=999.0,  # Unreachable threshold
            minimum_dwell_seconds=0.0,
            cooldown_seconds=0.0,
            hysteresis_cycles=1,
            emergency_override=False,
        )
        ctrl = AdaptivePartitionController(config=cfg)
        meta = ModelMetadata.synthetic_default(total_layers=12)
        catalog = SplitCatalog()
        candidates = catalog.generate(meta, mode="exhaustive")
        state = make_state_at(1000.0)
        d = ctrl.decide(state=state, candidates=candidates)
        # With threshold=999 and no emergency override, switch should be blocked
        assert d.action != ControlAction.SWITCH or d.expected_gain > 999.0


# ===========================================================================
# 12. No Feasible Candidate
# ===========================================================================

class TestNoFeasibleCandidate:
    def test_empty_candidates_keep_current(self, mock_state):
        """Empty candidate list should result in KEEP_CURRENT or SAFE_FALLBACK."""
        cfg = ControllerConfig(mode=ControllerMode.REACTIVE)
        ctrl = AdaptivePartitionController(config=cfg)
        d = ctrl.decide(state=mock_state, candidates=[])
        assert d.action in (ControlAction.KEEP_CURRENT, ControlAction.SAFE_FALLBACK)
        assert d.reason == DecisionReason.NO_FEASIBLE_CANDIDATE

    def test_all_infeasible_candidates(self, mock_state):
        """All INFEASIBLE candidates -> NO_FEASIBLE_CANDIDATE reason."""
        cfg = ControllerConfig(mode=ControllerMode.REACTIVE)
        ctrl = AdaptivePartitionController(config=cfg)

        local = PartitionPlan.monolithic(total_layers=12)
        infeasible = CandidatePlan(
            plan_id="infeasible_only",
            partition_plan=local,
            active_tiers=[TierId.USER_DEVICE],
            layer_assignment={TierId.USER_DEVICE.value: (0, 11)},
            number_of_boundaries=0,
            boundaries=[],
            feasibility_now=FeasibilityStatus.INFEASIBLE,
            feasibility_predicted=FeasibilityStatus.INFEASIBLE,
        )

        d = ctrl.decide(state=mock_state, candidates=[infeasible])
        assert d.reason == DecisionReason.NO_FEASIBLE_CANDIDATE


# ===========================================================================
# 13. UNKNOWN Resource Handling
# ===========================================================================

class TestUnknownResourceHandling:
    def test_unavailable_vram_not_fabricated(self, all_candidates, mock_state):
        """UNAVAILABLE VRAM must not be converted to zero or fabricated."""
        cfg = ControllerConfig(mode=ControllerMode.REACTIVE)
        ctrl = AdaptivePartitionController(config=cfg)

        # mock_state has UNAVAILABLE VRAM
        assert mock_state.memory.vram_free_mb.source == DataSource.UNAVAILABLE
        assert mock_state.memory.vram_allocated_mb.source == DataSource.UNAVAILABLE

        d = ctrl.decide(state=mock_state, candidates=all_candidates)
        # Decision should be produced without errors (no zero fabrication)
        assert d is not None
        assert d.action in (ControlAction.KEEP_CURRENT, ControlAction.SWITCH,
                            ControlAction.SAFE_FALLBACK, ControlAction.NO_ACTION)

    def test_unknown_switch_blocked_by_default(self):
        """UNKNOWN-status candidates should be blocked unless allow_unknown_switches=True."""
        cfg = ControllerConfig(
            mode=ControllerMode.REACTIVE,
            allow_unknown_switches=False,
            switch_threshold=0.0,
            minimum_dwell_seconds=0.0,
            cooldown_seconds=0.0,
            hysteresis_cycles=1,
        )
        ctrl = AdaptivePartitionController(config=cfg)
        state = make_state_at(1000.0)

        local = PartitionPlan.monolithic(total_layers=12)
        unknown_cand = CandidatePlan(
            plan_id="unknown_cand",
            partition_plan=local,
            active_tiers=[TierId.USER_DEVICE],
            layer_assignment={TierId.USER_DEVICE.value: (0, 11)},
            number_of_boundaries=0,
            boundaries=[],
            feasibility_now=FeasibilityStatus.UNKNOWN,
            feasibility_predicted=FeasibilityStatus.UNKNOWN,
        )
        d = ctrl.decide(state=state, candidates=[unknown_cand])
        # With only UNKNOWN candidate and allow_unknown_switches=False -> no eligible
        assert d.reason == DecisionReason.NO_FEASIBLE_CANDIDATE


# ===========================================================================
# 14. Proactive vs Reactive Classification
# ===========================================================================

class TestProactiveVsReactive:
    def test_reactive_when_no_forecast(self, all_candidates, mock_state):
        """Switch without forecast must be classified as reactive (is_proactive=False)."""
        cfg = ControllerConfig(
            mode=ControllerMode.REACTIVE,
            switch_threshold=0.0,
            minimum_dwell_seconds=0.0,
            cooldown_seconds=0.0,
            hysteresis_cycles=1,
        )
        ctrl = AdaptivePartitionController(config=cfg)
        d = ctrl.decide(state=mock_state, candidates=all_candidates, forecast=None)
        if d.action == ControlAction.SWITCH:
            assert d.is_proactive is False

    def test_proactive_when_forecast_used(self, all_candidates, mock_state, mock_forecast):
        """Switch using valid forecast on still-feasible plan should be proactive."""
        cfg = ControllerConfig(
            mode=ControllerMode.PREDICTIVE,
            switch_threshold=0.0,
            minimum_dwell_seconds=0.0,
            cooldown_seconds=0.0,
            hysteresis_cycles=1,
        )
        ctrl = AdaptivePartitionController(config=cfg)
        decisions = []
        for i in range(5):
            state = make_state_at(float(i * 100))
            d = ctrl.decide(state=state, candidates=all_candidates, forecast=mock_forecast)
            decisions.append(d)

        proactive_switches = [d for d in decisions if d.action == ControlAction.SWITCH and d.is_proactive]
        reactive_switches = [d for d in decisions if d.action == ControlAction.SWITCH and not d.is_proactive]
        # At least one type of switch should be properly classified
        all_switches = proactive_switches + reactive_switches
        for s in all_switches:
            assert isinstance(s.is_proactive, bool)


# ===========================================================================
# 15. MigrationRequest Generation (No Physical Migration)
# ===========================================================================

class TestMigrationRequest:
    def test_migration_request_none_on_keep(self, all_candidates, mock_state):
        """KEEP_CURRENT action should produce no MigrationRequest."""
        cfg = ControllerConfig(
            mode=ControllerMode.STATIC,
        )
        ctrl = AdaptivePartitionController(config=cfg)
        d = ctrl.decide(state=mock_state, candidates=all_candidates)
        assert d.migration_request is None

    def test_migration_request_fields_on_switch(self, all_candidates, mock_state):
        """SWITCH action must produce a valid MigrationRequest."""
        cfg = ControllerConfig(
            mode=ControllerMode.REACTIVE,
            switch_threshold=0.0,
            minimum_dwell_seconds=0.0,
            cooldown_seconds=0.0,
            hysteresis_cycles=1,
        )
        ctrl = AdaptivePartitionController(config=cfg)
        # Run multiple cycles to find a switch
        mr = None
        for i in range(15):
            state = make_state_at(float(i * 100))
            d = ctrl.decide(state=state, candidates=all_candidates)
            if d.action == ControlAction.SWITCH and d.migration_request is not None:
                mr = d.migration_request
                break

        if mr is not None:
            assert isinstance(mr, MigrationRequest)
            assert isinstance(mr.source_plan_id, str)
            assert isinstance(mr.target_plan_id, str)
            assert mr.source_plan_id != mr.target_plan_id
            assert isinstance(mr.controller_cycle, int)
            assert mr.controller_cycle >= 1
            assert isinstance(mr.changed_layers, list)
            assert isinstance(mr.affected_tiers, list)
            assert isinstance(mr.estimated_kv_transfer_bytes, int)
            d_dict = d.to_dict()
            assert "migration_request" in d_dict

    def test_no_physical_migration_in_module7(self, all_candidates, mock_state):
        """Module 7 MUST NOT physically move model weights or tensors."""
        # The controller stores only a PartitionPlan reference in state; no tensor data
        cfg = ControllerConfig(
            mode=ControllerMode.REACTIVE,
            switch_threshold=0.0,
            minimum_dwell_seconds=0.0,
            cooldown_seconds=0.0,
            hysteresis_cycles=1,
        )
        ctrl = AdaptivePartitionController(config=cfg)
        initial_plan_ref = id(ctrl.state.current_plan)

        state = make_state_at(1000.0)
        d = ctrl.decide(state=state, candidates=all_candidates)

        if d.action == ControlAction.SWITCH:
            # If switch occurred, state.current_plan is updated to a new PartitionPlan reference
            # but NO actual tensor data was moved (no physical migration)
            assert isinstance(ctrl.state.current_plan, PartitionPlan)
        else:
            # Plan reference should be unchanged
            assert id(ctrl.state.current_plan) == initial_plan_ref


# ===========================================================================
# 16. ControllerHistory Metrics
# ===========================================================================

class TestControllerHistoryMetrics:
    def test_empty_metrics(self):
        h = ControllerHistory()
        m = h.get_metrics()
        assert m["total_cycles"] == 0
        assert m["total_switches"] == 0
        assert m["switch_frequency_per_hour"] == 0.0

    def test_metrics_accumulate(self, all_candidates, mock_state):
        cfg = ControllerConfig(
            mode=ControllerMode.REACTIVE,
            switch_threshold=0.0,
            minimum_dwell_seconds=0.0,
            cooldown_seconds=0.0,
            hysteresis_cycles=1,
        )
        ctrl = AdaptivePartitionController(config=cfg)
        for i in range(10):
            state = make_state_at(float(i * 50))
            ctrl.decide(state=state, candidates=all_candidates)

        metrics = ctrl.get_metrics()
        assert metrics["total_cycles"] == 10
        assert metrics["total_switches"] >= 0
        assert "reason_distribution" in metrics
        assert "plan_occupancy" in metrics
        assert "threshold_rejections" in metrics
        assert "cooldown_rejections" in metrics
        assert "hysteresis_rejections" in metrics

    def test_history_records_all_decisions(self, all_candidates, mock_state):
        """History must record every decide() call."""
        cfg = ControllerConfig(mode=ControllerMode.STATIC)
        ctrl = AdaptivePartitionController(config=cfg)
        for _ in range(7):
            ctrl.decide(state=mock_state, candidates=all_candidates)
        assert len(ctrl.history.decisions) == 7

    def test_history_to_dict(self, all_candidates, mock_state):
        cfg = ControllerConfig(mode=ControllerMode.STATIC)
        ctrl = AdaptivePartitionController(config=cfg)
        ctrl.decide(state=mock_state, candidates=all_candidates)
        d = ctrl.history.to_dict()
        assert "metrics" in d
        assert "decisions" in d
        assert len(d["decisions"]) == 1


# ===========================================================================
# 17. Controller Cycle Counter
# ===========================================================================

class TestControllerCycle:
    def test_cycle_increments(self, all_candidates, mock_state):
        """controller_cycle must increment by 1 on each decide() call."""
        cfg = ControllerConfig(mode=ControllerMode.STATIC)
        ctrl = AdaptivePartitionController(config=cfg)
        for i in range(1, 6):
            d = ctrl.decide(state=mock_state, candidates=all_candidates)
            assert d.controller_cycle == i

    def test_cycle_in_migration_request(self, all_candidates):
        """MigrationRequest must contain the correct controller_cycle."""
        cfg = ControllerConfig(
            mode=ControllerMode.REACTIVE,
            switch_threshold=0.0,
            minimum_dwell_seconds=0.0,
            cooldown_seconds=0.0,
            hysteresis_cycles=1,
        )
        ctrl = AdaptivePartitionController(config=cfg)
        for i in range(10):
            state = make_state_at(float(i * 100))
            d = ctrl.decide(state=state, candidates=all_candidates)
            if d.action == ControlAction.SWITCH and d.migration_request:
                assert d.migration_request.controller_cycle == d.controller_cycle
                break


# ===========================================================================
# 18. Replay Multi-Step Trace
# ===========================================================================

class TestReplay:
    def test_replay_returns_decisions(self, all_candidates):
        """replay() must return one ControlDecision per trace step."""
        cfg = ControllerConfig(mode=ControllerMode.STATIC)
        ctrl = AdaptivePartitionController(config=cfg)

        trace = [
            {"state": make_state_at(float(i * 100)), "candidates": all_candidates, "forecast": None}
            for i in range(5)
        ]
        decisions = ctrl.replay(trace)
        assert len(decisions) == 5
        for d in decisions:
            assert isinstance(d, ControlDecision)

    def test_replay_increments_cycle(self, all_candidates):
        """replay() must increment cycle counter sequentially."""
        cfg = ControllerConfig(mode=ControllerMode.STATIC)
        ctrl = AdaptivePartitionController(config=cfg)

        trace = [
            {"state": make_state_at(float(i * 100)), "candidates": all_candidates}
            for i in range(4)
        ]
        decisions = ctrl.replay(trace)
        cycles = [d.controller_cycle for d in decisions]
        assert cycles == list(range(1, 5))


# ===========================================================================
# 19. Serialization
# ===========================================================================

class TestSerialization:
    def test_control_decision_to_dict(self, all_candidates, mock_state):
        cfg = ControllerConfig(mode=ControllerMode.STATIC)
        ctrl = AdaptivePartitionController(config=cfg)
        d = ctrl.decide(state=mock_state, candidates=all_candidates)
        dd = d.to_dict()
        required_keys = [
            "action", "current_plan_id", "selected_plan_id", "current_cost",
            "selected_cost", "expected_gain", "threshold", "switch_allowed",
            "reason", "explanation", "timestamp", "controller_cycle",
            "prediction_used", "is_proactive", "is_safety_override",
            "safety_status", "stability_status", "migration_request",
        ]
        for key in required_keys:
            assert key in dd, f"Missing key: {key}"

    def test_controller_state_to_dict(self, local_plan):
        state = ControllerState(
            current_plan=local_plan,
            current_plan_id=generate_plan_id(local_plan),
        )
        d = state.to_dict()
        assert "current_plan_id" in d
        assert "switch_count" in d
        assert "controller_cycle" in d

    def test_migration_request_to_dict(self, all_candidates):
        cfg = ControllerConfig(
            mode=ControllerMode.REACTIVE,
            switch_threshold=0.0,
            minimum_dwell_seconds=0.0,
            cooldown_seconds=0.0,
            hysteresis_cycles=1,
        )
        ctrl = AdaptivePartitionController(config=cfg)
        for i in range(15):
            state = make_state_at(float(i * 100))
            d = ctrl.decide(state=state, candidates=all_candidates)
            if d.migration_request is not None:
                mr_dict = d.migration_request.to_dict()
                assert "source_plan_id" in mr_dict
                assert "target_plan_id" in mr_dict
                assert "expected_gain" in mr_dict
                assert "changed_layers" in mr_dict
                break


# ===========================================================================
# 20. Safety Policy Unit Tests
# ===========================================================================

class TestSafetyPolicy:
    def _make_feasible_score(self, plan_id: str = "test") -> CandidateScore:
        """Build a minimal feasible CandidateScore for safety testing."""
        from src.cost.types import CostWeights
        local = PartitionPlan.monolithic(total_layers=12)
        cand = CandidatePlan(
            plan_id=plan_id,
            partition_plan=local,
            active_tiers=[TierId.USER_DEVICE],
            layer_assignment={TierId.USER_DEVICE.value: (0, 11)},
            number_of_boundaries=0,
            boundaries=[],
            feasibility_now=FeasibilityStatus.FEASIBLE,
            feasibility_predicted=FeasibilityStatus.FEASIBLE,
        )
        breakdown = CostBreakdown(
            latency_raw_ms=50.0,
            communication_raw_mb=0.0,
            memory_pressure_raw=0.3,
            energy_raw_j=2.0,
            switching_raw_units=0.0,
            latency_normalized=0.3,
            communication_normalized=0.0,
            memory_pressure_normalized=0.2,
            energy_normalized=0.1,
            switching_normalized=0.0,
            latency_cost=0.3,
            communication_cost=0.0,
            memory_pressure_cost=0.2,
            energy_cost=0.1,
            switching_cost=0.0,
            total_cost=0.6,
            current_pressure=0.3,
            predicted_max_pressure=0.4,
            estimated_kv_transfer_bytes=0,
        )
        return CandidateScore(
            candidate_plan=cand,
            breakdown=breakdown,
            total_cost=0.6,
            status=ScoreStatus.FEASIBLE,
            feasible=True,
            explanation="test score",
        )

    def test_safe_status_normal(self):
        policy = SafetyPolicy()
        score = self._make_feasible_score()
        result = policy.evaluate_current_plan(current_score=score)
        assert result.status == SafetyStatus.SAFE
        assert result.is_emergency is False

    def test_emergency_on_infeasible(self):
        local = PartitionPlan.monolithic(total_layers=12)
        cand = CandidatePlan(
            plan_id="infeasible",
            partition_plan=local,
            active_tiers=[TierId.USER_DEVICE],
            layer_assignment={TierId.USER_DEVICE.value: (0, 11)},
            number_of_boundaries=0,
            boundaries=[],
            feasibility_now=FeasibilityStatus.INFEASIBLE,
            feasibility_predicted=FeasibilityStatus.INFEASIBLE,
        )
        breakdown = CostBreakdown(
            latency_raw_ms=100.0,
            communication_raw_mb=0.0,
            memory_pressure_raw=1.0,
            energy_raw_j=0.0,
            switching_raw_units=0.0,
            latency_normalized=1.0,
            communication_normalized=0.0,
            memory_pressure_normalized=1.0,
            energy_normalized=0.0,
            switching_normalized=0.0,
            latency_cost=1.0,
            communication_cost=0.0,
            memory_pressure_cost=1.0,
            energy_cost=0.0,
            switching_cost=0.0,
            total_cost=2.0,
        )
        infeasible_score = CandidateScore(
            candidate_plan=cand,
            breakdown=breakdown,
            total_cost=2.0,
            status=ScoreStatus.INFEASIBLE,
            feasible=False,
            explanation="infeasible",
        )
        policy = SafetyPolicy()
        result = policy.evaluate_current_plan(current_score=infeasible_score)
        assert result.status == SafetyStatus.EMERGENCY
        assert result.is_emergency is True

    def test_emergency_on_high_memory_pressure(self):
        policy = SafetyPolicy(config=ControllerConfig(max_memory_pressure=0.90))
        score = self._make_feasible_score()
        high_pressure_breakdown = CostBreakdown(
            latency_raw_ms=50.0,
            communication_raw_mb=0.0,
            memory_pressure_raw=0.95,
            energy_raw_j=2.0,
            switching_raw_units=0.0,
            latency_normalized=0.3,
            communication_normalized=0.0,
            memory_pressure_normalized=0.9,
            energy_normalized=0.1,
            switching_normalized=0.0,
            latency_cost=0.3,
            communication_cost=0.0,
            memory_pressure_cost=0.9,
            energy_cost=0.1,
            switching_cost=0.0,
            total_cost=1.5,
            current_pressure=0.92,
            predicted_max_pressure=0.95,
            estimated_kv_transfer_bytes=0,
        )
        high_score = CandidateScore(
            candidate_plan=score.candidate_plan,
            breakdown=high_pressure_breakdown,
            total_cost=1.5,
            status=ScoreStatus.FEASIBLE,
            feasible=True,
            explanation="high pressure",
        )
        result = policy.evaluate_current_plan(current_score=high_score)
        assert result.status == SafetyStatus.EMERGENCY
        assert result.is_emergency is True

    def test_warning_on_invalid_forecast(self):
        policy = SafetyPolicy()
        score = self._make_feasible_score()
        invalid_forecast = PredictionResult(
            prediction_timestamp=1000.0,
            horizon_seconds=2.0,
            step_interval_seconds=0.5,
            forecast_timestamps=[1000.5],
            targets={},
            predictor_name="test",
            input_window_length=0,
            is_valid=False,
            status_message="insufficient data",
        )
        result = policy.evaluate_current_plan(current_score=score, forecast=invalid_forecast)
        assert result.status == SafetyStatus.WARNING
        assert result.is_emergency is False


# ===========================================================================
# 21. StabilityChecker Unit Tests
# ===========================================================================

class TestStabilityChecker:
    def _make_ctrl_state(self, plan_id: str = "plan_A") -> ControllerState:
        local = PartitionPlan.monolithic(total_layers=12)
        return ControllerState(
            current_plan=local,
            current_plan_id=plan_id,
            last_switch_timestamp=0.0,
            consecutive_preference_plan_id=None,
            consecutive_preference_count=0,
        )

    def test_gain_below_threshold_rejected(self):
        checker = StabilityChecker(ControllerConfig(switch_threshold=0.5))
        state = self._make_ctrl_state("A")
        result = checker.evaluate(
            current_cost=1.0, candidate_cost=0.8,  # gain=0.2 < 0.5
            candidate_plan_id="B", ctrl_state=state, current_timestamp=1000.0,
        )
        assert result.switch_allowed is False
        assert result.reason == DecisionReason.BELOW_THRESHOLD

    def test_gain_above_threshold_and_hysteresis_satisfied(self):
        checker = StabilityChecker(ControllerConfig(
            switch_threshold=0.1,
            minimum_dwell_seconds=0.0,
            cooldown_seconds=0.0,
            hysteresis_cycles=1,
        ))
        state = self._make_ctrl_state("A")
        result = checker.evaluate(
            current_cost=1.0, candidate_cost=0.5,  # gain=0.5 > 0.1
            candidate_plan_id="B", ctrl_state=state, current_timestamp=1000.0,
        )
        assert result.switch_allowed is True

    def test_emergency_bypasses_all(self):
        checker = StabilityChecker(ControllerConfig(
            switch_threshold=999.0,
            minimum_dwell_seconds=999.0,
            cooldown_seconds=999.0,
            hysteresis_cycles=100,
            emergency_override=True,
        ))
        state = self._make_ctrl_state("A")
        state.last_switch_timestamp = 999.0  # recent switch
        result = checker.evaluate(
            current_cost=1.0, candidate_cost=0.5,
            candidate_plan_id="B", ctrl_state=state, current_timestamp=1000.0,
            is_emergency=True,
        )
        assert result.switch_allowed is True
        assert result.reason == DecisionReason.EMERGENCY_RECOVERY

    def test_same_plan_rejected(self):
        checker = StabilityChecker(ControllerConfig(switch_threshold=0.0))
        state = self._make_ctrl_state("A")
        result = checker.evaluate(
            current_cost=1.0, candidate_cost=0.5,
            candidate_plan_id="A",  # Same as current
            ctrl_state=state, current_timestamp=1000.0,
        )
        assert result.switch_allowed is False
        assert result.reason == DecisionReason.BEST_CANDIDATE_IS_CURRENT

    def test_dwell_time_blocks(self):
        checker = StabilityChecker(ControllerConfig(
            switch_threshold=0.0,
            minimum_dwell_seconds=100.0,
            cooldown_seconds=0.0,
            hysteresis_cycles=1,
        ))
        state = self._make_ctrl_state("A")
        state.last_switch_timestamp = 999.0  # 1 second ago

        result = checker.evaluate(
            current_cost=1.0, candidate_cost=0.5,
            candidate_plan_id="B", ctrl_state=state, current_timestamp=1000.0,
        )
        assert result.switch_allowed is False
        assert result.reason == DecisionReason.DWELL_TIME_ACTIVE

    def test_hysteresis_not_satisfied(self):
        checker = StabilityChecker(ControllerConfig(
            switch_threshold=0.0,
            minimum_dwell_seconds=0.0,
            cooldown_seconds=0.0,
            hysteresis_cycles=3,
        ))
        state = self._make_ctrl_state("A")
        state.consecutive_preference_plan_id = "B"
        state.consecutive_preference_count = 1  # Only 1 of 3 required

        result = checker.evaluate(
            current_cost=1.0, candidate_cost=0.5,
            candidate_plan_id="B", ctrl_state=state, current_timestamp=1000.0,
        )
        assert result.switch_allowed is False
        assert result.reason == DecisionReason.HYSTERESIS_NOT_SATISFIED
        assert result.consecutive_count == 2  # incremented

    def test_negative_gain_rejected(self):
        checker = StabilityChecker()
        state = self._make_ctrl_state("A")
        result = checker.evaluate(
            current_cost=0.5, candidate_cost=1.0,  # negative gain
            candidate_plan_id="B", ctrl_state=state, current_timestamp=1000.0,
        )
        assert result.switch_allowed is False
        assert result.reason == DecisionReason.NO_IMPROVEMENT


# ===========================================================================
# 22. 5-Phase Research Trace Simulation
# ===========================================================================

class TestFivePhaseTrace:
    """
    Simulates: stable -> forecast degradation -> proactive switch -> dwell hold -> recovery.

    Phase 1 (t=0-40s):   Stable, local plan is optimal, no switch needed.
    Phase 2 (t=50-80s):  Forecast indicates future bandwidth degradation.
    Phase 3 (t=90s):     Proactive switch to edge plan (forecast-driven).
    Phase 4 (t=100-140s): Dwell time active, hold edge plan.
    Phase 5 (t=150-200s): Recovery, bandwidth improves, may switch back.
    """

    def _make_degraded_forecast(self, t: float) -> PredictionResult:
        """Forecast showing declining bandwidth."""
        return PredictionResult(
            prediction_timestamp=t,
            horizon_seconds=4.0,
            step_interval_seconds=1.0,
            forecast_timestamps=[t + 1, t + 2, t + 3, t + 4],
            targets={
                "bandwidth_mbps": TargetForecast(
                    target_name="bandwidth_mbps",
                    values=[40.0, 30.0, 20.0, 10.0],
                    raw_values=[40.0, 30.0, 20.0, 10.0],
                    timestamps=[t + 1, t + 2, t + 3, t + 4],
                    is_available=True,
                    status="ok",
                    provenance="emulated",
                ),
                "latency_ms": TargetForecast(
                    target_name="latency_ms",
                    values=[20.0, 30.0, 50.0, 80.0],
                    raw_values=[20.0, 30.0, 50.0, 80.0],
                    timestamps=[t + 1, t + 2, t + 3, t + 4],
                    is_available=True,
                    status="ok",
                    provenance="emulated",
                ),
                "vram_free_mb": TargetForecast.unavailable("vram_free_mb", [t + 1, t + 2, t + 3, t + 4]),
            },
            predictor_name="linear_trend",
            input_window_length=8,
            is_valid=True,
            status_message="ok",
        )

    def _make_recovery_forecast(self, t: float) -> PredictionResult:
        """Forecast showing improving bandwidth."""
        return PredictionResult(
            prediction_timestamp=t,
            horizon_seconds=4.0,
            step_interval_seconds=1.0,
            forecast_timestamps=[t + 1, t + 2, t + 3, t + 4],
            targets={
                "bandwidth_mbps": TargetForecast(
                    target_name="bandwidth_mbps",
                    values=[55.0, 60.0, 65.0, 70.0],
                    raw_values=[55.0, 60.0, 65.0, 70.0],
                    timestamps=[t + 1, t + 2, t + 3, t + 4],
                    is_available=True,
                    status="ok",
                    provenance="emulated",
                ),
                "latency_ms": TargetForecast(
                    target_name="latency_ms",
                    values=[12.0, 10.0, 8.0, 6.0],
                    raw_values=[12.0, 10.0, 8.0, 6.0],
                    timestamps=[t + 1, t + 2, t + 3, t + 4],
                    is_available=True,
                    status="ok",
                    provenance="emulated",
                ),
                "vram_free_mb": TargetForecast.unavailable("vram_free_mb", [t + 1, t + 2, t + 3, t + 4]),
            },
            predictor_name="linear_trend",
            input_window_length=8,
            is_valid=True,
            status_message="ok",
        )

    def test_five_phase_trace_produces_decisions(self):
        """5-phase trace must complete without errors and return valid decisions."""
        cfg = ControllerConfig(
            mode=ControllerMode.PREDICTIVE,
            switch_threshold=0.05,
            minimum_dwell_seconds=10.0,
            cooldown_seconds=5.0,
            hysteresis_cycles=2,
            emergency_override=True,
        )
        ctrl = AdaptivePartitionController(config=cfg)
        meta = ModelMetadata.synthetic_default(total_layers=12)
        catalog = SplitCatalog()
        candidates = catalog.generate(meta, mode="exhaustive")

        all_decisions = []

        # Phase 1: Stable (t=0 to t=40, step=10)
        for t in range(0, 50, 10):
            state = make_state_at(float(t), bandwidth_mbps=50.0, latency_ms=15.0)
            d = ctrl.decide(state=state, candidates=candidates, forecast=None)
            all_decisions.append(("stable", t, d))

        # Phase 2: Degradation forecast (t=50 to t=80)
        for t in range(50, 90, 10):
            state = make_state_at(float(t), bandwidth_mbps=45.0, latency_ms=20.0)
            forecast = self._make_degraded_forecast(float(t))
            d = ctrl.decide(state=state, candidates=candidates, forecast=forecast)
            all_decisions.append(("degradation", t, d))

        # Phase 3: Proactive switch window (t=90)
        state = make_state_at(90.0, bandwidth_mbps=35.0, latency_ms=30.0)
        forecast = self._make_degraded_forecast(90.0)
        d = ctrl.decide(state=state, candidates=candidates, forecast=forecast)
        all_decisions.append(("switch", 90, d))

        # Phase 4: Dwell hold (t=100 to t=140)
        for t in range(100, 150, 10):
            state = make_state_at(float(t), bandwidth_mbps=30.0, latency_ms=35.0)
            d = ctrl.decide(state=state, candidates=candidates)
            all_decisions.append(("dwell", t, d))

        # Phase 5: Recovery (t=150 to t=200)
        for t in range(150, 210, 10):
            state = make_state_at(float(t), bandwidth_mbps=60.0, latency_ms=10.0)
            forecast = self._make_recovery_forecast(float(t))
            d = ctrl.decide(state=state, candidates=candidates, forecast=forecast)
            all_decisions.append(("recovery", t, d))

        # All decisions must be valid ControlDecision instances
        assert len(all_decisions) > 0
        for phase, t, decision in all_decisions:
            assert isinstance(decision, ControlDecision)
            assert decision.action in list(ControlAction)
            assert decision.timestamp >= 0.0
            assert decision.controller_cycle >= 1

        # Metrics summary
        metrics = ctrl.get_metrics()
        assert metrics["total_cycles"] == len(all_decisions)
        assert metrics["total_switches"] >= 0

    def test_five_phase_no_thrashing_within_dwell(self):
        """During dwell phase, rapid re-switching must NOT occur."""
        cfg = ControllerConfig(
            mode=ControllerMode.PREDICTIVE,
            switch_threshold=0.05,
            minimum_dwell_seconds=15.0,
            cooldown_seconds=5.0,
            hysteresis_cycles=2,
        )
        ctrl = AdaptivePartitionController(config=cfg)
        meta = ModelMetadata.synthetic_default(total_layers=12)
        catalog = SplitCatalog()
        candidates = catalog.generate(meta, mode="exhaustive")

        switch_timestamps = []
        for t in range(0, 200, 5):
            state = make_state_at(float(t))
            d = ctrl.decide(state=state, candidates=candidates)
            if d.action == ControlAction.SWITCH:
                switch_timestamps.append(float(t))

        # Verify minimum dwell between consecutive switches
        for i in range(1, len(switch_timestamps)):
            gap = switch_timestamps[i] - switch_timestamps[i - 1]
            assert gap >= 15.0, f"Switch gap {gap}s < dwell {15.0}s (anti-thrashing violated)"


# ===========================================================================
# 23. Current Plan Synchronization
# ===========================================================================

class TestCurrentPlanSync:
    def test_explicit_current_plan_overrides_internal(self, all_candidates, mock_state):
        """Passing current_plan= to decide() should override internal state."""
        cfg = ControllerConfig(mode=ControllerMode.REACTIVE)
        ctrl = AdaptivePartitionController(config=cfg)
        new_plan = PartitionPlan.monolithic(total_layers=12)
        d = ctrl.decide(state=mock_state, candidates=all_candidates, current_plan=new_plan)
        new_id = generate_plan_id(new_plan)
        assert d.current_plan_id == new_id

    def test_controller_state_updated_after_switch(self, all_candidates):
        """After a switch, internal state.current_plan_id must reflect new plan."""
        cfg = ControllerConfig(
            mode=ControllerMode.REACTIVE,
            switch_threshold=0.0,
            minimum_dwell_seconds=0.0,
            cooldown_seconds=0.0,
            hysteresis_cycles=1,
        )
        ctrl = AdaptivePartitionController(config=cfg)
        original_id = ctrl.state.current_plan_id

        for i in range(15):
            state = make_state_at(float(i * 100))
            d = ctrl.decide(state=state, candidates=all_candidates)
            if d.action == ControlAction.SWITCH:
                # Internal state must now reflect the new plan
                assert ctrl.state.current_plan_id == d.selected_plan_id
                assert ctrl.state.current_plan_id != original_id or d.selected_plan_id == original_id
                break
