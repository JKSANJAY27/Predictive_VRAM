"""
ClosedLoopRuntime orchestrator integrating Modules 1 through 8.

Coordinates:
- Token-by-token split autoregressive inference (Module 1)
- Continuous source-tagged telemetry sampling (Module 2)
- Unified RuntimeState and StateBuffer updates (Module 3)
- Short-horizon predictive forecasting (Module 4)
- Feasible partition candidate enumeration (Module 5)
- Multi-objective candidate scoring (Module 6)
- Adaptive partition control and anti-thrashing (Module 7)
- Transactional physical migration and rollback (Module 8)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.controller.controller import AdaptivePartitionController
from src.controller.types import ControlAction, ControllerConfig, ControllerMode
from src.cost.model import CostModel
from src.cost.types import CostWeights
from src.migration.adapter import RuntimeAdapter
from src.migration.manager import MigrationManager
from src.migration.types import MigrationConfig, MigrationMode, MigrationStatus, VerificationMode
from src.orchestration.adapters import TelemetryAgent
from src.orchestration.config import OrchestrationConfig
from src.orchestration.cycle import ControlCycleRunner
from src.orchestration.errors import ErrorCategory
from src.orchestration.scenarios import ScenarioRegistry, SyntheticScenario
from src.orchestration.scheduler import ControlScheduler
from src.orchestration.trace import RuntimeTrace
from src.orchestration.types import (
    ControlCycle,
    ExecutionMode,
    MigrationWindowRecord,
    OrchestrationMetrics,
    OrchestrationMode,
    PredictionLeadRecord,
    TokenRecord,
)
from src.partitioning.catalog import SplitCatalog
from src.partitioning.metadata import ModelMetadata
from src.prediction import (
    KVCacheAwareMemoryPredictor,
    LastValuePredictor,
    LinearTrendPredictor,
    MovingAveragePredictor,
    Predictor,
)
from src.runtime.executor import DistributedInferenceExecutor
from src.runtime.model import LayeredTransformer
from src.runtime.partition import PartitionPlan
from src.runtime.tier import Tier, TierId
from src.state.buffer import StateBuffer
from src.state.types import RuntimeState


class ClosedLoopRuntime:
    """
    Main closed-loop runtime engine managing end-to-end split inference,
    continuous telemetry, predictive forecasting, candidate scoring,
    and transactional partition migration.
    """

    def __init__(
        self,
        config: OrchestrationConfig,
        executor: DistributedInferenceExecutor,
        telemetry_agent: TelemetryAgent,
        state_buffer: StateBuffer,
        predictor: Optional[Predictor],
        split_catalog: SplitCatalog,
        cost_model: CostModel,
        controller: AdaptivePartitionController,
        migration_manager: MigrationManager,
        model_metadata: Optional["ModelMetadata"] = None,
    ) -> None:
        self.config = config
        self.executor = executor
        self.telemetry_agent = telemetry_agent
        self.state_buffer = state_buffer
        self.predictor = predictor
        self.split_catalog = split_catalog
        self.cost_model = cost_model
        self.controller = controller
        self.migration_manager = migration_manager
        # ModelMetadata needed by catalog.generate() — synthetic default if not provided
        self.model_metadata = model_metadata or ModelMetadata(
            total_layers=4, hidden_size=64, num_heads=2, vocab_size=500, dtype_bytes=4,
        )

        self.scheduler = ControlScheduler(self.config)
        self.cycle_runner = ControlCycleRunner(
            config=self.config,
            telemetry_agent=self.telemetry_agent,
            state_buffer=self.state_buffer,
            predictor=self.predictor,
            split_catalog=self.split_catalog,
            model_metadata=self.model_metadata,
            cost_model=self.cost_model,
            controller=self.controller,
            migration_manager=self.migration_manager,
        )

    # -----------------------------------------------------------------------
    # Factory Constructor
    # -----------------------------------------------------------------------

    @classmethod
    def create_default(
        cls,
        config: Optional[OrchestrationConfig] = None,
        num_layers: int = 4,
        hidden_size: int = 64,
        initial_plan: Optional[PartitionPlan] = None,
    ) -> ClosedLoopRuntime:
        """
        Convenience factory constructing an operational synthetic closed-loop runtime.
        """
        cfg = config or OrchestrationConfig()
        torch.manual_seed(cfg.random_seed)

        # 1. Model & Executor
        model = LayeredTransformer.create_synthetic(
            num_layers=num_layers,
            hidden_size=hidden_size,
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

        # Setup initial plan
        plan = initial_plan or PartitionPlan(
            total_layers=num_layers,
            user_device=(0, 1),
            edge_a=(2, num_layers - 1),
        )
        executor._setup_tiers_for_plan(plan)
        executor.active_plan = plan

        # 2. Telemetry
        telemetry_agent = TelemetryAgent()

        # 3. State Buffer
        state_buffer = StateBuffer(capacity=100)

        # 4. Predictor
        predictor: Optional[Predictor] = None
        if cfg.predictor_type == "linear_trend":
            predictor = LinearTrendPredictor()
        elif cfg.predictor_type in ("kv_aware", "kv_cache_aware"):
            predictor = KVCacheAwareMemoryPredictor()
        elif cfg.predictor_type == "moving_average":
            predictor = MovingAveragePredictor()
        elif cfg.predictor_type == "last_value":
            predictor = LastValuePredictor()

        # 5. Split Catalog and ModelMetadata
        model_meta = ModelMetadata(
            total_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=2,
            vocab_size=500,
            dtype_bytes=4,
        )
        catalog = SplitCatalog()

        # 6. Cost Model
        weights = cfg.cost_weights
        if isinstance(weights, dict):
            weights = CostWeights.from_dict(weights)
        cost_model = CostModel(weights=weights)

        # 7. Controller
        ctrl_mode = (
            ControllerMode.STATIC if cfg.mode == OrchestrationMode.STATIC
            else ControllerMode.REACTIVE if cfg.mode == OrchestrationMode.REACTIVE
            else ControllerMode.PREDICTIVE
        )
        ctrl_config = ControllerConfig(
            mode=ctrl_mode,
            switch_threshold=0.05,
            minimum_dwell_seconds=0.0,
            cooldown_seconds=0.0,
            hysteresis_cycles=1,
        )
        controller = AdaptivePartitionController(
            config=ctrl_config,
            cost_model=cost_model,
            initial_plan=plan,
        )

        # 8. Migration Manager
        adapter = RuntimeAdapter(executor)
        v_mode = VerificationMode.FAST if cfg.migration_verification_mode == "FAST" else VerificationMode.DEEP
        m_mode = MigrationMode.SIMULATE if cfg.execution_mode == ExecutionMode.SIMULATION else MigrationMode.EXECUTE
        mig_cfg = MigrationConfig(
            execution_mode=m_mode,
            verification_mode=v_mode,
            rollback_enabled=cfg.rollback_enabled,
            emulated_bandwidth_delay=cfg.emulated_bandwidth_delay,
            fast_mode=True,
        )
        migration_mgr = MigrationManager(runtime_adapter=adapter, config=mig_cfg)

        return cls(
            config=cfg,
            executor=executor,
            telemetry_agent=telemetry_agent,
            state_buffer=state_buffer,
            predictor=predictor,
            split_catalog=catalog,
            model_metadata=model_meta,
            cost_model=cost_model,
            controller=controller,
            migration_manager=migration_mgr,
        )

    # -----------------------------------------------------------------------
    # Main Execution Loop
    # -----------------------------------------------------------------------

    def run_inference(
        self,
        prompt: str = "The quick brown fox",
        max_new_tokens: Optional[int] = None,
        scenario_name: Optional[str] = None,
    ) -> RuntimeTrace:
        """
        Execute closed-loop autoregressive inference with periodic control cycles.
        """
        torch.manual_seed(self.config.random_seed)
        start_wall_time = time.time()
        max_tokens = max_new_tokens or self.config.max_generated_tokens
        scenario = ScenarioRegistry.get(scenario_name or self.config.scenario_name)

        trace = RuntimeTrace(
            metadata={
                "experiment_id": self.config.experiment_id,
                "timestamp": start_wall_time,
                "mode": self.config.mode.value,
                "execution_mode": self.config.execution_mode.value,
                "model": self.config.model_name,
                "random_seed": self.config.random_seed,
                "scenario": scenario.name,
                "horizon": self.config.horizon,
                "control_interval_tokens": self.config.control_interval_tokens,
                "cost_weights": self.config.cost_weights,
            }
        )

        self.scheduler.reset()

        # Prepare tokens and generation state
        tok = self.executor.model.tokenizer
        if hasattr(tok, "encode"):
            input_ids = tok.encode(prompt, return_tensors="pt")
        else:
            input_ids = tok(prompt, return_tensors="pt")["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor([input_ids], dtype=torch.long)
        if hasattr(self.executor.model, "create_kv_cache"):
            kv_cache = self.executor.model.create_kv_cache()
        else:
            from transformers import DynamicCache
            kv_cache = DynamicCache()
        self.migration_manager.adapter.set_active_kv_cache(kv_cache)

        current_plan = self.migration_manager.adapter.get_current_plan()
        self.executor.active_plan = current_plan
        first_device = self.executor.tiers[TierId.USER_DEVICE].device

        current_input_ids = input_ids.to(first_device)
        past_length = 0
        itl_latencies: List[float] = []
        ttft: float = 0.0

        # Autoregressive decode loop
        for step in range(max_tokens):
            step_start = time.perf_counter()

            # 1. Update synthetic scenario conditions for this step
            conds = scenario.get_conditions(step, max_tokens)
            self.telemetry_agent.set_network_conditions(
                bandwidth_mbps=conds.bandwidth_mbps,
                rtt_ms=conds.rtt_ms,
                packet_loss=conds.packet_loss,
                jitter_ms=conds.jitter_ms,
            )
            if conds.inject_dropout:
                self.telemetry_agent.inject_network_dropout = True
            else:
                self.telemetry_agent.inject_network_dropout = False

            # 2. Execute single token forward pass through current partition
            current_plan = self.migration_manager.adapter.get_current_plan()
            active_tiers = current_plan.get_active_tiers()

            # Embedding at first tier
            hidden = self.executor.model.embed(current_input_ids, past_length=past_length)

            # Sequential tier execution
            activations_list = []
            for i, (tier_id, (start_l, end_l)) in enumerate(active_tiers):
                if i > 0:
                    prev_tier_id, _ = active_tiers[i - 1]
                    target_device = self.executor.tiers[tier_id].device
                    act_snap = self.telemetry_agent.activation_collector.collect(
                        tensor=hidden,
                        source_tier=prev_tier_id.value,
                        destination_tier=tier_id.value,
                        step=step,
                    )
                    activations_list.append(act_snap)
                    hidden = self.executor.transfer_manager.transfer(
                        tensor=hidden,
                        source_tier=prev_tier_id,
                        destination_tier=tier_id,
                        target_device=target_device,
                        step=step,
                    )

                hidden = self.executor.model.forward_layer_range(
                    hidden_states=hidden,
                    start_layer=start_l,
                    end_layer=end_l,
                    past_key_values=kv_cache,
                    attention_mask=None,
                )

            # Logits & greedy selection
            logits = self.executor.model.compute_logits(hidden)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            next_token_id = int(next_token.item())
            token_str = self.executor.model.tokenizer.decode([next_token_id])

            step_duration_ms = (time.perf_counter() - step_start) * 1000.0
            is_ttft = (step == 0)
            if is_ttft:
                ttft = step_duration_ms
            else:
                itl_latencies.append(step_duration_ms)

            # Record token
            plan_id = getattr(current_plan, "plan_id", str(current_plan))
            token_rec = TokenRecord(
                token_index=step,
                token_id=next_token_id,
                token_str=token_str,
                step_latency_ms=step_duration_ms,
                timestamp=time.time(),
                active_plan_id=plan_id,
                is_ttft=is_ttft,
            )
            trace.add_token(token_rec)

            # 3. Synchronize control cycle on token boundary if triggered
            if self.scheduler.should_trigger(step, time.time()):
                if self.scheduler.mark_cycle_started():
                    # Record recent pre-migration latencies
                    pre_itls = itl_latencies[-3:] if itl_latencies else [step_duration_ms]
                    pre_tokens = [t.token_id for t in trace.token_records[-3:]]
                    plan_before = self.migration_manager.adapter.get_current_plan()

                    # Run 14-step control cycle
                    cycle = self.cycle_runner.run_cycle(
                        inference_step=step,
                        current_plan=plan_before,
                        kv_cache=kv_cache,
                        activations=activations_list,
                    )
                    self.scheduler.mark_cycle_completed(step, time.time())
                    trace.add_cycle(cycle)

                    # If migration occurred, track migration window and lead time
                    if cycle.migration_result and cycle.migration_result.success:
                        window = MigrationWindowRecord(
                            token_index=step,
                            tokens_before=pre_tokens,
                            itl_before_ms=pre_itls,
                            migration_duration_ms=cycle.migration_result.timings.total_duration_ms,
                            tokens_after=[],  # will populate subsequently
                            itl_after_ms=[],
                            network_before_mbps=conds.bandwidth_mbps,
                        )
                        trace.add_migration_window(window)

                        # If proactive predictive switch:
                        if cycle.control_decision and cycle.control_decision.is_proactive:
                            lead_rec = PredictionLeadRecord(
                                prediction_time=cycle.cycle_timestamp,
                                decision_time=cycle.cycle_timestamp,
                                migration_start=cycle.migration_result.start_timestamp,
                                actual_degradation_time=None,
                                prediction_lead_time_s=0.5,  # Estimated lead time
                                decision_lead_time_s=0.5,
                            )
                            trace.add_prediction_lead(lead_rec)

                    # Stop condition check on critical error
                    if self.config.stop_on_failure and cycle.error_category != ErrorCategory.NONE:
                        break

            # Check time-based stop condition
            if (time.time() - start_wall_time) >= self.config.max_runtime_seconds:
                break

            # Prepare next decode step
            current_input_ids = next_token.to(first_device)
            past_length = input_ids.shape[1] + step

        # Compute aggregate metrics
        trace.summary = self._compute_metrics(trace, start_wall_time, itl_latencies, ttft)
        return trace

    # -----------------------------------------------------------------------
    # Simulation & Replay Modes
    # -----------------------------------------------------------------------

    def run_simulation(
        self,
        scenario_name: Optional[str] = None,
        num_steps: int = 32,
    ) -> RuntimeTrace:
        """
        Execute closed-loop simulation over a synthetic scenario without full tensor evaluation.
        """
        torch.manual_seed(self.config.random_seed)
        start_wall_time = time.time()
        scenario = ScenarioRegistry.get(scenario_name or self.config.scenario_name)

        trace = RuntimeTrace(
            metadata={
                "experiment_id": self.config.experiment_id,
                "timestamp": start_wall_time,
                "mode": self.config.mode.value,
                "execution_mode": ExecutionMode.SIMULATION.value,
                "scenario": scenario.name,
                "horizon": self.config.horizon,
                "control_interval_tokens": self.config.control_interval_tokens,
            }
        )

        self.scheduler.reset()
        sim_itls: List[float] = []
        ttft = 15.0

        for step in range(num_steps):
            conds = scenario.get_conditions(step, num_steps)
            self.telemetry_agent.set_network_conditions(
                bandwidth_mbps=conds.bandwidth_mbps,
                rtt_ms=conds.rtt_ms,
                packet_loss=conds.packet_loss,
                jitter_ms=conds.jitter_ms,
            )

            step_lat = 10.0 + (50.0 / max(0.5, conds.bandwidth_mbps))
            is_ttft = (step == 0)
            if not is_ttft:
                sim_itls.append(step_lat)

            current_plan = self.migration_manager.adapter.get_current_plan()
            token_rec = TokenRecord(
                token_index=step,
                token_id=42 + (step % 20),
                token_str=f"tok_{step}",
                step_latency_ms=step_lat,
                timestamp=time.time(),
                active_plan_id=str(current_plan),
                is_ttft=is_ttft,
            )
            trace.add_token(token_rec)

            if self.scheduler.should_trigger(step, time.time()):
                if self.scheduler.mark_cycle_started():
                    cycle = self.cycle_runner.run_cycle(
                        inference_step=step,
                        current_plan=current_plan,
                    )
                    self.scheduler.mark_cycle_completed(step, time.time())
                    trace.add_cycle(cycle)

        trace.summary = self._compute_metrics(trace, start_wall_time, sim_itls, ttft)
        return trace

    def run_replay(self, base_trace: RuntimeTrace) -> RuntimeTrace:
        """
        Replay pre-recorded observations from another trace through the control loop.
        """
        start_wall_time = time.time()
        replayed_trace = RuntimeTrace(
            metadata={
                "experiment_id": f"replay_{self.config.experiment_id}",
                "timestamp": start_wall_time,
                "mode": self.config.mode.value,
                "execution_mode": ExecutionMode.REPLAY.value,
                "base_experiment_id": base_trace.metadata.get("experiment_id"),
            }
        )

        self.scheduler.reset()
        sim_itls: List[float] = []
        ttft = 15.0

        for tok in base_trace.token_records:
            step = tok.token_index
            if not tok.is_ttft:
                sim_itls.append(tok.step_latency_ms)
            replayed_trace.add_token(tok)

            if self.scheduler.should_trigger(step, time.time()):
                if self.scheduler.mark_cycle_started():
                    current_plan = self.migration_manager.adapter.get_current_plan()
                    cycle = self.cycle_runner.run_cycle(
                        inference_step=step,
                        current_plan=current_plan,
                    )
                    self.scheduler.mark_cycle_completed(step, time.time())
                    replayed_trace.add_cycle(cycle)

        replayed_trace.summary = self._compute_metrics(replayed_trace, start_wall_time, sim_itls, ttft)
        return replayed_trace

    # -----------------------------------------------------------------------
    # Metrics Aggregator
    # -----------------------------------------------------------------------

    def _compute_metrics(
        self,
        trace: RuntimeTrace,
        start_wall: float,
        itl_list: List[float],
        ttft_ms: float,
    ) -> OrchestrationMetrics:
        """Compute end-to-end performance and control-plane overhead statistics."""
        e2e_ms = (time.time() - start_wall) * 1000.0
        completed = len(trace.token_records)

        mean_itl = sum(itl_list) / max(1, len(itl_list)) if itl_list else 0.0
        p95_itl = sorted(itl_list)[int(len(itl_list) * 0.95)] if itl_list else mean_itl

        total_switches = 0
        proactive_switches = 0
        reactive_switches = 0
        safety_overrides = 0
        successful_migs = 0
        rolled_back_migs = 0
        failed_migs = 0
        mig_time_total = 0.0
        mig_bytes_total = 0

        # Overhead breakdown
        overhead_telemetry = 0.0
        overhead_prediction = 0.0
        overhead_candidates = 0.0
        overhead_cost = 0.0
        overhead_controller = 0.0
        overhead_migration = 0.0

        for c in trace.cycles:
            timings = c.timings
            overhead_telemetry += timings.get("telemetry_ms", 0.0) + timings.get("state_conversion_ms", 0.0)
            overhead_prediction += timings.get("prediction_ms", 0.0)
            overhead_candidates += timings.get("candidate_gen_ms", 0.0)
            overhead_cost += timings.get("cost_scoring_ms", 0.0)
            overhead_controller += timings.get("controller_ms", 0.0)
            overhead_migration += timings.get("migration_ms", 0.0)

            if c.control_decision and c.control_decision.action in (ControlAction.SWITCH, ControlAction.SAFE_FALLBACK):
                total_switches += 1
                if c.control_decision.is_proactive:
                    proactive_switches += 1
                else:
                    reactive_switches += 1
                if c.control_decision.action == ControlAction.SAFE_FALLBACK:
                    safety_overrides += 1

            if c.migration_result:
                mig_time_total += c.migration_result.timings.total_duration_ms
                mig_bytes_total += c.migration_result.metrics.total_bytes_transferred
                if c.migration_result.status == MigrationStatus.COMPLETED:
                    successful_migs += 1
                elif c.migration_result.status == MigrationStatus.ROLLED_BACK:
                    rolled_back_migs += 1
                else:
                    failed_migs += 1

        total_overhead = (
            overhead_telemetry
            + overhead_prediction
            + overhead_candidates
            + overhead_cost
            + overhead_controller
            + overhead_migration
        )
        overhead_fraction = total_overhead / max(1.0, e2e_ms)

        # SLO violations: tokens exceeding target ITL
        slo_violations = sum(1 for itl in itl_list if itl > self.config.slo_itl_ms)

        return OrchestrationMetrics(
            completed_tokens=completed,
            ttft_ms=round(ttft_ms, 3),
            mean_itl_ms=round(mean_itl, 3),
            p95_itl_ms=round(p95_itl, 3),
            total_inference_latency_ms=round(sum(itl_list) + ttft_ms, 3),
            end_to_end_runtime_ms=round(e2e_ms, 3),
            total_control_cycles=len(trace.cycles),
            total_switches=total_switches,
            proactive_switches=proactive_switches,
            reactive_switches=reactive_switches,
            safety_overrides=safety_overrides,
            successful_migrations=successful_migs,
            rolled_back_migrations=rolled_back_migs,
            failed_migrations=failed_migs,
            total_migration_time_ms=round(mig_time_total, 3),
            total_migration_bytes=mig_bytes_total,
            telemetry_overhead_ms=round(overhead_telemetry, 3),
            prediction_overhead_ms=round(overhead_prediction, 3),
            candidate_generation_overhead_ms=round(overhead_candidates, 3),
            cost_model_overhead_ms=round(overhead_cost, 3),
            controller_overhead_ms=round(overhead_controller, 3),
            migration_execution_overhead_ms=round(overhead_migration, 3),
            total_control_plane_overhead_ms=round(total_overhead, 3),
            control_overhead_fraction=round(overhead_fraction, 4),
            slo_violations=slo_violations,
        )
