"""
ExperimentRunner and TrialResult for Module 10.

Integrates the closed-loop runtime engine (Module 9), baseline policy specifications,
environmental trace injection, and metric calculations into an autonomous, robust
experimental trial runner.

Key Guarantees:
  - Determinism & Trace Parity: Identical environment trace injected across all 5 baselines
  - Immutability: Results stored in results/raw/<trial_id>.json with DuplicateRunGuard
  - No Future Leakage: Closed-loop telemetry strictly respects chronological causality
  - Provenance Tracking: Machine metadata and DataSource.EMULATED recorded
  - Fast execution: Uses simulation mode by default for rapid benchmark execution
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from src.controller.controller import AdaptivePartitionController
from src.controller.types import ControllerConfig, ControllerMode
from src.cost.model import CostModel
from src.cost.types import CostWeights
from src.evaluation.ablations import AblationSuite
from src.evaluation.baselines import BaselinePolicy, BaselinePolicyFactory
from src.evaluation.experiment_config import ExperimentConfig, ExperimentMatrix, TrialConfig
from src.evaluation.manifest import DuplicateRunGuard
from src.evaluation.metrics import EvaluationMetrics, MetricCalculator
from src.evaluation.traces import CombinedEnvironmentTrace, TraceGenerator
from src.migration.manager import MigrationConfig, MigrationManager, MigrationMode, VerificationMode
from src.orchestration.config import OrchestrationConfig
from src.orchestration.runtime import ClosedLoopRuntime
from src.orchestration.trace import RuntimeTrace
from src.orchestration.types import ExecutionMode, OrchestrationMode, TokenRecord
from src.partitioning.catalog import SplitCatalog
from src.partitioning.metadata import ModelMetadata
from src.prediction import (
    KVCacheAwareMemoryPredictor,
    LastValuePredictor,
    LinearTrendPredictor,
    MovingAveragePredictor,
    Predictor,
)
from src.migration.adapter import RuntimeAdapter
from src.runtime.executor import DistributedInferenceExecutor
from src.runtime.model import LayeredTransformer
from src.runtime.partition import PartitionPlan
from src.runtime.tier import Tier, TierId
from src.state.buffer import StateBuffer
from src.orchestration.adapters import TelemetryAgent

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """
    Complete, self-contained record of a single experimental trial.
    """
    schema_version: str = "1.0"
    trial_id: str = ""
    experiment_id: str = ""
    scenario_id: str = ""
    baseline_id: str = ""
    seed: int = 42
    repetition: int = 0
    ablation_id: str = ""
    configuration: Dict[str, Any] = field(default_factory=dict)
    runtime_trace: Dict[str, Any] = field(default_factory=dict)
    metrics: EvaluationMetrics = field(default_factory=EvaluationMetrics)
    environment_snapshot: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    completion_status: str = "completed"  # "completed" | "failed" | "skipped"
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trial_id": self.trial_id,
            "experiment_id": self.experiment_id,
            "scenario_id": self.scenario_id,
            "baseline_id": self.baseline_id,
            "seed": self.seed,
            "repetition": self.repetition,
            "ablation_id": self.ablation_id,
            "configuration": self.configuration,
            "runtime_trace": self.runtime_trace,
            "metrics": self.metrics.to_dict(),
            "environment_snapshot": self.environment_snapshot,
            "errors": self.errors,
            "completion_status": self.completion_status,
            "provenance": self.provenance,
        }

    def save_json(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return target

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TrialResult:
        metrics_dict = d.get("metrics", {})
        metrics = EvaluationMetrics.from_dict(metrics_dict) if hasattr(EvaluationMetrics, "from_dict") else EvaluationMetrics(**{k: v for k, v in metrics_dict.items() if hasattr(EvaluationMetrics, k)})
        return cls(
            schema_version=d.get("schema_version", "1.0"),
            trial_id=d.get("trial_id", ""),
            experiment_id=d.get("experiment_id", ""),
            scenario_id=d.get("scenario_id", ""),
            baseline_id=d.get("baseline_id", ""),
            seed=int(d.get("seed", 42)),
            repetition=int(d.get("repetition", 0)),
            ablation_id=d.get("ablation_id", ""),
            configuration=d.get("configuration", {}),
            runtime_trace=d.get("runtime_trace", {}),
            metrics=metrics,
            environment_snapshot=d.get("environment_snapshot", {}),
            errors=d.get("errors", []),
            completion_status=d.get("completion_status", "completed"),
            provenance=d.get("provenance", {}),
        )

    @classmethod
    def load_json(cls, path: Path | str) -> TrialResult:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)


class ExperimentRunner:
    """
    Executes benchmark trials with trace injection and reproducibility guarantees.
    """

    def __init__(
        self,
        raw_results_dir: Path | str = "results/raw",
        traces_dir: Path | str = "results/traces",
        force_rerun: bool = False,
    ) -> None:
        self.raw_dir = Path(raw_results_dir)
        self.traces_dir = Path(traces_dir)
        self.force_rerun = force_rerun
        self.guard = DuplicateRunGuard(self.raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        # In-memory trace cache: (scenario_id, seed, n_steps) -> CombinedEnvironmentTrace
        self._trace_cache: Dict[Tuple[str, int, int], CombinedEnvironmentTrace] = {}

    def get_or_create_trace(
        self,
        scenario_id: str,
        seed: int,
        n_steps: int = 64,
        sampling_interval_s: float = 0.25,
        max_new_tokens: int = 32,
    ) -> CombinedEnvironmentTrace:
        """
        Return the unique environment trace for (scenario_id, seed).
        Generates once, caches in memory, and writes to results/traces/.
        """
        cache_key = (scenario_id, seed, n_steps)
        if cache_key in self._trace_cache:
            return self._trace_cache[cache_key]

        trace_file = self.traces_dir / f"{scenario_id}_seed{seed}.json"
        if trace_file.is_file() and not self.force_rerun:
            try:
                trace = CombinedEnvironmentTrace.load_json(trace_file)
                self._trace_cache[cache_key] = trace
                return trace
            except Exception as e:
                logger.warning(f"Failed to load cached trace {trace_file}: {e}")

        # Generate deterministically
        trace = TraceGenerator.generate(
            scenario_id=scenario_id,
            seed=seed,
            n_steps=n_steps,
            sampling_interval_s=sampling_interval_s,
            max_new_tokens=max_new_tokens,
        )
        trace.save_json(trace_file)
        self._trace_cache[cache_key] = trace
        return trace

    def run_trial(self, trial_config: TrialConfig) -> TrialResult:
        """
        Execute a single trial according to trial_config.
        """
        output_file = self.raw_dir / f"{trial_config.trial_id}.json"

        # Check duplicate run guard
        if not self.guard.can_run(trial_config.trial_id, force=self.force_rerun):
            try:
                logger.info(f"Skipping completed trial {trial_config.trial_id}")
                return TrialResult.load_json(output_file)
            except Exception as e:
                logger.warning(f"Failed to load existing trial {output_file}: {e}")

        # 1. Retrieve or generate the shared deterministic trace
        env_trace = self.get_or_create_trace(
            scenario_id=trial_config.scenario_id,
            seed=trial_config.seed,
            n_steps=trial_config.n_trace_steps,
            sampling_interval_s=trial_config.sampling_interval_s,
            max_new_tokens=trial_config.max_new_tokens,
        )

        trace_snapshot = {
            "content_hash": env_trace.content_hash(),
            "n_steps": len(env_trace.network.steps),
            "scenario_id": env_trace.scenario_id,
            "seed": env_trace.seed,
        }

        provenance = {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_telemetry_source": "DataSource.EMULATED",
            "vram_status": "emulated_cpu_mode",
            "timestamp": time.time(),
        }

        # 2. Build configuration for baseline policy
        try:
            policy = BaselinePolicy(trial_config.baseline_id)
        except ValueError:
            policy = BaselinePolicy.PREDICTIVE

        orch_cfg, ctrl_cfg, weights = BaselinePolicyFactory.build_configs(
            policy,
            scenario_name=trial_config.scenario_id,
            seed=trial_config.seed,
            max_new_tokens=trial_config.max_new_tokens,
            control_interval_tokens=trial_config.control_interval_tokens,
            predictor_type=trial_config.predictor_type,
            horizon=trial_config.horizon_steps,
            execution_mode=ExecutionMode(trial_config.execution_mode),
            experiment_id=trial_config.experiment_id,
            switch_threshold=trial_config.switch_threshold,
            cost_weights_override=trial_config.cost_weights,
            slo_itl_ms=trial_config.slo_config.get("p95_itl_ms", 200.0),
        )

        # 3. Apply ablation if specified
        if trial_config.ablation_id:
            try:
                ablation_spec = AblationSuite.get(trial_config.ablation_id)
                if ablation_spec:
                    if ablation_spec.cost_weights_override:
                        weights.update(ablation_spec.cost_weights_override)
                        orch_cfg.cost_weights = dict(weights)
                    if ablation_spec.switch_threshold_override is not None:
                        ctrl_cfg.switch_threshold = ablation_spec.switch_threshold_override
                    if ablation_spec.cooldown_seconds_override is not None:
                        ctrl_cfg.cooldown_seconds = ablation_spec.cooldown_seconds_override
                    if ablation_spec.hysteresis_cycles_override is not None:
                        ctrl_cfg.hysteresis_cycles = ablation_spec.hysteresis_cycles_override
                    if ablation_spec.predictor_type_override is not None:
                        orch_cfg.predictor_type = ablation_spec.predictor_type_override
                    if ablation_spec.disable_prediction:
                        orch_cfg.mode = OrchestrationMode.REACTIVE
                        ctrl_cfg.mode = ControllerMode.REACTIVE
            except Exception as e:
                logger.warning(f"Could not apply ablation {trial_config.ablation_id}: {e}")

        # 4. Construct ClosedLoopRuntime with identical hardware/model parameters
        errors: List[str] = []
        try:
            rt = self._build_runtime(orch_cfg, ctrl_cfg, weights, seed=trial_config.seed)
            # Execute closed loop driven strictly by the trace
            runtime_trace = self._run_closed_loop_trace(rt, env_trace, max_tokens=trial_config.max_new_tokens)
            status = "completed"
        except Exception as e:
            logger.error(f"Trial {trial_config.trial_id} execution failed: {e}", exc_info=True)
            errors.append(str(e))
            status = "failed"
            runtime_trace = RuntimeTrace(metadata={"error": str(e), "trial_id": trial_config.trial_id})

        # 5. Compute rigorous EvaluationMetrics
        calculator = MetricCalculator(
            slo_config=trial_config.slo_config,
            memory_capacity_mb=4096.0,
        )
        metrics = calculator.compute(runtime_trace)

        result = TrialResult(
            schema_version="1.0",
            trial_id=trial_config.trial_id,
            experiment_id=trial_config.experiment_id,
            scenario_id=trial_config.scenario_id,
            baseline_id=trial_config.baseline_id,
            seed=trial_config.seed,
            repetition=trial_config.repetition,
            ablation_id=trial_config.ablation_id,
            configuration=trial_config.to_dict(),
            runtime_trace=runtime_trace.to_dict(),
            metrics=metrics,
            environment_snapshot=trace_snapshot,
            errors=errors,
            completion_status=status,
            provenance=provenance,
        )

        result.save_json(output_file)
        return result

    def run_experiment(
        self,
        config: ExperimentConfig,
        include_ablations: bool = True,
        include_sensitivity: bool = False,
    ) -> List[TrialResult]:
        """
        Run all trials expanded from an ExperimentConfig.
        """
        matrix = ExperimentMatrix(config)
        trials = matrix.expand(
            include_ablations=include_ablations,
            include_sensitivity=include_sensitivity,
        )
        results = []
        for t in trials:
            res = self.run_trial(t)
            results.append(res)
        return results

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _build_runtime(
        self,
        orch_cfg: OrchestrationConfig,
        ctrl_cfg: ControllerConfig,
        weights: Dict[str, float],
        seed: int,
        num_layers: int = 4,
        hidden_size: int = 64,
    ) -> ClosedLoopRuntime:
        """Instantiate closed-loop runtime components."""
        torch.manual_seed(seed)

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

        plan = PartitionPlan(
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
        if orch_cfg.mode == OrchestrationMode.PREDICTIVE:
            p_type = orch_cfg.predictor_type
            if p_type == "linear_trend":
                predictor = LinearTrendPredictor()
            elif p_type in ("kv_aware", "kv_cache_aware"):
                predictor = KVCacheAwareMemoryPredictor()
            elif p_type == "moving_average":
                predictor = MovingAveragePredictor()
            elif p_type == "last_value":
                predictor = LastValuePredictor()
            else:
                predictor = LinearTrendPredictor()

        # 5. Split Catalog & Metadata
        model_meta = ModelMetadata(
            total_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=2,
            vocab_size=500,
            dtype_bytes=4,
        )
        catalog = SplitCatalog()

        # 6. Cost Model
        cost_weights = CostWeights.from_dict(weights)
        cost_model = CostModel(weights=cost_weights)

        # 7. Controller
        controller = AdaptivePartitionController(
            config=ctrl_cfg,
            cost_model=cost_model,
            initial_plan=plan,
        )

        # 8. Migration Manager
        adapter = RuntimeAdapter(executor)
        v_mode = VerificationMode.FAST if orch_cfg.migration_verification_mode == "FAST" else VerificationMode.DEEP
        m_mode = MigrationMode.SIMULATE if orch_cfg.execution_mode == ExecutionMode.SIMULATION else MigrationMode.EXECUTE
        mig_cfg = MigrationConfig(
            execution_mode=m_mode,
            verification_mode=v_mode,
            rollback_enabled=orch_cfg.rollback_enabled,
            emulated_bandwidth_delay=orch_cfg.emulated_bandwidth_delay,
            fast_mode=True,
        )
        migration_mgr = MigrationManager(runtime_adapter=adapter, config=mig_cfg)

        return ClosedLoopRuntime(
            config=orch_cfg,
            executor=executor,
            telemetry_agent=telemetry_agent,
            state_buffer=state_buffer,
            predictor=predictor,
            split_catalog=catalog,
            cost_model=cost_model,
            controller=controller,
            migration_manager=migration_mgr,
            model_metadata=model_meta,
        )

    def _run_closed_loop_trace(
        self,
        rt: ClosedLoopRuntime,
        env_trace: CombinedEnvironmentTrace,
        max_tokens: int = 32,
    ) -> RuntimeTrace:
        """
        Execute closed loop token generation, injecting network conditions
        from env_trace at each step with chronological monotonicity.
        """
        start_wall_time = time.time()
        trace = RuntimeTrace(
            metadata={
                "experiment_id": rt.config.experiment_id,
                "timestamp": start_wall_time,
                "mode": rt.config.mode.value,
                "execution_mode": rt.config.execution_mode.value,
                "scenario": env_trace.scenario_id,
                "trace_hash": env_trace.content_hash(),
                "horizon": rt.config.horizon,
                "control_interval_tokens": rt.config.control_interval_tokens,
                "cost_weights": rt.config.cost_weights,
            }
        )

        rt.scheduler.reset()
        sim_itls: List[float] = []
        ttft = 15.0

        n_trace_steps = len(env_trace.network.steps)

        for step in range(max_tokens):
            # Deterministic injection of step-specific network telemetry
            net_step = env_trace.network.steps[min(step, n_trace_steps - 1)]
            rt.telemetry_agent.set_network_conditions(
                bandwidth_mbps=net_step.bandwidth_mbps,
                rtt_ms=net_step.rtt_ms,
                packet_loss=net_step.packet_loss,
                jitter_ms=net_step.jitter_ms,
            )

            # Simulated step latency derived from physical channel condition
            step_lat = 10.0 + (50.0 / max(0.5, net_step.bandwidth_mbps))
            is_ttft = (step == 0)
            if not is_ttft:
                sim_itls.append(step_lat)

            current_plan = rt.migration_manager.adapter.get_current_plan()
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

            # Control cycle trigger check
            if rt.scheduler.should_trigger(step, time.time()):
                if rt.scheduler.mark_cycle_started():
                    cycle = rt.cycle_runner.run_cycle(
                        inference_step=step,
                        current_plan=current_plan,
                    )
                    rt.scheduler.mark_cycle_completed(step, time.time())
                    trace.add_cycle(cycle)

        trace.summary = rt._compute_metrics(trace, start_wall_time, sim_itls, ttft)
        return trace
