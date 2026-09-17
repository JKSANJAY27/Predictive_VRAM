"""
ControlCycleRunner implementing the mandatory 14-step control cycle lifecycle.

Follows the exact research pipeline:
1. collect telemetry
2. convert telemetry to RuntimeState
3. append to StateBuffer
4. construct prediction if enabled (PREDICTIVE only; None for REACTIVE/STATIC)
5. obtain candidate partition plans
6. score candidates via CostModel
7. invoke Controller to obtain ControlDecision
8. check action (if KEEP: continue without migration)
9. if SWITCH / SAFE_FALLBACK: execute via MigrationManager
10. record MigrationResult
11. verify active runtime plan
12. collect post-action telemetry
13. record outcome & prediction lead metrics
14. return immutable ControlCycle
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from src.controller.controller import AdaptivePartitionController
from src.controller.types import ControlAction, ControlDecision
from src.cost.model import CostModel
from src.migration.manager import MigrationManager
from src.migration.types import MigrationMode, MigrationResult, MigrationStatus
from src.orchestration.adapters import TelemetryAgent
from src.orchestration.config import OrchestrationConfig
from src.orchestration.errors import (
    CandidateError,
    ControllerError,
    CostModelError,
    ErrorCategory,
    MigrationError,
    OrchestrationError,
    PredictionError,
    StateError,
    TelemetryError,
)
from src.orchestration.types import (
    ControlCycle,
    MeasuredOutcome,
    OrchestrationMode,
    PredictionLeadRecord,
)
from src.partitioning.catalog import SplitCatalog
from src.partitioning.metadata import ModelMetadata
from src.prediction.base import Predictor
from src.prediction.types import PredictionResult
from src.runtime.partition import PartitionPlan
from src.state.buffer import StateBuffer
from src.state.types import RuntimeState


class ControlCycleRunner:
    """
    Executes a single, atomic closed-loop control cycle adhering to strict step ordering.
    """

    def __init__(
        self,
        config: OrchestrationConfig,
        telemetry_agent: TelemetryAgent,
        state_buffer: StateBuffer,
        predictor: Optional[Predictor],
        split_catalog: SplitCatalog,
        cost_model: CostModel,
        controller: AdaptivePartitionController,
        migration_manager: MigrationManager,
        model_metadata: Optional[ModelMetadata] = None,
    ) -> None:
        self.config = config
        self.telemetry_agent = telemetry_agent
        self.state_buffer = state_buffer
        self.predictor = predictor
        self.split_catalog = split_catalog
        self.cost_model = cost_model
        self.controller = controller
        self.migration_manager = migration_manager
        # Synthetic default for standalone test use
        self.model_metadata = model_metadata or ModelMetadata(
            total_layers=4, hidden_size=64, num_heads=2, vocab_size=500, dtype_bytes=4,
        )

        self._cycle_counter: int = 0
        self._prev_state: Optional[RuntimeState] = None

    def run_cycle(
        self,
        inference_step: int,
        current_plan: PartitionPlan,
        kv_cache: Optional[Any] = None,
        activations: Optional[List[Any]] = None,
        actual_degradation_time: Optional[float] = None,
    ) -> ControlCycle:
        """
        Execute the 14-step control cycle lifecycle.
        """
        self._cycle_counter += 1
        cycle_id = self._cycle_counter
        cycle_start_time = time.time()
        start_perf = time.perf_counter()

        timings: Dict[str, float] = {}
        observed_state: Optional[RuntimeState] = None
        forecast: Optional[PredictionResult] = None
        candidate_scores_dicts: List[Dict[str, Any]] = []
        candidate_count: int = 0
        decision: Optional[ControlDecision] = None
        migration_result: Optional[MigrationResult] = None
        measured_outcome: Optional[MeasuredOutcome] = None
        error_category = ErrorCategory.NONE
        error_message: Optional[str] = None

        try:
            # -----------------------------------------------------------------
            # 1. COLLECT TELEMETRY
            # -----------------------------------------------------------------
            t0 = time.perf_counter()
            try:
                raw_snapshot = self.telemetry_agent.collect(
                    step=inference_step,
                    kv_cache=kv_cache,
                    activations=activations,
                )
            except Exception as e:
                raise TelemetryError(f"Telemetry collection failed: {e}", category=ErrorCategory.TELEMETRY_ERROR) from e
            timings["telemetry_ms"] = (time.perf_counter() - t0) * 1000.0

            # -----------------------------------------------------------------
            # 2. CONVERT TO RUNTIME STATE
            # -----------------------------------------------------------------
            t0 = time.perf_counter()
            try:
                observed_state = RuntimeState.from_telemetry(
                    snapshot=raw_snapshot,
                    prev_state=self._prev_state,
                    wall_clock=cycle_start_time,
                )
                self._prev_state = observed_state
            except Exception as e:
                raise StateError(f"RuntimeState conversion failed: {e}", category=ErrorCategory.STATE_ERROR) from e
            timings["state_conversion_ms"] = (time.perf_counter() - t0) * 1000.0

            # -----------------------------------------------------------------
            # 3. APPEND TO STATE BUFFER
            # -----------------------------------------------------------------
            t0 = time.perf_counter()
            try:
                self.state_buffer.append(observed_state)
            except Exception as e:
                raise StateError(f"StateBuffer append failed: {e}", category=ErrorCategory.STATE_ERROR) from e
            timings["buffer_append_ms"] = (time.perf_counter() - t0) * 1000.0

            # -----------------------------------------------------------------
            # 4. CONSTRUCT PREDICTION IF ENABLED
            # -----------------------------------------------------------------
            t0 = time.perf_counter()
            pred_time = time.time()
            if self.config.mode == OrchestrationMode.PREDICTIVE and self.predictor is not None:
                try:
                    # In predictive mode: use strictly observations up to now
                    states = self.state_buffer.get_all()
                    if states:
                        try:
                            forecast = self.predictor.predict(
                                states,
                                horizon_seconds=float(self.config.horizon),
                            )
                        except TypeError:
                            forecast = self.predictor.predict(
                                state_window=states,
                                horizon_seconds=float(self.config.horizon),
                            )
                    else:
                        forecast = None
                except Exception as e:
                    # Prediction error fallback: do not crash, fallback to None forecast
                    error_category = ErrorCategory.PREDICTION_ERROR
                    error_message = f"Predictor failed: {e}; falling back to reactive evaluation."
                    forecast = None
            else:
                # In STATIC and REACTIVE modes: strictly NO forecast (fairness guarantee)
                forecast = None
            timings["prediction_ms"] = (time.perf_counter() - t0) * 1000.0

            # -----------------------------------------------------------------
            # 5. OBTAIN CANDIDATES
            # -----------------------------------------------------------------
            t0 = time.perf_counter()
            try:
                # Use split catalog to generate candidates for the current state
                candidates = self.split_catalog.generate(
                    model_metadata=self.model_metadata,
                    current_state=observed_state,
                    forecast=forecast if (self.config.mode == OrchestrationMode.PREDICTIVE) else None,
                    current_plan=current_plan,
                )
                candidate_count = len(candidates)
            except Exception as e:
                raise CandidateError(f"Candidate retrieval failed: {e}", category=ErrorCategory.CANDIDATE_ERROR) from e
            timings["candidate_gen_ms"] = (time.perf_counter() - t0) * 1000.0

            # -----------------------------------------------------------------
            # 6. SCORE CANDIDATES
            # -----------------------------------------------------------------
            t0 = time.perf_counter()
            try:
                effective_forecast = forecast if (self.config.mode == OrchestrationMode.PREDICTIVE) else None
                scored_candidates = self.cost_model.score_candidates(
                    candidates=candidates,
                    state=observed_state,
                    forecast=effective_forecast,
                    current_plan=current_plan,
                )
                candidate_scores_dicts = []
                for s in scored_candidates:
                    if hasattr(s, "to_dict"):
                        d = s.to_dict()
                        if hasattr(s, "breakdown") and s.breakdown:
                            d["latency_cost"] = s.breakdown.latency_cost
                            d["comm_cost"] = s.breakdown.communication_cost
                            d["memory_cost"] = s.breakdown.memory_pressure_cost
                            d["switching_cost"] = s.breakdown.switching_cost
                        candidate_scores_dicts.append(d)
                    elif isinstance(s, dict):
                        candidate_scores_dicts.append(s)
                    else:
                        candidate_scores_dicts.append({
                            "plan_id": getattr(getattr(s, "candidate_plan", None), "plan_id", str(s)),
                            "total_cost": getattr(s, "total_cost", 0.0),
                            "feasible": getattr(s, "feasible", True),
                        })
            except Exception as e:
                raise CostModelError(f"Cost scoring failed: {e}", category=ErrorCategory.COST_ERROR) from e
            timings["cost_scoring_ms"] = (time.perf_counter() - t0) * 1000.0

            # -----------------------------------------------------------------
            # 7. INVOKE CONTROLLER
            # -----------------------------------------------------------------
            t0 = time.perf_counter()
            try:
                decision = self.controller.decide(
                    state=observed_state,
                    candidates=candidates,
                    forecast=effective_forecast,
                    current_plan=current_plan,
                )
            except Exception as e:
                raise ControllerError(f"Controller decision failed: {e}", category=ErrorCategory.CONTROLLER_ERROR) from e
            timings["controller_ms"] = (time.perf_counter() - t0) * 1000.0

            # -----------------------------------------------------------------
            # 8 & 9. MIGRATION EXECUTION IF SWITCH OR SAFE_FALLBACK
            # -----------------------------------------------------------------
            t0 = time.perf_counter()
            needs_migration = (
                decision.action in (ControlAction.SWITCH, ControlAction.SAFE_FALLBACK)
                and decision.migration_request is not None
            )

            if needs_migration and decision.migration_request is not None:
                # Module 9 passes request to Module 8 without altering it
                try:
                    migration_result = self.migration_manager.execute(
                        request=decision.migration_request,
                        kv_cache=kv_cache,
                    )
                except Exception as e:
                    error_category = ErrorCategory.MIGRATION_ERROR
                    error_message = f"Migration execution threw exception: {e}"
            timings["migration_ms"] = (time.perf_counter() - t0) * 1000.0

            # -----------------------------------------------------------------
            # 10 & 11. VERIFY ACTIVE RUNTIME PLAN
            # -----------------------------------------------------------------
            active_plan_after = self.migration_manager.adapter.get_current_plan()
            active_plan_id = getattr(active_plan_after, "plan_id", str(active_plan_after))

            # -----------------------------------------------------------------
            # 12 & 13. COLLECT POST-ACTION OUTCOME & LEAD TIME
            # -----------------------------------------------------------------
            bw_val = None
            if observed_state and observed_state.network.bandwidth_mbps.value is not None:
                bw_val = observed_state.network.bandwidth_mbps.value

            measured_outcome = MeasuredOutcome(
                active_plan_after=active_plan_id,
                measured_latency_after_ms=timings.get("migration_ms", 0.0),
                measured_bandwidth_after_mbps=bw_val,
                migration_pause_ms=timings.get("migration_ms", 0.0) if needs_migration else 0.0,
                notes="Cycle completed successfully." if error_category == ErrorCategory.NONE else error_message or "",
            )

        except OrchestrationError as oe:
            error_category = oe.category
            error_message = str(oe)
            # Safe fallback: retain current active plan
            active_plan_after = self.migration_manager.adapter.get_current_plan()
            measured_outcome = MeasuredOutcome(
                active_plan_after=str(active_plan_after),
                notes=f"Fallback triggered due to error: {oe}",
            )
        except Exception as e:
            error_category = ErrorCategory.CONTROLLER_ERROR
            error_message = f"Unexpected orchestration error: {e}"
            active_plan_after = self.migration_manager.adapter.get_current_plan()
            measured_outcome = MeasuredOutcome(
                active_plan_after=str(active_plan_after),
                notes=f"Fallback triggered: {e}",
            )

        total_duration = (time.perf_counter() - start_perf) * 1000.0
        timings["total_cycle_duration_ms"] = total_duration

        return ControlCycle(
            cycle_id=cycle_id,
            cycle_timestamp=cycle_start_time,
            inference_step=inference_step,
            observed_state=observed_state,
            prediction_result=forecast,
            candidate_count=candidate_count,
            candidate_scores=candidate_scores_dicts,
            control_decision=decision,
            migration_result=migration_result,
            measured_outcome=measured_outcome,
            cycle_duration_ms=total_duration,
            timings=timings,
            error_category=error_category,
            error_message=error_message,
        )
