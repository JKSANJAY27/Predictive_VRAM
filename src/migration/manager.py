"""
MigrationManager: Central transactional migration coordinator for Module 8.

Executes atomic, transactional transitions between model partition plans,
coordinating validation, planning, transfer, verification, commit, and rollback.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from src.controller.types import MigrationRequest
from src.migration.adapter import RuntimeAdapter
from src.migration.planner import MigrationPlanner
from src.migration.rollback import RollbackManager
from src.migration.state import KVCacheTransferManager, LayerTransferManager
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
    MigrationResult,
    MigrationStatus,
    VerificationMode,
)
from src.migration.validator import MigrationValidator
from src.migration.verifier import MigrationVerifier
from src.telemetry.types import DataSource


class MigrationManager:
    """
    Transactional coordinator for physical and emulated partition migrations.
    """

    def __init__(
        self,
        runtime_adapter: RuntimeAdapter,
        config: Optional[MigrationConfig] = None,
        transfer_provider: Optional[MigrationTransferProvider] = None,
    ) -> None:
        self.adapter = runtime_adapter
        self.config = config or MigrationConfig()

        if transfer_provider is not None:
            self.transfer_provider = transfer_provider
        elif self.config.emulated_bandwidth_delay:
            self.transfer_provider = EmulatedNetworkTransfer(
                bandwidth_mbps=self.config.emulated_bandwidth_mbps,
                latency_ms=self.config.emulated_latency_ms,
                fast_mode=self.config.fast_mode,
            )
        else:
            self.transfer_provider = LocalTensorTransfer()

        # Subordinate managers
        tier_devices = self.adapter.get_tier_devices()
        self.layer_manager = LayerTransferManager(
            model=self.adapter.model,
            tier_devices=tier_devices,
            transfer_provider=self.transfer_provider,
        )
        self.kv_manager = KVCacheTransferManager(
            tier_devices=tier_devices,
            transfer_provider=self.transfer_provider,
        )

        # Concurrency guard
        self._in_progress = False

        # Test-only failure injection flags
        self.fail_on_stage: Optional[str] = None
        self.fail_on_layer: Optional[int] = None
        self.fail_kv: bool = False
        self.fail_verification: bool = False

    def validate(self, request: MigrationRequest) -> Any:
        """Validate request against current active runtime plan."""
        return MigrationValidator.validate(request, self.adapter.get_current_plan())

    def simulate(
        self,
        request: MigrationRequest,
        kv_cache: Optional[Any] = None,
    ) -> MigrationResult:
        """
        Simulate a migration without modifying runtime placement or model weights.
        """
        sim_config = MigrationConfig(
            execution_mode=MigrationMode.SIMULATE,
            synchronization_point=self.config.synchronization_point,
            verification_mode=self.config.verification_mode,
            rollback_enabled=self.config.rollback_enabled,
            single_flight=self.config.single_flight,
            emulated_bandwidth_delay=self.config.emulated_bandwidth_delay,
            fast_mode=self.config.fast_mode,
        )
        sim_manager = MigrationManager(
            runtime_adapter=self.adapter,
            config=sim_config,
            transfer_provider=self.transfer_provider,
        )
        return sim_manager.execute(request, kv_cache=kv_cache)

    def execute(
        self,
        request: MigrationRequest,
        kv_cache: Optional[Any] = None,
    ) -> MigrationResult:
        """
        Execute a complete transactional migration.
        """
        start_wall = time.time()
        start_perf = time.perf_counter()
        req_id = getattr(request, "request_id", None) or str(uuid.uuid4())[:8]

        events: List[str] = []
        timings = MigrationPhaseTimings()
        metrics = MigrationMetrics()
        comparison = MigrationComparison()

        # 1. Concurrency Check (Single Flight)
        if self.config.single_flight and self._in_progress:
            return MigrationResult(
                request_id=req_id,
                source_plan=request.source_plan,
                target_plan=request.target_plan,
                source_plan_id=request.source_plan_id,
                target_plan_id=request.target_plan_id,
                status=MigrationStatus.FAILED,
                success=False,
                start_timestamp=start_wall,
                end_timestamp=time.time(),
                timings=timings,
                metrics=metrics,
                comparison=comparison,
                events=["rejected_busy"],
                error_message="MIGRATION_BUSY: concurrent migration currently in progress.",
            )

        if not self.adapter.acquire_migration_lock(timeout=2.0):
            return MigrationResult(
                request_id=req_id,
                source_plan=request.source_plan,
                target_plan=request.target_plan,
                source_plan_id=request.source_plan_id,
                target_plan_id=request.target_plan_id,
                status=MigrationStatus.FAILED,
                success=False,
                start_timestamp=start_wall,
                end_timestamp=time.time(),
                timings=timings,
                metrics=metrics,
                comparison=comparison,
                events=["rejected_lock_timeout"],
                error_message="MIGRATION_LOCK_TIMEOUT: could not acquire runtime quiescence lock.",
            )

        self._in_progress = True
        events.append(MigrationEvent.MIGRATION_STARTED.value)

        try:
            # -----------------------------------------------------------------
            # 2. VALIDATION
            # -----------------------------------------------------------------
            t0 = time.perf_counter()
            current_plan = self.adapter.get_current_plan()
            val_res = MigrationValidator.validate(request, current_plan)
            t_val = (time.perf_counter() - t0) * 1000.0

            if val_res.is_idempotent:
                # Target already active: safe no-op
                events.append("idempotent_no_action")
                self.adapter.release_migration_lock()
                self._in_progress = False
                return MigrationResult(
                    request_id=req_id,
                    source_plan=request.source_plan,
                    target_plan=request.target_plan,
                    source_plan_id=request.source_plan_id,
                    target_plan_id=request.target_plan_id,
                    status=MigrationStatus.ALREADY_AT_TARGET,
                    success=True,
                    start_timestamp=start_wall,
                    end_timestamp=time.time(),
                    timings=MigrationPhaseTimings(validation_ms=t_val, total_duration_ms=t_val),
                    metrics=metrics,
                    comparison=comparison,
                    events=events,
                    error_message=val_res.reason,
                    provenance=DataSource.MEASURED,
                )

            if not val_res.is_valid:
                events.append(MigrationEvent.MIGRATION_FAILED.value)
                self.adapter.release_migration_lock()
                self._in_progress = False
                return MigrationResult(
                    request_id=req_id,
                    source_plan=request.source_plan,
                    target_plan=request.target_plan,
                    source_plan_id=request.source_plan_id,
                    target_plan_id=request.target_plan_id,
                    status=MigrationStatus.FAILED,
                    success=False,
                    start_timestamp=start_wall,
                    end_timestamp=time.time(),
                    timings=MigrationPhaseTimings(validation_ms=t_val, total_duration_ms=t_val),
                    metrics=metrics,
                    comparison=comparison,
                    events=events,
                    error_message=val_res.reason,
                    provenance=DataSource.MEASURED,
                )

            events.append(MigrationEvent.VALIDATION_COMPLETED.value)

            # -----------------------------------------------------------------
            # 3. PREPARATION & PLANNING
            # -----------------------------------------------------------------
            t0 = time.perf_counter()
            plan = MigrationPlanner.plan(request.source_plan, request.target_plan)
            t_prep = (time.perf_counter() - t0) * 1000.0
            events.append(MigrationEvent.PREPARATION_COMPLETED.value)

            if self.fail_on_stage == "preparation":
                raise RuntimeError("Injected preparation failure triggered.")

            # SIMULATION MODE BRANCH:
            if self.config.execution_mode == MigrationMode.SIMULATE:
                # Estimate sizes without moving anything
                est_model_bytes = sum(
                    sum(p.numel() * p.element_size() for p in self.adapter.model.blocks[l].parameters())
                    for l in plan.affected_layers
                )
                active_kv = kv_cache or self.adapter.get_kv_cache()
                est_kv_bytes = 0
                if active_kv is not None and hasattr(active_kv, "layers"):
                    for l in plan.affected_layers:
                        if l < len(active_kv.layers):
                            k = getattr(active_kv.layers[l], "keys", None)
                            v = getattr(active_kv.layers[l], "values", None)
                            if k is not None and v is not None:
                                est_kv_bytes += (k.numel() * k.element_size()) + (v.numel() * v.element_size())

                sim_metrics = MigrationMetrics(
                    model_bytes_transferred=est_model_bytes,
                    kv_cache_bytes_transferred=est_kv_bytes,
                    total_bytes_transferred=est_model_bytes + est_kv_bytes,
                    transfer_count=len(plan.affected_layers),
                    affected_layers=plan.affected_layers,
                    affected_tiers=[t.value for t in plan.affected_tiers],
                )
                t_total = (time.perf_counter() - start_perf) * 1000.0
                events.append("simulation_completed")
                self.adapter.release_migration_lock()
                self._in_progress = False
                return MigrationResult(
                    request_id=req_id,
                    source_plan=request.source_plan,
                    target_plan=request.target_plan,
                    source_plan_id=request.source_plan_id,
                    target_plan_id=request.target_plan_id,
                    status=MigrationStatus.COMPLETED,
                    success=True,
                    start_timestamp=start_wall,
                    end_timestamp=time.time(),
                    timings=MigrationPhaseTimings(
                        validation_ms=t_val,
                        preparation_ms=t_prep,
                        total_duration_ms=t_total,
                    ),
                    metrics=sim_metrics,
                    comparison=MigrationComparison(
                        predicted_switching_cost=request.switching_cost,
                        actual_migration_duration_ms=t_total,
                        predicted_kv_bytes=request.estimated_kv_transfer_bytes,
                        actual_kv_bytes=est_kv_bytes,
                        notes="Simulated execution: no physical tensors modified.",
                    ),
                    events=events,
                    provenance=DataSource.ESTIMATED,
                )

            # -----------------------------------------------------------------
            # 4. MODEL LAYER TRANSFER
            # -----------------------------------------------------------------
            events.append(MigrationEvent.MODEL_TRANSFER_STARTED.value)
            t0 = time.perf_counter()
            layer_report = self.layer_manager.transfer_layers(
                plan=plan,
                fail_on_layer=self.fail_on_layer,
            )
            t_model = (time.perf_counter() - t0) * 1000.0
            events.append(MigrationEvent.MODEL_TRANSFER_COMPLETED.value)

            if self.fail_on_stage == "model_transfer":
                raise RuntimeError("Injected model transfer failure triggered.")

            # -----------------------------------------------------------------
            # 5. KV-CACHE TRANSFER
            # -----------------------------------------------------------------
            events.append(MigrationEvent.KV_TRANSFER_STARTED.value)
            t0 = time.perf_counter()
            active_kv = kv_cache or self.adapter.get_kv_cache()
            kv_report = self.kv_manager.transfer_cache(
                kv_cache=active_kv,
                plan=plan,
                fail_kv=self.fail_kv,
            )
            t_kv = (time.perf_counter() - t0) * 1000.0
            events.append(MigrationEvent.KV_TRANSFER_COMPLETED.value)

            if self.fail_on_stage == "kv_transfer":
                raise RuntimeError("Injected KV-cache transfer failure triggered.")

            # -----------------------------------------------------------------
            # 6. VERIFICATION
            # -----------------------------------------------------------------
            events.append(MigrationEvent.VERIFICATION_STARTED.value)
            t0 = time.perf_counter()
            ver_res = MigrationVerifier.verify(
                model=self.adapter.model,
                plan=plan,
                tier_devices=self.adapter.get_tier_devices(),
                kv_cache=active_kv,
                mode=self.config.verification_mode,
                fail_verification=self.fail_verification,
            )
            t_ver = (time.perf_counter() - t0) * 1000.0

            if not ver_res.passed:
                events.append("verification_failed")
                raise RuntimeError(f"Post-migration verification failed: {'; '.join(ver_res.issues)}")

            events.append(MigrationEvent.VERIFICATION_COMPLETED.value)

            # -----------------------------------------------------------------
            # 7. ATOMIC COMMIT
            # -----------------------------------------------------------------
            events.append(MigrationEvent.COMMIT_STARTED.value)
            t0 = time.perf_counter()
            self.adapter.set_active_plan(request.target_plan)
            t_commit = (time.perf_counter() - t0) * 1000.0
            events.append(MigrationEvent.COMMIT_COMPLETED.value)
            events.append(MigrationEvent.MIGRATION_COMPLETED.value)

            # Assemble metrics
            total_bytes = layer_report.total_bytes + kv_report.total_bytes
            metrics = MigrationMetrics(
                model_bytes_transferred=layer_report.total_bytes,
                kv_cache_bytes_transferred=kv_report.total_bytes,
                total_bytes_transferred=total_bytes,
                transfer_count=len(layer_report.layers_transferred) + len(kv_report.layers_transferred),
                affected_layers=plan.affected_layers,
                affected_tiers=[t.value for t in plan.affected_tiers],
                per_layer_bytes=layer_report.per_layer_bytes,
                per_tier_bytes={t.value: layer_report.total_bytes for t in plan.affected_tiers},
            )

            t_total = (time.perf_counter() - start_perf) * 1000.0
            timings = MigrationPhaseTimings(
                validation_ms=t_val,
                preparation_ms=t_prep,
                model_transfer_ms=t_model,
                kv_transfer_ms=t_kv,
                verification_ms=t_ver,
                commit_ms=t_commit,
                rollback_ms=0.0,
                total_duration_ms=t_total,
            )

            comparison = MigrationComparison(
                predicted_switching_cost=request.switching_cost,
                actual_migration_duration_ms=t_total,
                predicted_kv_bytes=request.estimated_kv_transfer_bytes,
                actual_kv_bytes=kv_report.total_bytes,
                model_bytes_discrepancy=0,
                notes="Transactional migration committed successfully.",
            )

            return MigrationResult(
                request_id=req_id,
                source_plan=request.source_plan,
                target_plan=request.target_plan,
                source_plan_id=request.source_plan_id,
                target_plan_id=request.target_plan_id,
                status=MigrationStatus.COMPLETED,
                success=True,
                start_timestamp=start_wall,
                end_timestamp=time.time(),
                timings=timings,
                metrics=metrics,
                comparison=comparison,
                events=events,
                error_message=None,
                rollback_performed=False,
                provenance=layer_report.provenance,
            )

        except Exception as ex:
            events.append(MigrationEvent.MIGRATION_FAILED.value)
            # -----------------------------------------------------------------
            # ROLLBACK PATH
            # -----------------------------------------------------------------
            events.append(MigrationEvent.ROLLBACK_STARTED.value)
            t0 = time.perf_counter()
            active_kv = kv_cache or self.adapter.get_kv_cache()
            rb_report = RollbackManager.execute_rollback(
                layer_manager=self.layer_manager,
                kv_manager=self.kv_manager,
                runtime_adapter=self.adapter,
                source_plan=request.source_plan,
                kv_cache=active_kv,
            )
            t_rb = (time.perf_counter() - t0) * 1000.0
            events.append(MigrationEvent.ROLLBACK_COMPLETED.value)

            final_status = MigrationStatus.ROLLED_BACK if rb_report.success else MigrationStatus.FAILED_PARTIAL
            t_total = (time.perf_counter() - start_perf) * 1000.0

            timings = MigrationPhaseTimings(
                rollback_ms=t_rb,
                total_duration_ms=t_total,
            )

            return MigrationResult(
                request_id=req_id,
                source_plan=request.source_plan,
                target_plan=request.target_plan,
                source_plan_id=request.source_plan_id,
                target_plan_id=request.target_plan_id,
                status=final_status,
                success=False,
                start_timestamp=start_wall,
                end_timestamp=time.time(),
                timings=timings,
                metrics=metrics,
                comparison=comparison,
                events=events,
                error_message=f"Migration failed and rolled back: {ex}",
                rollback_performed=True,
                provenance=DataSource.MEASURED,
            )

        finally:
            self._in_progress = False
            self.adapter.release_migration_lock()
