"""
MetricCalculator and EvaluationMetrics for Module 10.

Computes the full metric suite from a RuntimeTrace (Module 9 output).
All functions are pure — no side effects, no I/O.

Metric categories:
  Inference      : TTFT, mean/median/p95/p99 ITL, E2E, tokens/sec
  Memory         : headroom, pressure, violations, OOM
  Network        : BW stats, latency, packet loss, jitter, comm bytes
  Control        : cycle count, per-component timing, overhead fraction
  Migration      : counts, durations, bytes moved
  Stability      : switch rate, dwell time, thrashing, rejection counts
  Prediction     : MAE, RMSE, median/max AE, lead time
  Safety         : overrides, fallbacks, violations, recovery time
  Proactive      : proactive vs reactive switch classification
  Lead-time      : prediction/decision/migration-start lead times
  SLO            : violation rate, tokens/requests outside SLO
  Migration impact: pre/post ITL, memory, network per migration
  Cost validation : predicted vs measured bytes/duration errors
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.orchestration.trace import RuntimeTrace
from src.orchestration.types import (
    ControlCycle,
    TokenRecord,
    MigrationWindowRecord,
    PredictionLeadRecord,
)
from src.controller.types import ControlAction, DecisionReason


# ---------------------------------------------------------------------------
# Helper percentile
# ---------------------------------------------------------------------------

def _percentile(data: Sequence[float], p: float) -> float:
    """Return the p-th percentile (0-100) using nearest-rank method."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    if p <= 0:
        return sorted_data[0]
    if p >= 100:
        return sorted_data[-1]
    idx = (p / 100) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])


def _safe_mean(data: Sequence[float]) -> float:
    return statistics.mean(data) if data else 0.0


def _safe_median(data: Sequence[float]) -> float:
    return statistics.median(data) if data else 0.0


# ---------------------------------------------------------------------------
# EvaluationMetrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluationMetrics:
    """Complete metric record for one trial."""

    # --- Inference ---
    ttft_ms: float = 0.0
    mean_itl_ms: float = 0.0
    median_itl_ms: float = 0.0
    p95_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0
    end_to_end_latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    completed_tokens: int = 0

    # --- Memory ---
    min_vram_headroom_mb: Optional[float] = None
    max_memory_pressure: Optional[float] = None
    memory_safety_violations: int = 0
    oom_count: int = 0

    # --- Network ---
    avg_bandwidth_mbps: Optional[float] = None
    min_bandwidth_mbps: Optional[float] = None
    avg_rtt_ms: Optional[float] = None
    p95_rtt_ms: Optional[float] = None
    avg_packet_loss: Optional[float] = None
    avg_jitter_ms: Optional[float] = None
    total_communication_bytes: int = 0

    # --- Control ---
    controller_cycles: int = 0
    total_controller_time_ms: float = 0.0
    total_prediction_time_ms: float = 0.0
    total_candidate_gen_time_ms: float = 0.0
    total_cost_eval_time_ms: float = 0.0
    total_control_plane_overhead_ms: float = 0.0
    control_overhead_fraction: float = 0.0

    # --- Migration ---
    migration_count: int = 0
    successful_migrations: int = 0
    failed_migrations: int = 0
    rollback_count: int = 0
    total_migration_time_ms: float = 0.0
    mean_migration_time_ms: float = 0.0
    p95_migration_time_ms: float = 0.0
    total_model_bytes_moved: int = 0
    total_kv_bytes_moved: int = 0
    total_migration_bytes: int = 0

    # --- Stability ---
    switch_count: int = 0
    total_switches: int = 0
    proactive_switches: int = 0
    reactive_switches: int = 0
    switches_per_hour: float = 0.0
    avg_dwell_time_tokens: float = 0.0
    min_dwell_time_tokens: float = 0.0
    thrashing_event_count: int = 0
    threshold_rejection_count: int = 0
    cooldown_rejection_count: int = 0
    hysteresis_rejection_count: int = 0

    # --- Prediction ---
    prediction_mae: Optional[float] = None
    prediction_rmse: Optional[float] = None
    prediction_median_ae: Optional[float] = None
    prediction_max_ae: Optional[float] = None
    avg_prediction_lead_time_s: Optional[float] = None

    # --- Safety ---
    safety_override_count: int = 0
    fallback_count: int = 0
    unsafe_plan_attempts: int = 0
    memory_safety_violation_count: int = 0

    # --- Proactive ---
    proactive_switch_count: int = 0
    reactive_switch_count: int = 0
    safety_override_switch_count: int = 0
    proactive_switch_rate: float = 0.0

    # --- Lead time ---
    mean_prediction_lead_time_s: Optional[float] = None
    mean_decision_lead_time_s: Optional[float] = None
    mean_migration_start_lead_time_s: Optional[float] = None

    # --- SLO ---
    slo_p95_itl_violated: bool = False
    slo_ttft_violated: bool = False
    slo_memory_violated: bool = False
    slo_violation_count: int = 0
    slo_violation_rate: float = 0.0

    # --- Cost model validation ---
    cost_model_byte_mae: Optional[float] = None
    cost_model_byte_rmse: Optional[float] = None
    cost_model_duration_mae: Optional[float] = None
    cost_model_duration_rmse: Optional[float] = None

    # --- Error / status ---
    component_errors: int = 0
    completion_status: str = "completed"
    provenance_notes: str = "emulated"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> EvaluationMetrics:
        valid = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in valid})


# ---------------------------------------------------------------------------
# MetricCalculator
# ---------------------------------------------------------------------------

class MetricCalculator:
    """
    Computes EvaluationMetrics from a RuntimeTrace.

    Usage:
        calc = MetricCalculator(slo_config={"p95_itl_ms": 200, "ttft_ms": 500})
        metrics = calc.compute(trace)
    """

    def __init__(
        self,
        slo_config: Optional[Dict[str, float]] = None,
        memory_capacity_mb: float = 4096.0,
        max_memory_pressure_threshold: float = 0.90,
        thrashing_window: int = 5,          # switches in N cycles = thrashing
        thrashing_threshold: int = 3,
    ) -> None:
        self.slo = slo_config or {}
        self.capacity_mb = memory_capacity_mb
        self.max_pressure = max_memory_pressure_threshold
        self.thrashing_window = thrashing_window
        self.thrashing_threshold = thrashing_threshold

    def compute(self, trace: RuntimeTrace) -> EvaluationMetrics:
        m = EvaluationMetrics()
        self._inference_metrics(trace, m)
        self._memory_metrics(trace, m)
        self._network_metrics(trace, m)
        self._control_metrics(trace, m)
        self._migration_metrics(trace, m)
        self._stability_metrics(trace, m)
        self._prediction_metrics(trace, m)
        self._safety_metrics(trace, m)
        self._proactive_metrics(trace, m)
        self._lead_time_metrics(trace, m)
        self._slo_metrics(trace, m)
        self._cost_validation_metrics(trace, m)
        m.component_errors = sum(
            1 for c in trace.cycles if c.error_category.value != "none"
        )
        return m

    # ---- Inference ----------------------------------------------------------

    def _inference_metrics(self, trace: RuntimeTrace, m: EvaluationMetrics) -> None:
        tokens = trace.token_records
        if not tokens:
            return

        m.completed_tokens = len(tokens)

        # TTFT: latency of first token
        if tokens:
            m.ttft_ms = tokens[0].step_latency_ms

        # ITL: latency of tokens 1..N (inter-token latency)
        itl_ms = [t.step_latency_ms for t in tokens[1:]] if len(tokens) > 1 else []
        if itl_ms:
            m.mean_itl_ms   = _safe_mean(itl_ms)
            m.median_itl_ms = _safe_median(itl_ms)
            m.p95_itl_ms    = _percentile(itl_ms, 95)
            m.p99_itl_ms    = _percentile(itl_ms, 99)

        # E2E latency
        if len(tokens) >= 2:
            m.end_to_end_latency_ms = tokens[-1].timestamp - tokens[0].timestamp
            if m.end_to_end_latency_ms > 0:
                m.tokens_per_second = 1000.0 * len(tokens) / m.end_to_end_latency_ms

    # ---- Memory -------------------------------------------------------------

    def _memory_metrics(self, trace: RuntimeTrace, m: EvaluationMetrics) -> None:
        pressures: List[float] = []
        headrooms: List[float] = []

        for c in trace.cycles:
            if c.observed_state is not None:
                mem = getattr(c.observed_state, "memory", None) or getattr(getattr(c.observed_state, "telemetry", None), "memory", None)
                if mem is not None:
                    used = None
                    vram_alloc = getattr(mem, "vram_allocated_mb", None)
                    ram_alloc = getattr(mem, "ram_used_mb", None)
                    if vram_alloc is not None and getattr(vram_alloc, "value", None) is not None:
                        used = float(vram_alloc.value)
                    elif ram_alloc is not None and getattr(ram_alloc, "value", None) is not None:
                        used = float(ram_alloc.value)
                    elif hasattr(mem, "used_mb"):
                        used = float(mem.used_mb)

                    if used is not None:
                        pressure = used / max(1.0, self.capacity_mb)
                        headroom = max(0.0, self.capacity_mb - used)
                        pressures.append(pressure)
                        headrooms.append(headroom)
                        if pressure > self.max_pressure:
                            m.memory_safety_violations += 1

        if pressures:
            m.max_memory_pressure = max(pressures)
        if headrooms:
            m.min_vram_headroom_mb = min(headrooms)

        m.memory_safety_violation_count = m.memory_safety_violations
        m.provenance_notes = "emulated"  # CPU-only environment

    # ---- Network ------------------------------------------------------------

    def _network_metrics(self, trace: RuntimeTrace, m: EvaluationMetrics) -> None:
        bws: List[float] = []
        rtts: List[float] = []
        losses: List[float] = []
        jitters: List[float] = []

        for c in trace.cycles:
            if c.observed_state is not None:
                net = getattr(c.observed_state, "network", None) or getattr(getattr(c.observed_state, "telemetry", None), "network", None)
                if net is not None:
                    bw = getattr(getattr(net, "bandwidth_mbps", None), "value", getattr(net, "bandwidth_mbps", None))
                    rtt = getattr(getattr(net, "latency_ms", None), "value", getattr(getattr(net, "rtt_ms", None), "value", getattr(net, "rtt_ms", None)))
                    loss = getattr(getattr(net, "packet_loss", None), "value", getattr(getattr(net, "packet_loss_rate", None), "value", getattr(net, "packet_loss", None)))
                    jitter = getattr(getattr(net, "jitter_ms", None), "value", getattr(net, "jitter_ms", None))
                    if bw is not None:
                        bws.append(float(bw))
                    if rtt is not None:
                        rtts.append(float(rtt))
                    if loss is not None:
                        losses.append(float(loss))
                    if jitter is not None:
                        jitters.append(float(jitter))

        if bws:
            m.avg_bandwidth_mbps = _safe_mean(bws)
            m.min_bandwidth_mbps = min(bws)
        if rtts:
            m.avg_rtt_ms = _safe_mean(rtts)
            m.p95_rtt_ms = _percentile(rtts, 95)
        if losses:
            m.avg_packet_loss = _safe_mean(losses)
        if jitters:
            m.avg_jitter_ms = _safe_mean(jitters)

        # Communication bytes from migration records
        for c in trace.cycles:
            if c.migration_result:
                m.total_communication_bytes += getattr(
                    c.migration_result, "total_bytes_transferred", 0
                )

    # ---- Control ------------------------------------------------------------

    def _control_metrics(self, trace: RuntimeTrace, m: EvaluationMetrics) -> None:
        m.controller_cycles = len(trace.cycles)
        pred_times:  List[float] = []
        cgen_times:  List[float] = []
        cost_times:  List[float] = []
        ctrl_times:  List[float] = []

        for c in trace.cycles:
            t = c.timings
            if "prediction_ms" in t:
                pred_times.append(t["prediction_ms"])
            if "candidate_gen_ms" in t:
                cgen_times.append(t["candidate_gen_ms"])
            if "cost_eval_ms" in t:
                cost_times.append(t["cost_eval_ms"])
            ctrl_times.append(c.cycle_duration_ms)

        m.total_prediction_time_ms      = sum(pred_times)
        m.total_candidate_gen_time_ms   = sum(cgen_times)
        m.total_cost_eval_time_ms       = sum(cost_times)
        m.total_controller_time_ms      = sum(ctrl_times)
        m.total_control_plane_overhead_ms = (
            m.total_prediction_time_ms
            + m.total_candidate_gen_time_ms
            + m.total_cost_eval_time_ms
        )

        if m.end_to_end_latency_ms > 0:
            m.control_overhead_fraction = (
                m.total_control_plane_overhead_ms / m.end_to_end_latency_ms
            )

    # ---- Migration ----------------------------------------------------------

    def _migration_metrics(self, trace: RuntimeTrace, m: EvaluationMetrics) -> None:
        durations: List[float] = []

        for c in trace.cycles:
            mr = c.migration_result
            if mr is None:
                continue
            m.migration_count += 1
            status = getattr(mr, "status", None)
            if status is not None:
                sv = status.value if hasattr(status, "value") else str(status)
                if sv in ("completed", "already_at_target"):
                    m.successful_migrations += 1
                elif sv in ("rolled_back",):
                    m.rollback_count += 1
                    m.failed_migrations += 1
                elif sv in ("failed",):
                    m.failed_migrations += 1

            dur = getattr(mr, "migration_duration_ms", 0.0) or 0.0
            durations.append(dur)
            m.total_migration_time_ms += dur

            model_bytes = getattr(mr, "model_bytes_transferred", 0) or 0
            kv_bytes    = getattr(mr, "kv_bytes_transferred", 0) or 0
            m.total_model_bytes_moved += model_bytes
            m.total_kv_bytes_moved    += kv_bytes
            m.total_migration_bytes   += model_bytes + kv_bytes

        if durations:
            m.mean_migration_time_ms = _safe_mean(durations)
            m.p95_migration_time_ms  = _percentile(durations, 95)

    # ---- Stability ----------------------------------------------------------

    def _stability_metrics(self, trace: RuntimeTrace, m: EvaluationMetrics) -> None:
        switch_steps: List[int] = []

        for c in trace.cycles:
            dec = c.control_decision
            if dec is None:
                continue
            action = dec.action
            av = action.value if hasattr(action, "value") else str(action)
            reason = getattr(dec, "reason", None)
            rv = reason.value if reason and hasattr(reason, "value") else str(reason)

            if av == "switch":
                m.switch_count += 1
                m.total_switches += 1
                switch_steps.append(c.inference_step)
            if rv == "below_threshold":
                m.threshold_rejection_count += 1
            if rv == "cooldown_active":
                m.cooldown_rejection_count += 1
            if rv == "hysteresis_not_satisfied":
                m.hysteresis_rejection_count += 1

        # Switches per hour — use end-to-end span
        if m.end_to_end_latency_ms > 0:
            hours = m.end_to_end_latency_ms / (1000 * 3600)
            m.switches_per_hour = m.switch_count / max(hours, 1e-9)

        # Dwell time (tokens between switches)
        if switch_steps and len(switch_steps) >= 2:
            dwells = [b - a for a, b in zip(switch_steps, switch_steps[1:])]
            m.avg_dwell_time_tokens = _safe_mean(dwells)
            m.min_dwell_time_tokens = min(dwells)

        # Thrashing: more than threshold switches in any window
        if len(switch_steps) >= self.thrashing_threshold:
            for i in range(len(switch_steps) - self.thrashing_threshold + 1):
                window = switch_steps[i : i + self.thrashing_threshold]
                if window[-1] - window[0] <= self.thrashing_window:
                    m.thrashing_event_count += 1

    # ---- Prediction ---------------------------------------------------------

    def _prediction_metrics(self, trace: RuntimeTrace, m: EvaluationMetrics) -> None:
        errors: List[float] = []

        for c in trace.cycles:
            pr = c.prediction_result
            if pr is None:
                continue
            actual = getattr(pr, "actual_value", None)
            predicted = getattr(pr, "predicted_value", None)
            if actual is not None and predicted is not None:
                try:
                    err = abs(float(actual) - float(predicted))
                    errors.append(err)
                except (TypeError, ValueError):
                    pass

        if errors:
            m.prediction_mae        = _safe_mean(errors)
            m.prediction_rmse       = math.sqrt(_safe_mean([e * e for e in errors]))
            m.prediction_median_ae  = _safe_median(errors)
            m.prediction_max_ae     = max(errors)

    # ---- Safety -------------------------------------------------------------

    def _safety_metrics(self, trace: RuntimeTrace, m: EvaluationMetrics) -> None:
        for c in trace.cycles:
            dec = c.control_decision
            if dec is None:
                continue
            if getattr(dec, "is_safety_override", False):
                m.safety_override_count += 1
            action = dec.action
            av = action.value if hasattr(action, "value") else str(action)
            if av == "safe_fallback":
                m.fallback_count += 1

    # ---- Proactive ----------------------------------------------------------

    def _proactive_metrics(self, trace: RuntimeTrace, m: EvaluationMetrics) -> None:
        for c in trace.cycles:
            dec = c.control_decision
            if dec is None:
                continue
            action = dec.action
            av = action.value if hasattr(action, "value") else str(action)
            if av != "switch":
                continue

            if getattr(dec, "is_safety_override", False):
                m.safety_override_switch_count += 1
            elif getattr(dec, "is_proactive", False):
                m.proactive_switch_count += 1
                m.proactive_switches += 1
            else:
                m.reactive_switch_count += 1
                m.reactive_switches += 1

        total = m.switch_count
        if total > 0:
            m.proactive_switch_rate = m.proactive_switch_count / total

    # ---- Lead time ----------------------------------------------------------

    def _lead_time_metrics(self, trace: RuntimeTrace, m: EvaluationMetrics) -> None:
        pred_leads: List[float] = []
        dec_leads:  List[float] = []
        mig_leads:  List[float] = []

        for lead in trace.prediction_leads:
            if lead.prediction_lead_time_s is not None:
                pred_leads.append(lead.prediction_lead_time_s)
            if lead.decision_lead_time_s is not None:
                dec_leads.append(lead.decision_lead_time_s)
            # migration start lead = decision_time - migration_start
            if lead.migration_start and lead.actual_degradation_time:
                mig_lead = lead.actual_degradation_time - (lead.migration_start or 0)
                if mig_lead > 0:
                    mig_leads.append(mig_lead)

        if pred_leads:
            m.mean_prediction_lead_time_s = _safe_mean(pred_leads)
            m.avg_prediction_lead_time_s  = _safe_mean(pred_leads)
        if dec_leads:
            m.mean_decision_lead_time_s = _safe_mean(dec_leads)
        if mig_leads:
            m.mean_migration_start_lead_time_s = _safe_mean(mig_leads)

    # ---- SLO ----------------------------------------------------------------

    def _slo_metrics(self, trace: RuntimeTrace, m: EvaluationMetrics) -> None:
        p95_limit = self.slo.get("p95_itl_ms", self.slo.get("slo_itl_ms", None))
        ttft_limit = self.slo.get("ttft_ms", None)
        mem_limit  = self.slo.get("memory_pressure_max", None)

        if p95_limit is not None and m.p95_itl_ms > p95_limit:
            m.slo_p95_itl_violated = True
        if ttft_limit is not None and m.ttft_ms > ttft_limit:
            m.slo_ttft_violated = True
        if mem_limit is not None and m.max_memory_pressure is not None:
            if m.max_memory_pressure > mem_limit:
                m.slo_memory_violated = True

        non_ttft = [tok for tok in trace.token_records if not tok.is_ttft]
        if p95_limit is not None and non_ttft:
            v_count = sum(1 for tok in non_ttft if tok.step_latency_ms > p95_limit)
            m.slo_violation_count = v_count
            m.slo_violation_rate = v_count / len(non_ttft)
        else:
            violations = sum([m.slo_p95_itl_violated, m.slo_ttft_violated, m.slo_memory_violated])
            m.slo_violation_count = violations
            m.slo_violation_rate  = violations / 3.0 if self.slo else 0.0

    # ---- Cost model validation ----------------------------------------------

    def _cost_validation_metrics(self, trace: RuntimeTrace, m: EvaluationMetrics) -> None:
        byte_errors:     List[float] = []
        duration_errors: List[float] = []

        for c in trace.cycles:
            mr  = c.migration_result
            req = c.control_decision.migration_request if c.control_decision else None
            if mr is None or req is None:
                continue
            # Predicted bytes from MigrationRequest
            predicted_bytes = getattr(req, "estimated_kv_transfer_bytes", 0)
            actual_bytes    = getattr(mr, "total_bytes_transferred", 0)
            if predicted_bytes and actual_bytes:
                byte_errors.append(abs(actual_bytes - predicted_bytes))

        if byte_errors:
            m.cost_model_byte_mae  = _safe_mean(byte_errors)
            m.cost_model_byte_rmse = math.sqrt(_safe_mean([e * e for e in byte_errors]))
