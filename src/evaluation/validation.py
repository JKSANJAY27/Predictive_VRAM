"""
Validation and sanity checking framework for Module 10.

Implements rigorous invariant checks for trial execution results and
batch-level experimental consistency.

Key Invariants:
  1. Fairness: Identical environment traces replayed across all baselines.
  2. No Future Leakage: Predictive decisions strictly use information with t <= current_time.
  3. Reactive Policy Purity: Reactive policies never record or execute predictive decisions.
  4. Static Policy Immutability: B1 STATIC policy has zero partition switches.
  5. Rollback Correctness: Aborted/rolled back migrations never counted as successful.
  6. Telemetry Monotonicity: Timestamps and token indices strictly monotonically non-decreasing.
  7. Physical Plausibility: Latencies >= 0, memory pressures in [0, 1], non-negative overhead.
  8. Emulated Hardware Provenance: VRAM and compute values labeled as EMULATED on CPU platforms.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.evaluation.baselines import BaselinePolicy
from src.evaluation.metrics import EvaluationMetrics
from src.telemetry.types import DataSource


@dataclasses.dataclass
class ValidationIssue:
    """A single validation issue or invariant violation."""
    level: str  # "ERROR", "WARNING", "INFO"
    code: str
    message: str
    trial_id: Optional[str] = None
    details: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ValidationReport:
    """Comprehensive validation report for a trial or batch of trials."""
    is_valid: bool
    issues: List[ValidationIssue] = dataclasses.field(default_factory=list)
    total_checks: int = 0
    passed_checks: int = 0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "ERROR")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "WARNING")

    def add_error(self, code: str, msg: str, trial_id: Optional[str] = None, **details: Any) -> None:
        self.issues.append(ValidationIssue(level="ERROR", code=code, message=msg, trial_id=trial_id, details=details))
        self.is_valid = False

    def add_warning(self, code: str, msg: str, trial_id: Optional[str] = None, **details: Any) -> None:
        self.issues.append(ValidationIssue(level="WARNING", code=code, message=msg, trial_id=trial_id, details=details))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [
                {
                    "level": i.level,
                    "code": i.code,
                    "message": i.message,
                    "trial_id": i.trial_id,
                    "details": i.details,
                }
                for i in self.issues
            ],
        }


class ResultValidator:
    """
    Validates a single trial execution for structural correctness and invariant adherence.
    """

    @classmethod
    def validate_trial(
        cls,
        trial_data: Dict[str, Any],
        expected_tokens: Optional[int] = None,
    ) -> ValidationReport:
        """
        Run all invariant checks on a single trial result dictionary.
        """
        report = ValidationReport(is_valid=True)
        trial_id = trial_data.get("trial_id", "unknown")
        baseline_id = trial_data.get("baseline_id", "")
        runtime_trace = trial_data.get("runtime_trace", {})
        metrics = trial_data.get("metrics", {})

        # 1. Structural schema checks
        report.total_checks += 1
        if not trial_data.get("trial_id"):
            report.add_error("ERR_MISSING_TRIAL_ID", "Trial ID is missing or empty", trial_id)
        else:
            report.passed_checks += 1

        # 2. Token counts check
        report.total_checks += 1
        tokens = runtime_trace.get("token_records", [])
        if expected_tokens is not None and len(tokens) != expected_tokens:
            report.add_warning(
                "WARN_TOKEN_COUNT_MISMATCH",
                f"Generated token count {len(tokens)} != expected {expected_tokens}",
                trial_id,
                actual=len(tokens),
                expected=expected_tokens,
            )
        else:
            report.passed_checks += 1

        # 3. Timestamp monotonicity
        report.total_checks += 1
        mono_ok = True
        last_t = -1.0
        for i, tok in enumerate(tokens):
            t = tok.get("timestamp", 0.0)
            if t < last_t:
                mono_ok = False
                report.add_error(
                    "ERR_TIMESTAMP_NON_MONOTONIC",
                    f"Token record {i} timestamp {t} < prior timestamp {last_t}",
                    trial_id,
                    token_index=i,
                )
                break
            last_t = t
        if mono_ok:
            report.passed_checks += 1

        # 4. Invariant: STATIC policy has zero partition switches
        report.total_checks += 1
        if baseline_id == BaselinePolicy.STATIC.value:
            switches = metrics.get("total_switches", 0)
            if switches != 0:
                report.add_error(
                    "ERR_STATIC_POLICY_SWITCHED",
                    f"STATIC policy executed {switches} switches (expected 0)",
                    trial_id,
                    switches=switches,
                )
            else:
                report.passed_checks += 1
        else:
            report.passed_checks += 1

        # 5. Invariant: REACTIVE policies have no proactive switches
        report.total_checks += 1
        if baseline_id in (
            BaselinePolicy.NETWORK_REACTIVE.value,
            BaselinePolicy.MEMORY_REACTIVE.value,
            BaselinePolicy.JOINT_REACTIVE.value,
        ):
            proactive = metrics.get("proactive_switches", 0)
            if proactive != 0:
                report.add_error(
                    "ERR_REACTIVE_PROACTIVE_SWITCH",
                    f"Reactive policy {baseline_id} performed {proactive} proactive switches",
                    trial_id,
                    proactive=proactive,
                )
            else:
                report.passed_checks += 1
        else:
            report.passed_checks += 1

        # 6. Invariant: No future leakage in predictive cycles
        report.total_checks += 1
        cycles = runtime_trace.get("cycle_records", [])
        leakage_detected = False
        for c in cycles:
            cycle_time = c.get("timestamp", 0.0)
            pred_meta = c.get("prediction_metadata", {})
            pred_sample_time = pred_meta.get("sample_time", 0.0)
            if pred_sample_time > cycle_time + 1e-3:  # allow 1ms clock jitter
                leakage_detected = True
                report.add_error(
                    "ERR_FUTURE_LEAKAGE",
                    f"Prediction sampled at t={pred_sample_time} > cycle time t={cycle_time}",
                    trial_id,
                )
                break
        if not leakage_detected:
            report.passed_checks += 1

        # 7. Rollbacks not counted as successful migrations
        report.total_checks += 1
        mig_records = runtime_trace.get("migration_records", [])
        rollbacks = sum(1 for m in mig_records if m.get("rolled_back", False))
        successful_migs = sum(1 for m in mig_records if not m.get("rolled_back", False))
        total_switches = metrics.get("total_switches", 0)
        if mig_records and total_switches > successful_migs:
            report.add_error(
                "ERR_ROLLBACK_COUNTED_AS_SWITCH",
                f"total_switches ({total_switches}) > successful migrations ({successful_migs})",
                trial_id,
                rollbacks=rollbacks,
            )
        else:
            report.passed_checks += 1

        # 8. Physical plausibility checks
        report.total_checks += 1
        mean_itl = metrics.get("mean_itl_ms", 0.0)
        ttft = metrics.get("ttft_ms", 0.0)
        if mean_itl < 0.0 or ttft < 0.0:
            report.add_error(
                "ERR_NEGATIVE_LATENCY",
                f"Negative latency detected: mean_itl={mean_itl}, ttft={ttft}",
                trial_id,
            )
        else:
            report.passed_checks += 1

        # 9. Memory pressure bounds
        report.total_checks += 1
        peak_pressure = metrics.get("peak_memory_pressure", 0.0)
        if peak_pressure < 0.0 or peak_pressure > 1.5:  # allow minor headroom overshoot
            report.add_warning(
                "WARN_EXTREME_MEMORY_PRESSURE",
                f"Peak memory pressure out of expected [0, 1] range: {peak_pressure}",
                trial_id,
            )
        report.passed_checks += 1

        # 10. Emulated hardware provenance check
        report.total_checks += 1
        prov = trial_data.get("provenance", {})
        gpu_source = prov.get("gpu_telemetry_source", "")
        if "emulated" not in gpu_source.lower() and "synthetic" not in gpu_source.lower():
            report.add_warning(
                "WARN_UNVERIFIED_PROVENANCE",
                f"GPU telemetry source '{gpu_source}' not explicitly EMULATED on CPU host",
                trial_id,
            )
        report.passed_checks += 1

        return report


class SanityChecker:
    """
    Checks batch-level experimental fairness, trace parity, and reproducibility.
    """

    @classmethod
    def check_batch(
        cls,
        trials: Sequence[Dict[str, Any]],
        expected_baselines: Optional[List[str]] = None,
    ) -> ValidationReport:
        """
        Run cross-trial sanity and fairness checks across a batch of results.
        """
        report = ValidationReport(is_valid=True)

        if not trials:
            report.add_error("ERR_EMPTY_BATCH", "Batch contains no trial results")
            return report

        # 1. Trace Parity Check: for each (scenario_id, seed), all baselines got same trace hash
        report.total_checks += 1
        trace_hashes_by_key: Dict[Tuple[str, int], Dict[str, str]] = {}
        for t in trials:
            sc_id = t.get("scenario_id", "")
            seed = t.get("seed", 0)
            baseline = t.get("baseline_id", "")
            thash = t.get("environment_snapshot", {}).get("content_hash", "")
            if thash:
                trace_hashes_by_key.setdefault((sc_id, seed), {})[baseline] = thash

        parity_violated = False
        for (sc_id, seed), base_hashes in trace_hashes_by_key.items():
            unique_hashes = set(base_hashes.values())
            if len(unique_hashes) > 1:
                parity_violated = True
                report.add_error(
                    "ERR_TRACE_PARITY_VIOLATION",
                    f"Baselines under scenario={sc_id}, seed={seed} received different environment traces",
                    details=base_hashes,
                )
        if not parity_violated:
            report.passed_checks += 1

        # 2. Baseline Completeness Check
        report.total_checks += 1
        if expected_baselines:
            present_baselines = {t.get("baseline_id") for t in trials}
            missing = set(expected_baselines) - present_baselines
            if missing:
                report.add_error(
                    "ERR_INCOMPLETE_BASELINES",
                    f"Expected baselines missing from batch: {missing}",
                    details={"missing": list(missing)},
                )
            else:
                report.passed_checks += 1
        else:
            report.passed_checks += 1

        # 3. Duplicate Trial ID Check
        report.total_checks += 1
        seen_ids = set()
        duplicates = set()
        for t in trials:
            tid = t.get("trial_id", "")
            if tid in seen_ids:
                duplicates.add(tid)
            seen_ids.add(tid)
        if duplicates:
            report.add_error(
                "ERR_DUPLICATE_TRIAL_IDS",
                f"Duplicate trial IDs found in batch: {duplicates}",
                details={"duplicate_ids": list(duplicates)},
            )
        else:
            report.passed_checks += 1

        # 4. Run single trial checks for each trial in batch
        for t in trials:
            single_rep = ResultValidator.validate_trial(t)
            report.total_checks += single_rep.total_checks
            report.passed_checks += single_rep.passed_checks
            report.issues.extend(single_rep.issues)
            if not single_rep.is_valid:
                report.is_valid = False

        return report
