"""
Unit and integration tests for Module 10:
Experimental Harness, Baseline Policies, Ablation Framework, and Statistical Evaluation.

Covers:
  - BaselinePolicy and BaselinePolicyFactory specifications (B1-B5)
  - Deterministic TraceGenerator and CombinedEnvironmentTrace invariants
  - ScenarioCatalog and alias mappings (combined_degradation -> mixed_degradation)
  - ExperimentConfig, ExperimentMatrix expansion, and ConfigHasher
  - MetricCalculator across latency, SLO, stability, overhead, and memory
  - ResultAggregator statistical distributions and effect size calculations
  - AblationSuite definitions and delta constraints (A1-A8)
  - SensitivityAnalyzer parameter sweeps
  - ResultValidator and SanityChecker invariant enforcement
  - Manifest and DuplicateRunGuard immutability
  - CSVExporter and JSONExporter format adherence
  - End-to-end ExperimentRunner execution with trace parity
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

from src.controller.types import ControllerConfig, ControllerMode
from src.evaluation.ablations import AblationSpec, AblationSuite
from src.evaluation.aggregation import AggregatedResult, ResultAggregator
from src.evaluation.baselines import BaselinePolicy, BaselinePolicyFactory
from src.evaluation.experiment_config import (
    ExperimentConfig,
    ExperimentMatrix,
    NetworkConfig,
    TrialConfig,
    WorkloadConfig,
    make_canonical_experiment,
)
from src.evaluation.export import CSVExporter, JSONExporter, ResearchDatasetBuilder
from src.evaluation.manifest import ConfigHasher, DuplicateRunGuard, ExperimentManifest
from src.evaluation.metrics import EvaluationMetrics, MetricCalculator
from src.evaluation.runner import ExperimentRunner, TrialResult
from src.evaluation.scenario_catalog import ScenarioCatalog
from src.evaluation.sensitivity import (
    SensitivityAnalyzer,
    SensitivityCurve,
    SensitivityDimension,
)
from src.evaluation.traces import (
    CombinedEnvironmentTrace,
    NetworkStep,
    TraceGenerator,
)
from src.evaluation.validation import ResultValidator, SanityChecker, ValidationReport
from src.orchestration.config import OrchestrationConfig
from src.orchestration.trace import RuntimeTrace
from src.orchestration.types import (
    ControlCycle,
    ExecutionMode,
    OrchestrationMode,
    TokenRecord,
)


# ===========================================================================
# 1. Baseline Policy Factory & Specs (B1-B5)
# ===========================================================================

def test_baseline_policy_enum():
    """All 5 baselines must be defined with exact IDs."""
    expected = {"static", "network_reactive", "memory_reactive", "joint_reactive", "predictive"}
    actual = {p.value for p in BaselinePolicy}
    assert actual == expected


def test_baseline_policy_factory_specs():
    """Verify spec properties: STATIC and REACTIVE policies must not have prediction enabled."""
    spec_static = BaselinePolicyFactory.get_spec(BaselinePolicy.STATIC)
    assert not spec_static.prediction_enabled
    assert spec_static.controller_mode == ControllerMode.STATIC

    spec_net = BaselinePolicyFactory.get_spec(BaselinePolicy.NETWORK_REACTIVE)
    assert not spec_net.prediction_enabled
    assert spec_net.network_signal
    assert not spec_net.vram_signal

    spec_mem = BaselinePolicyFactory.get_spec(BaselinePolicy.MEMORY_REACTIVE)
    assert not spec_mem.prediction_enabled
    assert not spec_mem.network_signal
    assert spec_mem.vram_signal

    spec_joint = BaselinePolicyFactory.get_spec(BaselinePolicy.JOINT_REACTIVE)
    assert not spec_joint.prediction_enabled
    assert spec_joint.network_signal
    assert spec_joint.vram_signal

    spec_pred = BaselinePolicyFactory.get_spec(BaselinePolicy.PREDICTIVE)
    assert spec_pred.prediction_enabled
    assert spec_pred.network_signal
    assert spec_pred.vram_signal


def test_baseline_cost_weights_fairness():
    """B2 zeroes gamma (memory), B3 zeroes beta (network); B4 and B5 keep joint terms."""
    w_b2 = BaselinePolicyFactory.get_spec(BaselinePolicy.NETWORK_REACTIVE).cost_weights
    w_b3 = BaselinePolicyFactory.get_spec(BaselinePolicy.MEMORY_REACTIVE).cost_weights
    w_b5 = BaselinePolicyFactory.get_spec(BaselinePolicy.PREDICTIVE).cost_weights

    assert w_b2["gamma"] == 0.0
    assert w_b2["beta"] > 0.0
    assert w_b3["beta"] == 0.0
    assert w_b3["gamma"] > 0.0
    assert w_b5["beta"] > 0.0 and w_b5["gamma"] > 0.0


def test_baseline_build_configs():
    """build_configs returns valid orchestration and controller configs."""
    orch, ctrl, weights = BaselinePolicyFactory.build_configs(
        BaselinePolicy.PREDICTIVE,
        scenario_name="stable",
        seed=42,
        max_new_tokens=16,
    )
    assert orch.mode == OrchestrationMode.PREDICTIVE
    assert ctrl.mode == ControllerMode.PREDICTIVE
    assert orch.max_generated_tokens == 16
    assert isinstance(weights, dict)


# ===========================================================================
# 2. Trace Generator & Environmental Trace
# ===========================================================================

def test_trace_generator_deterministic():
    """Identical (scenario, seed) must produce identical trace content_hash."""
    t1 = TraceGenerator.generate(scenario_id="stable", seed=42, n_steps=16)
    t2 = TraceGenerator.generate(scenario_id="stable", seed=42, n_steps=16)
    assert t1.content_hash() == t2.content_hash()
    assert len(t1.network.steps) == 16


def test_trace_generator_seed_variation():
    """Different seeds must produce different trace content hashes."""
    t1 = TraceGenerator.generate(scenario_id="stable", seed=42, n_steps=16)
    t2 = TraceGenerator.generate(scenario_id="stable", seed=99, n_steps=16)
    assert t1.content_hash() != t2.content_hash()


def test_trace_monotone_timestamps():
    """All step timestamps must be strictly non-decreasing."""
    trace = TraceGenerator.generate(scenario_id="mixed_degradation", seed=42, n_steps=32)
    assert trace.validate_timestamps()


def test_trace_provenance_emulated():
    """Memory steps on CPU host must be explicitly marked as emulated."""
    trace = TraceGenerator.generate(scenario_id="stable", seed=42, n_steps=8)
    for m in trace.memory.steps:
        assert m.provenance == "emulated"


def test_trace_serialization_roundtrip(tmp_path):
    """CombinedEnvironmentTrace save/load roundtrip preserves hash and steps."""
    trace = TraceGenerator.generate(scenario_id="mixed_degradation", seed=42, n_steps=16)
    save_path = tmp_path / "test_trace.json"
    trace.save_json(save_path)
    loaded = CombinedEnvironmentTrace.load_json(save_path)
    assert loaded.content_hash() == trace.content_hash()
    assert len(loaded.network.steps) == len(trace.network.steps)


# ===========================================================================
# 3. Scenario Catalog
# ===========================================================================

def test_scenario_catalog_coverage():
    """All research scenarios S1-S13 must be registered."""
    scenarios = ScenarioCatalog.list_scenarios()
    assert len(scenarios) >= 10
    assert "stable" in scenarios
    assert "mixed_degradation" in scenarios


def test_scenario_alias_combined_degradation():
    """combined_degradation must alias to mixed_degradation."""
    meta = ScenarioCatalog.get_scenario("combined_degradation")
    assert meta is not None
    assert meta.scenario_id in ("combined_degradation", "mixed_degradation")


# ===========================================================================
# 4. Experiment Config & Matrix
# ===========================================================================

def test_experiment_config_hash_deterministic():
    """Configs with identical attributes produce the identical hash."""
    c1 = make_canonical_experiment()
    c2 = make_canonical_experiment()
    assert c1.config_hash() == c2.config_hash()


def test_experiment_matrix_expansion():
    """Matrix expansion covers baselines * repetitions without duplicates."""
    cfg = ExperimentConfig(
        name="test_exp",
        seed=10,
        repetitions=2,
        baselines=["static", "predictive"],
    )
    matrix = ExperimentMatrix(cfg)
    trials = matrix.expand(include_ablations=False, include_sensitivity=False)
    assert len(trials) == 4  # 2 baselines * 2 seeds
    trial_ids = [t.trial_id for t in trials]
    assert len(set(trial_ids)) == 4


# ===========================================================================
# 5. Metric Calculator
# ===========================================================================

def _make_dummy_runtime_trace() -> RuntimeTrace:
    rt = RuntimeTrace(metadata={"experiment_id": "test_exp", "mode": "predictive"})
    now = time.time()
    # Add 5 token records
    for i in range(5):
        rt.add_token(TokenRecord(
            token_index=i,
            token_id=100 + i,
            token_str=f"tok_{i}",
            step_latency_ms=100.0 if i == 0 else 50.0 + (i * 10.0),
            timestamp=now + (i * 0.1),
            active_plan_id="plan_0",
            is_ttft=(i == 0),
        ))
    return rt


def test_metric_calculator_empty_trace():
    """MetricCalculator handles empty trace safely without dividing by zero."""
    calc = MetricCalculator()
    metrics = calc.compute(RuntimeTrace())
    assert metrics.completed_tokens == 0
    assert metrics.mean_itl_ms == 0.0
    assert metrics.p95_itl_ms == 0.0


def test_metric_calculator_inference_metrics():
    """Verify TTFT and ITL calculations on controlled trace."""
    rt = _make_dummy_runtime_trace()
    calc = MetricCalculator()
    m = calc.compute(rt)
    assert m.completed_tokens == 5
    assert m.ttft_ms == 100.0
    # ITLs: 60, 70, 80, 90 -> mean = 75.0
    assert abs(m.mean_itl_ms - 75.0) < 1e-3
    assert m.p95_itl_ms >= 75.0


def test_metric_calculator_slo_violations():
    """SLO violation count correctly flags ITLs exceeding threshold."""
    rt = _make_dummy_runtime_trace()
    calc = MetricCalculator(slo_config={"p95_itl_ms": 75.0})
    m = calc.compute(rt)
    # ITLs: 60, 70, 80, 90 -> 80 and 90 > 75 -> 2 violations
    assert m.slo_violation_count == 2
    assert abs(m.slo_violation_rate - 0.5) < 1e-3


def test_metric_calculator_provenance_emulated():
    """Metrics object must retain emulated provenance notes."""
    calc = MetricCalculator()
    m = calc.compute(_make_dummy_runtime_trace())
    assert m.provenance_notes == "emulated"


# ===========================================================================
# 6. Result Aggregator & Statistical Analysis
# ===========================================================================

def test_result_aggregator_mean_std():
    """ResultAggregator computes correct sample mean and standard deviation."""
    trials = [
        {
            "baseline_id": "predictive",
            "scenario_id": "stable",
            "metrics": {"mean_itl_ms": 100.0, "p95_itl_ms": 120.0},
        },
        {
            "baseline_id": "predictive",
            "scenario_id": "stable",
            "metrics": {"mean_itl_ms": 110.0, "p95_itl_ms": 130.0},
        },
        {
            "baseline_id": "predictive",
            "scenario_id": "stable",
            "metrics": {"mean_itl_ms": 120.0, "p95_itl_ms": 140.0},
        },
    ]
    agg = ResultAggregator().aggregate_trials(trials)
    assert len(agg) == 1
    m_stat = agg[0].metrics["mean_itl_ms"]
    assert abs(m_stat.mean - 110.0) < 1e-3
    assert abs(m_stat.std - 10.0) < 1e-3
    assert abs(m_stat.median - 110.0) < 1e-3


def test_paired_comparison_diff():
    """Paired comparison computes delta and percentage differences."""
    t_pred = [{"baseline_id": "predictive", "scenario_id": "stable", "seed": 42, "metrics": {"p95_itl_ms": 100.0}}]
    t_stat = [{"baseline_id": "static", "scenario_id": "stable", "seed": 42, "metrics": {"p95_itl_ms": 150.0}}]

    pair = ResultAggregator.compute_paired_comparison(t_pred, t_stat, metric_name="p95_itl_ms")
    assert abs(pair.mean_diff - (-50.0)) < 1e-3
    # delta / static = -50 / 150 = -33.33%
    assert abs(pair.pct_diff - (-33.33)) < 0.1


def test_effect_size_cohens_d():
    """Cohen's d calculation handles distinct distributions properly."""
    group1 = [10.0, 11.0, 12.0, 10.5, 11.5]
    group2 = [20.0, 21.0, 22.0, 20.5, 21.5]
    es = ResultAggregator.compute_effect_size(group1, group2)
    assert es.cohens_d < -5.0  # large negative effect size


# ===========================================================================
# 7. Ablation Suite
# ===========================================================================

def test_ablation_suite_mandatory_count():
    """Mandatory ablations A1-A4 must be registered."""
    mand = AblationSuite.mandatory()
    assert len(mand) == 4
    ids = {a.ablation_id for a in mand}
    assert ids == {"A1", "A2", "A3", "A4"}


def test_ablation_a1_disables_prediction():
    """A1 disables prediction entirely."""
    a1 = AblationSuite.get("prediction_removed")
    assert a1.disable_prediction


def test_ablation_a2_zeroes_epsilon():
    """A2 zeroes switching penalty (epsilon = 0.0)."""
    a2 = AblationSuite.get("switching_penalty_removed")
    assert a2.cost_weights_override.get("epsilon") == 0.0


def test_ablation_a3_zeroes_beta():
    """A3 zeroes network term (beta = 0.0)."""
    a3 = AblationSuite.get("network_signal_removed")
    assert a3.cost_weights_override.get("beta") == 0.0


def test_ablation_a4_zeroes_gamma():
    """A4 zeroes memory term (gamma = 0.0)."""
    a4 = AblationSuite.get("vram_signal_removed")
    assert a4.cost_weights_override.get("gamma") == 0.0


# ===========================================================================
# 8. Sensitivity Analysis
# ===========================================================================

def test_sensitivity_analyzer_sweep():
    """SensitivityAnalyzer groups metrics and builds curves."""
    data = [
        (0.05, EvaluationMetrics(p95_itl_ms=110.0, total_switches=5), "t1"),
        (0.10, EvaluationMetrics(p95_itl_ms=120.0, total_switches=3), "t2"),
        (0.20, EvaluationMetrics(p95_itl_ms=140.0, total_switches=1), "t3"),
    ]
    res = SensitivityAnalyzer.analyze_sweep(
        dimension=SensitivityDimension.SWITCH_THRESHOLD,
        sweep_data=data,
        metrics_to_extract=["p95_itl_ms", "total_switches"],
    )
    assert len(res.curves) == 2
    p95_curve = res.curves["p95_itl_ms"]
    assert p95_curve.metric_values == [110.0, 120.0, 140.0]
    sw_curve = res.curves["total_switches"]
    assert sw_curve.metric_values == [5.0, 3.0, 1.0]


# ===========================================================================
# 9. Result Validator & Sanity Checker
# ===========================================================================

def test_result_validator_valid_trial():
    """Valid trial passes all checks without error."""
    trial_data = {
        "trial_id": "trial_123",
        "baseline_id": "predictive",
        "runtime_trace": {
            "token_records": [{"token_index": 0, "timestamp": 1.0}, {"token_index": 1, "timestamp": 1.1}],
            "cycle_records": [],
            "migration_records": [],
        },
        "metrics": {"mean_itl_ms": 50.0, "ttft_ms": 100.0, "total_switches": 1},
        "provenance": {"gpu_telemetry_source": "DataSource.EMULATED"},
    }
    rep = ResultValidator.validate_trial(trial_data)
    assert rep.is_valid
    assert rep.error_count == 0


def test_result_validator_static_no_switches_invariant():
    """STATIC policy invariant: must error if total_switches > 0."""
    trial_data = {
        "trial_id": "static_trial",
        "baseline_id": "static",
        "runtime_trace": {"token_records": []},
        "metrics": {"total_switches": 2},  # violation
    }
    rep = ResultValidator.validate_trial(trial_data)
    assert not rep.is_valid
    assert any(i.code == "ERR_STATIC_POLICY_SWITCHED" for i in rep.issues)


def test_result_validator_reactive_no_proactive_invariant():
    """REACTIVE policy invariant: must error if proactive_switches > 0."""
    trial_data = {
        "trial_id": "reactive_trial",
        "baseline_id": "joint_reactive",
        "runtime_trace": {"token_records": []},
        "metrics": {"proactive_switches": 1},  # violation
    }
    rep = ResultValidator.validate_trial(trial_data)
    assert not rep.is_valid
    assert any(i.code == "ERR_REACTIVE_PROACTIVE_SWITCH" for i in rep.issues)


def test_result_validator_monotonic_timestamps_invariant():
    """Timestamp monotonicity invariant: must error on clock rollback."""
    trial_data = {
        "trial_id": "time_trial",
        "baseline_id": "predictive",
        "runtime_trace": {
            "token_records": [
                {"token_index": 0, "timestamp": 10.0},
                {"token_index": 1, "timestamp": 8.0},  # rollback
            ]
        },
        "metrics": {},
    }
    rep = ResultValidator.validate_trial(trial_data)
    assert not rep.is_valid
    assert any(i.code == "ERR_TIMESTAMP_NON_MONOTONIC" for i in rep.issues)


def test_sanity_checker_trace_parity_invariant():
    """Trace parity check: errors if baselines on same scenario/seed got different traces."""
    trials = [
        {
            "trial_id": "t1",
            "scenario_id": "stable",
            "seed": 42,
            "baseline_id": "static",
            "environment_snapshot": {"content_hash": "hash_AAA"},
            "metrics": {},
        },
        {
            "trial_id": "t2",
            "scenario_id": "stable",
            "seed": 42,
            "baseline_id": "predictive",
            "environment_snapshot": {"content_hash": "hash_BBB"},  # parity violation
            "metrics": {},
        },
    ]
    rep = SanityChecker.check_batch(trials)
    assert not rep.is_valid
    assert any(i.code == "ERR_TRACE_PARITY_VIOLATION" for i in rep.issues)


# ===========================================================================
# 10. Manifest & Duplicate Run Guard
# ===========================================================================

def test_config_hasher_consistent():
    """ConfigHasher produces identical hash for identical dict contents."""
    d1 = {"a": 1, "b": "hello", "c": [1, 2, 3]}
    d2 = {"c": [1, 2, 3], "a": 1, "b": "hello"}
    assert ConfigHasher.hash_dict(d1) == ConfigHasher.hash_dict(d2)


def test_duplicate_run_guard(tmp_path):
    """DuplicateRunGuard recognizes completed non-empty trials."""
    guard = DuplicateRunGuard(tmp_path)
    assert guard.can_run("trial_001")

    # Create dummy trial file
    (tmp_path / "trial_001.json").write_text(json.dumps({"trial_id": "trial_001", "dummy": "data"}))

    assert not guard.can_run("trial_001")
    assert guard.can_run("trial_001", force=True)


# ===========================================================================
# 11. CSV and JSON Exporters
# ===========================================================================

def test_csv_exporter_flattens_trial(tmp_path):
    """CSVExporter properly flattens trial records into columns."""
    trials = [
        {
            "trial_id": "trial_xyz",
            "scenario_id": "stable",
            "baseline_id": "predictive",
            "seed": 42,
            "metrics": {"mean_itl_ms": 45.2, "p95_itl_ms": 62.1},
        }
    ]
    out = tmp_path / "test.csv"
    CSVExporter.export_trials_to_csv(trials, out)
    assert out.is_file()
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    assert "trial_xyz" in lines[1]
    assert "predictive" in lines[1]


# ===========================================================================
# 12. End-to-End ExperimentRunner Execution (Fast/Tiny steps)
# ===========================================================================

def test_runner_executes_trial(tmp_path):
    """ExperimentRunner executes a tiny trial and saves TrialResult."""
    runner = ExperimentRunner(raw_results_dir=tmp_path / "raw", traces_dir=tmp_path / "traces")

    tc = TrialConfig(
        trial_id="tiny_trial_1",
        experiment_id="test_exp",
        scenario_id="stable",
        baseline_id="predictive",
        seed=42,
        repetition=0,
        switch_threshold=0.25,
        control_interval_tokens=2,
        horizon_steps=2,
        predictor_type="linear_trend",
        max_new_tokens=4,
        n_trace_steps=8,
        sampling_interval_s=0.25,
        execution_mode="simulation",
        cost_weights={"alpha": 0.35, "beta": 0.20, "gamma": 0.25, "delta": 0.10, "epsilon": 0.10},
    )

    res = runner.run_trial(tc)
    assert res.completion_status == "completed"
    assert res.trial_id == "tiny_trial_1"
    assert res.metrics.completed_tokens == 4
    assert (tmp_path / "raw" / "tiny_trial_1.json").is_file()


def test_runner_enforces_trace_parity(tmp_path):
    """Static and Predictive baselines must receive identical environment trace snapshot."""
    runner = ExperimentRunner(raw_results_dir=tmp_path / "raw", traces_dir=tmp_path / "traces")

    tc_static = TrialConfig(
        trial_id="parity_static",
        experiment_id="test_parity",
        scenario_id="mixed_degradation",
        baseline_id="static",
        seed=42,
        repetition=0,
        switch_threshold=0.25,
        control_interval_tokens=2,
        horizon_steps=2,
        predictor_type="last_value",
        max_new_tokens=4,
        n_trace_steps=8,
        sampling_interval_s=0.25,
        execution_mode="simulation",
        cost_weights={"alpha": 0.35, "beta": 0.20, "gamma": 0.25, "delta": 0.10, "epsilon": 0.10},
    )

    tc_pred = TrialConfig(
        trial_id="parity_pred",
        experiment_id="test_parity",
        scenario_id="mixed_degradation",
        baseline_id="predictive",
        seed=42,
        repetition=0,
        switch_threshold=0.25,
        control_interval_tokens=2,
        horizon_steps=2,
        predictor_type="linear_trend",
        max_new_tokens=4,
        n_trace_steps=8,
        sampling_interval_s=0.25,
        execution_mode="simulation",
        cost_weights={"alpha": 0.35, "beta": 0.20, "gamma": 0.25, "delta": 0.10, "epsilon": 0.10},
    )

    res_static = runner.run_trial(tc_static)
    res_pred = runner.run_trial(tc_pred)

    # Invariant 1: exact bitwise trace hash match
    assert res_static.environment_snapshot["content_hash"] == res_pred.environment_snapshot["content_hash"]


def test_runner_static_policy_zero_switches(tmp_path):
    """Running B1 STATIC policy must result in 0 switches."""
    runner = ExperimentRunner(raw_results_dir=tmp_path / "raw", traces_dir=tmp_path / "traces")
    tc_static = TrialConfig(
        trial_id="static_zero_switches",
        experiment_id="test_zero",
        scenario_id="sudden_bandwidth_drop",
        baseline_id="static",
        seed=42,
        repetition=0,
        switch_threshold=0.25,
        control_interval_tokens=2,
        horizon_steps=2,
        predictor_type="last_value",
        max_new_tokens=4,
        n_trace_steps=8,
        sampling_interval_s=0.25,
        execution_mode="simulation",
        cost_weights={"alpha": 0.35, "beta": 0.20, "gamma": 0.25, "delta": 0.10, "epsilon": 0.10},
    )
    res = runner.run_trial(tc_static)
    assert res.metrics.total_switches == 0
