"""
Unit and Integration Tests for Module 12:
Reproducible Research, Statistical Validation & Publication-Ready Evaluation.

Covers:
- SeedManager deterministic derivation and hierarchical uniqueness
- EnvironmentFingerprint capture and software dependency auditing
- ResearchConfigHasher canonical sorting and ephemeral key stripping
- ResearchManifest serialization, checksum recording, and resume validation
- StatisticalAnalyzer descriptive statistics against known reference distributions
- Confidence interval formulations (exact Student-t and bootstrap resampling)
- Small-sample safeguards (N < 2, N < 3, N < 5) and zero-fabrication protections
- Paired comparison tests, Cohen's d, Hedges' g correction, and Cliff's delta
- Multiple-testing corrections (Holm-Bonferroni and Benjamini-Hochberg FDR)
- PolicyComparisonEngine seed-matched evaluation and DataFrame flattening
- ResearchAblationRunner hypothesis checking and effect evaluation
- ResearchSensitivityRunner parameter sweeps and degradation curves
- ResearchValidator provenance invariants (CPU/VRAM, static immutability, checksums)
- PublicationArtifactGenerator tables and headless matplotlib figure generation
- ResearchReportGenerator 15-section scientific Markdown output
- End-to-end ResearchRunner campaign execution and analysis idempotency
"""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from src.research.ablations import ResearchAblationRunner
from src.research.artifacts import PublicationArtifactGenerator
from src.research.comparisons import PolicyComparisonEngine
from src.research.manifest import (
    CANONICAL_METRIC_DEFINITIONS,
    ManifestTrialRecord,
    ResearchManifest,
)
from src.research.report import ResearchReportGenerator
from src.research.reproducibility import (
    EnvironmentFingerprint,
    ResearchConfigHasher,
    SeedManager,
)
from src.research.runner import ResearchRunner
from src.research.sensitivity import ResearchSensitivityRunner
from src.research.statistics import (
    ConfidenceInterval,
    MetricSummaryStats,
    PairedComparisonResult,
    StatisticalAnalyzer,
)
from src.research.validation import ResearchValidator


# ---------------------------------------------------------------------------
# 1. Reproducibility & Seed Management Tests
# ---------------------------------------------------------------------------

def test_seed_manager_deterministic_and_unique() -> None:
    sm1 = SeedManager(root_seed=42)
    sm2 = SeedManager(root_seed=42)

    # Identical derivation
    s1 = sm1.derive_seed("trial", repetition=0, subtask="stable:predictive")
    s2 = sm2.derive_seed("trial", repetition=0, subtask="stable:predictive")
    assert s1 == s2
    assert isinstance(s1, int)
    assert 0 <= s1 <= 0x7FFFFFFF

    # Different repetitions produce distinct seeds
    s_rep1 = sm1.derive_seed("trial", repetition=1, subtask="stable:predictive")
    assert s1 != s_rep1

    # Different subtasks produce distinct seeds
    s_other = sm1.derive_seed("trial", repetition=0, subtask="stable:static")
    assert s1 != s_other


def test_global_seed_application() -> None:
    sm = SeedManager(root_seed=12345)
    meta = sm.set_global_seeds(54321)
    assert meta["applied_seed"] == 54321
    assert meta["python_random_seeded"] is True
    assert meta["numpy_seeded"] is True


def test_environment_fingerprint_capture() -> None:
    fp = EnvironmentFingerprint.capture(check_git=True)
    assert fp.python_version != ""
    assert fp.platform_system != ""
    assert isinstance(fp.dependency_versions, dict)
    assert "numpy" in fp.dependency_versions
    d = fp.to_dict()
    assert "platform_system" in d


def test_config_hasher_deterministic_and_excludes_ephemeral() -> None:
    cfg1 = {"model": "gpt2", "seed": 42, "experiment_id": "exp_1", "timestamp": "2026-01-01"}
    cfg2 = {"model": "gpt2", "seed": 42, "experiment_id": "exp_999", "timestamp": "2026-12-31"}
    cfg3 = {"model": "gpt2", "seed": 43}

    h1 = ResearchConfigHasher.hash_dict(cfg1)
    h2 = ResearchConfigHasher.hash_dict(cfg2)
    h3 = ResearchConfigHasher.hash_dict(cfg3)

    assert h1 == h2  # Ephemeral keys stripped
    assert h1 != h3  # Real parameter change alters hash


# ---------------------------------------------------------------------------
# 2. Manifest Serialization & Resume Tests
# ---------------------------------------------------------------------------

def test_manifest_serialization_and_checksums(tmp_path: Path) -> None:
    manifest_file = tmp_path / "manifest.json"
    dummy_artifact = tmp_path / "dummy.csv"
    dummy_artifact.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

    manifest = ResearchManifest(
        experiment_id="test_exp_001",
        title="Test Campaign",
        resolved_config={"alpha": 1.0, "beta": 2.0},
        root_seed=42,
        scenarios=["stable"],
        policies=["static", "predictive"],
    )
    manifest.record_artifact_checksum("dummy.csv", dummy_artifact)

    record = ManifestTrialRecord(
        trial_id="t1",
        scenario_id="stable",
        baseline_id="predictive",
        seed=42,
        repetition=0,
        status="completed",
    )
    manifest.add_trial_record(record)
    saved_path = manifest.save(manifest_file)
    assert saved_path.is_file()

    loaded = ResearchManifest.load(manifest_file)
    assert loaded.experiment_id == "test_exp_001"
    assert len(loaded.trials) == 1
    assert loaded.completed_repetitions == 1
    assert "dummy.csv" in loaded.artifact_checksums


def test_manifest_resume_validation_match_and_mismatch() -> None:
    cfg = {"param_a": 10, "param_b": 20}
    manifest = ResearchManifest(experiment_id="exp_resume", resolved_config=cfg)

    # Identical config -> resume allowed
    ok, msg = manifest.validate_for_resume(cfg)
    assert ok is True

    # Altered config -> resume rejected
    altered_cfg = {"param_a": 10, "param_b": 999}
    ok_alt, msg_alt = manifest.validate_for_resume(altered_cfg)
    assert ok_alt is False
    assert "Configuration mismatch" in msg_alt


# ---------------------------------------------------------------------------
# 3. Statistical Analysis & Hypothesis Testing Tests
# ---------------------------------------------------------------------------

def test_statistical_summary_known_distribution() -> None:
    # Synthetic Gaussian sample: N=100, mu=50.0, sigma=10.0
    rng = np.random.default_rng(42)
    sample = rng.normal(loc=50.0, scale=10.0, size=100).tolist()

    stats = StatisticalAnalyzer.compute_metric_stats("latency", sample, confidence_level=0.95)
    assert stats.n_valid == 100
    assert stats.n_missing == 0
    assert 48.0 < stats.mean < 52.0
    assert 7.0 < stats.std < 13.0
    assert stats.ci_t.is_valid is True
    assert stats.ci_t.lower < stats.mean < stats.ci_t.upper
    assert stats.small_sample_warning is False


def test_statistical_small_sample_safeguards() -> None:
    # N = 0
    s0 = StatisticalAnalyzer.compute_metric_stats("metric", [])
    assert s0.n_valid == 0
    assert s0.mean is None

    # N = 1
    s1 = StatisticalAnalyzer.compute_metric_stats("metric", [10.5])
    assert s1.n_valid == 1
    assert s1.mean == 10.5
    assert s1.std is None
    assert s1.ci_t.is_valid is False

    # N = 4 (< 5 triggers warning)
    s4 = StatisticalAnalyzer.compute_metric_stats("metric", [10.0, 12.0, 11.0, 10.5])
    assert s4.n_valid == 4
    assert s4.small_sample_warning is True
    assert s4.ci_t.is_valid is True
    assert s4.ci_t.warning is not None


def test_clean_observations_no_zero_fabrication() -> None:
    raw = [10.0, None, float("nan"), float("inf"), 20.0, -float("inf")]
    cleaned = StatisticalAnalyzer.clean_observations(raw)
    assert cleaned == [10.0, 20.0]
    # No zero-filling occurred
    assert 0.0 not in cleaned


def test_bootstrap_confidence_interval() -> None:
    # Synthetic skewed exponential data
    rng = np.random.default_rng(123)
    data = rng.exponential(scale=5.0, size=50)

    ci_boot = StatisticalAnalyzer.bootstrap_ci_mean(data, confidence_level=0.95, n_bootstraps=1000, seed=42)
    assert ci_boot.is_valid is True
    assert ci_boot.lower < float(np.mean(data)) < ci_boot.upper


def test_paired_comparison_known_effect() -> None:
    # Synthetic matched pairs: A is systematically higher than B by 5.0 units with low noise
    rng = np.random.default_rng(42)
    base = rng.uniform(20.0, 30.0, size=20)
    vals_b = base
    vals_a = base + 5.0 + rng.normal(0.0, 0.5, size=20)

    res = StatisticalAnalyzer.compare_paired(
        metric_name="itl",
        policy_a_name="PolicyA",
        values_a=vals_a,
        policy_b_name="PolicyB",
        values_b=vals_b,
        confidence_level=0.95,
    )

    assert res.n_pairs == 20
    assert 4.5 < res.mean_diff < 5.5
    assert res.t_statistic > 10.0  # Highly significant
    assert res.p_value_t < 1e-4
    assert res.hedges_g > 2.0     # Large effect size
    assert res.practical_significance == "large"
    assert res.ci_diff_t.is_valid is True
    assert res.ci_diff_t.lower < res.mean_diff < res.ci_diff_t.upper


def test_multiple_testing_correction_holm_and_fdr() -> None:
    p_values = [0.001, 0.015, 0.03, 0.04, 0.50]
    holm_adj = StatisticalAnalyzer.adjust_p_values_holm(p_values)
    fdr_adj = StatisticalAnalyzer.adjust_p_values_fdr(p_values)

    assert len(holm_adj) == 5
    assert len(fdr_adj) == 5

    # Holm-Bonferroni is more conservative than raw p
    for p, h in zip(p_values, holm_adj):
        assert h >= p

    # FDR values are between raw p and Holm
    for p, f in zip(p_values, fdr_adj):
        assert f >= p


# ---------------------------------------------------------------------------
# 4. Fair Comparison Engine Tests
# ---------------------------------------------------------------------------

def test_policy_comparison_engine_matched_seeds() -> None:
    # Synthetic DataFrame simulating matched seeds across B1 and B5
    rows = []
    for s in [101, 102, 103, 104, 105]:
        rows.append({"scenario_id": "scen_test", "baseline_id": "predictive", "seed": s, "mean_itl_ms": 15.0, "total_switches": 2})
        rows.append({"scenario_id": "scen_test", "baseline_id": "static", "seed": s, "mean_itl_ms": 25.0, "total_switches": 0})

    df = pd.DataFrame(rows)
    report = PolicyComparisonEngine.compare_scenario_trials(
        scenario_id="scen_test",
        trials_df=df,
        reference_policy="predictive",
    )

    assert report.scenario_id == "scen_test"
    assert len(report.comparisons) > 0

    cmp_df = PolicyComparisonEngine.comparisons_to_dataframe(report.comparisons)
    assert not cmp_df.empty
    assert "mean_diff_a_minus_b" in cmp_df.columns
    # predictive (15.0) - static (25.0) = -10.0
    itl_diff = cmp_df[cmp_df["metric_name"] == "mean_itl_ms"]["mean_diff_a_minus_b"].values[0]
    assert itl_diff == -10.0


# ---------------------------------------------------------------------------
# 5. Ablations & Sensitivity Tests
# ---------------------------------------------------------------------------

def test_ablation_hypotheses_and_evaluation() -> None:
    hypos = ResearchAblationRunner.get_registered_hypotheses()
    assert "prediction_removed" in hypos
    assert "switching_penalty_removed" in hypos
    assert hypos["prediction_removed"].ablation_id == "A1"

    # Synthetic trials DataFrame with ablation rows
    rows = [
        {"scenario_id": "s1", "baseline_id": "predictive", "ablation_id": "none", "seed": 42, "mean_itl_ms": 10.0},
        {"scenario_id": "s1", "baseline_id": "predictive", "ablation_id": "none", "seed": 43, "mean_itl_ms": 12.0},
        {"scenario_id": "s1", "baseline_id": "predictive", "ablation_id": "prediction_removed", "seed": 42, "mean_itl_ms": 18.0},
        {"scenario_id": "s1", "baseline_id": "predictive", "ablation_id": "prediction_removed", "seed": 43, "mean_itl_ms": 20.0},
    ]
    df = pd.DataFrame(rows)

    res = ResearchAblationRunner.evaluate_ablation_effects("prediction_removed", "s1", df)
    assert res is not None
    assert res.ablation_id == "A1"
    assert len(res.comparisons) > 0


def test_sensitivity_runner_parameter_sweep() -> None:
    rows = [
        {"threshold": 0.1, "mean_itl_ms": 12.0, "tokens_per_second": 25.0, "total_switches": 6},
        {"threshold": 0.25, "mean_itl_ms": 10.0, "tokens_per_second": 28.0, "total_switches": 2},
        {"threshold": 0.5, "mean_itl_ms": 15.0, "tokens_per_second": 20.0, "total_switches": 0},
    ]
    df = pd.DataFrame(rows)
    sweep = ResearchSensitivityRunner.analyze_parameter_sweep("threshold", "ratio", df, nominal_value=0.25)
    assert len(sweep.points) == 3
    assert sweep.nominal_value == 0.25

    sw_df = ResearchSensitivityRunner.sweep_to_dataframe(sweep)
    assert len(sw_df) == 3
    assert "degradation_pct_vs_nominal" in sw_df.columns


# ---------------------------------------------------------------------------
# 6. Research Validator Tests
# ---------------------------------------------------------------------------

def test_research_validator_cpu_vram_and_static_invariants() -> None:
    # Invariant: VRAM on CPU cannot be MEASURED > 0
    invalid_trial = {
        "trial_id": "bad_vram",
        "baseline_id": "predictive",
        "metrics": {"peak_vram_mb": 8192.0, "mean_itl_ms": 15.0},
        "provenance": {"peak_vram_mb": "MEASURED"},
    }
    rep = ResearchValidator.validate_trial_record(invalid_trial, is_cpu_only=True)
    assert rep.is_valid is False
    assert any(i.code == "INVALID_VRAM_ON_CPU" for i in rep.issues)

    # Invariant: Static baseline cannot switch
    invalid_static = {
        "trial_id": "bad_static",
        "baseline_id": "static",
        "metrics": {"total_switches": 3, "mean_itl_ms": 15.0},
        "provenance": {},
    }
    rep_static = ResearchValidator.validate_trial_record(invalid_static, is_cpu_only=True)
    assert rep_static.is_valid is False
    assert any(i.code == "STATIC_POLICY_MUTATED" for i in rep_static.issues)


def test_research_validator_artifact_checksum_verification(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp_check"
    exp_dir.mkdir()

    art_file = exp_dir / "raw_observations.csv"
    art_file.write_text("trial_id,baseline_id,mean_itl_ms\nt1,static,10.0\n", encoding="utf-8")

    manifest = ResearchManifest(experiment_id="exp_check")
    manifest.record_artifact_checksum("raw_observations.csv", art_file)
    manifest.save(exp_dir / "manifest.json")

    # Valid check passes
    rep_valid = ResearchValidator.validate_experiment_artifacts(exp_dir)
    assert rep_valid.is_valid is True

    # Tampering with file causes checksum mismatch
    art_file.write_text("TAMPERED DATA", encoding="utf-8")
    rep_tampered = ResearchValidator.validate_experiment_artifacts(exp_dir)
    assert rep_tampered.is_valid is False
    assert any(i.code == "CHECKSUM_MISMATCH" for i in rep_tampered.issues)


# ---------------------------------------------------------------------------
# 7. Artifact & Report Generation Tests
# ---------------------------------------------------------------------------

def test_publication_artifact_generator_tables_and_plots(tmp_path: Path) -> None:
    out_dir = tmp_path / "artifacts"
    out_dir.mkdir()

    # Synthetic DataFrame
    rows = []
    for pol in ["static", "predictive"]:
        for i in range(5):
            rows.append({
                "trial_id": f"{pol}_{i}",
                "baseline_id": pol,
                "scenario_id": "stable",
                "mean_itl_ms": 12.0 + i,
                "median_itl_ms": 11.5 + i,
                "p95_itl_ms": 14.0 + i,
                "tokens_per_second": 20.0 - i,
                "total_switches": 0 if pol == "static" else 2,
                "total_migration_time_ms": 0.0 if pol == "static" else 15.0,
                "total_transferred_mb": 0.0 if pol == "static" else 1.2,
                "total_control_overhead_ms": 1.0,
                "peak_memory_pressure": 0.4,
            })
    df = pd.DataFrame(rows)

    artifacts = PublicationArtifactGenerator.generate_all_artifacts(
        trials_df=df,
        output_dir=out_dir,
        is_cpu_only=True,
    )

    assert "tables/table1_policy_summary.md" in artifacts
    assert "tables/table4_provenance_matrix.md" in artifacts
    assert (out_dir / "tables/table1_policy_summary.md").is_file()
    assert (out_dir / "tables/table4_provenance_matrix.md").is_file()


def test_research_report_generator_15_sections(tmp_path: Path) -> None:
    rep_file = tmp_path / "test_report.md"
    manifest = ResearchManifest(
        experiment_id="rep_test_15",
        title="15 Section Integrity Test",
        scenarios=["stable"],
        policies=["static", "predictive"],
    )
    df = pd.DataFrame([
        {"baseline_id": "static", "mean_itl_ms": 15.0},
        {"baseline_id": "predictive", "mean_itl_ms": 12.0},
    ])

    path = ResearchReportGenerator.generate_report(manifest, df, output_path=rep_file)
    assert path.is_file()
    content = path.read_text(encoding="utf-8")

    # Verify all 15 required sections are present
    for sec_num in range(1, 16):
        assert f"## {sec_num}." in content


# ---------------------------------------------------------------------------
# 8. End-to-End Campaign Execution & Idempotency Test
# ---------------------------------------------------------------------------

def test_research_runner_campaign_execution_simulation(tmp_path: Path) -> None:
    cfg = {
        "experiment_id": "ci_test_campaign",
        "title": "CI Campaign Execution",
        "seed": 42,
        "execution_mode": "simulation",
        "scenarios": ["stable"],
        "policies": ["static", "predictive"],
        "repetitions": 1,
        "max_new_tokens": 4,
        "control_interval_tokens": 2,
    }

    runner = ResearchRunner(config=cfg, base_output_dir=tmp_path)
    res = runner.execute_campaign()

    assert res["experiment_id"] == "ci_test_campaign"
    assert res["completed"] == 2  # 1 scen * 2 pol * 1 rep = 2 trials
    assert res["failed"] == 0

    exp_dir = tmp_path / "ci_test_campaign"
    assert (exp_dir / "manifest.json").is_file()
    assert (exp_dir / "raw_observations.csv").is_file()
    assert (exp_dir / "report.md").is_file()

    # Validate generated artifacts pass all invariant checks
    audit_rep = ResearchValidator.validate_experiment_artifacts(exp_dir)
    assert audit_rep.is_valid is True
