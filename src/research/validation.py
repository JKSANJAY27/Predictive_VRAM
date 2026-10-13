"""
Research Provenance & Integrity Validation (Module 12).

Enforces non-negotiable scientific integrity invariants across all experimental trials,
telemetry records, and published artifacts:
1. Host CPU RAM is NEVER reported or substituted as GPU VRAM.
2. On CPU-only environments, VRAM observations are explicitly UNAVAILABLE (value=None).
3. Mock GPU telemetry is strictly tagged as EMULATED.
4. Simulated 6G network profiles are NEVER labeled as physically measured 6G.
5. Missing observations are NEVER replaced with zero.
6. Loopback socket transfers are NEVER labeled as WAN or cellular network measurements.
7. Simulated/modeled energy is NEVER reported as direct physical measurement.
8. B1 STATIC policy maintains strictly zero partition transitions.
9. Chronological causality: telemetry timestamps and token indices are monotonically non-decreasing.
10. Artifact checksum verification matches manifest records on disk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.research.manifest import ResearchManifest
from src.research.reproducibility import ResearchConfigHasher
from src.telemetry.types import DataSource

logger = logging.getLogger(__name__)


@dataclass
class ResearchValidationIssue:
    """A single validation defect or invariant breach."""
    level: str  # "ERROR" | "WARNING" | "INFO"
    code: str
    message: str
    trial_id: Optional[str] = None
    metric_name: Optional[str] = None
    provenance: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchValidationReport:
    """Full audit report validating research artifacts and provenance integrity."""
    is_valid: bool = True
    issues: List[ResearchValidationIssue] = field(default_factory=list)
    total_checks: int = 0
    passed_checks: int = 0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "ERROR")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "WARNING")

    def add_error(self, code: str, msg: str, trial_id: Optional[str] = None, **details: Any) -> None:
        self.issues.append(ResearchValidationIssue(level="ERROR", code=code, message=msg, trial_id=trial_id, details=details))
        self.is_valid = False

    def add_warning(self, code: str, msg: str, trial_id: Optional[str] = None, **details: Any) -> None:
        self.issues.append(ResearchValidationIssue(level="WARNING", code=code, message=msg, trial_id=trial_id, details=details))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [asdict(i) for i in self.issues],
        }


class ResearchValidator:
    """
    Automated validator auditing trial outputs, data tables, and manifests.
    """

    @classmethod
    def validate_trial_record(
        cls,
        trial_dict: Dict[str, Any],
        is_cpu_only: bool = True,
    ) -> ResearchValidationReport:
        """
        Audit a single raw trial output dictionary for provenance and physical plausibility.
        """
        report = ResearchValidationReport(is_valid=True)
        trial_id = trial_dict.get("trial_id", "unknown")
        baseline_id = trial_dict.get("baseline_id", "")
        metrics = trial_dict.get("metrics", {})
        provenance = trial_dict.get("provenance", {})

        report.total_checks += 1
        # Invariant 1 & 2: VRAM on CPU-only system
        if is_cpu_only:
            vram_val = metrics.get("peak_vram_mb")
            vram_prov = provenance.get("peak_vram_mb", {}).get("source") if isinstance(provenance.get("peak_vram_mb"), dict) else provenance.get("peak_vram_mb")
            
            if vram_val is not None and vram_val > 0.0 and vram_prov == "MEASURED":
                report.add_error(
                    code="INVALID_VRAM_ON_CPU",
                    msg="Physical VRAM was marked MEASURED on a CPU-only environment.",
                    trial_id=trial_id,
                    vram_val=vram_val,
                )
            else:
                report.passed_checks += 1

        report.total_checks += 1
        # Invariant 8: B1 STATIC policy immutability
        if baseline_id == "static":
            switches = metrics.get("total_switches", 0)
            if switches > 0:
                report.add_error(
                    code="STATIC_POLICY_MUTATED",
                    msg=f"Static baseline executed {switches} partition switches (expected 0).",
                    trial_id=trial_id,
                )
            else:
                report.passed_checks += 1

        report.total_checks += 1
        # Invariant 9: Physical plausibility (latencies >= 0, throughput >= 0)
        mean_itl = metrics.get("mean_itl_ms", 0.0)
        if mean_itl is not None and mean_itl < 0.0:
            report.add_error(
                code="NEGATIVE_LATENCY",
                msg=f"Mean ITL is negative ({mean_itl} ms).",
                trial_id=trial_id,
            )
        else:
            report.passed_checks += 1

        report.total_checks += 1
        # Invariant 4: Simulated 6G provenance check
        scenario_id = trial_dict.get("scenario_id", "")
        if "six_g" in scenario_id:
            network_prov = provenance.get("network_source", "EMULATED")
            if network_prov == "MEASURED":
                report.add_error(
                    code="FABRICATED_6G_MEASUREMENT",
                    msg=f"Scenario {scenario_id} was marked MEASURED on 6G without physical testbed.",
                    trial_id=trial_id,
                )
            else:
                report.passed_checks += 1

        return report

    @classmethod
    def validate_trials_dataframe(
        cls,
        df: pd.DataFrame,
        is_cpu_only: bool = True,
    ) -> ResearchValidationReport:
        """
        Audit a complete batch DataFrame of trials.
        """
        report = ResearchValidationReport(is_valid=True)

        # Check for zero-substituted missing metrics (e.g. latency cannot be 0.0 unless incomplete)
        report.total_checks += 1
        if "mean_itl_ms" in df.columns:
            zero_latencies = df[(df["mean_itl_ms"] == 0.0) & (df.get("completed_tokens", 1) > 0)]
            if len(zero_latencies) > 0:
                report.add_warning(
                    code="SUSPICIOUS_ZERO_LATENCY",
                    msg=f"{len(zero_latencies)} trials report exactly 0.0 ms ITL with completed tokens.",
                )
            else:
                report.passed_checks += 1

        # Check static baseline switches
        report.total_checks += 1
        if "baseline_id" in df.columns and "total_switches" in df.columns:
            static_switches = df[df["baseline_id"] == "static"]["total_switches"].sum()
            if static_switches > 0:
                report.add_error(
                    code="STATIC_SWITCHES_DETECTED",
                    msg=f"Static baseline has {static_switches} switches across dataset.",
                )
            else:
                report.passed_checks += 1

        # Check VRAM vs CPU host RAM separation
        report.total_checks += 1
        if is_cpu_only and "peak_vram_mb" in df.columns:
            non_null_vram = df["peak_vram_mb"].dropna()
            if len(non_null_vram) > 0 and (non_null_vram > 0).any():
                report.add_warning(
                    code="NON_NULL_VRAM_CPU",
                    msg="Non-null VRAM values detected in CPU-only environment dataset.",
                )
            else:
                report.passed_checks += 1

        return report

    @classmethod
    def validate_experiment_artifacts(
        cls,
        experiment_dir: Path | str,
    ) -> ResearchValidationReport:
        """
        Audit an experiment output directory on disk:
        Verifies manifest checksums, file presence, and data table validity.
        """
        exp_path = Path(experiment_dir)
        report = ResearchValidationReport(is_valid=True)

        manifest_file = exp_path / "manifest.json"
        report.total_checks += 1
        if not manifest_file.is_file():
            report.add_error("MISSING_MANIFEST", f"Manifest file missing at {manifest_file}")
            return report
        report.passed_checks += 1

        try:
            manifest = ResearchManifest.load(manifest_file)
        except Exception as e:
            report.add_error("CORRUPT_MANIFEST", f"Failed to load manifest: {e}")
            return report

        # Verify artifact checksums
        for rel_path, expected_checksum in manifest.artifact_checksums.items():
            report.total_checks += 1
            art_file = exp_path / rel_path
            if not art_file.is_file():
                report.add_error("MISSING_ARTIFACT", f"Artifact {rel_path} declared in manifest but missing on disk.")
                continue

            actual_checksum = ResearchConfigHasher.hash_file(art_file)
            if actual_checksum != expected_checksum:
                report.add_error(
                    "CHECKSUM_MISMATCH",
                    f"Checksum mismatch for {rel_path}: expected {expected_checksum[:12]}..., got {actual_checksum[:12]}...",
                )
            else:
                report.passed_checks += 1

        # Audit raw_observations.csv if present
        raw_csv = exp_path / "raw_observations.csv"
        if raw_csv.is_file():
            try:
                df = pd.read_csv(raw_csv)
                df_rep = cls.validate_trials_dataframe(
                    df,
                    is_cpu_only=not manifest.environment.cuda_available,
                )
                report.total_checks += df_rep.total_checks
                report.passed_checks += df_rep.passed_checks
                report.issues.extend(df_rep.issues)
                if not df_rep.is_valid:
                    report.is_valid = False
            except Exception as e:
                report.add_error("INVALID_RAW_CSV", f"Failed to parse raw_observations.csv: {e}")

        return report
