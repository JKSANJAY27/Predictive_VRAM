"""
Research Experiment Manifest (Module 12).

Provides immutable, auditable metadata records for complete experimental campaigns.
Captures system environment, resolved configurations, cryptographic hashes,
metric definitions, provenance, and output artifact checksums.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.research.reproducibility import EnvironmentFingerprint, ResearchConfigHasher

logger = logging.getLogger(__name__)


# Standard Metric Definitions and Measurement Units for Module 12
CANONICAL_METRIC_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "ttft_ms": {
        "unit": "milliseconds",
        "category": "Inference Latency",
        "description": "Time to first token generation after request arrival.",
        "aggregation": "mean",
    },
    "mean_itl_ms": {
        "unit": "milliseconds",
        "category": "Inference Latency",
        "description": "Mean inter-token latency across all generated tokens in trial.",
        "aggregation": "mean",
    },
    "median_itl_ms": {
        "unit": "milliseconds",
        "category": "Inference Latency",
        "description": "Median inter-token latency across all generated tokens in trial.",
        "aggregation": "median",
    },
    "p95_itl_ms": {
        "unit": "milliseconds",
        "category": "Inference Latency",
        "description": "95th percentile inter-token latency (tail latency).",
        "aggregation": "p95",
    },
    "p99_itl_ms": {
        "unit": "milliseconds",
        "category": "Inference Latency",
        "description": "99th percentile inter-token latency (tail latency).",
        "aggregation": "p99",
    },
    "end_to_end_latency_ms": {
        "unit": "milliseconds",
        "category": "Inference Latency",
        "description": "Total generation time from prompt start to final token completion.",
        "aggregation": "mean",
    },
    "tokens_per_second": {
        "unit": "tokens/sec",
        "category": "Throughput",
        "description": "Effective generation throughput (total tokens / generation duration).",
        "aggregation": "mean",
    },
    "total_switches": {
        "unit": "count",
        "category": "Adaptation & Stability",
        "description": "Total partition transitions executed during trial.",
        "aggregation": "sum",
    },
    "switch_frequency_per_token": {
        "unit": "switches/token",
        "category": "Adaptation & Stability",
        "description": "Ratio of total switches to completed generation tokens.",
        "aggregation": "mean",
    },
    "total_migration_time_ms": {
        "unit": "milliseconds",
        "category": "Migration Overhead",
        "description": "Cumulative time spent transferring state during partition migrations.",
        "aggregation": "sum",
    },
    "total_transferred_mb": {
        "unit": "megabytes",
        "category": "Migration Overhead",
        "description": "Total tensor state transferred over network during migrations.",
        "aggregation": "sum",
    },
    "slo_violation_rate": {
        "unit": "fraction [0.0-1.0]",
        "category": "SLO Attainment",
        "description": "Fraction of token intervals exceeding the configured SLO target.",
        "aggregation": "mean",
    },
    "total_control_overhead_ms": {
        "unit": "milliseconds",
        "category": "Controller Overhead",
        "description": "Total execution time spent evaluating costs, prediction, and candidate plans.",
        "aggregation": "sum",
    },
    "control_overhead_fraction": {
        "unit": "fraction [0.0-1.0]",
        "category": "Controller Overhead",
        "description": "Ratio of controller overhead to total generation time.",
        "aggregation": "mean",
    },
    "peak_memory_pressure": {
        "unit": "ratio [0.0-1.0]",
        "category": "Memory Safety",
        "description": "Peak ratio of allocated memory to available headroom.",
        "aggregation": "max",
    },
    "peak_vram_mb": {
        "unit": "megabytes",
        "category": "Memory Safety",
        "description": "Peak physical GPU allocated memory (UNAVAILABLE on CPU platforms).",
        "aggregation": "max",
    },
    "host_ram_mb": {
        "unit": "megabytes",
        "category": "Memory Safety",
        "description": "Host physical RAM consumption (reported separately from VRAM).",
        "aggregation": "mean",
    },
    "prediction_mae_latency": {
        "unit": "milliseconds",
        "category": "Prediction Quality",
        "description": "Mean absolute error between forecasted latency and actual observed latency.",
        "aggregation": "mean",
    },
    "prediction_rmse_latency": {
        "unit": "milliseconds",
        "category": "Prediction Quality",
        "description": "Root mean squared error between forecasted latency and actual observed latency.",
        "aggregation": "mean",
    },
}


@dataclass
class ManifestTrialRecord:
    """Record of an individual trial execution within an experiment."""
    trial_id: str
    scenario_id: str
    baseline_id: str
    seed: int
    repetition: int
    ablation_id: str = "none"
    status: str = "planned"  # "planned" | "completed" | "failed" | "skipped"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    execution_duration_sec: Optional[float] = None
    output_artifact_path: Optional[str] = None
    error_message: Optional[str] = None
    error_trace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchManifest:
    """
    Complete, publication-ready experiment manifest.
    """
    schema_version: str = "1.0"
    experiment_id: str = ""
    title: str = "Predictive VRAM & Network-Aware Split Inference Study"
    created_at_utc: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    environment: EnvironmentFingerprint = field(default_factory=EnvironmentFingerprint.capture)
    resolved_config: Dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    root_seed: int = 42
    seed_derivation_method: str = "sha256(root_seed:component:repetition:subtask)"
    scenarios: List[str] = field(default_factory=list)
    policies: List[str] = field(default_factory=list)
    requested_repetitions: int = 1
    completed_repetitions: int = 0
    failed_repetitions: int = 0
    skipped_repetitions: int = 0
    metric_definitions: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: CANONICAL_METRIC_DEFINITIONS.copy()
    )
    data_source_provenance: Dict[str, str] = field(default_factory=dict)
    artifact_checksums: Dict[str, str] = field(default_factory=dict)
    trials: List[ManifestTrialRecord] = field(default_factory=list)
    exceptions: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.config_hash and self.resolved_config:
            self.config_hash = ResearchConfigHasher.hash_dict(self.resolved_config)

    def validate_for_resume(self, new_config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Verify whether an experiment can be resumed with the provided configuration.
        Requires configuration hash to strictly match the original resolved configuration.
        """
        new_hash = ResearchConfigHasher.hash_dict(new_config)
        if new_hash != self.config_hash:
            return (
                False,
                f"Configuration mismatch: Manifest hash is {self.config_hash[:16]} "
                f"but provided configuration hash is {new_hash[:16]}.",
            )
        return True, "Configuration verified: matching hash."

    def get_completed_trial_ids(self) -> Set[str]:
        return {t.trial_id for t in self.trials if t.status == "completed"}

    def add_trial_record(self, record: ManifestTrialRecord) -> None:
        # Update or append trial
        for i, t in enumerate(self.trials):
            if t.trial_id == record.trial_id:
                self.trials[i] = record
                self._update_counts()
                return
        self.trials.append(record)
        self._update_counts()

    def record_artifact_checksum(self, relative_path: str, absolute_path: Path | str) -> None:
        """Compute and store SHA-256 checksum for a generated artifact file."""
        p = Path(absolute_path)
        if p.is_file():
            self.artifact_checksums[relative_path] = ResearchConfigHasher.hash_file(p)

    def _update_counts(self) -> None:
        self.completed_repetitions = sum(1 for t in self.trials if t.status == "completed")
        self.failed_repetitions = sum(1 for t in self.trials if t.status == "failed")
        self.skipped_repetitions = sum(1 for t in self.trials if t.status == "skipped")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["environment"] = self.environment.to_dict()
        d["trials"] = [t.to_dict() for t in self.trials]
        return d

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return target

    @classmethod
    def load(cls, path: Path | str) -> ResearchManifest:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)

        env_dict = d.pop("environment", {})
        env = EnvironmentFingerprint(**env_dict)

        trials_raw = d.pop("trials", [])
        trials = [ManifestTrialRecord(**t) for t in trials_raw]

        manifest = cls(**d, environment=env)
        manifest.trials = trials
        return manifest
