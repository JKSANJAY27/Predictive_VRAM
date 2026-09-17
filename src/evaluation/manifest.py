"""
Experiment manifest, duplicate run guard, and config hashing for Module 10.

Ensures experimental immutability, provenance tracking, and deduplication:
  - ExperimentManifest: tracks all trials planned, running, and completed
  - DuplicateRunGuard: prevents accidental overwriting of completed runs
  - ConfigHasher: produces deterministic, canonical hashes for configs
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from src.evaluation.experiment_config import ExperimentConfig, TrialConfig


class ConfigHasher:
    """Produces deterministic SHA-256 hashes from configs and dictionaries."""

    @staticmethod
    def hash_dict(d: Dict[str, Any], keys_to_ignore: Optional[Sequence[str]] = None) -> str:
        """Hash dictionary content after sorting keys and stripping volatile fields."""
        ignore = set(keys_to_ignore or ["experiment_id", "timestamp", "created_at", "run_id"])
        clean_d = {k: v for k, v in d.items() if k not in ignore}
        serialized = json.dumps(clean_d, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def hash_experiment_config(config: ExperimentConfig) -> str:
        """Derive 16-char hex digest from an ExperimentConfig."""
        return ConfigHasher.hash_dict(config.to_dict())[:16]

    @staticmethod
    def hash_trial_config(trial: TrialConfig) -> str:
        """Derive 24-char hex digest from a TrialConfig."""
        d = trial.to_dict()
        d.pop("trial_id", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:24]


class DuplicateRunGuard:
    """
    Guards trial result files against accidental overwriting.

    Result files stored in results/raw/<trial_id>.json are immutable.
    """

    def __init__(self, raw_results_dir: Path | str = "results/raw") -> None:
        self.raw_dir = Path(raw_results_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def is_completed(self, trial_id: str) -> bool:
        """Check whether trial_id has a non-empty, completed output file."""
        trial_path = self.raw_dir / f"{trial_id}.json"
        if not trial_path.is_file():
            return False
        try:
            return trial_path.stat().st_size > 32
        except OSError:
            return False

    def can_run(self, trial_id: str, force: bool = False) -> bool:
        """Return True if trial should be executed, False if it can be skipped."""
        if force:
            return True
        return not self.is_completed(trial_id)

    def get_completed_trial_ids(self) -> Set[str]:
        """Scan directory and return all completed trial IDs."""
        completed = set()
        if not self.raw_dir.is_dir():
            return completed
        for p in self.raw_dir.glob("*.json"):
            try:
                if p.stat().st_size > 32:
                    completed.add(p.stem)
            except OSError:
                continue
        return completed


@dataclass
class ManifestTrialEntry:
    """Single trial record within an ExperimentManifest."""
    trial_id: str
    scenario_id: str
    baseline_id: str
    seed: int
    repetition: int
    status: str = "planned"  # "planned" | "running" | "completed" | "failed" | "skipped"
    output_path: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_s: Optional[float] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentManifest:
    """
    Execution manifest logging all trials, environment specifications,
    and completion statuses for an experiment.
    """
    schema_version: str = "1.0"
    experiment_id: str = ""
    name: str = ""
    created_at: float = field(default_factory=time.time)
    status: str = "planned"  # "planned" | "running" | "completed" | "partial"
    environment: Dict[str, Any] = field(default_factory=dict)
    trials: Dict[str, ManifestTrialEntry] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.environment:
            self.environment = self._capture_environment()

    @staticmethod
    def _capture_environment() -> Dict[str, Any]:
        """Record system environment for experimental provenance."""
        env = {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }
        try:
            import torch
            env["torch_version"] = torch.__version__
            env["cuda_available"] = torch.cuda.is_available()
        except ImportError:
            env["torch_version"] = "unavailable"
            env["cuda_available"] = False
        return env

    @classmethod
    def create_from_config(cls, config: ExperimentConfig) -> ExperimentManifest:
        """Create a planned manifest from an ExperimentConfig."""
        from src.evaluation.experiment_config import ExperimentMatrix

        matrix = ExperimentMatrix(config)
        trial_configs = matrix.expand(
            include_ablations=bool(config.ablations),
            include_sensitivity=False,
        )

        manifest = cls(
            experiment_id=config.experiment_id,
            name=config.name,
        )
        for tc in trial_configs:
            manifest.trials[tc.trial_id] = ManifestTrialEntry(
                trial_id=tc.trial_id,
                scenario_id=tc.scenario_id,
                baseline_id=tc.baseline_id,
                seed=tc.seed,
                repetition=tc.repetition,
                status="planned",
            )
        return manifest

    def register_trial_start(self, trial_id: str) -> None:
        if trial_id in self.trials:
            entry = self.trials[trial_id]
            entry.status = "running"
            entry.started_at = time.time()

    def register_trial_completed(self, trial_id: str, output_path: str) -> None:
        if trial_id in self.trials:
            entry = self.trials[trial_id]
            entry.status = "completed"
            entry.completed_at = time.time()
            if entry.started_at:
                entry.duration_s = entry.completed_at - entry.started_at
            entry.output_path = str(output_path)

    def register_trial_failed(self, trial_id: str, error_msg: str) -> None:
        if trial_id in self.trials:
            entry = self.trials[trial_id]
            entry.status = "failed"
            entry.completed_at = time.time()
            entry.error_message = str(error_msg)

    def register_trial_skipped(self, trial_id: str, output_path: Optional[str] = None) -> None:
        if trial_id in self.trials:
            entry = self.trials[trial_id]
            entry.status = "skipped"
            entry.output_path = output_path

    def update_overall_status(self) -> str:
        statuses = [t.status for t in self.trials.values()]
        if not statuses:
            self.status = "planned"
        elif all(s == "completed" for s in statuses):
            self.status = "completed"
        elif any(s in ("completed", "running") for s in statuses):
            self.status = "partial"
        else:
            self.status = "planned"
        return self.status

    def to_dict(self) -> Dict[str, Any]:
        self.update_overall_status()
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "created_at": self.created_at,
            "status": self.status,
            "environment": self.environment,
            "trial_count": len(self.trials),
            "trials": {k: v.to_dict() for k, v in self.trials.items()},
        }

    def save_json(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path | str) -> ExperimentManifest:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        manifest = cls(
            schema_version=d.get("schema_version", "1.0"),
            experiment_id=d.get("experiment_id", ""),
            name=d.get("name", ""),
            created_at=d.get("created_at", time.time()),
            status=d.get("status", "planned"),
            environment=d.get("environment", {}),
        )
        for tid, t_dict in d.get("trials", {}).items():
            manifest.trials[tid] = ManifestTrialEntry(
                trial_id=t_dict.get("trial_id", tid),
                scenario_id=t_dict.get("scenario_id", ""),
                baseline_id=t_dict.get("baseline_id", ""),
                seed=t_dict.get("seed", 0),
                repetition=t_dict.get("repetition", 0),
                status=t_dict.get("status", "planned"),
                output_path=t_dict.get("output_path"),
                started_at=t_dict.get("started_at"),
                completed_at=t_dict.get("completed_at"),
                duration_s=t_dict.get("duration_s"),
                error_message=t_dict.get("error_message"),
            )
        return manifest
