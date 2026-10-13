"""
Research Reproducibility Layer (Module 12).

Provides deterministic seed management, environment fingerprinting, and
configuration snapshot hashing. Guarantees that every experiment trial can be
independently replicated or audited.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False


class SeedManager:
    """
    Hierarchical deterministic random seed derivation and runtime state management.

    Generates reproducible per-repetition and per-task seeds derived from a
    root experiment seed using cryptographic hashing, ensuring no correlation
    between parallel or sequential trial streams.
    """

    def __init__(self, root_seed: int = 42) -> None:
        self.root_seed = int(root_seed)

    def derive_seed(self, component: str, repetition: int = 0, subtask: Optional[str] = None) -> int:
        """
        Derive a 32-bit positive integer seed deterministically from the root seed.
        """
        key = f"{self.root_seed}:{component}:{repetition}:{subtask or ''}"
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        # Return positive 31-bit integer (compatible with Python, NumPy, PyTorch)
        derived = int.from_bytes(digest[:4], byteorder="big") & 0x7FFFFFFF
        return derived

    def set_global_seeds(self, seed: int) -> Dict[str, Any]:
        """
        Set deterministic random seeds across Python, NumPy, and PyTorch (if available).
        Returns metadata documenting applied determinism settings.
        """
        random.seed(seed)
        np.random.seed(seed)

        torch_seeded = False
        cuda_deterministic = False

        if torch is not None:
            torch.manual_seed(seed)
            torch_seeded = True
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                # Attempt to enforce deterministic algorithms
                try:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                    cuda_deterministic = True
                except Exception:
                    pass

        return {
            "applied_seed": seed,
            "python_random_seeded": True,
            "numpy_seeded": True,
            "torch_seeded": torch_seeded,
            "cuda_deterministic": cuda_deterministic,
        }


@dataclass
class EnvironmentFingerprint:
    """
    Comprehensive snapshot of host hardware, operating system, and software dependencies.
    """
    python_version: str = field(default_factory=lambda: sys.version.replace("\n", " "))
    python_executable: str = field(default_factory=lambda: sys.executable)
    platform_system: str = field(default_factory=platform.system)
    platform_release: str = field(default_factory=platform.release)
    platform_machine: str = field(default_factory=platform.machine)
    processor: str = field(default_factory=platform.processor)
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None
    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_device_names: List[str] = field(default_factory=list)
    host_ram_total_mb: Optional[float] = None
    host_ram_available_mb: Optional[float] = None
    cpu_count_logical: Optional[int] = None
    cpu_count_physical: Optional[int] = None
    dependency_versions: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def capture(cls, check_git: bool = True) -> EnvironmentFingerprint:
        """Capture the current environment fingerprint."""
        git_commit = None
        git_dirty = None

        if check_git:
            try:
                import subprocess
                res = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
                if res.returncode == 0:
                    git_commit = res.stdout.strip()

                status_res = subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
                if status_res.returncode == 0:
                    git_dirty = len(status_res.stdout.strip()) > 0
            except Exception:
                git_commit = "unavailable"
                git_dirty = False

        cuda_avail = False
        cuda_count = 0
        cuda_names: List[str] = []
        if torch is not None:
            try:
                cuda_avail = torch.cuda.is_available()
                if cuda_avail:
                    cuda_count = torch.cuda.device_count()
                    cuda_names = [torch.cuda.get_device_name(i) for i in range(cuda_count)]
            except Exception:
                pass

        ram_total = None
        ram_avail = None
        cpu_log = os.cpu_count()
        cpu_phys = None
        if PSUTIL_AVAILABLE and psutil is not None:
            try:
                vm = psutil.virtual_memory()
                ram_total = round(vm.total / (1024 * 1024), 2)
                ram_avail = round(vm.available / (1024 * 1024), 2)
                cpu_phys = psutil.cpu_count(logical=False)
            except Exception:
                pass

        # Capture key dependencies
        deps: Dict[str, str] = {}
        for mod_name in ["torch", "transformers", "accelerate", "numpy", "pandas", "scipy", "psutil", "matplotlib", "pytest"]:
            try:
                mod = sys.modules.get(mod_name)
                if mod is None:
                    mod = __import__(mod_name)
                deps[mod_name] = getattr(mod, "__version__", "unknown")
            except Exception:
                deps[mod_name] = "not_installed"

        return cls(
            git_commit=git_commit,
            git_dirty=git_dirty,
            cuda_available=cuda_avail,
            cuda_device_count=cuda_count,
            cuda_device_names=cuda_names,
            host_ram_total_mb=ram_total,
            host_ram_available_mb=ram_avail,
            cpu_count_logical=cpu_log,
            cpu_count_physical=cpu_phys,
            dependency_versions=deps,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResearchConfigHasher:
    """Produces canonical, deterministic hashes of configurations and nested dicts."""

    EXCLUDED_KEYS = {"experiment_id", "timestamp", "created_at", "run_id", "secret_key", "auth_token"}

    @classmethod
    def hash_dict(cls, data: Dict[str, Any]) -> str:
        """
        Produce a canonical SHA-256 hash from a dictionary after stripping
        ephemeral and sensitive fields.
        """
        def _sanitize(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in sorted(obj.items()) if k not in cls.EXCLUDED_KEYS}
            elif isinstance(obj, (list, tuple)):
                return [_sanitize(x) for x in obj]
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            return str(obj)

        sanitized = _sanitize(data)
        encoded = json.dumps(sanitized, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def hash_file(cls, path: Path | str) -> str:
        """Calculate SHA-256 checksum of an arbitrary file."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
