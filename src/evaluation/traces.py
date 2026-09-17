"""
Environmental trace abstractions for Module 10 experimental evaluation.

Provides typed, serialisable trace containers for:
  NetworkTrace   — per-step network conditions (BW, RTT, loss, jitter)
  MemoryTrace    — per-step memory state (used MB, capacity, KV bytes)
  ComputeTrace   — per-step CPU/edge load fractions
  WorkloadTrace  — prompt + generation configuration
  CombinedEnvironmentTrace — all four with shared timestamps

Design invariant:
  A CombinedEnvironmentTrace is generated ONCE per (scenario_id, seed) and
  serialised. All five baselines replay the IDENTICAL trace — environmental
  randomness is NOT re-sampled per-baseline.

Provenance:
  All traces generated on this CPU-only machine are labelled
  DataSource.EMULATED for VRAM-related values.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Per-step records
# ---------------------------------------------------------------------------

@dataclass
class NetworkStep:
    """One discrete network observation."""
    timestamp: float
    bandwidth_mbps: float
    rtt_ms: float
    packet_loss: float        # fraction [0, 1]
    jitter_ms: float
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "bandwidth_mbps": self.bandwidth_mbps,
            "rtt_ms": self.rtt_ms,
            "packet_loss": self.packet_loss,
            "jitter_ms": self.jitter_ms,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> NetworkStep:
        return cls(
            timestamp=float(d["timestamp"]),
            bandwidth_mbps=float(d["bandwidth_mbps"]),
            rtt_ms=float(d["rtt_ms"]),
            packet_loss=float(d["packet_loss"]),
            jitter_ms=float(d["jitter_ms"]),
            notes=d.get("notes", ""),
        )


@dataclass
class MemoryStep:
    """One discrete memory observation."""
    timestamp: float
    used_mb: float
    capacity_mb: float
    kv_cache_bytes: int
    provenance: str = "emulated"   # "measured" | "emulated" | "estimated"
    notes: str = ""

    @property
    def pressure(self) -> float:
        if self.capacity_mb <= 0:
            return 0.0
        return self.used_mb / self.capacity_mb

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "used_mb": self.used_mb,
            "capacity_mb": self.capacity_mb,
            "kv_cache_bytes": self.kv_cache_bytes,
            "provenance": self.provenance,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MemoryStep:
        return cls(
            timestamp=float(d["timestamp"]),
            used_mb=float(d["used_mb"]),
            capacity_mb=float(d["capacity_mb"]),
            kv_cache_bytes=int(d["kv_cache_bytes"]),
            provenance=d.get("provenance", "emulated"),
            notes=d.get("notes", ""),
        )


@dataclass
class ComputeStep:
    """One discrete compute-load observation."""
    timestamp: float
    cpu_fraction: float        # [0, 1]
    edge_load_fraction: float  # [0, 1]
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cpu_fraction": self.cpu_fraction,
            "edge_load_fraction": self.edge_load_fraction,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ComputeStep:
        return cls(
            timestamp=float(d["timestamp"]),
            cpu_fraction=float(d["cpu_fraction"]),
            edge_load_fraction=float(d["edge_load_fraction"]),
            notes=d.get("notes", ""),
        )


# ---------------------------------------------------------------------------
# Top-level trace containers
# ---------------------------------------------------------------------------

@dataclass
class NetworkTrace:
    """Complete network-condition time series for one trial."""
    steps: List[NetworkStep] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": [s.to_dict() for s in self.steps]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> NetworkTrace:
        return cls(steps=[NetworkStep.from_dict(s) for s in d.get("steps", [])])


@dataclass
class MemoryTrace:
    """Complete memory time series for one trial."""
    steps: List[MemoryStep] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": [s.to_dict() for s in self.steps]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MemoryTrace:
        return cls(steps=[MemoryStep.from_dict(s) for s in d.get("steps", [])])


@dataclass
class ComputeTrace:
    """Complete compute-load time series for one trial."""
    steps: List[ComputeStep] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": [s.to_dict() for s in self.steps]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ComputeTrace:
        return cls(steps=[ComputeStep.from_dict(s) for s in d.get("steps", [])])


@dataclass
class WorkloadTrace:
    """Prompt + generation configuration for a trial (constant across baselines)."""
    prompt: str = "The quick brown fox"
    max_new_tokens: int = 32
    temperature: float = 0.0
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> WorkloadTrace:
        return cls(
            prompt=d.get("prompt", "The quick brown fox"),
            max_new_tokens=int(d.get("max_new_tokens", 32)),
            temperature=float(d.get("temperature", 0.0)),
            seed=int(d.get("seed", 42)),
        )


@dataclass
class CombinedEnvironmentTrace:
    """
    Single unified environmental trace shared across all baselines in a comparison.

    Invariant: generate once, replay identically through B1-B5.
    Timestamps are guaranteed monotonically non-decreasing.
    """
    scenario_id: str
    seed: int
    sampling_interval_s: float
    n_steps: int
    provenance: str                         # "synthetic" | "replay" | "recorded"
    network: NetworkTrace = field(default_factory=NetworkTrace)
    memory: MemoryTrace   = field(default_factory=MemoryTrace)
    compute: ComputeTrace = field(default_factory=ComputeTrace)
    workload: WorkloadTrace = field(default_factory=WorkloadTrace)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "sampling_interval_s": self.sampling_interval_s,
            "n_steps": self.n_steps,
            "provenance": self.provenance,
            "network": self.network.to_dict(),
            "memory": self.memory.to_dict(),
            "compute": self.compute.to_dict(),
            "workload": self.workload.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CombinedEnvironmentTrace:
        return cls(
            scenario_id=d["scenario_id"],
            seed=int(d["seed"]),
            sampling_interval_s=float(d["sampling_interval_s"]),
            n_steps=int(d["n_steps"]),
            provenance=d.get("provenance", "synthetic"),
            network=NetworkTrace.from_dict(d.get("network", {})),
            memory=MemoryTrace.from_dict(d.get("memory", {})),
            compute=ComputeTrace.from_dict(d.get("compute", {})),
            workload=WorkloadTrace.from_dict(d.get("workload", {})),
            metadata=d.get("metadata", {}),
        )

    def save_json(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path | str) -> CombinedEnvironmentTrace:
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def content_hash(self) -> str:
        """SHA-256 of the serialised trace (for integrity verification)."""
        raw = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def validate_timestamps(self) -> bool:
        """Assert that all three sub-traces have monotone timestamps."""
        def mono(seq: Sequence[float]) -> bool:
            return all(b >= a for a, b in zip(seq, seq[1:]))

        ok = True
        if self.network.steps:
            ok &= mono([s.timestamp for s in self.network.steps])
        if self.memory.steps:
            ok &= mono([s.timestamp for s in self.memory.steps])
        if self.compute.steps:
            ok &= mono([s.timestamp for s in self.compute.steps])
        return ok


# ---------------------------------------------------------------------------
# Deterministic trace generator
# ---------------------------------------------------------------------------

class TraceGenerator:
    """
    Generates a CombinedEnvironmentTrace from a ScenarioCatalog entry and seed.

    All randomness is derived from the supplied seed — no hidden global state.
    """

    # Memory emulation parameters
    _BASE_MEMORY_MB   = 2048.0
    _CAPACITY_MB      = 4096.0
    _KV_BYTES_PER_TOK = 512

    @classmethod
    def generate(
        cls,
        scenario_id: str,
        seed: int,
        n_steps: int = 64,
        sampling_interval_s: float = 0.25,
        prompt: str = "The quick brown fox",
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> CombinedEnvironmentTrace:
        """
        Deterministically generate a CombinedEnvironmentTrace.

        Uses the existing SyntheticScenario generator for network conditions
        and derives memory/compute from the scenario profile.
        """
        from src.orchestration.scenarios import ScenarioRegistry

        # Seed a local RNG — does not touch global numpy/torch state
        rng = random.Random(seed)

        try:
            scenario = ScenarioRegistry.get(scenario_id)
        except KeyError:
            # Fall back to stable if unknown
            scenario = ScenarioRegistry.get("stable")

        net_steps: List[NetworkStep] = []
        mem_steps: List[MemoryStep] = []
        cmp_steps: List[ComputeStep] = []

        t = 0.0
        base_kv = 0

        for i in range(n_steps):
            # Network from existing scenario generator
            cond = scenario.get_conditions(i, n_steps)

            # Add small deterministic jitter to make traces distinguishable
            noise = rng.uniform(-0.01, 0.01)

            net_steps.append(NetworkStep(
                timestamp=round(t, 6),
                bandwidth_mbps=max(0.1, cond.bandwidth_mbps + noise),
                rtt_ms=max(0.5, cond.rtt_ms),
                packet_loss=min(1.0, max(0.0, cond.packet_loss)),
                jitter_ms=max(0.0, cond.jitter_ms),
                notes=cond.notes,
            ))

            # Memory: emulated growth tied to sequence length
            kv_growth = cls._KV_BYTES_PER_TOK * i
            # Scenario-aware growth multiplier
            growth_factor = cls._memory_growth_factor(scenario_id, i, n_steps)
            used_mb = min(
                cls._CAPACITY_MB * 0.95,
                cls._BASE_MEMORY_MB + growth_factor * i * 0.5,
            )
            mem_steps.append(MemoryStep(
                timestamp=round(t, 6),
                used_mb=round(used_mb, 2),
                capacity_mb=cls._CAPACITY_MB,
                kv_cache_bytes=kv_growth,
                provenance="emulated",
                notes=f"step_{i}",
            ))

            # Compute: edge load from scenario
            edge_load = cls._edge_load(scenario_id, i, n_steps)
            cpu_frac  = 0.2 + rng.uniform(0.0, 0.1)
            cmp_steps.append(ComputeStep(
                timestamp=round(t, 6),
                cpu_fraction=min(1.0, round(cpu_frac, 3)),
                edge_load_fraction=round(edge_load, 3),
                notes=f"step_{i}",
            ))

            t += sampling_interval_s

        trace = CombinedEnvironmentTrace(
            scenario_id=scenario_id,
            seed=seed,
            sampling_interval_s=sampling_interval_s,
            n_steps=n_steps,
            provenance="synthetic",
            network=NetworkTrace(steps=net_steps),
            memory=MemoryTrace(steps=mem_steps),
            compute=ComputeTrace(steps=cmp_steps),
            workload=WorkloadTrace(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed,
            ),
            metadata={
                "generator": "TraceGenerator",
                "scenario_id": scenario_id,
                "seed": seed,
                "n_steps": n_steps,
                "sampling_interval_s": sampling_interval_s,
            },
        )
        return trace

    @staticmethod
    def _memory_growth_factor(scenario_id: str, step: int, total: float) -> float:
        """Return per-step memory growth multiplier for the scenario."""
        p = step / max(1, total - 1)
        if "kv_cache" in scenario_id or "memory" in scenario_id:
            return 2.0 + p * 3.0      # aggressive growth
        if "combined" in scenario_id or "mixed" in scenario_id:
            if p < 0.2:
                return 0.5
            elif p < 0.65:
                return 1.0 + p * 2.0
            else:
                return max(0.3, 2.0 - p * 1.5)
        return 0.5 + p * 0.5          # stable / slow

    @staticmethod
    def _edge_load(scenario_id: str, step: int, total: float) -> float:
        p = step / max(1, total - 1)
        if "edge_compute" in scenario_id or "saturation" in scenario_id:
            return 0.9 if p > 0.33 else 0.2
        if "mixed" in scenario_id or "combined" in scenario_id:
            return 0.2 + p * 0.4
        return 0.2
