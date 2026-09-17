"""
ScenarioCatalog for Module 10: formal scenario registry with metadata.

Defines 13 scenarios S1-S13 including the canonical combined_degradation
(S9) which aliases the existing mixed_degradation 5-phase scenario.

Each scenario entry has:
  - scenario_id, description, seed, duration, sampling_interval
  - network/memory/compute profile labels
  - phase list (human-readable)
  - provenance ("synthetic")

The underlying generators live in src.orchestration.scenarios; this
catalog adds structured metadata and the three new scenarios that extend it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ScenarioMetadata:
    """Structured metadata for one scenario entry."""
    scenario_id: str
    description: str
    seed: int
    duration_steps: int
    sampling_interval_s: float
    network_profile: str
    memory_profile: str
    compute_profile: str
    workload_profile: str
    provenance: str           # "synthetic" | "emulated" | "replay"
    phases: List[str] = field(default_factory=list)
    tags: List[str]  = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id":        self.scenario_id,
            "description":        self.description,
            "seed":               self.seed,
            "duration_steps":     self.duration_steps,
            "sampling_interval_s": self.sampling_interval_s,
            "network_profile":    self.network_profile,
            "memory_profile":     self.memory_profile,
            "compute_profile":    self.compute_profile,
            "workload_profile":   self.workload_profile,
            "provenance":         self.provenance,
            "phases":             self.phases,
            "tags":               self.tags,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ScenarioMetadata:
        return cls(
            scenario_id=d["scenario_id"],
            description=d["description"],
            seed=int(d.get("seed", 42)),
            duration_steps=int(d.get("duration_steps", 64)),
            sampling_interval_s=float(d.get("sampling_interval_s", 0.25)),
            network_profile=d.get("network_profile", "stable"),
            memory_profile=d.get("memory_profile", "stable"),
            compute_profile=d.get("compute_profile", "low"),
            workload_profile=d.get("workload_profile", "standard"),
            provenance=d.get("provenance", "synthetic"),
            phases=d.get("phases", []),
            tags=d.get("tags", []),
        )


# ---------------------------------------------------------------------------
# Canonical catalog — S1 to S13
# ---------------------------------------------------------------------------

_CATALOG: Dict[str, ScenarioMetadata] = {

    "stable": ScenarioMetadata(
        scenario_id="stable",
        description=(
            "S1 STABLE: High bandwidth (100 Mbps), constant low memory, "
            "low compute. No degradation. Reference baseline scenario."
        ),
        seed=42, duration_steps=64, sampling_interval_s=0.25,
        network_profile="stable_high", memory_profile="stable_flat",
        compute_profile="low", workload_profile="standard",
        provenance="synthetic",
        phases=["stable"],
        tags=["baseline", "reference"],
    ),

    "kv_cache_driven_memory_growth": ScenarioMetadata(
        scenario_id="kv_cache_driven_memory_growth",
        description=(
            "S2 KV_CACHE_DRIVEN_MEMORY_GROWTH: High network bandwidth, "
            "KV-cache grows linearly with sequence length. "
            "Tests memory-pressure response without network stress."
        ),
        seed=43, duration_steps=64, sampling_interval_s=0.25,
        network_profile="stable_high", memory_profile="kv_growth",
        compute_profile="low", workload_profile="standard",
        provenance="synthetic",
        phases=["kv_growth"],
        tags=["memory", "kv_cache"],
    ),

    "linear_memory_growth": ScenarioMetadata(
        scenario_id="linear_memory_growth",
        description=(
            "S3 LINEAR_MEMORY_GROWTH: Stable network. Memory pressure increases "
            "linearly independent of KV cache. Tests memory-aware controller."
        ),
        seed=44, duration_steps=64, sampling_interval_s=0.25,
        network_profile="stable_high", memory_profile="linear_growth",
        compute_profile="low", workload_profile="standard",
        provenance="synthetic",
        phases=["memory_growth"],
        tags=["memory"],
    ),

    "bandwidth_degradation": ScenarioMetadata(
        scenario_id="bandwidth_degradation",
        description=(
            "S4 BANDWIDTH_DEGRADATION: Smooth monotonic ramp-down from 100 to 5 Mbps. "
            "Memory stable. Tests network-aware controller."
        ),
        seed=45, duration_steps=64, sampling_interval_s=0.25,
        network_profile="gradual_degradation", memory_profile="stable_flat",
        compute_profile="low", workload_profile="standard",
        provenance="synthetic",
        phases=["degradation"],
        tags=["network"],
    ),

    "bandwidth_recovery": ScenarioMetadata(
        scenario_id="bandwidth_recovery",
        description=(
            "S5 BANDWIDTH_RECOVERY: Network starts degraded (5 Mbps) and "
            "recovers to 100 Mbps at midpoint. Tests adaptation to improving conditions."
        ),
        seed=46, duration_steps=64, sampling_interval_s=0.25,
        network_profile="recovery", memory_profile="stable_flat",
        compute_profile="low", workload_profile="standard",
        provenance="synthetic",
        phases=["degraded", "recovery"],
        tags=["network", "recovery"],
    ),

    "sudden_bandwidth_drop": ScenarioMetadata(
        scenario_id="sudden_bandwidth_drop",
        description=(
            "S6 SUDDEN_BANDWIDTH_DROP: Step-function drop from 100 to 2 Mbps "
            "at 33% of run. Tests controller reaction speed."
        ),
        seed=47, duration_steps=64, sampling_interval_s=0.25,
        network_profile="sudden_drop", memory_profile="stable_flat",
        compute_profile="low", workload_profile="standard",
        provenance="synthetic",
        phases=["stable", "sudden_drop"],
        tags=["network", "step_change"],
    ),

    "oscillating_bandwidth": ScenarioMetadata(
        scenario_id="oscillating_bandwidth",
        description=(
            "S7 OSCILLATING_BANDWIDTH: Square-wave BW oscillation (4-step period). "
            "Tests hysteresis, cooldown, and thrashing prevention."
        ),
        seed=48, duration_steps=64, sampling_interval_s=0.25,
        network_profile="oscillation", memory_profile="stable_flat",
        compute_profile="low", workload_profile="standard",
        provenance="synthetic",
        phases=["oscillation"],
        tags=["network", "stability", "oscillation"],
    ),

    "edge_compute_saturation": ScenarioMetadata(
        scenario_id="edge_compute_saturation",
        description=(
            "S8 EDGE_COMPUTE_SATURATION: Edge node CPU saturates at 33% of run. "
            "Moderate network (50 Mbps). Tests compute-aware adaptation."
        ),
        seed=49, duration_steps=64, sampling_interval_s=0.25,
        network_profile="moderate", memory_profile="stable_flat",
        compute_profile="saturation", workload_profile="standard",
        provenance="synthetic",
        phases=["pre_saturation", "saturation"],
        tags=["compute", "edge"],
    ),

    "combined_degradation": ScenarioMetadata(
        scenario_id="combined_degradation",
        description=(
            "S9 COMBINED_DEGRADATION (CANONICAL): 5-phase research scenario. "
            "Phase 1: stable. Phase 2: KV growth + BW decline. "
            "Phase 3: predicted threshold breach. "
            "Phase 4: bottleneck sustained. Phase 5: recovery. "
            "Aliases the existing mixed_degradation scenario."
        ),
        seed=42, duration_steps=64, sampling_interval_s=0.25,
        network_profile="mixed_degradation", memory_profile="kv_growth",
        compute_profile="medium", workload_profile="standard",
        provenance="synthetic",
        phases=["stable", "kv_growth+bw_decline", "predicted_risk",
                "bottleneck", "recovery"],
        tags=["canonical", "combined", "research"],
    ),

    "telemetry_dropout": ScenarioMetadata(
        scenario_id="telemetry_dropout",
        description=(
            "S10 TELEMETRY_DROPOUT: Intermittent metric unavailability "
            "(DataSource.UNAVAILABLE injected) during middle 25% of run. "
            "Tests graceful handling of missing observations."
        ),
        seed=50, duration_steps=64, sampling_interval_s=0.25,
        network_profile="stable_high", memory_profile="stable_flat",
        compute_profile="low", workload_profile="standard",
        provenance="synthetic",
        phases=["normal", "dropout", "normal"],
        tags=["robustness", "dropout"],
    ),

    "prediction_error": ScenarioMetadata(
        scenario_id="prediction_error",
        description=(
            "S11 PREDICTION_ERROR: Channel behavior diverges sharply from "
            "linear extrapolation. Tests predictor resilience and graceful "
            "degradation when forecasts are inaccurate."
        ),
        seed=51, duration_steps=64, sampling_interval_s=0.25,
        network_profile="chaotic", memory_profile="stable_flat",
        compute_profile="low", workload_profile="standard",
        provenance="synthetic",
        phases=["gradual_decline", "chaotic"],
        tags=["prediction", "robustness"],
    ),

    "migration_cost_dominant": ScenarioMetadata(
        scenario_id="migration_cost_dominant",
        description=(
            "S12 MIGRATION_COST_DOMINANT: Network fluctuates just above "
            "the decision threshold, making migration overhead potentially "
            "dominant. Tests switching-penalty effectiveness."
        ),
        seed=52, duration_steps=64, sampling_interval_s=0.25,
        network_profile="near_threshold_oscillation", memory_profile="stable_flat",
        compute_profile="low", workload_profile="standard",
        provenance="synthetic",
        phases=["near_threshold"],
        tags=["stability", "switching_cost"],
    ),

    "mixed_stress": ScenarioMetadata(
        scenario_id="mixed_stress",
        description=(
            "S13 MIXED_STRESS: Concurrent network degradation, memory growth, "
            "and edge compute saturation. Maximum stress scenario."
        ),
        seed=53, duration_steps=64, sampling_interval_s=0.25,
        network_profile="sudden_drop", memory_profile="kv_growth",
        compute_profile="saturation", workload_profile="standard",
        provenance="synthetic",
        phases=["full_stress"],
        tags=["stress", "combined"],
    ),
}

# Canonical alias: mixed_degradation <-> combined_degradation
if "combined_degradation" in _CATALOG and "mixed_degradation" not in _CATALOG:
    _CATALOG["mixed_degradation"] = _CATALOG["combined_degradation"]

# Alias: combined_degradation → mixed_degradation for scenario generator lookup
_SCENARIO_GENERATOR_ALIASES: Dict[str, str] = {
    "combined_degradation":      "mixed_degradation",
    "bandwidth_degradation":     "gradual_bandwidth_degradation",
    "bandwidth_recovery":        "recovery",
    "oscillating_bandwidth":     "bandwidth_oscillation",
    "edge_compute_saturation":   "edge_compute_saturation",
    "linear_memory_growth":      "kv_cache_growth",   # use KV growth as proxy
    "kv_cache_driven_memory_growth": "kv_cache_growth",
    "migration_cost_dominant":   "bandwidth_oscillation",
    "mixed_stress":              "mixed_degradation",
}


class ScenarioCatalog:
    """
    Registry and factory for all 13 formal experimental scenarios.

    Usage:
        meta   = ScenarioCatalog.get("combined_degradation")
        gen_id = ScenarioCatalog.generator_id("combined_degradation")
        names  = ScenarioCatalog.list_all()
    """

    @classmethod
    def get(cls, scenario_id: str) -> ScenarioMetadata:
        if scenario_id not in _CATALOG:
            raise KeyError(
                f"Unknown scenario '{scenario_id}'. "
                f"Available: {cls.list_all()}"
            )
        return _CATALOG[scenario_id]

    @classmethod
    def generator_id(cls, scenario_id: str) -> str:
        """
        Return the scenario generator key to pass to ScenarioRegistry.get().

        Some catalog IDs are aliases for existing generator names.
        """
        return _SCENARIO_GENERATOR_ALIASES.get(scenario_id, scenario_id)

    @classmethod
    def list_all(cls) -> List[str]:
        return sorted(_CATALOG.keys())

    list_scenarios = list_all
    get_scenario = get

    @classmethod
    def describe_all(cls) -> Dict[str, str]:
        return {k: v.description for k, v in _CATALOG.items()}

    @classmethod
    def get_canonical(cls) -> ScenarioMetadata:
        """Return the canonical combined_degradation (S9) scenario."""
        return cls.get("combined_degradation")

    @classmethod
    def save_catalog_json(cls, path: Path | str) -> None:
        """Persist the full catalog to JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.to_dict() for k, v in _CATALOG.items()}
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
