"""
ExperimentConfig, ExperimentMatrix, and TrialConfig for Module 10.

ExperimentConfig  — structured experiment specification (YAML/JSON)
ExperimentMatrix  — expands cross-products of (scenarios, baselines, seeds, …)
                    into a flat, deduplicated list of TrialConfig objects
TrialConfig       — fully-specified single-trial descriptor with unique trial_id
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class WorkloadConfig:
    prompt: str = "The quick brown fox"
    max_new_tokens: int = 32
    temperature: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> WorkloadConfig:
        return cls(
            prompt=d.get("prompt", "The quick brown fox"),
            max_new_tokens=int(d.get("max_new_tokens", 32)),
            temperature=float(d.get("temperature", 0.0)),
        )


@dataclass
class TopologyConfig:
    tiers: int = 3
    n_layers: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return {"tiers": self.tiers, "n_layers": self.n_layers}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TopologyConfig:
        return cls(tiers=int(d.get("tiers", 3)), n_layers=int(d.get("n_layers", 4)))


@dataclass
class ExecutionConfig:
    mode: str = "simulation"   # "simulation" | "execution" | "replay"

    def to_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ExecutionConfig:
        return cls(mode=d.get("mode", "simulation"))


@dataclass
class ControlConfig:
    interval_tokens: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return {"interval_tokens": self.interval_tokens}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ControlConfig:
        return cls(interval_tokens=int(d.get("interval_tokens", 4)))


@dataclass
class PredictionConfig:
    enabled: bool = True
    predictor: str = "linear_trend"
    horizon_steps: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "predictor": self.predictor,
            "horizon_steps": self.horizon_steps,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PredictionConfig:
        return cls(
            enabled=bool(d.get("enabled", True)),
            predictor=d.get("predictor", "linear_trend"),
            horizon_steps=int(d.get("horizon_steps", 4)),
        )


@dataclass
class NetworkConfig:
    scenario: str = "stable"
    n_steps: int = 64
    sampling_interval_s: float = 0.25

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "n_steps": self.n_steps,
            "sampling_interval_s": self.sampling_interval_s,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> NetworkConfig:
        return cls(
            scenario=d.get("scenario", "stable"),
            n_steps=int(d.get("n_steps", 64)),
            sampling_interval_s=float(d.get("sampling_interval_s", 0.25)),
        )


@dataclass
class MemoryConfig:
    scenario: str = "stable"

    def to_dict(self) -> Dict[str, Any]:
        return {"scenario": self.scenario}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MemoryConfig:
        return cls(scenario=d.get("scenario", "stable"))


@dataclass
class ComputeConfig:
    load: str = "low"   # "low" | "medium" | "high" | "saturation"

    def to_dict(self) -> Dict[str, Any]:
        return {"load": self.load}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ComputeConfig:
        return cls(load=d.get("load", "low"))


@dataclass
class SLOConfig:
    p95_itl_ms: float = 200.0
    ttft_ms: float = 500.0
    memory_pressure_max: float = 0.90

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p95_itl_ms": self.p95_itl_ms,
            "ttft_ms": self.ttft_ms,
            "memory_pressure_max": self.memory_pressure_max,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SLOConfig:
        return cls(
            p95_itl_ms=float(d.get("p95_itl_ms", 200.0)),
            ttft_ms=float(d.get("ttft_ms", 500.0)),
            memory_pressure_max=float(d.get("memory_pressure_max", 0.90)),
        )


@dataclass
class SensitivityConfig:
    threshold_values: List[float]   = field(default_factory=lambda: [0.00, 0.02, 0.05, 0.10, 0.20])
    horizon_steps:    List[int]     = field(default_factory=lambda: [2, 4, 8, 20])
    control_intervals: List[int]    = field(default_factory=lambda: [1, 2, 4, 8])
    predictor_types:  List[str]     = field(default_factory=lambda: [
        "last_value", "linear_trend", "moving_average"
    ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold_values": self.threshold_values,
            "horizon_steps": self.horizon_steps,
            "control_intervals": self.control_intervals,
            "predictor_types": self.predictor_types,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SensitivityConfig:
        return cls(
            threshold_values=d.get("threshold_values", [0.00, 0.02, 0.05, 0.10, 0.20]),
            horizon_steps=d.get("horizon_steps", [2, 4, 8, 20]),
            control_intervals=d.get("control_intervals", [1, 2, 4, 8]),
            predictor_types=d.get("predictor_types", ["last_value", "linear_trend", "moving_average"]),
        )


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """
    Complete, serialisable experiment specification.

    Load from YAML (requires PyYAML) or JSON.  Save to JSON.
    experiment_id is derived from a hash of the normalised config so that
    two runs with the same parameters get the same ID.
    """
    name: str
    seed: int = 42
    repetitions: int = 1

    workload:    WorkloadConfig    = field(default_factory=WorkloadConfig)
    topology:    TopologyConfig    = field(default_factory=TopologyConfig)
    execution:   ExecutionConfig   = field(default_factory=ExecutionConfig)
    control:     ControlConfig     = field(default_factory=ControlConfig)
    prediction:  PredictionConfig  = field(default_factory=PredictionConfig)
    network:     NetworkConfig     = field(default_factory=NetworkConfig)
    memory:      MemoryConfig      = field(default_factory=MemoryConfig)
    compute:     ComputeConfig     = field(default_factory=ComputeConfig)
    slo:         SLOConfig         = field(default_factory=SLOConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)

    baselines:   List[str] = field(default_factory=lambda: [
        "static", "network_reactive", "memory_reactive", "joint_reactive", "predictive"
    ])
    ablations:   List[str] = field(default_factory=list)
    cost_weights: Dict[str, float] = field(default_factory=lambda: {
        "alpha": 0.35, "beta": 0.20, "gamma": 0.25, "delta": 0.10, "epsilon": 0.10,
    })
    experiment_id: str = ""

    def __post_init__(self) -> None:
        if not self.experiment_id:
            self.experiment_id = _compute_config_hash(self)[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "1.0",
            "name": self.name,
            "seed": self.seed,
            "repetitions": self.repetitions,
            "workload":    self.workload.to_dict(),
            "topology":    self.topology.to_dict(),
            "execution":   self.execution.to_dict(),
            "control":     self.control.to_dict(),
            "prediction":  self.prediction.to_dict(),
            "network":     self.network.to_dict(),
            "memory":      self.memory.to_dict(),
            "compute":     self.compute.to_dict(),
            "slo":         self.slo.to_dict(),
            "sensitivity": self.sensitivity.to_dict(),
            "baselines":   self.baselines,
            "ablations":   self.ablations,
            "cost_weights": self.cost_weights,
            "experiment_id": self.experiment_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ExperimentConfig:
        obj = cls(
            name=d.get("name", "unnamed"),
            seed=int(d.get("seed", 42)),
            repetitions=int(d.get("repetitions", 1)),
            workload=WorkloadConfig.from_dict(d.get("workload", {})),
            topology=TopologyConfig.from_dict(d.get("topology", {})),
            execution=ExecutionConfig.from_dict(d.get("execution", {})),
            control=ControlConfig.from_dict(d.get("control", {})),
            prediction=PredictionConfig.from_dict(d.get("prediction", {})),
            network=NetworkConfig.from_dict(d.get("network", {})),
            memory=MemoryConfig.from_dict(d.get("memory", {})),
            compute=ComputeConfig.from_dict(d.get("compute", {})),
            slo=SLOConfig.from_dict(d.get("slo", {})),
            sensitivity=SensitivityConfig.from_dict(d.get("sensitivity", {})),
            baselines=d.get("baselines", ["static", "network_reactive", "memory_reactive",
                                          "joint_reactive", "predictive"]),
            ablations=d.get("ablations", []),
            cost_weights=d.get("cost_weights", {}),
            experiment_id=d.get("experiment_id", ""),
        )
        return obj

    def save_json(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path | str) -> ExperimentConfig:
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def load_yaml(cls, path: Path | str) -> ExperimentConfig:
        try:
            import yaml  # type: ignore
        except ImportError:
            raise ImportError("PyYAML is required to load YAML configs. Install with: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    def config_hash(self) -> str:
        return _compute_config_hash(self)


def _compute_config_hash(cfg: ExperimentConfig) -> str:
    """SHA-256 of JSON-sorted, experiment_id-stripped config."""
    d = cfg.to_dict()
    d.pop("experiment_id", None)
    raw = json.dumps(d, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# TrialConfig
# ---------------------------------------------------------------------------

@dataclass
class TrialConfig:
    """
    Fully-specified single-trial descriptor.

    Every trial has a unique, deterministic trial_id derived from its
    (experiment_id, scenario_id, baseline_id, seed, repetition_index).
    """
    trial_id: str
    experiment_id: str
    scenario_id: str
    baseline_id: str
    seed: int
    repetition: int

    # Effective per-trial settings (may differ from base config for sweeps)
    switch_threshold: float
    control_interval_tokens: int
    horizon_steps: int
    predictor_type: str
    max_new_tokens: int
    n_trace_steps: int
    sampling_interval_s: float
    execution_mode: str
    cost_weights: Dict[str, float]
    ablation_id: str = ""
    slo_config: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "experiment_id": self.experiment_id,
            "scenario_id": self.scenario_id,
            "baseline_id": self.baseline_id,
            "seed": self.seed,
            "repetition": self.repetition,
            "switch_threshold": self.switch_threshold,
            "control_interval_tokens": self.control_interval_tokens,
            "horizon_steps": self.horizon_steps,
            "predictor_type": self.predictor_type,
            "max_new_tokens": self.max_new_tokens,
            "n_trace_steps": self.n_trace_steps,
            "sampling_interval_s": self.sampling_interval_s,
            "execution_mode": self.execution_mode,
            "cost_weights": self.cost_weights,
            "ablation_id": self.ablation_id,
            "slo_config": self.slo_config,
            "metadata": self.metadata,
        }


def _make_trial_id(
    experiment_id: str,
    scenario_id: str,
    baseline_id: str,
    seed: int,
    repetition: int,
    ablation_id: str = "",
    **kwargs: Any,
) -> str:
    parts = [experiment_id, scenario_id, baseline_id, str(seed), str(repetition)]
    if ablation_id:
        parts.append(ablation_id)
    # Include any sweep-specific kwargs that distinguish the trial
    for k, v in sorted(kwargs.items()):
        parts.append(f"{k}={v}")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


# ---------------------------------------------------------------------------
# ExperimentMatrix
# ---------------------------------------------------------------------------

class ExperimentMatrix:
    """
    Expands an ExperimentConfig into a flat, deduplicated list of TrialConfig.

    Expansion covers:
      scenarios × baselines × repetitions/seeds
    plus (optionally):
      ablations × sensitivity sweeps
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._cfg = config

    def expand(
        self,
        include_ablations: bool = False,
        include_sensitivity: bool = False,
    ) -> List[TrialConfig]:
        """Return the full flat trial list."""
        trials: List[TrialConfig] = []
        seen: set = set()

        base_seeds = [self._cfg.seed + i for i in range(self._cfg.repetitions)]

        scenario_ids = [self._cfg.network.scenario]

        for scenario_id, baseline_id, (rep_idx, seed) in itertools.product(
            scenario_ids,
            self._cfg.baselines,
            enumerate(base_seeds),
        ):
            t = self._make_trial(
                scenario_id=scenario_id,
                baseline_id=baseline_id,
                seed=seed,
                repetition=rep_idx,
            )
            if t.trial_id not in seen:
                seen.add(t.trial_id)
                trials.append(t)

        if include_ablations:
            for abl_id in self._cfg.ablations:
                for seed_idx, seed in enumerate(base_seeds):
                    t = self._make_trial(
                        scenario_id=scenario_ids[0],
                        baseline_id="predictive",
                        seed=seed,
                        repetition=seed_idx,
                        ablation_id=abl_id,
                    )
                    if t.trial_id not in seen:
                        seen.add(t.trial_id)
                        trials.append(t)

        if include_sensitivity:
            trials.extend(self._expand_sensitivity(seen, base_seeds[0]))

        return trials

    def _expand_sensitivity(
        self, seen: set, seed: int
    ) -> List[TrialConfig]:
        """Generate threshold / horizon / interval sensitivity trials."""
        sc = self._cfg.sensitivity
        extras: List[TrialConfig] = []

        for theta in sc.threshold_values:
            t = self._make_trial(
                scenario_id=self._cfg.network.scenario,
                baseline_id="predictive",
                seed=seed,
                repetition=0,
                switch_threshold=theta,
            )
            if t.trial_id not in seen:
                seen.add(t.trial_id)
                extras.append(t)

        for h in sc.horizon_steps:
            t = self._make_trial(
                scenario_id=self._cfg.network.scenario,
                baseline_id="predictive",
                seed=seed,
                repetition=0,
                horizon_steps=h,
            )
            if t.trial_id not in seen:
                seen.add(t.trial_id)
                extras.append(t)

        for k in sc.control_intervals:
            t = self._make_trial(
                scenario_id=self._cfg.network.scenario,
                baseline_id="predictive",
                seed=seed,
                repetition=0,
                control_interval_tokens=k,
            )
            if t.trial_id not in seen:
                seen.add(t.trial_id)
                extras.append(t)

        return extras

    def _make_trial(
        self,
        scenario_id: str,
        baseline_id: str,
        seed: int,
        repetition: int,
        ablation_id: str = "",
        switch_threshold: Optional[float] = None,
        horizon_steps: Optional[int] = None,
        control_interval_tokens: Optional[int] = None,
    ) -> TrialConfig:
        cfg = self._cfg
        eff_threshold = switch_threshold if switch_threshold is not None else 0.25
        eff_horizon   = horizon_steps   if horizon_steps   is not None else cfg.prediction.horizon_steps
        eff_interval  = control_interval_tokens if control_interval_tokens is not None else cfg.control.interval_tokens

        trial_id = _make_trial_id(
            cfg.experiment_id, scenario_id, baseline_id, seed, repetition,
            ablation_id=ablation_id,
            theta=round(eff_threshold, 4),
            h=eff_horizon,
            k=eff_interval,
        )

        return TrialConfig(
            trial_id=trial_id,
            experiment_id=cfg.experiment_id,
            scenario_id=scenario_id,
            baseline_id=baseline_id,
            seed=seed,
            repetition=repetition,
            switch_threshold=eff_threshold,
            control_interval_tokens=eff_interval,
            horizon_steps=eff_horizon,
            predictor_type=cfg.prediction.predictor,
            max_new_tokens=cfg.workload.max_new_tokens,
            n_trace_steps=cfg.network.n_steps,
            sampling_interval_s=cfg.network.sampling_interval_s,
            execution_mode=cfg.execution.mode,
            cost_weights=dict(cfg.cost_weights),
            ablation_id=ablation_id,
            slo_config=cfg.slo.to_dict(),
        )


def make_canonical_experiment() -> ExperimentConfig:
    """
    Return the canonical ExperimentConfig:
    combined_degradation, all 5 baselines, seed=42, 1 repetition.
    """
    return ExperimentConfig(
        name="canonical_combined_degradation",
        seed=42,
        repetitions=1,
        workload=WorkloadConfig(max_new_tokens=32),
        network=NetworkConfig(scenario="combined_degradation", n_steps=64),
        memory=MemoryConfig(scenario="combined_degradation"),
        baselines=["static", "network_reactive", "memory_reactive", "joint_reactive", "predictive"],
        ablations=["prediction_removed", "switching_penalty_removed",
                   "network_signal_removed", "vram_signal_removed"],
    )
