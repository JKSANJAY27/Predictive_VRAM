"""
Feature extraction and vectorization for RuntimeState sequences.

Deterministic feature extraction for downstream state prediction and control.
NO trainable models, ML weights, or auto-fitting normalization are implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.state.types import RuntimeState
from src.telemetry.types import DataSource, TaggedValue

# ---------------------------------------------------------------------------
# Feature Column Definitions (Deterministic Ordering)
# ---------------------------------------------------------------------------

BASE_FEATURE_NAMES: Tuple[str, ...] = (
    "bandwidth_mbps",
    "latency_ms",
    "packet_loss",
    "jitter_ms",
    "vram_allocated_mb",
    "vram_free_mb",
    "ram_used_mb",
    "ram_available_mb",
    "gpu_utilization",
    "cpu_utilization",
    "kv_cache_bytes",
    "kv_cache_growth_bytes",
    "generated_tokens",
    "context_length",
    "generation_rate",
    "latest_activation_bytes",
    "latest_transfer_latency_ms",
    "energy_proxy",
)

DERIVED_FEATURE_NAMES: Tuple[str, ...] = (
    "kv_cache_growth_rate",
    "memory_growth_rate",
    "token_growth_rate",
    "elapsed_time",
)

FEATURE_NAMES: Tuple[str, ...] = BASE_FEATURE_NAMES + DERIVED_FEATURE_NAMES


# ---------------------------------------------------------------------------
# FeatureVector Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureVector:
    """
    Fixed-length vector representation of a RuntimeState with provenance metadata.

    Attributes:
        values: Extracted scalar values (None if unavailable/missing).
        availability_mask: Binary list (1 = available, 0 = unavailable / None).
        source_mask: Data source string for each feature (e.g. 'measured', 'unavailable').
        feature_names: Exact feature names in order.
    """
    values: Tuple[Optional[float], ...]
    availability_mask: Tuple[int, ...]
    source_mask: Tuple[str, ...]
    feature_names: Tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self) -> None:
        n = len(self.feature_names)
        if len(self.values) != n:
            raise ValueError(f"values length {len(self.values)} does not match feature_names {n}")
        if len(self.availability_mask) != n:
            raise ValueError(f"availability_mask length {len(self.availability_mask)} does not match feature_names {n}")
        if len(self.source_mask) != n:
            raise ValueError(f"source_mask length {len(self.source_mask)} does not match feature_names {n}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "values": list(self.values),
            "availability_mask": list(self.availability_mask),
            "source_mask": list(self.source_mask),
        }

    def to_dense(self, fill_value: float = 0.0) -> List[float]:
        """Return a dense float list, replacing missing (None) values with fill_value."""
        return [float(v) if v is not None else fill_value for v in self.values]

    def get(self, name: str) -> Optional[float]:
        """Get feature value by name."""
        idx = self.feature_names.index(name)
        return self.values[idx]

    def is_available(self, name: str) -> bool:
        """Return True if named feature is available."""
        idx = self.feature_names.index(name)
        return bool(self.availability_mask[idx])

    def get_source(self, name: str) -> str:
        """Return DataSource string for named feature."""
        idx = self.feature_names.index(name)
        return self.source_mask[idx]


# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Deterministic feature extractor for RuntimeState.

    Provides stable column ordering, availability masking, provenance masking,
    derived rate features across temporal sequences, and deterministic normalization.
    """

    FEATURE_NAMES = FEATURE_NAMES
    BASE_FEATURE_NAMES = BASE_FEATURE_NAMES
    DERIVED_FEATURE_NAMES = DERIVED_FEATURE_NAMES

    @classmethod
    def to_vector(
        cls,
        state: RuntimeState,
        prev_state: Optional[RuntimeState] = None,
        start_timestamp: Optional[float] = None,
    ) -> FeatureVector:
        """
        Extract a single deterministic FeatureVector from a RuntimeState.

        Derived features are computed using prev_state and start_timestamp if provided.
        """
        raw_values: List[Optional[float]] = []
        avail_mask: List[int] = []
        src_mask: List[str] = []

        def _append_tagged(tv: Optional[TaggedValue]) -> None:
            if tv is None or tv.source == DataSource.UNAVAILABLE or tv.value is None:
                raw_values.append(None)
                avail_mask.append(0)
                src_mask.append(DataSource.UNAVAILABLE.value if tv is None else tv.source.value)
            else:
                raw_values.append(float(tv.value))
                avail_mask.append(1)
                src_mask.append(tv.source.value)

        # 1-4: Network
        _append_tagged(state.network.bandwidth_mbps)
        _append_tagged(state.network.latency_ms)
        _append_tagged(state.network.packet_loss)
        _append_tagged(state.network.jitter_ms)

        # 5-8: Memory
        _append_tagged(state.memory.vram_allocated_mb)
        _append_tagged(state.memory.vram_free_mb)
        _append_tagged(state.memory.ram_used_mb)
        _append_tagged(state.memory.ram_available_mb)

        # 9-10: Compute
        _append_tagged(state.compute.gpu_utilization)
        _append_tagged(state.compute.cpu_utilization)

        # 11-12: Inference KV
        _append_tagged(state.inference.kv_cache_bytes)
        _append_tagged(state.inference.kv_cache_growth_bytes)

        # 13-15: Inference Progress
        raw_values.append(float(state.inference.generated_tokens))
        avail_mask.append(1)
        src_mask.append(DataSource.MEASURED.value)

        raw_values.append(float(state.inference.context_length))
        avail_mask.append(1)
        src_mask.append(DataSource.MEASURED.value)

        if state.inference.generation_rate is not None:
            raw_values.append(float(state.inference.generation_rate))
            avail_mask.append(1)
            src_mask.append(DataSource.ESTIMATED.value)
        else:
            raw_values.append(None)
            avail_mask.append(0)
            src_mask.append(DataSource.UNAVAILABLE.value)

        # 16-17: Activation
        _append_tagged(state.activation.latest_activation_bytes)
        _append_tagged(state.activation.latest_transfer_latency_ms)

        # 18: Energy
        _append_tagged(state.energy.energy_proxy)

        # --- Derived Features (19-22) ---
        # 19: kv_cache_growth_rate
        # 20: memory_growth_rate
        # 21: token_growth_rate
        # 22: elapsed_time

        # If first sample (prev_state is None), growth rates return 0.0 if base value is available, else None
        dt: Optional[float] = None
        if prev_state is not None:
            dt = state.timestamp - prev_state.timestamp

        # 19: kv_cache_growth_rate
        cur_kv = state.inference.kv_cache_bytes.value
        if prev_state is None:
            if cur_kv is not None and state.inference.kv_cache_bytes.source != DataSource.UNAVAILABLE:
                raw_values.append(0.0)
                avail_mask.append(1)
                src_mask.append(DataSource.ESTIMATED.value)
            else:
                raw_values.append(None)
                avail_mask.append(0)
                src_mask.append(DataSource.UNAVAILABLE.value)
        else:
            prev_kv = prev_state.inference.kv_cache_bytes.value
            if dt is not None and dt > 0 and cur_kv is not None and prev_kv is not None and state.inference.kv_cache_bytes.source != DataSource.UNAVAILABLE and prev_state.inference.kv_cache_bytes.source != DataSource.UNAVAILABLE:
                raw_values.append(float((cur_kv - prev_kv) / dt))
                avail_mask.append(1)
                src_mask.append(DataSource.ESTIMATED.value)
            else:
                raw_values.append(None)
                avail_mask.append(0)
                src_mask.append(DataSource.UNAVAILABLE.value)

        # 20: memory_growth_rate (ram_used_mb)
        cur_mem = state.memory.ram_used_mb.value
        if prev_state is None:
            if cur_mem is not None and state.memory.ram_used_mb.source != DataSource.UNAVAILABLE:
                raw_values.append(0.0)
                avail_mask.append(1)
                src_mask.append(DataSource.ESTIMATED.value)
            else:
                raw_values.append(None)
                avail_mask.append(0)
                src_mask.append(DataSource.UNAVAILABLE.value)
        else:
            prev_mem = prev_state.memory.ram_used_mb.value
            if dt is not None and dt > 0 and cur_mem is not None and prev_mem is not None and state.memory.ram_used_mb.source != DataSource.UNAVAILABLE and prev_state.memory.ram_used_mb.source != DataSource.UNAVAILABLE:
                raw_values.append(float((cur_mem - prev_mem) / dt))
                avail_mask.append(1)
                src_mask.append(DataSource.ESTIMATED.value)
            else:
                raw_values.append(None)
                avail_mask.append(0)
                src_mask.append(DataSource.UNAVAILABLE.value)

        # 21: token_growth_rate
        cur_tok = state.inference.generated_tokens
        if prev_state is None:
            raw_values.append(0.0)
            avail_mask.append(1)
            src_mask.append(DataSource.ESTIMATED.value)
        else:
            prev_tok = prev_state.inference.generated_tokens
            if dt is not None and dt > 0 and cur_tok >= prev_tok:
                raw_values.append(float((cur_tok - prev_tok) / dt))
                avail_mask.append(1)
                src_mask.append(DataSource.ESTIMATED.value)
            else:
                raw_values.append(None)
                avail_mask.append(0)
                src_mask.append(DataSource.UNAVAILABLE.value)

        # 22: elapsed_time
        base_t = start_timestamp if start_timestamp is not None else state.timestamp
        elapsed = state.timestamp - base_t
        if elapsed >= 0:
            raw_values.append(float(elapsed))
            avail_mask.append(1)
            src_mask.append(DataSource.MEASURED.value)
        else:
            raw_values.append(None)
            avail_mask.append(0)
            src_mask.append(DataSource.UNAVAILABLE.value)

        return FeatureVector(
            values=tuple(raw_values),
            availability_mask=tuple(avail_mask),
            source_mask=tuple(src_mask),
            feature_names=FEATURE_NAMES,
        )

    @classmethod
    def from_states(cls, states: Sequence[RuntimeState]) -> List[FeatureVector]:
        """Extract a sequence of FeatureVectors from an ordered sequence of RuntimeStates."""
        if not states:
            return []
        start_t = states[0].timestamp
        vectors: List[FeatureVector] = []
        for i, s in enumerate(states):
            prev = states[i - 1] if i > 0 else None
            vec = cls.to_vector(s, prev_state=prev, start_timestamp=start_t)
            vectors.append(vec)
        return vectors

    @classmethod
    def to_matrix(
        cls,
        states: Sequence[RuntimeState],
        fill_value: Optional[float] = None,
        as_numpy: bool = False,
    ) -> Any:
        """
        Convert a sequence of RuntimeStates into a 2D feature matrix (N x M).

        If fill_value is provided, replaces None with fill_value.
        If as_numpy is True, attempts to convert to numpy.ndarray.
        """
        vectors = cls.from_states(states)
        matrix: List[List[Any]] = []
        for vec in vectors:
            if fill_value is not None:
                matrix.append(vec.to_dense(fill_value=fill_value))
            else:
                matrix.append(list(vec.values))

        if as_numpy:
            try:
                import numpy as np
                return np.array(matrix, dtype=float)
            except ImportError:
                return matrix
        return matrix

    # -----------------------------------------------------------------------
    # Deterministic Normalization (Strictly Externally Supplied Parameters)
    # -----------------------------------------------------------------------

    @staticmethod
    def normalize_min_max(
        values: Sequence[Optional[float]],
        min_vals: Sequence[float],
        max_vals: Sequence[float],
    ) -> List[Optional[float]]:
        """
        Min-Max normalization using externally supplied statistics:
        norm = (x - min) / (max - min).

        If max == min, returns 0.0.
        If value is None, preserves None.
        """
        if len(values) != len(min_vals) or len(values) != len(max_vals):
            raise ValueError("Input values and min/max parameter lengths must match.")

        normalized: List[Optional[float]] = []
        for v, min_v, max_v in zip(values, min_vals, max_vals):
            if v is None:
                normalized.append(None)
                continue
            span = max_v - min_v
            if span == 0:
                normalized.append(0.0)
            else:
                normalized.append(float((v - min_v) / span))
        return normalized

    @staticmethod
    def normalize_z_score(
        values: Sequence[Optional[float]],
        mean_vals: Sequence[float],
        std_vals: Sequence[float],
    ) -> List[Optional[float]]:
        """
        Z-score standardization using externally supplied statistics:
        z = (x - mean) / std.

        If std == 0, returns 0.0.
        If value is None, preserves None.
        """
        if len(values) != len(mean_vals) or len(values) != len(std_vals):
            raise ValueError("Input values and mean/std parameter lengths must match.")

        standardized: List[Optional[float]] = []
        for v, mean_v, std_v in zip(values, mean_vals, std_vals):
            if v is None:
                standardized.append(None)
                continue
            if std_v == 0:
                standardized.append(0.0)
            else:
                standardized.append(float((v - mean_v) / std_v))
        return standardized
