"""
Physical domain constraints for forecast trajectories.

Ensures predicted quantities respect obvious physical laws (e.g. non-negative
bandwidth and latency, packet loss in [0, 1]) without blindly destroying
unconstrained predictions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

# Domain bounds: target_name -> (min_value, max_value)
PHYSICAL_BOUNDS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "bandwidth_mbps": (0.0, None),
    "latency_ms": (0.0, None),
    "packet_loss": (0.0, 1.0),
    "jitter_ms": (0.0, None),
    "vram_allocated_mb": (0.0, None),
    "vram_free_mb": (0.0, None),
    "vram_pressure": (0.0, 1.0),
    "ram_used_mb": (0.0, None),
    "ram_available_mb": (0.0, None),
    "cpu_utilization": (0.0, 100.0),
    "gpu_utilization": (0.0, 100.0),
    "kv_cache_bytes": (0.0, None),
    "generated_tokens": (0.0, None),
    "context_length": (0.0, None),
    "generation_rate": (0.0, None),
}


def apply_physical_constraints(
    target_name: str,
    raw_values: Sequence[Optional[float]],
) -> Tuple[List[Optional[float]], bool]:
    """
    Apply physical domain bounds to raw forecast values.

    Args:
        target_name: Name of target variable.
        raw_values: Sequence of raw (unconstrained) predictions.

    Returns:
        A tuple of:
            - constrained_values: list of clamped values (None values preserved).
            - constraint_applied: bool indicating if any value was altered.
    """
    bounds = PHYSICAL_BOUNDS.get(target_name)
    if bounds is None:
        return list(raw_values), False

    min_val, max_val = bounds
    constrained: List[Optional[float]] = []
    altered = False

    for v in raw_values:
        if v is None:
            constrained.append(None)
            continue

        c = v
        if min_val is not None and c < min_val:
            c = min_val
            altered = True
        if max_val is not None and c > max_val:
            c = max_val
            altered = True

        constrained.append(c)

    return constrained, altered
