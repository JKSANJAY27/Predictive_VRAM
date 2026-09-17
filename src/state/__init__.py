"""
Unified Runtime State and Rolling StateBuffer package.

Converts raw TelemetrySnapshot observations into clean, normalized,
timestamp-ordered runtime state suitable for future prediction and control.
"""

from src.state.buffer import StateBuffer
from src.state.features import (
    BASE_FEATURE_NAMES,
    DERIVED_FEATURE_NAMES,
    FEATURE_NAMES,
    FeatureExtractor,
    FeatureVector,
)
from src.state.types import (
    ActivationState,
    ComputeState,
    EnergyState,
    InferenceState,
    MemoryState,
    NetworkState,
    RuntimeState,
)

__all__ = [
    "RuntimeState",
    "NetworkState",
    "MemoryState",
    "ComputeState",
    "InferenceState",
    "ActivationState",
    "EnergyState",
    "StateBuffer",
    "FeatureVector",
    "FeatureExtractor",
    "FEATURE_NAMES",
    "BASE_FEATURE_NAMES",
    "DERIVED_FEATURE_NAMES",
]
