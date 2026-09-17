"""
Cost Model and Candidate Scoring package.

Module 6 evaluates candidate partition plans against the multi-objective cost function:
    J(a) = alpha * L(a) + beta * C(a) + gamma * M(a) + delta * E(a) + epsilon * P_switch(a)

Translating runtime telemetry, resource forecasts, and structural plan differences into
transparent, inspectable, normalized cost breakdowns for the upcoming Module 7 controller.
"""

from src.cost.communication import CommunicationBreakdown, CommunicationCostModel
from src.cost.energy import EnergyBreakdown, EnergyCostModel
from src.cost.latency import LatencyBreakdown, LatencyCostModel
from src.cost.memory import MemoryPressureBreakdown, MemoryPressureCostModel
from src.cost.model import CostModel
from src.cost.normalization import CostNormalizer, NormalizationConfig
from src.cost.switching import SwitchingBreakdown, SwitchingCostModel
from src.cost.types import (
    CandidateScore,
    CostBreakdown,
    CostModelCalibration,
    CostWeights,
    ScoreStatus,
)

__all__ = [
    "CandidateScore",
    "CommunicationBreakdown",
    "CommunicationCostModel",
    "CostBreakdown",
    "CostModel",
    "CostModelCalibration",
    "CostNormalizer",
    "CostWeights",
    "EnergyBreakdown",
    "EnergyCostModel",
    "LatencyBreakdown",
    "LatencyCostModel",
    "MemoryPressureBreakdown",
    "MemoryPressureCostModel",
    "NormalizationConfig",
    "ScoreStatus",
    "SwitchingBreakdown",
    "SwitchingCostModel",
]
