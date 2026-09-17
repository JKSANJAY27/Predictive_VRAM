"""
Split Catalog and Feasible Candidate Plan Generation package.

Module 5 translates model metadata, tier capacities, and resource forecasts into
a structured catalog of feasible candidate PartitionPlans.
"""

from src.partitioning.candidate import CandidatePlan
from src.partitioning.catalog import SplitCatalog
from src.partitioning.comparison import PlanDifference, compare_plans
from src.partitioning.feasibility import (
    FeasibilityStatus,
    evaluate_plan_feasibility,
)
from src.partitioning.formatting import format_candidate_table
from src.partitioning.memory_estimator import (
    TierMemoryRequirement,
    estimate_plan_memory,
)
from src.partitioning.metadata import ModelMetadata, TierCapacity
from src.partitioning.plan_id import generate_plan_id
from src.partitioning.structural import enumerate_structural_plans

__all__ = [
    "CandidatePlan",
    "SplitCatalog",
    "PlanDifference",
    "compare_plans",
    "FeasibilityStatus",
    "evaluate_plan_feasibility",
    "format_candidate_table",
    "TierMemoryRequirement",
    "estimate_plan_memory",
    "ModelMetadata",
    "TierCapacity",
    "generate_plan_id",
    "enumerate_structural_plans",
]
