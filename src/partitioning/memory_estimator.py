"""
Memory requirement estimation for transformer partition plans.

Computes static parameter volume, dynamic KV-cache expansion, and safety margins
per execution tier for a candidate PartitionPlan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from src.partitioning.metadata import ModelMetadata
from src.runtime.partition import PartitionPlan
from src.runtime.tier import TierId
from src.telemetry.types import DataSource


@dataclass(frozen=True)
class TierMemoryRequirement:
    """
    Projected memory allocation on a single execution tier under a specific plan.
    """
    tier_id: TierId
    layer_count: int
    param_memory_mb: float
    kv_cache_memory_mb: float
    safety_margin_mb: float
    total_required_mb: float
    provenance: DataSource = DataSource.ESTIMATED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier_id": self.tier_id.value,
            "layer_count": self.layer_count,
            "param_memory_mb": self.param_memory_mb,
            "kv_cache_memory_mb": self.kv_cache_memory_mb,
            "safety_margin_mb": self.safety_margin_mb,
            "total_required_mb": self.total_required_mb,
            "provenance": self.provenance.value,
        }


def estimate_plan_memory(
    plan: PartitionPlan,
    model_metadata: ModelMetadata,
    context_length: int = 128,
    safety_margin_mb: float = 256.0,
) -> Dict[TierId, TierMemoryRequirement]:
    """
    Estimate memory required by each active tier in a PartitionPlan.

    Args:
        plan: The candidate PartitionPlan.
        model_metadata: Structural dimensions of the transformer.
        context_length: Token sequence length (prompt + generated tokens).
        safety_margin_mb: Buffer for runtime overhead, activation buffers, etc.

    Returns:
        Mapping from TierId to TierMemoryRequirement for all active tiers.
    """
    requirements: Dict[TierId, TierMemoryRequirement] = {}
    active_tiers = plan.get_active_tiers()

    dtype_b = model_metadata.dtype_bytes
    h_size = model_metadata.hidden_size
    n_heads = model_metadata.num_heads
    head_dim = model_metadata.head_dim

    # Bytes per token per layer for KV-cache: 2 (K & V) * n_heads * head_dim * dtype_bytes
    kv_bytes_per_token_per_layer = 2 * n_heads * head_dim * dtype_b
    kv_bytes_per_layer = kv_bytes_per_token_per_layer * max(1, context_length)

    first_tier = active_tiers[0][0]
    last_tier = active_tiers[-1][0]

    for tier_id, (start, end) in active_tiers:
        num_layers = end - start + 1

        # Parameter memory
        param_bytes = 0
        if model_metadata.per_layer_params:
            for l_idx in range(start, end + 1):
                param_bytes += model_metadata.per_layer_params[l_idx] * dtype_b
        else:
            # Approximate standard layer size
            layer_params = model_metadata.total_parameters // model_metadata.total_layers
            param_bytes = layer_params * num_layers * dtype_b

        # Add embeddings to tier holding layer 0
        if tier_id == first_tier and model_metadata.embeddings_params:
            param_bytes += model_metadata.embeddings_params * dtype_b

        # Add lm_head and final norm to tier holding final layer
        if tier_id == last_tier and model_metadata.head_params:
            param_bytes += model_metadata.head_params * dtype_b

        # KV-cache memory for assigned layers
        kv_bytes = kv_bytes_per_layer * num_layers

        param_mb = param_bytes / (1024.0 * 1024.0)
        kv_mb = kv_bytes / (1024.0 * 1024.0)
        total_mb = param_mb + kv_mb + safety_margin_mb

        requirements[tier_id] = TierMemoryRequirement(
            tier_id=tier_id,
            layer_count=num_layers,
            param_memory_mb=round(param_mb, 2),
            kv_cache_memory_mb=round(kv_mb, 2),
            safety_margin_mb=round(safety_margin_mb, 2),
            total_required_mb=round(total_mb, 2),
            provenance=DataSource.ESTIMATED,
        )

    return requirements
