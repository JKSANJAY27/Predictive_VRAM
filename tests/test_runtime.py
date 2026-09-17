"""
Integration tests for DistributedInferenceExecutor and split-inference execution.
"""

import pytest
import torch

from src.runtime.executor import DistributedInferenceExecutor
from src.runtime.model import LayeredTransformer
from src.runtime.partition import PartitionPlan
from src.runtime.tier import Tier, TierId
from src.runtime.transfer import TransferManager


def test_synthetic_model_initialization(synthetic_model: LayeredTransformer):
    assert synthetic_model.num_layers == 4
    assert synthetic_model.hidden_size == 64
    assert synthetic_model.num_heads == 2


def test_monolithic_generation(synthetic_model: LayeredTransformer, standard_tiers: dict[TierId, Tier]):
    transfer_mgr = TransferManager()
    executor = DistributedInferenceExecutor(
        model=synthetic_model,
        tiers=standard_tiers,
        transfer_manager=transfer_mgr,
    )
    plan = PartitionPlan.monolithic(total_layers=4)

    res = executor.generate(
        prompt="Test prompt",
        partition_plan=plan,
        max_new_tokens=4,
        temperature=0.0,
    )

    assert res.generated_token_count == 4
    assert res.total_transfers == 0
    assert res.total_transfer_bytes == 0
    assert res.execution_time_seconds > 0
    assert res.kv_cache_summary["cached_layers_count"] == 4


def test_two_tier_generation(synthetic_model: LayeredTransformer, standard_tiers: dict[TierId, Tier]):
    transfer_mgr = TransferManager()
    executor = DistributedInferenceExecutor(
        model=synthetic_model,
        tiers=standard_tiers,
        transfer_manager=transfer_mgr,
    )
    plan = PartitionPlan.two_tier(total_layers=4, cut_layer=1)  # [0, 1] on User, [2, 3] on EdgeA

    max_tokens = 5
    res = executor.generate(
        prompt="Test prompt",
        partition_plan=plan,
        max_new_tokens=max_tokens,
        temperature=0.0,
    )

    assert res.generated_token_count == max_tokens
    # Exactly 1 boundary transfer per token step: 5 transfers total
    assert res.total_transfers == max_tokens
    assert res.total_transfer_bytes > 0
    for record in res.transfer_records:
        assert record.source_tier == TierId.USER_DEVICE
        assert record.destination_tier == TierId.EDGE_A


def test_three_tier_generation(synthetic_model: LayeredTransformer, standard_tiers: dict[TierId, Tier]):
    transfer_mgr = TransferManager()
    executor = DistributedInferenceExecutor(
        model=synthetic_model,
        tiers=standard_tiers,
        transfer_manager=transfer_mgr,
    )
    plan = PartitionPlan.three_tier(total_layers=4, cut1=0, cut2=2)  # [0,0] User, [1,2] EdgeA, [3,3] EdgeB

    max_tokens = 4
    res = executor.generate(
        prompt="Split prompt",
        partition_plan=plan,
        max_new_tokens=max_tokens,
        temperature=0.0,
    )

    assert res.generated_token_count == max_tokens
    # 2 boundaries per token step: User->EdgeA and EdgeA->EdgeB
    assert res.total_transfers == max_tokens * 2


def test_deterministic_parity_across_partitions(
    synthetic_model: LayeredTransformer, standard_tiers: dict[TierId, Tier]
):
    """
    CRITICAL RESEARCH VERIFICATION:
    Monolithic, 2-tier split, and 3-tier split MUST produce identical token sequences
    under deterministic settings (temperature=0.0).
    """
    executor = DistributedInferenceExecutor(
        model=synthetic_model,
        tiers=standard_tiers,
    )

    prompt = "Predictive VRAM and network-aware split inference"
    tokens_to_gen = 6

    # 1. Monolithic
    plan_mono = PartitionPlan.monolithic(total_layers=4)
    res_mono = executor.generate(prompt=prompt, partition_plan=plan_mono, max_new_tokens=tokens_to_gen, temperature=0.0)

    # 2. Two-Tier
    plan_2t = PartitionPlan.two_tier(total_layers=4, cut_layer=1)
    res_2t = executor.generate(prompt=prompt, partition_plan=plan_2t, max_new_tokens=tokens_to_gen, temperature=0.0)

    # 3. Three-Tier
    plan_3t = PartitionPlan.three_tier(total_layers=4, cut1=0, cut2=2)
    res_3t = executor.generate(prompt=prompt, partition_plan=plan_3t, max_new_tokens=tokens_to_gen, temperature=0.0)

    # Verify parity
    assert res_mono.generated_tokens == res_2t.generated_tokens, (
        f"Mismatch between Monolithic and 2-Tier: {res_mono.generated_tokens} vs {res_2t.generated_tokens}"
    )
    assert res_mono.generated_tokens == res_3t.generated_tokens, (
        f"Mismatch between Monolithic and 3-Tier: {res_mono.generated_tokens} vs {res_3t.generated_tokens}"
    )
    assert res_mono.generated_text == res_3t.generated_text
