"""
Unit tests for PartitionPlan and validation logic.
"""

import pytest

from src.runtime.partition import PartitionPlan, TransferBoundary
from src.runtime.tier import TierId


def test_valid_monolithic_partition():
    plan = PartitionPlan.monolithic(total_layers=12)
    assert plan.total_layers == 12
    assert plan.user_device == (0, 11)
    assert plan.edge_a is None
    assert plan.edge_b is None
    assert len(plan.get_active_tiers()) == 1
    assert len(plan.get_transfer_boundaries()) == 0
    assert plan.get_tier_for_layer(0) == TierId.USER_DEVICE
    assert plan.get_tier_for_layer(11) == TierId.USER_DEVICE


def test_valid_two_tier_partition():
    plan = PartitionPlan.two_tier(total_layers=12, cut_layer=5)
    assert plan.user_device == (0, 5)
    assert plan.edge_a == (6, 11)
    assert plan.edge_b is None

    active = plan.get_active_tiers()
    assert len(active) == 2
    assert active[0][0] == TierId.USER_DEVICE
    assert active[1][0] == TierId.EDGE_A

    boundaries = plan.get_transfer_boundaries()
    assert len(boundaries) == 1
    assert boundaries[0] == TransferBoundary(
        source_tier=TierId.USER_DEVICE,
        destination_tier=TierId.EDGE_A,
        cut_layer=5,
    )
    assert plan.get_tier_for_layer(3) == TierId.USER_DEVICE
    assert plan.get_tier_for_layer(7) == TierId.EDGE_A


def test_valid_three_tier_partition():
    plan = PartitionPlan.three_tier(total_layers=12, cut1=3, cut2=7)
    assert plan.user_device == (0, 3)
    assert plan.edge_a == (4, 7)
    assert plan.edge_b == (8, 11)

    active = plan.get_active_tiers()
    assert len(active) == 3
    boundaries = plan.get_transfer_boundaries()
    assert len(boundaries) == 2
    assert boundaries[0].source_tier == TierId.USER_DEVICE
    assert boundaries[0].destination_tier == TierId.EDGE_A
    assert boundaries[0].cut_layer == 3
    assert boundaries[1].source_tier == TierId.EDGE_A
    assert boundaries[1].destination_tier == TierId.EDGE_B
    assert boundaries[1].cut_layer == 7


def test_invalid_negative_layers():
    with pytest.raises(ValueError, match="total_layers must be positive"):
        PartitionPlan(total_layers=-1, user_device=(0, 0))


def test_invalid_no_active_tiers():
    with pytest.raises(ValueError, match="at least one active tier"):
        PartitionPlan(total_layers=12, user_device=None, edge_a=None, edge_b=None)


def test_invalid_out_of_bounds_layer():
    with pytest.raises(ValueError, match="outside valid model layer boundaries"):
        PartitionPlan(total_layers=12, user_device=(0, 15))


def test_invalid_inverted_range():
    with pytest.raises(ValueError, match="start layer must be <= end layer"):
        PartitionPlan(total_layers=12, user_device=(5, 2), edge_a=(6, 11))


def test_missing_layers_in_sequence():
    # Missing layer 4
    with pytest.raises(ValueError, match="Missing layers"):
        PartitionPlan(
            total_layers=12,
            user_device=(0, 3),
            edge_a=(5, 11),
        )


def test_missing_final_layers():
    # Covers only up to layer 9 out of 12
    with pytest.raises(ValueError, match="Missing final layers"):
        PartitionPlan(
            total_layers=12,
            user_device=(0, 4),
            edge_a=(5, 9),
        )


def test_duplicated_overlapping_layers():
    # Layer 4 assigned to both user_device and edge_a
    with pytest.raises(ValueError, match="Duplicated/overlapping layers"):
        PartitionPlan(
            total_layers=12,
            user_device=(0, 4),
            edge_a=(4, 11),
        )


def test_layer_lookup_out_of_bounds():
    plan = PartitionPlan.monolithic(total_layers=12)
    with pytest.raises(IndexError):
        plan.get_tier_for_layer(12)
    with pytest.raises(IndexError):
        plan.get_tier_for_layer(-1)
