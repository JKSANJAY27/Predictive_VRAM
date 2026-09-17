"""
Shared test fixtures for unit and integration testing.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from src.runtime.model import LayeredTransformer
from src.runtime.tier import Tier, TierId
from src.runtime.transfer import TransferManager


@pytest.fixture
def synthetic_model() -> LayeredTransformer:
    """Create a fast, 4-layer synthetic model for sub-second deterministic testing."""
    torch.manual_seed(42)
    return LayeredTransformer.create_synthetic(
        num_layers=4,
        hidden_size=64,
        num_heads=2,
        vocab_size=500,
        device="cpu",
    )


@pytest.fixture
def standard_tiers() -> dict[TierId, Tier]:
    """Default 3-tier setup."""
    return {
        TierId.USER_DEVICE: Tier(tier_id=TierId.USER_DEVICE, name="User Device", device="cpu"),
        TierId.EDGE_A: Tier(tier_id=TierId.EDGE_A, name="Edge Node A", device="cpu"),
        TierId.EDGE_B: Tier(tier_id=TierId.EDGE_B, name="Edge Node B", device="cpu"),
    }


@pytest.fixture
def transfer_manager() -> TransferManager:
    """Fresh TransferManager instance."""
    return TransferManager()
