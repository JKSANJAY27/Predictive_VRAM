"""
Unit tests for TransferManager and TransferRecord.
"""

import pytest
import torch

from src.runtime.tier import TierId
from src.runtime.transfer import TransferManager, TransferRecord


def test_transfer_records_telemetry(transfer_manager: TransferManager):
    t = torch.randn(1, 4, 64, dtype=torch.float32)
    # 1 * 4 * 64 * 4 bytes = 1024 bytes

    result = transfer_manager.transfer(
        tensor=t,
        source_tier=TierId.USER_DEVICE,
        destination_tier=TierId.EDGE_A,
        step=0,
    )

    assert transfer_manager.total_transfers == 1
    assert transfer_manager.total_bytes == 1024
    assert result.shape == (1, 4, 64)

    history = transfer_manager.get_history()
    assert len(history) == 1
    record = history[0]
    assert record.source_tier == TierId.USER_DEVICE
    assert record.destination_tier == TierId.EDGE_A
    assert record.tensor_shape == (1, 4, 64)
    assert record.num_elements == 256
    assert record.byte_size == 1024
    assert record.step == 0


def test_transfer_multiple_steps(transfer_manager: TransferManager):
    for s in range(5):
        t = torch.randn(1, 1, 64, dtype=torch.float32)  # 256 bytes each
        transfer_manager.transfer(
            tensor=t,
            source_tier=TierId.EDGE_A,
            destination_tier=TierId.EDGE_B,
            step=s,
        )

    assert transfer_manager.total_transfers == 5
    assert transfer_manager.total_bytes == 5 * 256

    transfer_manager.clear_history()
    assert transfer_manager.total_transfers == 0
    assert transfer_manager.total_bytes == 0


def test_transfer_rejects_non_tensor(transfer_manager: TransferManager):
    with pytest.raises(TypeError, match="Expected torch.Tensor"):
        transfer_manager.transfer(
            tensor="not a tensor",  # type: ignore
            source_tier=TierId.USER_DEVICE,
            destination_tier=TierId.EDGE_A,
        )
