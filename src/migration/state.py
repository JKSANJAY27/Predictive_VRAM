"""
Model layer and KV-cache transfer managers for Module 8.

Coordinates moving model parameters and dynamic attention cache tensors
for affected layers, maintaining exact backups for instant rollback.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.migration.planner import MigrationPlan
from src.migration.transfer import MigrationTransferProvider
from src.runtime.model import LayeredTransformer
from src.runtime.tier import TierId
from src.telemetry.types import DataSource


@dataclass
class LayerTransferReport:
    """Report of model layer parameter migration."""
    layers_transferred: List[int] = field(default_factory=list)
    total_bytes: int = 0
    duration_ms: float = 0.0
    per_layer_bytes: Dict[int, int] = field(default_factory=dict)
    provenance: DataSource = DataSource.MEASURED


@dataclass
class KVCacheTransferReport:
    """Report of KV-cache tensor migration."""
    layers_transferred: List[int] = field(default_factory=list)
    total_bytes: int = 0
    duration_ms: float = 0.0
    per_layer_bytes: Dict[int, int] = field(default_factory=dict)
    provenance: DataSource = DataSource.MEASURED


class LayerTransferManager:
    """
    Coordinates migration of model layer blocks across execution tiers.
    """

    def __init__(
        self,
        model: LayeredTransformer,
        tier_devices: Dict[TierId, str],
        transfer_provider: MigrationTransferProvider,
    ) -> None:
        self.model = model
        self.tier_devices = tier_devices
        self.transfer_provider = transfer_provider
        self._backup_devices: Dict[int, str] = {}
        self._backup_state_dicts: Dict[int, Dict[str, torch.Tensor]] = {}

    def transfer_layers(
        self,
        plan: MigrationPlan,
        fail_on_layer: Optional[int] = None,
    ) -> LayerTransferReport:
        """
        Migrate parameters of affected layers to their target tier devices.

        Args:
            plan: Concrete MigrationPlan.
            fail_on_layer: Optional layer index to simulate an injected failure.
        """
        report = LayerTransferReport()
        self._backup_devices.clear()
        self._backup_state_dicts.clear()

        for l_idx in plan.affected_layers:
            if fail_on_layer is not None and l_idx == fail_on_layer:
                raise RuntimeError(f"Injected layer transfer failure on layer {l_idx}")

            block = self.model.blocks[l_idx]
            current_dev = str(next(block.parameters()).device)
            self._backup_devices[l_idx] = current_dev

            target_tier = plan.get_target_tier(l_idx)
            source_tier = plan.get_source_tier(l_idx)
            target_device = self.tier_devices.get(target_tier, "cpu")

            # Store backup state dict for deep verification / rollback
            self._backup_state_dicts[l_idx] = {
                k: v.clone() for k, v in block.state_dict().items()
            }

            # Execute transfer
            _, b_count, dur_ms, prov = self.transfer_provider.transfer_module(
                module=block,
                source_tier=source_tier,
                destination_tier=target_tier,
                target_device=target_device,
            )

            report.layers_transferred.append(l_idx)
            report.total_bytes += b_count
            report.duration_ms += dur_ms
            report.per_layer_bytes[l_idx] = b_count
            report.provenance = prov

        return report

    def rollback(self) -> None:
        """Restore all transferred layers to their previous devices and state dicts."""
        for l_idx, dev in self._backup_devices.items():
            block = self.model.blocks[l_idx]
            block.to(dev)
            if l_idx in self._backup_state_dicts:
                block.load_state_dict(self._backup_state_dicts[l_idx])
        self._backup_devices.clear()
        self._backup_state_dicts.clear()


class KVCacheTransferManager:
    """
    Coordinates migration of DynamicCache attention entries for affected layers.
    """

    def __init__(
        self,
        tier_devices: Dict[TierId, str],
        transfer_provider: MigrationTransferProvider,
    ) -> None:
        self.tier_devices = tier_devices
        self.transfer_provider = transfer_provider
        self._backup_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor, str]] = {}

    def transfer_cache(
        self,
        kv_cache: Optional[Any],
        plan: MigrationPlan,
        fail_kv: bool = False,
    ) -> KVCacheTransferReport:
        """
        Migrate key/value cache tensors for affected layers only.
        """
        report = KVCacheTransferReport()
        self._backup_kv.clear()

        if kv_cache is None or not hasattr(kv_cache, "layers"):
            return report

        if fail_kv:
            raise RuntimeError("Injected KV-cache transfer failure")

        for l_idx in plan.affected_layers:
            if l_idx >= len(kv_cache.layers):
                continue

            layer_cache = kv_cache.layers[l_idx]
            k = getattr(layer_cache, "keys", None)
            v = getattr(layer_cache, "values", None)

            if k is None or v is None or not isinstance(k, torch.Tensor):
                continue

            target_tier = plan.get_target_tier(l_idx)
            source_tier = plan.get_source_tier(l_idx)
            target_device = self.tier_devices.get(target_tier, "cpu")

            # Backup original references and devices
            self._backup_kv[l_idx] = (k, v, str(k.device))

            # Transfer keys
            k_moved, k_bytes, k_dur, prov = self.transfer_provider.transfer_tensor(
                k, source_tier, target_tier, target_device
            )
            # Transfer values
            v_moved, v_bytes, v_dur, _ = self.transfer_provider.transfer_tensor(
                v, source_tier, target_tier, target_device
            )

            # Update cache layer in-place
            layer_cache.keys = k_moved
            layer_cache.values = v_moved

            total_l_bytes = k_bytes + v_bytes
            report.layers_transferred.append(l_idx)
            report.total_bytes += total_l_bytes
            report.duration_ms += (k_dur + v_dur)
            report.per_layer_bytes[l_idx] = total_l_bytes
            report.provenance = prov

        return report

    def rollback(self, kv_cache: Optional[Any]) -> None:
        """Restore previous key and value tensor references in DynamicCache."""
        if kv_cache is None or not hasattr(kv_cache, "layers"):
            return
        for l_idx, (orig_k, orig_v, _) in self._backup_kv.items():
            if l_idx < len(kv_cache.layers):
                kv_cache.layers[l_idx].keys = orig_k
                kv_cache.layers[l_idx].values = orig_v
        self._backup_kv.clear()
