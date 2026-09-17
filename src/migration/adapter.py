"""
Runtime adapter isolating migration execution from executor specifics in Module 8.

Exposes a clean interface for reading/setting active partition plans,
moving layers, accessing KV-cache, and coordinating quiescence.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.runtime.executor import DistributedInferenceExecutor
from src.runtime.model import LayeredTransformer
from src.runtime.partition import PartitionPlan
from src.runtime.tier import Tier, TierId


class RuntimeAdapter:
    """
    Standard interface bridging the migration engine to DistributedInferenceExecutor.
    """

    def __init__(self, executor: DistributedInferenceExecutor) -> None:
        self.executor = executor
        self._lock = threading.RLock()
        self._current_plan: Optional[PartitionPlan] = getattr(executor, "active_plan", None)
        self._active_kv_cache: Optional[Any] = None

    @property
    def model(self) -> LayeredTransformer:
        return self.executor.model

    @property
    def tiers(self) -> Dict[TierId, Tier]:
        return self.executor.tiers

    def get_tier_devices(self) -> Dict[TierId, str]:
        return {t_id: t.device for t_id, t in self.executor.tiers.items()}

    def get_current_plan(self) -> PartitionPlan:
        """Return the currently active PartitionPlan."""
        if hasattr(self.executor, "active_plan") and self.executor.active_plan is not None:
            return self.executor.active_plan
        if self._current_plan is not None:
            return self._current_plan
        # Default: 12-layer monolithic
        return PartitionPlan.monolithic(total_layers=len(self.model.blocks))

    def set_active_plan(self, plan: PartitionPlan) -> None:
        """Update active PartitionPlan and reconfigure tier ranges."""
        self._current_plan = plan
        if hasattr(self.executor, "active_plan"):
            self.executor.active_plan = plan
        self.executor._setup_tiers_for_plan(plan)

    def get_layer(self, layer_idx: int) -> nn.Module:
        """Retrieve a specific transformer block."""
        return self.model.blocks[layer_idx]

    def get_kv_cache(self) -> Optional[Any]:
        """Retrieve active DynamicCache instance if currently executing generation."""
        if hasattr(self.executor, "active_kv_cache"):
            return self.executor.active_kv_cache
        return self._active_kv_cache

    def set_active_kv_cache(self, kv_cache: Optional[Any]) -> None:
        """Register active DynamicCache for migration coordination."""
        self._active_kv_cache = kv_cache
        if hasattr(self.executor, "active_kv_cache"):
            self.executor.active_kv_cache = kv_cache

    def acquire_migration_lock(self, timeout: float = 5.0) -> bool:
        """Acquire synchronization lock to ensure inference quiescence."""
        return self._lock.acquire(timeout=timeout)

    def release_migration_lock(self) -> None:
        """Release synchronization lock after migration completion or rollback."""
        try:
            self._lock.release()
        except RuntimeError:
            pass

    def verify_runtime(self) -> bool:
        """Check that runtime layer devices match current active plan ranges."""
        plan = self.get_current_plan()
        for tier_id in (TierId.USER_DEVICE, TierId.EDGE_A, TierId.EDGE_B):
            rng = plan.get_tier_range(tier_id)
            if rng is not None:
                expected_device = self.executor.tiers[tier_id].device
                for l in range(rng[0], rng[1] + 1):
                    dev = str(next(self.model.blocks[l].parameters()).device)
                    if not dev.startswith(expected_device):
                        return False
        return True
