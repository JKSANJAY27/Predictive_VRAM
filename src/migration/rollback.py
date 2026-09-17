"""
Rollback orchestration for Module 8.

Guarantees transactional integrity by reversing staged model parameters,
KV-cache placements, and execution routes upon migration failure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.migration.state import KVCacheTransferManager, LayerTransferManager
from src.runtime.partition import PartitionPlan


@dataclass(frozen=True)
class RollbackReport:
    """Outcome of a rollback operation."""
    success: bool
    issues: List[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "issues": self.issues,
            "duration_ms": round(self.duration_ms, 3),
        }


class RollbackManager:
    """
    Manages state restoration when migration fails prior to transaction commit.
    """

    @classmethod
    def execute_rollback(
        cls,
        layer_manager: LayerTransferManager,
        kv_manager: KVCacheTransferManager,
        runtime_adapter: Any,
        source_plan: PartitionPlan,
        kv_cache: Optional[Any] = None,
    ) -> RollbackReport:
        """
        Reverse all staged changes and restore the runtime to the source plan.
        """
        start = time.perf_counter()
        issues: List[str] = []

        try:
            # 1. Restore model layer parameters and device placement
            layer_manager.rollback()
        except Exception as e:
            issues.append(f"Layer rollback failed: {e}")

        try:
            # 2. Restore KV-cache references and device placement
            kv_manager.rollback(kv_cache)
        except Exception as e:
            issues.append(f"KV-cache rollback failed: {e}")

        try:
            # 3. Ensure runtime routing is restored to source plan
            runtime_adapter.set_active_plan(source_plan)
        except Exception as e:
            issues.append(f"Route restoration failed: {e}")

        duration_ms = (time.perf_counter() - start) * 1000.0
        success = len(issues) == 0

        return RollbackReport(
            success=success,
            issues=issues,
            duration_ms=duration_ms,
        )
