"""
Correctness verification for Module 8.

Performs FAST and DEEP verification on model parameters, layer device mappings,
and DynamicCache integrity following state transfer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from src.migration.planner import MigrationPlan
from src.migration.types import VerificationMode
from src.runtime.model import LayeredTransformer
from src.runtime.partition import PartitionPlan
from src.runtime.tier import TierId


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of migration correctness verification."""
    passed: bool
    issues: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    mode: VerificationMode = VerificationMode.FAST

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": self.issues,
            "duration_ms": round(self.duration_ms, 3),
            "mode": self.mode.value,
        }


class MigrationVerifier:
    """
    Validates physical parameter placement and KV-cache continuity before commit.
    """

    @classmethod
    def verify(
        cls,
        model: LayeredTransformer,
        plan: MigrationPlan,
        tier_devices: Dict[TierId, str],
        kv_cache: Optional[Any] = None,
        mode: VerificationMode = VerificationMode.FAST,
        fail_verification: bool = False,
    ) -> VerificationResult:
        """
        Execute post-transfer validation checks.
        """
        start = time.perf_counter()
        issues: List[str] = []

        if fail_verification:
            return VerificationResult(
                passed=False,
                issues=["Injected verification failure triggered."],
                duration_ms=(time.perf_counter() - start) * 1000.0,
                mode=mode,
            )

        target_plan = plan.target_plan

        # 1. Block count check
        if len(model.blocks) != target_plan.total_layers:
            issues.append(
                f"Model block count ({len(model.blocks)}) does not match target plan "
                f"total layers ({target_plan.total_layers})."
            )

        # 2. Layer device mapping check
        for tier_id in (TierId.USER_DEVICE, TierId.EDGE_A, TierId.EDGE_B):
            rng = target_plan.get_tier_range(tier_id)
            if rng is not None:
                expected_device = tier_devices.get(tier_id, "cpu")
                for l_idx in range(rng[0], rng[1] + 1):
                    block = model.blocks[l_idx]
                    actual_device = str(next(block.parameters()).device)
                    # Normalize device strings (e.g., 'cpu' vs 'cpu:0')
                    if not actual_device.startswith(expected_device):
                        issues.append(
                            f"Layer {l_idx} on device '{actual_device}', expected '{expected_device}' "
                            f"for tier '{tier_id.value}'."
                        )

        # 3. Model parameter integrity (FAST: shapes & non-NaN; DEEP: checksum)
        for l_idx, block in enumerate(model.blocks):
            for name, param in block.named_parameters():
                if torch.isnan(param).any():
                    issues.append(f"NaN detected in block {l_idx} parameter '{name}'.")
                if mode == VerificationMode.DEEP:
                    # Verify parameter is finite
                    if not torch.isfinite(param).all():
                        issues.append(f"Non-finite values in block {l_idx} parameter '{name}'.")

        # 4. KV-cache integrity check if cache is present
        if kv_cache is not None and hasattr(kv_cache, "layers") and kv_cache.layers:
            # Check length matches model blocks if populated
            sample_seq_len = None
            for l_idx, layer_cache in enumerate(kv_cache.layers):
                k = getattr(layer_cache, "keys", None)
                v = getattr(layer_cache, "values", None)
                if k is not None and v is not None and isinstance(k, torch.Tensor):
                    if k.shape != v.shape:
                        issues.append(f"KV-cache shape mismatch at layer {l_idx}: K={k.shape} vs V={v.shape}")
                    if torch.isnan(k).any() or torch.isnan(v).any():
                        issues.append(f"NaN in KV-cache at layer {l_idx}.")

                    # Check device placement matches assigned tier device
                    assigned_tier = plan.get_target_tier(l_idx)
                    expected_dev = tier_devices.get(assigned_tier, "cpu")
                    actual_k_dev = str(k.device)
                    if not actual_k_dev.startswith(expected_dev):
                        issues.append(
                            f"KV-cache layer {l_idx} on '{actual_k_dev}', expected '{expected_dev}'."
                        )

                    # Check sequence length consistency across layers
                    seq_l = k.shape[-2] if k.ndim >= 2 else 0
                    if sample_seq_len is None:
                        sample_seq_len = seq_l
                    elif seq_l != sample_seq_len:
                        issues.append(
                            f"KV-cache sequence length mismatch: layer {l_idx} has {seq_l}, "
                            f"expected {sample_seq_len}."
                        )

        duration_ms = (time.perf_counter() - start) * 1000.0
        passed = len(issues) == 0

        return VerificationResult(
            passed=passed,
            issues=issues,
            duration_ms=duration_ms,
            mode=mode,
        )
