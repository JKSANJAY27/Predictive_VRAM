"""
Request validation and idempotency checking for Module 8.

Verifies structural integrity of MigrationRequest, prevents stale execution,
and handles duplicate/idempotent transition requests safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.controller.types import MigrationRequest
from src.runtime.partition import PartitionPlan


@dataclass(frozen=True)
class ValidationResult:
    """Result of MigrationRequest validation."""
    is_valid: bool
    is_idempotent: bool = False
    reason: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.is_valid and not self.is_idempotent


class MigrationValidator:
    """
    Validates MigrationRequest contracts prior to transaction staging.
    """

    @classmethod
    def validate(
        cls,
        request: MigrationRequest,
        current_plan: PartitionPlan,
    ) -> ValidationResult:
        """
        Validate MigrationRequest against current active runtime plan.

        Args:
            request: The MigrationRequest emitted by Module 7.
            current_plan: The active PartitionPlan currently loaded in runtime.

        Returns:
            ValidationResult with validation status, idempotency flag, and reason.
        """
        if request is None:
            return ValidationResult(is_valid=False, reason="MigrationRequest cannot be None.")

        # 1. Source and target plan existence
        if request.source_plan is None:
            return ValidationResult(is_valid=False, reason="request.source_plan is None.")
        if request.target_plan is None:
            return ValidationResult(is_valid=False, reason="request.target_plan is None.")

        # 2. Structural validity of target plan
        try:
            request.target_plan.validate()
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                reason=f"target_plan failed structural validation: {e}",
            )

        # 3. Layer count matching
        if request.source_plan.total_layers != request.target_plan.total_layers:
            return ValidationResult(
                is_valid=False,
                reason=(
                    f"Layer count mismatch between source ({request.source_plan.total_layers}) "
                    f"and target ({request.target_plan.total_layers})."
                ),
            )

        # 4. Idempotency: target matches current active plan
        if current_plan == request.target_plan:
            return ValidationResult(
                is_valid=True,
                is_idempotent=True,
                reason="Current runtime plan is already identical to target plan (idempotent).",
            )

        # 5. Stale request check: active runtime plan does not match request source plan
        if current_plan != request.source_plan:
            return ValidationResult(
                is_valid=False,
                reason=(
                    f"Stale migration request: active runtime plan '{current_plan}' does not "
                    f"match request source_plan '{request.source_plan}'."
                ),
            )

        # 6. Check non-negative transfer estimations
        if request.estimated_kv_transfer_bytes < 0:
            return ValidationResult(
                is_valid=False,
                reason=f"estimated_kv_transfer_bytes cannot be negative: {request.estimated_kv_transfer_bytes}",
            )
        if request.switching_cost < 0:
            return ValidationResult(
                is_valid=False,
                reason=f"switching_cost cannot be negative: {request.switching_cost}",
            )

        # 7. Check changed layers bounds
        total = request.source_plan.total_layers
        for l in request.changed_layers:
            if not (0 <= l < total):
                return ValidationResult(
                    is_valid=False,
                    reason=f"Changed layer index {l} out of bounds [0, {total - 1}].",
                )

        return ValidationResult(is_valid=True, is_idempotent=False, reason=None)
