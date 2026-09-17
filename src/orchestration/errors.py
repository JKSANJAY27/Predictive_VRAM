"""
Explicit exception and error classification hierarchy for Module 9 orchestration.
"""

from __future__ import annotations

from src.orchestration.types import ErrorCategory


class OrchestrationError(Exception):
    """Base exception for all closed-loop orchestration failures."""
    category: ErrorCategory = ErrorCategory.NONE

    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.NONE) -> None:
        super().__init__(message)
        self.category = category


class TelemetryError(OrchestrationError):
    """Failure occurred while sampling system telemetry."""
    category = ErrorCategory.TELEMETRY_ERROR


class StateError(OrchestrationError):
    """Failure occurred while converting or buffering RuntimeState."""
    category = ErrorCategory.STATE_ERROR


class PredictionError(OrchestrationError):
    """Failure occurred inside the predictive forecasting engine."""
    category = ErrorCategory.PREDICTION_ERROR


class CandidateError(OrchestrationError):
    """Failure occurred while querying or filtering candidate partition plans."""
    category = ErrorCategory.CANDIDATE_ERROR


class CostModelError(OrchestrationError):
    """Failure occurred while evaluating multi-objective candidate scores."""
    category = ErrorCategory.COST_ERROR


class ControllerError(OrchestrationError):
    """Failure occurred inside the partition controller."""
    category = ErrorCategory.CONTROLLER_ERROR


class MigrationError(OrchestrationError):
    """Failure occurred during physical migration coordination or state transfer."""
    category = ErrorCategory.MIGRATION_ERROR


class VerificationError(OrchestrationError):
    """Failure occurred during post-migration verification."""
    category = ErrorCategory.VERIFICATION_ERROR
