"""
Forecasting evaluation metrics, threshold lead-time analysis, and abrupt change detection.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple


def filter_valid_pairs(
    actuals: Sequence[Optional[float]],
    predictions: Sequence[Optional[float]],
) -> Tuple[List[float], List[float]]:
    """Filter out any pairs where either actual or prediction is None."""
    acts: List[float] = []
    preds: List[float] = []
    for a, p in zip(actuals, predictions):
        if a is not None and p is not None:
            acts.append(float(a))
            preds.append(float(p))
    return acts, preds


def mean_absolute_error(
    actuals: Sequence[Optional[float]],
    predictions: Sequence[Optional[float]],
) -> float:
    """Mean Absolute Error (MAE)."""
    acts, preds = filter_valid_pairs(actuals, predictions)
    if not acts:
        return 0.0
    return float(sum(abs(a - p) for a, p in zip(acts, preds)) / len(acts))


def root_mean_squared_error(
    actuals: Sequence[Optional[float]],
    predictions: Sequence[Optional[float]],
) -> float:
    """Root Mean Squared Error (RMSE)."""
    acts, preds = filter_valid_pairs(actuals, predictions)
    if not acts:
        return 0.0
    mse = sum((a - p) ** 2 for a, p in zip(acts, preds)) / len(acts)
    return float(math.sqrt(mse))


def mean_absolute_percentage_error(
    actuals: Sequence[Optional[float]],
    predictions: Sequence[Optional[float]],
    epsilon: float = 1e-4,
) -> Optional[float]:
    """
    Mean Absolute Percentage Error (MAPE).
    Returns None if values are near zero to prevent division by zero / exploding percentages.
    """
    acts, preds = filter_valid_pairs(actuals, predictions)
    if not acts:
        return None
    # If any actual value is too close to zero, MAPE is mathematically inappropriate
    if any(abs(a) < epsilon for a in acts):
        return None
    pct_errors = [abs((a - p) / a) for a, p in zip(acts, preds)]
    return float(sum(pct_errors) / len(pct_errors) * 100.0)


def median_absolute_error(
    actuals: Sequence[Optional[float]],
    predictions: Sequence[Optional[float]],
) -> float:
    """Median Absolute Error."""
    acts, preds = filter_valid_pairs(actuals, predictions)
    if not acts:
        return 0.0
    errors = sorted(abs(a - p) for a, p in zip(acts, preds))
    n = len(errors)
    mid = n // 2
    if n % 2 == 1:
        return float(errors[mid])
    return float((errors[mid - 1] + errors[mid]) / 2.0)


def max_absolute_error(
    actuals: Sequence[Optional[float]],
    predictions: Sequence[Optional[float]],
) -> float:
    """Maximum Absolute Error across all evaluated points."""
    acts, preds = filter_valid_pairs(actuals, predictions)
    if not acts:
        return 0.0
    return float(max(abs(a - p) for a, p in zip(acts, preds)))


def calculate_metrics(
    actuals: Sequence[Optional[float]],
    predictions: Sequence[Optional[float]],
) -> Dict[str, Optional[float]]:
    """Compute standard evaluation metrics suite for a forecast series."""
    acts, preds = filter_valid_pairs(actuals, predictions)
    if not acts:
        return {
            "sample_count": 0,
            "mae": None,
            "rmse": None,
            "mape": None,
            "median_ae": None,
            "max_ae": None,
        }

    return {
        "sample_count": len(acts),
        "mae": mean_absolute_error(acts, preds),
        "rmse": root_mean_squared_error(acts, preds),
        "mape": mean_absolute_percentage_error(acts, preds),
        "median_ae": median_absolute_error(acts, preds),
        "max_ae": max_absolute_error(acts, preds),
    }


def calculate_lead_time(
    actual_timestamps: Sequence[float],
    actual_values: Sequence[Optional[float]],
    predicted_crossing_time: Optional[float],
    threshold: float,
    mode: str = "lower",
) -> Optional[float]:
    """
    Calculate prediction lead-time: actual crossing timestamp minus predicted crossing timestamp.

    A positive lead time indicates proactive detection before the system crossed the boundary.
    A negative lead time indicates late detection (after actual crossing).
    Returns None if actual threshold crossing never occurs or prediction never triggered.

    Args:
        actual_timestamps: Observed timestamps.
        actual_values: Observed metric values.
        predicted_crossing_time: Timestamp when predictor forecasted crossing would occur.
        threshold: Critical boundary value.
        mode: 'lower' (value falls below threshold) or 'upper' (value rises above threshold).
    """
    if predicted_crossing_time is None:
        return None

    actual_crossing_time: Optional[float] = None
    for t, v in zip(actual_timestamps, actual_values):
        if v is None:
            continue
        if mode == "lower" and v <= threshold:
            actual_crossing_time = t
            break
        elif mode == "upper" and v >= threshold:
            actual_crossing_time = t
            break

    if actual_crossing_time is None:
        return None

    return float(actual_crossing_time - predicted_crossing_time)


class AbruptChangeDetector:
    """
    Online residual tracking utility to detect sudden network or memory shifts.
    """

    def __init__(self, sensitivity_sigma: float = 3.0, min_history: int = 5) -> None:
        self._sensitivity = sensitivity_sigma
        self._min_history = min_history
        self._residuals: List[float] = []

    def update(self, actual: float, predicted: float) -> Tuple[bool, float]:
        """
        Record a new residual and evaluate whether it represents an abrupt change / spike.

        Returns:
            (is_abrupt_change, residual)
        """
        residual = abs(actual - predicted)
        is_spike = False

        if len(self._residuals) >= self._min_history:
            mean = sum(self._residuals) / len(self._residuals)
            var = sum((r - mean) ** 2 for r in self._residuals) / len(self._residuals)
            std = math.sqrt(var) if var > 1e-9 else 0.1

            if residual > mean + self._sensitivity * std:
                is_spike = True

        self._residuals.append(residual)
        if len(self._residuals) > 50:
            self._residuals.pop(0)

        return is_spike, residual
