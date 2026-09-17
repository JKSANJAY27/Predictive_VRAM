"""
Transparent baseline time-series predictors:
- LastValuePredictor: y_hat(t+Delta) = y(t)
- LinearTrendPredictor: y_hat(t+Delta) = y(t) + s * Delta
- MovingAveragePredictor: rolling mean over recent observations
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from src.prediction.base import Predictor


class LastValuePredictor(Predictor):
    """
    Persistence / Last-Value Baseline Predictor:
    Forecasts all future steps as equal to the latest observed value.
    """

    @property
    def name(self) -> str:
        return "last_value"

    def forecast_series(
        self,
        timestamps: Sequence[float],
        values: Sequence[Optional[float]],
        forecast_timestamps: Sequence[float],
        target_name: str,
    ) -> Tuple[List[Optional[float]], Optional[float]]:
        # Find latest non-None value
        valid_pairs = [(t, v) for t, v in zip(timestamps, values) if v is not None]
        if not valid_pairs:
            return [None] * len(forecast_timestamps), None

        last_val = valid_pairs[-1][1]
        preds: List[Optional[float]] = [float(last_val)] * len(forecast_timestamps)

        # Compute error estimate: mean absolute step-to-step delta over recent observations
        if len(valid_pairs) > 1:
            diffs = [abs(valid_pairs[i][1] - valid_pairs[i - 1][1]) for i in range(1, len(valid_pairs))]
            error_est = float(sum(diffs) / len(diffs))
        else:
            error_est = 0.0

        return preds, error_est


class LinearTrendPredictor(Predictor):
    """
    Linear Trend Baseline Predictor:
    Estimates recent slope s and projects: y_hat(t + Delta) = y(t) + s * Delta.
    Uses ordinary least-squares on the recent window for stability against single-sample noise.
    """

    def __init__(
        self,
        targets: Optional[Sequence[str]] = None,
        minimum_history_length: int = 3,
        window_size: Optional[int] = 10,
    ) -> None:
        super().__init__(targets=targets, minimum_history_length=minimum_history_length)
        self._window_size = window_size

    @property
    def name(self) -> str:
        return "linear_trend"

    def forecast_series(
        self,
        timestamps: Sequence[float],
        values: Sequence[Optional[float]],
        forecast_timestamps: Sequence[float],
        target_name: str,
    ) -> Tuple[List[Optional[float]], Optional[float]]:
        valid_pairs = [(t, v) for t, v in zip(timestamps, values) if v is not None]
        if len(valid_pairs) < self.minimum_history_length:
            return [None] * len(forecast_timestamps), None

        # Take the most recent window_size points
        if self._window_size is not None and len(valid_pairs) > self._window_size:
            valid_pairs = valid_pairs[-self._window_size:]

        ts = [p[0] for p in valid_pairs]
        ys = [p[1] for p in valid_pairs]
        n = len(ts)

        t_mean = sum(ts) / n
        y_mean = sum(ys) / n

        num = sum((t - t_mean) * (y - y_mean) for t, y in zip(ts, ys))
        den = sum((t - t_mean) ** 2 for t in ts)

        if den == 0.0:
            # All timestamps identical; fall back to last value
            slope = 0.0
            intercept = ys[-1]
        else:
            slope = num / den
            intercept = y_mean - slope * t_mean

        preds: List[Optional[float]] = [float(intercept + slope * ft) for ft in forecast_timestamps]

        # Error estimate: in-sample MAE of linear fit
        residuals = [abs(y - (intercept + slope * t)) for t, y in zip(ts, ys)]
        error_est = float(sum(residuals) / n) if n > 0 else 0.0

        return preds, error_est


class MovingAveragePredictor(Predictor):
    """
    Moving Average Baseline Predictor:
    Forecasts future steps as the arithmetic mean of the recent k observations.
    """

    def __init__(
        self,
        targets: Optional[Sequence[str]] = None,
        minimum_history_length: int = 3,
        window_size: int = 5,
    ) -> None:
        super().__init__(targets=targets, minimum_history_length=minimum_history_length)
        self._window_size = max(1, window_size)

    @property
    def name(self) -> str:
        return "moving_average"

    def forecast_series(
        self,
        timestamps: Sequence[float],
        values: Sequence[Optional[float]],
        forecast_timestamps: Sequence[float],
        target_name: str,
    ) -> Tuple[List[Optional[float]], Optional[float]]:
        valid_vals = [v for v in values if v is not None]
        if len(valid_vals) < self.minimum_history_length:
            return [None] * len(forecast_timestamps), None

        k_vals = valid_vals[-self._window_size:]
        mean_val = float(sum(k_vals) / len(k_vals))

        preds: List[Optional[float]] = [mean_val] * len(forecast_timestamps)

        # Error estimate: mean absolute deviation from rolling mean
        mad = sum(abs(v - mean_val) for v in k_vals) / len(k_vals)
        return preds, float(mad)
