"""
Chronological walk-forward time-series evaluation and structured prediction tracing.

Guarantees zero temporal leakage: predictors are only exposed to history up to t_pred.
Evaluates future forecast accuracy against subsequent ground truth observations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from src.prediction.base import Predictor, extract_target_series
from src.prediction.metrics import calculate_metrics
from src.prediction.types import PredictionResult
from src.state.types import RuntimeState


@dataclass(frozen=True)
class PredictionTracePoint:
    """
    Evaluation record pairing a single forecasted step with its realized ground truth.
    """
    prediction_timestamp: float
    forecast_timestamp: float
    target_name: str
    predicted_value: Optional[float]
    actual_value: Optional[float]
    error: Optional[float]
    is_available: bool
    predictor_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_timestamp": self.prediction_timestamp,
            "forecast_timestamp": self.forecast_timestamp,
            "target_name": self.target_name,
            "predicted_value": self.predicted_value,
            "actual_value": self.actual_value,
            "error": self.error,
            "is_available": self.is_available,
            "predictor_name": self.predictor_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PredictionTracePoint:
        return cls(
            prediction_timestamp=data["prediction_timestamp"],
            forecast_timestamp=data["forecast_timestamp"],
            target_name=data["target_name"],
            predicted_value=data["predicted_value"],
            actual_value=data["actual_value"],
            error=data.get("error"),
            is_available=data["is_available"],
            predictor_name=data["predictor_name"],
        )


class PredictionTrace:
    """
    Container for historical prediction points recorded during walk-forward evaluation.
    """

    def __init__(self, predictor_name: str) -> None:
        self.predictor_name = predictor_name
        self.points: List[PredictionTracePoint] = []

    def add_point(self, point: PredictionTracePoint) -> None:
        self.points.append(point)

    def __len__(self) -> int:
        return len(self.points)

    def get_target_points(self, target_name: str) -> List[PredictionTracePoint]:
        return [p for p in self.points if p.target_name == target_name]

    def compute_metrics(self, target_name: Optional[str] = None) -> Dict[str, Any]:
        """Compute MAE, RMSE, and summary statistics across collected trace points."""
        targets = [target_name] if target_name is not None else list({p.target_name for p in self.points})
        metrics_by_target: Dict[str, Any] = {}

        for tgt in targets:
            pts = self.get_target_points(tgt)
            if not pts:
                continue

            acts = [p.actual_value for p in pts]
            preds = [p.predicted_value for p in pts]
            m = calculate_metrics(acts, preds)
            metrics_by_target[tgt] = m

        return metrics_by_target

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predictor_name": self.predictor_name,
            "point_count": len(self.points),
            "points": [p.to_dict() for p in self.points],
            "metrics": self.compute_metrics(),
        }

    def save_json(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: Union[str, Path]) -> PredictionTrace:
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        trace = cls(predictor_name=data["predictor_name"])
        for p_dict in data.get("points", []):
            trace.add_point(PredictionTracePoint.from_dict(p_dict))
        return trace


def evaluate_walk_forward(
    predictor: Predictor,
    trace: Sequence[RuntimeState],
    horizon_seconds: float = 2.0,
    min_history: int = 5,
    step_interval_seconds: Optional[float] = None,
) -> PredictionTrace:
    """
    Perform rigorous walk-forward time-series evaluation on an ordered RuntimeState trace.

    At each discrete step i:
    1. Feeds trace[:i+1] to the predictor (strictly past information).
    2. Predicts future values over horizon_seconds.
    3. Finds corresponding realized observations in trace[i+1:].
    4. Logs PredictionTracePoint with forecast, ground truth, and error.
    """
    pred_trace = PredictionTrace(predictor_name=predictor.name)
    n = len(trace)
    if n < min_history + 1:
        return pred_trace

    # Build fast timestamp lookup for ground truth matching
    ts_map = {s.timestamp: s for s in trace}
    all_timestamps = [s.timestamp for s in trace]

    for i in range(min_history - 1, n - 1):
        window = trace[: i + 1]
        t_pred = window[-1].timestamp

        res: PredictionResult = predictor.predict(
            state_window=window,
            horizon_seconds=horizon_seconds,
            step_interval_seconds=step_interval_seconds,
        )

        if not res.is_valid:
            continue

        for tgt_name, forecast in res.targets.items():
            if not forecast.is_available:
                # Record unavailable point
                for f_ts in forecast.timestamps:
                    pred_trace.add_point(
                        PredictionTracePoint(
                            prediction_timestamp=t_pred,
                            forecast_timestamp=f_ts,
                            target_name=tgt_name,
                            predicted_value=None,
                            actual_value=None,
                            error=None,
                            is_available=False,
                            predictor_name=predictor.name,
                        )
                    )
                continue

            for f_ts, p_val in zip(forecast.timestamps, forecast.values):
                # Find closest actual timestamp in subsequent states
                subsequent_ts = [t for t in all_timestamps if t >= f_ts]
                if not subsequent_ts:
                    # Beyond available trace duration
                    continue

                best_ts = min(subsequent_ts, key=lambda t: abs(t - f_ts))
                # Only pair if close enough (within half a step)
                if abs(best_ts - f_ts) > (step_interval_seconds or 1.0):
                    continue

                actual_state = ts_map[best_ts]
                _, act_vals, act_avail, _ = extract_target_series([actual_state], tgt_name)
                act_val = act_vals[0] if (act_avail and act_vals[0] is not None) else None

                err = float(act_val - p_val) if (act_val is not None and p_val is not None) else None

                pred_trace.add_point(
                    PredictionTracePoint(
                        prediction_timestamp=t_pred,
                        forecast_timestamp=f_ts,
                        target_name=tgt_name,
                        predicted_value=p_val,
                        actual_value=act_val,
                        error=err,
                        is_available=True,
                        predictor_name=predictor.name,
                    )
                )

    return pred_trace
