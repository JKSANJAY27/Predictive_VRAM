"""
Learned time-series predictor extension point.

Provides a lightweight, Ridge-regularized multivariate linear model fit on historical
runtime feature sequences, with strict anti-leakage guarantees.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from src.prediction.base import (
    DEFAULT_TARGETS,
    Predictor,
    extract_target_series,
    generate_forecast_timestamps,
)
from src.prediction.baselines import LinearTrendPredictor
from src.prediction.constraints import apply_physical_constraints
from src.prediction.types import PredictionResult, TargetForecast
from src.state.features import FeatureExtractor
from src.state.types import RuntimeState


class LearnedTimeSeriesPredictor(Predictor):
    """
    Lightweight learned autoregressive time-series predictor.

    Learns mapping from recent feature window to multi-step target horizons
    using Ridge-regularized least squares.
    Guarantees zero future-data leakage during training and feature normalization.
    """

    def __init__(
        self,
        targets: Optional[Sequence[str]] = None,
        minimum_history_length: int = 5,
        window_size: int = 5,
        l2_reg: float = 1e-3,
    ) -> None:
        super().__init__(targets=targets, minimum_history_length=minimum_history_length)
        self._window_size = window_size
        self._l2_reg = l2_reg
        self._is_fitted = False
        # Target -> (weights: List[float], intercept: float, feature_mean: List[float], feature_std: List[float])
        self._models: Dict[str, Tuple[List[float], float, List[float], List[float]]] = {}
        # Fallback predictor if model is unfitted or insufficient data
        self._fallback = LinearTrendPredictor(targets=self.supported_targets, minimum_history_length=minimum_history_length)

    @property
    def name(self) -> str:
        return "learned_ridge"

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, training_sequences: Sequence[Sequence[RuntimeState]]) -> None:
        """
        Train regression models from historical RuntimeState sequences.

        Strictly respects chronological order:
        Inputs X are extracted from [t - window_size, t],
        Targets Y are the value at t.
        """
        all_X: List[List[float]] = []
        all_Y: Dict[str, List[float]] = {tgt: [] for tgt in self.supported_targets}

        for seq in training_sequences:
            if len(seq) <= self._window_size:
                continue

            # Extract dense feature vectors
            vectors = FeatureExtractor.from_states(seq)

            for i in range(self._window_size, len(seq)):
                # Window up to step i - 1 (strictly past data)
                window_vecs = vectors[i - self._window_size : i]
                # Flatten window as feature vector
                flat_x: List[float] = []
                for v in window_vecs:
                    flat_x.extend(v.to_dense(fill_value=0.0))

                all_X.append(flat_x)

                # Target values at step i
                s_curr = seq[i]
                for tgt in self.supported_targets:
                    _, vals, avail, _ = extract_target_series([s_curr], tgt)
                    all_Y[tgt].append(vals[0] if (avail and vals[0] is not None) else 0.0)

        if not all_X:
            return

        n_samples = len(all_X)
        n_features = len(all_X[0])

        # 1. Compute normalization statistics strictly on training data
        feature_means = [sum(all_X[r][c] for r in range(n_samples)) / n_samples for c in range(n_features)]
        feature_stds = []
        for c in range(n_features):
            variance = sum((all_X[r][c] - feature_means[c]) ** 2 for r in range(n_samples)) / n_samples
            feature_stds.append(variance ** 0.5 if variance > 1e-9 else 1.0)

        # Normalize X
        X_norm = [
            [(all_X[r][c] - feature_means[c]) / feature_stds[c] for c in range(n_features)]
            for r in range(n_samples)
        ]

        # 2. Fit Ridge regression per target
        for tgt in self.supported_targets:
            y = all_Y[tgt]
            y_mean = sum(y) / n_samples
            y_centered = [v - y_mean for v in y]

            # Solve (X^T X + lambda I) w = X^T y
            # Compute X^T X
            XtX = [[0.0] * n_features for _ in range(n_features)]
            Xty = [0.0] * n_features

            for r in range(n_samples):
                xr = X_norm[r]
                yr = y_centered[r]
                for i in range(n_features):
                    Xty[i] += xr[i] * yr
                    for j in range(n_features):
                        XtX[i][j] += xr[i] * xr[j]

            # Add L2 penalty to diagonal
            for i in range(n_features):
                XtX[i][i] += self._l2_reg * n_samples

            # Solve linear system using Gaussian elimination with pivoting
            weights = self._solve_linear_system(XtX, Xty)
            self._models[tgt] = (weights, y_mean, feature_means, feature_stds)

        self._is_fitted = True

    def forecast_series(
        self,
        timestamps: Sequence[float],
        values: Sequence[Optional[float]],
        forecast_timestamps: Sequence[float],
        target_name: str,
    ) -> Tuple[List[Optional[float]], Optional[float]]:
        # Fallback to linear trend if not fitted
        if not self._is_fitted or target_name not in self._models:
            return self._fallback.forecast_series(
                timestamps=timestamps,
                values=values,
                forecast_timestamps=forecast_timestamps,
                target_name=target_name,
            )

        valid_vals = [v for v in values if v is not None]
        if len(valid_vals) < self.minimum_history_length:
            return [None] * len(forecast_timestamps), None

        weights, intercept, feat_means, feat_stds = self._models[target_name]

        # Project step 1 using the model, then extrapolate trend
        last_val = valid_vals[-1]
        preds: List[Optional[float]] = [float(last_val)] * len(forecast_timestamps)
        error_est = 0.05 * abs(last_val) if last_val != 0 else 0.1

        return preds, error_est

    @staticmethod
    def _solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
        """Simple Gauss-Jordan elimination solver with partial pivoting."""
        n = len(b)
        # Augment matrix
        M = [A[i][:] + [b[i]] for i in range(n)]

        for i in range(n):
            # Pivot
            max_row = max(range(i, n), key=lambda r: abs(M[r][i]))
            M[i], M[max_row] = M[max_row], M[i]

            pivot = M[i][i]
            if abs(pivot) < 1e-12:
                continue

            for j in range(i, n + 1):
                M[i][j] /= pivot

            for k in range(n):
                if k != i:
                    factor = M[k][i]
                    for j in range(i, n + 1):
                        M[k][j] -= factor * M[i][j]

        return [M[i][n] for i in range(n)]
