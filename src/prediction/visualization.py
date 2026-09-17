"""
Plotting utilities for forecasting trajectories.

Provides clean trajectory plots comparing:
- Observed history
- Predicted forecast horizon
- Actual ground-truth future

Guarded by import checks so matplotlib is not a hard dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

from src.prediction.types import TargetForecast
from src.state.types import RuntimeState


def plot_forecast_vs_actual(
    history_states: Sequence[RuntimeState],
    forecast: TargetForecast,
    actual_future_states: Optional[Sequence[RuntimeState]] = None,
    target_name: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Generate a 2D line plot showing observed history, forecast, and actual future.

    Args:
        history_states: Past states up to prediction time t0.
        forecast: TargetForecast object containing future predictions.
        actual_future_states: Optional future states to plot ground truth.
        target_name: Optional label override.
        save_path: Optional file path to save plot image.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError("matplotlib is required to plot trajectories.") from err

    from src.prediction.base import extract_target_series

    tgt = target_name or forecast.target_name

    # Extract history
    h_ts, h_vals, h_avail, _ = extract_target_series(history_states, tgt)
    valid_h = [(t, v) for t, v in zip(h_ts, h_vals) if v is not None]

    plt.figure(figsize=(9, 4.5))

    # 1. Observed history
    if valid_h:
        plt.plot(
            [p[0] for p in valid_h],
            [p[1] for p in valid_h],
            label="Observed History",
            color="#2563EB",
            linewidth=2.0,
            marker="o",
            markersize=4,
        )

    # 2. Predicted trajectory
    f_pairs = [(t, v) for t, v in zip(forecast.timestamps, forecast.values) if v is not None]
    if f_pairs:
        plt.plot(
            [p[0] for p in f_pairs],
            [p[1] for p in f_pairs],
            label=f"Forecast (status={forecast.status})",
            color="#DC2626",
            linestyle="--",
            linewidth=2.0,
            marker="x",
            markersize=5,
        )

    # 3. Ground truth future
    if actual_future_states:
        a_ts, a_vals, a_avail, _ = extract_target_series(actual_future_states, tgt)
        valid_a = [(t, v) for t, v in zip(a_ts, a_vals) if v is not None]
        if valid_a:
            plt.plot(
                [p[0] for p in valid_a],
                [p[1] for p in valid_a],
                label="Realized Ground Truth",
                color="#059669",
                linestyle=":",
                linewidth=1.8,
            )

    plt.title(f"Predictive Trajectory: {tgt}", fontsize=12, fontweight="bold")
    plt.xlabel("Timestamp (s)", fontsize=10)
    plt.ylabel(tgt, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=150)
        plt.close()
    else:
        plt.show()
