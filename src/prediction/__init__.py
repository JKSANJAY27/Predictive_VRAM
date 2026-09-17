"""
Predictive VRAM and Network Forecasting Engine.

Module 4 provides predictive time-series forecasting for short-horizon resource adaptation.
"""

from src.prediction.base import (
    DEFAULT_TARGETS,
    Predictor,
    extract_target_series,
    generate_forecast_timestamps,
)
from src.prediction.baselines import (
    LastValuePredictor,
    LinearTrendPredictor,
    MovingAveragePredictor,
)
from src.prediction.constraints import (
    PHYSICAL_BOUNDS,
    apply_physical_constraints,
)
from src.prediction.evaluation import (
    PredictionTrace,
    PredictionTracePoint,
    evaluate_walk_forward,
)
from src.prediction.learned import LearnedTimeSeriesPredictor
from src.prediction.memory import (
    KVCacheAwareMemoryPredictor,
    estimate_time_to_threshold,
)
from src.prediction.metrics import (
    AbruptChangeDetector,
    calculate_lead_time,
    calculate_metrics,
    max_absolute_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    root_mean_squared_error,
)
from src.prediction.registry import (
    create_predictor_from_config,
    get_predictor,
    list_available_predictors,
    register_predictor,
)
from src.prediction.synthetic import (
    SCENARIO_NAMES,
    SyntheticTraceGenerator,
)
from src.prediction.types import (
    PredictionResult,
    TargetForecast,
)
from src.prediction.visualization import plot_forecast_vs_actual

__all__ = [
    "Predictor",
    "PredictionResult",
    "TargetForecast",
    "DEFAULT_TARGETS",
    "extract_target_series",
    "generate_forecast_timestamps",
    "LastValuePredictor",
    "LinearTrendPredictor",
    "MovingAveragePredictor",
    "KVCacheAwareMemoryPredictor",
    "estimate_time_to_threshold",
    "LearnedTimeSeriesPredictor",
    "PHYSICAL_BOUNDS",
    "apply_physical_constraints",
    "mean_absolute_error",
    "root_mean_squared_error",
    "mean_absolute_percentage_error",
    "median_absolute_error",
    "max_absolute_error",
    "calculate_metrics",
    "calculate_lead_time",
    "AbruptChangeDetector",
    "SyntheticTraceGenerator",
    "SCENARIO_NAMES",
    "PredictionTrace",
    "PredictionTracePoint",
    "evaluate_walk_forward",
    "get_predictor",
    "register_predictor",
    "create_predictor_from_config",
    "list_available_predictors",
    "plot_forecast_vs_actual",
]
