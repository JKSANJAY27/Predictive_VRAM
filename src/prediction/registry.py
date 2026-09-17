"""
Registry and factory for dynamically instantiating predictors from configuration.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type

from src.prediction.base import Predictor
from src.prediction.baselines import (
    LastValuePredictor,
    LinearTrendPredictor,
    MovingAveragePredictor,
)
from src.prediction.learned import LearnedTimeSeriesPredictor
from src.prediction.memory import KVCacheAwareMemoryPredictor

# Internal registry mapping type string to class
_REGISTRY: Dict[str, Type[Predictor]] = {
    "last_value": LastValuePredictor,
    "linear_trend": LinearTrendPredictor,
    "moving_average": MovingAveragePredictor,
    "kv_aware_memory": KVCacheAwareMemoryPredictor,
    "learned": LearnedTimeSeriesPredictor,
    "learned_ridge": LearnedTimeSeriesPredictor,
}


def register_predictor(name: str, predictor_cls: Type[Predictor]) -> None:
    """Register a new predictor class in the factory registry."""
    _REGISTRY[name.lower()] = predictor_cls


def list_available_predictors() -> List[str]:
    """Return all registered predictor names."""
    return sorted(list(_REGISTRY.keys()))


def get_predictor(
    predictor_type: str,
    targets: Optional[List[str]] = None,
    minimum_history_length: int = 3,
    **kwargs: Any,
) -> Predictor:
    """
    Instantiate a Predictor by registered name.

    Args:
        predictor_type: Name string (e.g. 'linear_trend', 'last_value').
        targets: Optional target variables list.
        minimum_history_length: Minimum observations required before forecasting.
        **kwargs: Additional predictor-specific parameters.
    """
    key = predictor_type.lower()
    cls = _REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown predictor type '{predictor_type}'. "
            f"Available types: {list_available_predictors()}"
        )

    return cls(
        targets=targets,
        minimum_history_length=minimum_history_length,
        **kwargs,
    )


def create_predictor_from_config(config: Dict[str, Any]) -> Predictor:
    """
    Instantiate a predictor from a configuration dictionary.

    Expected schema:
        type: str (e.g. 'linear_trend')
        targets: Optional[List[str]]
        minimum_history: int
        ... (additional parameters)
    """
    p_cfg = config.get("predictor", config)
    p_type = p_cfg.get("type", "linear_trend")
    targets = p_cfg.get("targets")
    min_hist = p_cfg.get("minimum_history", 3)

    # Filter out known keys and pass remaining to constructor
    extra_kwargs = {
        k: v for k, v in p_cfg.items()
        if k not in ("type", "targets", "minimum_history")
    }

    return get_predictor(
        predictor_type=p_type,
        targets=targets,
        minimum_history_length=min_hist,
        **extra_kwargs,
    )
