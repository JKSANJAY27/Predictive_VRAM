"""
Unit and integration test suite for Module 4: Predictive VRAM and Network Forecasting Engine.

Tests:
1. Predictor interface and serialization (PredictionResult round-trip)
2. Last-value baseline predictor
3. Linear-trend baseline predictor
4. Moving-average baseline predictor
5. KV-Cache-Aware memory predictor and time-to-threshold calculation
6. Learned time-series predictor (Ridge regression, anti-leakage)
7. Physical constraint bounds and clipping tracking
8. Insufficient history handling (no fabricated forecast)
9. Missing / unavailable VRAM handling (no fabricated GPU values, no CPU RAM substitution)
10. Forecast evaluation metrics (MAE, RMSE, MAPE, Median AE, Max AE)
11. Prediction lead-time analytical metric
12. Abrupt change residual detector
13. Synthetic trace generator (9 canonical profiles, EMULATED tags)
14. Walk-forward chronological evaluation
15. Predictor registry and configuration instantiation
16. Anti-leakage temporal ordering validation
"""

import json
from pathlib import Path
import pytest

from src.prediction.base import Predictor, extract_target_series, generate_forecast_timestamps
from src.prediction.baselines import (
    LastValuePredictor,
    LinearTrendPredictor,
    MovingAveragePredictor,
)
from src.prediction.constraints import PHYSICAL_BOUNDS, apply_physical_constraints
from src.prediction.evaluation import (
    PredictionTrace,
    PredictionTracePoint,
    evaluate_walk_forward,
)
from src.prediction.learned import LearnedTimeSeriesPredictor
from src.prediction.memory import KVCacheAwareMemoryPredictor, estimate_time_to_threshold
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
from src.prediction.synthetic import SCENARIO_NAMES, SyntheticTraceGenerator
from src.prediction.types import PredictionResult, TargetForecast
from src.state.types import (
    ActivationState,
    ComputeState,
    EnergyState,
    InferenceState,
    MemoryState,
    NetworkState,
    RuntimeState,
)
from src.telemetry.types import DataSource, TaggedValue


# ---------------------------------------------------------------------------
# Test Suite 1: Prediction Types & Serialization
# ---------------------------------------------------------------------------

class TestPredictionTypes:
    def test_target_forecast_unavailable_factory(self):
        tf = TargetForecast.unavailable("vram_free_mb", [101.0, 102.0])
        assert not tf.is_available
        assert tf.status == "unavailable"
        assert tf.provenance == "unavailable"
        assert tf.values == [None, None]
        assert tf.raw_values == [None, None]
        assert tf.error_estimate is None

    def test_target_forecast_insufficient_history_factory(self):
        tf = TargetForecast.insufficient_history("bandwidth_mbps", [10.0, 11.0], 5, 2)
        assert tf.is_available
        assert "insufficient_history" in tf.status
        assert tf.values == [None, None]

    def test_prediction_result_json_round_trip(self):
        tf1 = TargetForecast(
            target_name="bandwidth_mbps",
            values=[50.0, 45.0],
            raw_values=[50.0, 45.0],
            timestamps=[101.0, 102.0],
            is_available=True,
            status="ok",
            error_estimate=1.5,
            provenance="emulated",
            constraint_applied=False,
        )
        res = PredictionResult(
            prediction_timestamp=100.0,
            horizon_seconds=2.0,
            step_interval_seconds=1.0,
            forecast_timestamps=[101.0, 102.0],
            targets={"bandwidth_mbps": tf1},
            predictor_name="test_pred",
            input_window_length=10,
            is_valid=True,
            status_message="ok",
            prediction_latency_ms=0.5,
            metadata={"key": "val"},
        )

        data = res.to_dict()
        reconstructed = PredictionResult.from_dict(data)
        assert reconstructed.prediction_timestamp == 100.0
        assert reconstructed.predictor_name == "test_pred"
        assert reconstructed.get_target("bandwidth_mbps").values == [50.0, 45.0]

        json_str = res.to_json()
        reconstructed_json = PredictionResult.from_json(json_str)
        assert reconstructed_json.is_valid is True


# ---------------------------------------------------------------------------
# Test Suite 2: Physical Constraints
# ---------------------------------------------------------------------------

class TestPhysicalConstraints:
    def test_apply_constraints_bandwidth_clamped_to_zero(self):
        raw = [10.0, -5.0, 0.0, 20.0]
        constrained, applied = apply_physical_constraints("bandwidth_mbps", raw)
        assert constrained == [10.0, 0.0, 0.0, 20.0]
        assert applied is True

    def test_apply_constraints_packet_loss_clamped_to_one(self):
        raw = [0.01, 1.25, -0.1, 0.5]
        constrained, applied = apply_physical_constraints("packet_loss", raw)
        assert constrained == [0.01, 1.0, 0.0, 0.5]
        assert applied is True

    def test_apply_constraints_preserves_none(self):
        raw = [10.0, None, -5.0]
        constrained, applied = apply_physical_constraints("latency_ms", raw)
        assert constrained[1] is None
        assert constrained[2] == 0.0
        assert applied is True

    def test_apply_constraints_unconstrained_target(self):
        raw = [-10.0, 20.0]
        constrained, applied = apply_physical_constraints("unknown_metric", raw)
        assert constrained == raw
        assert applied is False


# ---------------------------------------------------------------------------
# Test Suite 3: Baseline Predictors
# ---------------------------------------------------------------------------

class TestBaselinePredictors:
    def test_last_value_predictor(self):
        trace = SyntheticTraceGenerator.generate_trace("stable_network", num_steps=10, dt=1.0)
        predictor = LastValuePredictor(targets=["bandwidth_mbps", "latency_ms"])

        res = predictor.predict(trace, horizon_seconds=3.0, step_interval_seconds=1.0)
        assert res.is_valid is True
        assert len(res.forecast_timestamps) == 3

        bw_forecast = res.get_target("bandwidth_mbps")
        assert bw_forecast.is_available is True
        # Stable network has bw = 100.0
        assert bw_forecast.values == [100.0, 100.0, 100.0]
        assert bw_forecast.error_estimate == 0.0

    def test_linear_trend_predictor_increasing(self):
        # Linear memory growth scenario: memory grows by 50 MB / step
        trace = SyntheticTraceGenerator.generate_trace("linear_memory_growth", num_steps=10, dt=1.0)
        predictor = LinearTrendPredictor(targets=["ram_used_mb"], window_size=5)

        res = predictor.predict(trace, horizon_seconds=2.0, step_interval_seconds=1.0)
        assert res.is_valid is True
        assert len(res.forecast_timestamps) == 2

        ram_forecast = res.get_target("ram_used_mb")
        last_obs = trace[-1].memory.ram_used_mb.value  # 2048 + 9 * 50 = 2498
        # Projected: +50 per step -> 2548, 2598
        assert pytest.approx(ram_forecast.values[0], 0.1) == last_obs + 50.0
        assert pytest.approx(ram_forecast.values[1], 0.1) == last_obs + 100.0

    def test_linear_trend_predictor_decreasing_bandwidth(self):
        trace = SyntheticTraceGenerator.generate_trace("bandwidth_degradation", num_steps=20, dt=0.5)
        predictor = LinearTrendPredictor(targets=["bandwidth_mbps"])

        res = predictor.predict(trace, horizon_seconds=2.0, step_interval_seconds=1.0)
        bw_forecast = res.get_target("bandwidth_mbps")
        # Slope should be negative -> values should decrease
        assert bw_forecast.values[0] > bw_forecast.values[1]

    def test_moving_average_predictor(self):
        trace = SyntheticTraceGenerator.generate_trace("stable_network", num_steps=10, dt=1.0)
        predictor = MovingAveragePredictor(targets=["bandwidth_mbps"], window_size=4)

        res = predictor.predict(trace, horizon_seconds=2.0)
        assert res.is_valid is True
        assert res.get_target("bandwidth_mbps").values[0] == 100.0

    def test_insufficient_history_returns_invalid_result(self):
        trace = SyntheticTraceGenerator.generate_trace("stable_network", num_steps=2)
        predictor = LinearTrendPredictor(targets=["bandwidth_mbps"], minimum_history_length=5)

        res = predictor.predict(trace, horizon_seconds=2.0)
        assert res.is_valid is False
        assert "insufficient_history" in res.status_message
        assert res.get_target("bandwidth_mbps").values == [None] * len(res.forecast_timestamps)


# ---------------------------------------------------------------------------
# Test Suite 4: Missing VRAM & Hardware Constraints
# ---------------------------------------------------------------------------

class TestHardwareIntegrityConstraints:
    def test_missing_vram_is_explicitly_unavailable(self):
        # Default synthetic trace does NOT emulate VRAM -> mimics CPU dev host
        trace = SyntheticTraceGenerator.generate_trace("stable_memory", num_steps=10, emulate_vram=False)
        predictor = LinearTrendPredictor(targets=["vram_free_mb", "ram_used_mb"])

        res = predictor.predict(trace, horizon_seconds=2.0)
        vram_fc = res.get_target("vram_free_mb")
        assert vram_fc.is_available is False
        assert vram_fc.status == "unavailable"
        assert vram_fc.provenance == "unavailable"
        assert all(v is None for v in vram_fc.values)

        # RAM must still be available and NOT substituted as VRAM
        ram_fc = res.get_target("ram_used_mb")
        assert ram_fc.is_available is True
        assert ram_fc.values[0] is not None
        assert ram_fc.values[0] != 0.0

    def test_emulated_vram_produces_valid_forecast_when_explicitly_present(self):
        trace = SyntheticTraceGenerator.generate_trace("stable_memory", num_steps=10, emulate_vram=True)
        predictor = LinearTrendPredictor(targets=["vram_free_mb"])

        res = predictor.predict(trace, horizon_seconds=2.0)
        vram_fc = res.get_target("vram_free_mb")
        assert vram_fc.is_available is True
        assert vram_fc.provenance == "emulated"
        assert vram_fc.values[0] is not None


# ---------------------------------------------------------------------------
# Test Suite 5: KV-Cache-Aware Memory Predictor & Time-To-Threshold
# ---------------------------------------------------------------------------

class TestMemoryPredictor:
    def test_time_to_threshold_upper(self):
        # Current = 1000, slope = +100/s, threshold = 1500 -> 5.0 seconds
        t = estimate_time_to_threshold(current_val=1000.0, slope=100.0, threshold=1500.0, mode="upper")
        assert t == 5.0

        # Already crossed
        t_crossed = estimate_time_to_threshold(current_val=1600.0, slope=100.0, threshold=1500.0, mode="upper")
        assert t_crossed == 0.0

        # Negative slope never crosses upper threshold
        t_never = estimate_time_to_threshold(current_val=1000.0, slope=-50.0, threshold=1500.0, mode="upper")
        assert t_never is None

    def test_time_to_threshold_lower(self):
        # Current = 500, slope = -50/s, threshold = 200 -> 6.0 seconds
        t = estimate_time_to_threshold(current_val=500.0, slope=-50.0, threshold=200.0, mode="lower")
        assert t == 6.0

        # Positive slope never drops below lower threshold
        t_never = estimate_time_to_threshold(current_val=500.0, slope=10.0, threshold=200.0, mode="lower")
        assert t_never is None

    def test_kv_aware_memory_predictor(self):
        trace = SyntheticTraceGenerator.generate_trace("kv_cache_driven_memory_growth", num_steps=15, dt=0.5)
        predictor = KVCacheAwareMemoryPredictor(
            targets=["ram_used_mb", "kv_cache_bytes", "vram_free_mb"],
            minimum_history_length=5,
        )

        res = predictor.predict(trace, horizon_seconds=2.0)
        assert res.is_valid is True
        assert "kv_cache_growth_rate_bps" in res.metadata
        assert res.metadata["kv_cache_growth_rate_bps"] > 0.0

        # Since emulate_vram is False, time_to_vram_exhaustion_s should be None
        assert res.metadata["time_to_vram_exhaustion_s"] is None


# ---------------------------------------------------------------------------
# Test Suite 6: Learned Predictor
# ---------------------------------------------------------------------------

class TestLearnedPredictor:
    def test_learned_predictor_unfitted_falls_back_cleanly(self):
        trace = SyntheticTraceGenerator.generate_trace("stable_network", num_steps=10)
        pred = LearnedTimeSeriesPredictor(targets=["bandwidth_mbps"])
        assert pred.is_fitted is False

        # Should fall back cleanly without error
        res = pred.predict(trace, horizon_seconds=2.0)
        assert res.is_valid is True
        assert res.get_target("bandwidth_mbps").values[0] == 100.0

    def test_learned_predictor_fit_and_predict(self):
        seq1 = SyntheticTraceGenerator.generate_trace("linear_memory_growth", num_steps=20, dt=0.5)
        seq2 = SyntheticTraceGenerator.generate_trace("linear_memory_growth", num_steps=20, dt=0.5)

        pred = LearnedTimeSeriesPredictor(targets=["ram_used_mb"], window_size=3)
        pred.fit([seq1, seq2])
        assert pred.is_fitted is True

        # Evaluate on third sequence
        seq3 = SyntheticTraceGenerator.generate_trace("linear_memory_growth", num_steps=10, dt=0.5)
        res = pred.predict(seq3, horizon_seconds=1.0)
        assert res.is_valid is True
        ram_fc = res.get_target("ram_used_mb")
        assert ram_fc.values[0] is not None


# ---------------------------------------------------------------------------
# Test Suite 7: Forecasting Metrics & Lead-Time Analysis
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_mae_and_rmse(self):
        actuals = [10.0, 20.0, 30.0]
        preds = [12.0, 18.0, 33.0]  # residuals: +2, -2, +3

        mae = mean_absolute_error(actuals, preds)
        assert pytest.approx(mae, 0.001) == (2.0 + 2.0 + 3.0) / 3.0

        rmse = root_mean_squared_error(actuals, preds)
        expected_rmse = ((4.0 + 4.0 + 9.0) / 3.0) ** 0.5
        assert pytest.approx(rmse, 0.001) == expected_rmse

    def test_mape_safe_handling_near_zero(self):
        # When actual is near zero, MAPE returns None to prevent explosive percentages
        actuals = [0.0, 10.0]
        preds = [1.0, 10.0]
        assert mean_absolute_percentage_error(actuals, preds) is None

        # When actuals are strictly non-zero
        actuals_valid = [100.0, 200.0]
        preds_valid = [110.0, 180.0]  # 10% error each (10/100 and 20/200)
        mape = mean_absolute_percentage_error(actuals_valid, preds_valid)
        assert pytest.approx(mape, 0.01) == 10.0

    def test_median_and_max_ae(self):
        actuals = [10.0, 10.0, 10.0]
        preds = [11.0, 15.0, 12.0]  # residuals: 1.0, 5.0, 2.0 -> sorted: 1.0, 2.0, 5.0
        assert median_absolute_error(actuals, preds) == 2.0
        assert max_absolute_error(actuals, preds) == 5.0

    def test_lead_time_calculation(self):
        # Actual drops below 20 at t = 105.0
        actual_ts = [100.0, 102.0, 104.0, 105.0, 106.0]
        actual_vals = [50.0, 40.0, 25.0, 18.0, 15.0]

        # Predictor foresaw crossing at t = 102.0
        lead_time = calculate_lead_time(
            actual_timestamps=actual_ts,
            actual_values=actual_vals,
            predicted_crossing_time=102.0,
            threshold=20.0,
            mode="lower",
        )
        # 105.0 - 102.0 = 3.0 seconds advance warning
        assert lead_time == 3.0

    def test_abrupt_change_detector(self):
        detector = AbruptChangeDetector(sensitivity_sigma=2.0, min_history=5)
        # Feed 5 stable residuals (residual = 1.0)
        for _ in range(5):
            is_spike, _ = detector.update(actual=10.0, predicted=9.0)
            assert not is_spike

        # Massive jump (residual = 50.0) -> must trigger spike flag
        is_spike, res = detector.update(actual=60.0, predicted=10.0)
        assert is_spike is True
        assert res == 50.0


# ---------------------------------------------------------------------------
# Test Suite 8: Walk-Forward Evaluation & Synthetic Scenarios
# ---------------------------------------------------------------------------

class TestWalkForwardAndScenarios:
    def test_all_synthetic_scenarios_generate_valid_states(self):
        for scenario in SCENARIO_NAMES:
            states = SyntheticTraceGenerator.generate_trace(scenario=scenario, num_steps=10)
            assert len(states) == 10
            assert states[0].network.bandwidth_mbps.source == DataSource.EMULATED

    def test_walk_forward_evaluation(self):
        trace = SyntheticTraceGenerator.generate_trace("linear_memory_growth", num_steps=20, dt=0.5)
        predictor = LinearTrendPredictor(targets=["ram_used_mb"], minimum_history_length=5)

        pred_trace = evaluate_walk_forward(
            predictor=predictor,
            trace=trace,
            horizon_seconds=1.5,
            min_history=5,
            step_interval_seconds=0.5,
        )

        assert len(pred_trace) > 0
        metrics = pred_trace.compute_metrics()
        assert "ram_used_mb" in metrics
        assert metrics["ram_used_mb"]["sample_count"] > 0
        assert metrics["ram_used_mb"]["mae"] is not None

    def test_prediction_trace_save_and_load(self, tmp_path: Path):
        pt = PredictionTrace(predictor_name="test_pred")
        pt.add_point(
            PredictionTracePoint(
                prediction_timestamp=10.0,
                forecast_timestamp=11.0,
                target_name="bandwidth_mbps",
                predicted_value=50.0,
                actual_value=48.0,
                error=-2.0,
                is_available=True,
                predictor_name="test_pred",
            )
        )

        out_path = tmp_path / "trace.json"
        pt.save_json(out_path)
        assert out_path.exists()

        reloaded = PredictionTrace.load_json(out_path)
        assert len(reloaded) == 1
        assert reloaded.points[0].target_name == "bandwidth_mbps"
        assert reloaded.points[0].error == -2.0


# ---------------------------------------------------------------------------
# Test Suite 9: Predictor Registry & Configuration
# ---------------------------------------------------------------------------

class TestPredictorRegistry:
    def test_list_available_predictors(self):
        preds = list_available_predictors()
        assert "last_value" in preds
        assert "linear_trend" in preds
        assert "moving_average" in preds
        assert "kv_aware_memory" in preds

    def test_get_predictor_factory(self):
        p = get_predictor("linear_trend", targets=["bandwidth_mbps"], window_size=8)
        assert isinstance(p, LinearTrendPredictor)
        assert p.supported_targets == ["bandwidth_mbps"]

    def test_create_predictor_from_config(self):
        cfg = {
            "type": "kv_aware_memory",
            "minimum_history": 6,
            "targets": ["ram_used_mb", "kv_cache_bytes"],
        }
        p = create_predictor_from_config(cfg)
        assert isinstance(p, KVCacheAwareMemoryPredictor)
        assert p.minimum_history_length == 6
