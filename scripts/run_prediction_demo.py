"""
Demonstration runner for the Predictive VRAM and Network Forecasting Engine.

Demonstrates:
- Ingesting states into rolling StateBuffer
- Querying short-horizon forecasts (e.g., 2.0s)
- Displaying observed vs. forecasted trajectories
- Showing explicit UNAVAILABLE status for VRAM on CPU hardware
- Diagnostic error estimates
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prediction.registry import get_predictor
from src.prediction.synthetic import SyntheticTraceGenerator
from src.state.buffer import StateBuffer


def main() -> None:
    print("=" * 80)
    print("  PREDICTIVE VRAM & NETWORK FORECASTING ENGINE DEMO (MODULE 4)")
    print("=" * 80)

    # 1. Generate synthetic scenario (bandwidth degradation + memory growth)
    print("\n[Step 1] Generating synthetic runtime trace ('bandwidth_degradation')...")
    states = SyntheticTraceGenerator.generate_trace(
        scenario="bandwidth_degradation",
        num_steps=15,
        dt=0.5,
        start_timestamp=100.0,
        emulate_vram=False,  # Emulates real development host where VRAM is unavailable
    )

    # 2. Populate rolling StateBuffer
    print(f"[Step 2] Ingesting {len(states)} states into StateBuffer (capacity=64)...")
    buffer = StateBuffer(capacity=64)
    for s in states:
        buffer.append(s)

    window = buffer.get_window(10)["states"]
    t_curr = window[-1].timestamp
    print(f"Current Runtime Clock: t = {t_curr:.1f}s | Observed History Window: {len(window)} steps")

    # 3. Instantiate Predictors
    predictor = get_predictor(
        predictor_type="kv_aware_memory",
        targets=["bandwidth_mbps", "latency_ms", "vram_free_mb", "ram_used_mb", "kv_cache_bytes"],
        minimum_history_length=4,
    )

    # 4. Generate 2.0-second forecast
    horizon = 2.0
    print(f"\n[Step 3] Running Predictor: '{predictor.name}' for {horizon}s Horizon...")
    result = predictor.predict(window, horizon_seconds=horizon, step_interval_seconds=0.5)

    print(f"Prediction Status: {result.status_message} | Latency: {result.prediction_latency_ms:.2f} ms")
    print(f"Forecast Timestamps: {[round(t, 2) for t in result.forecast_timestamps]}")

    # 5. Display forecast details per target
    print("\n" + "-" * 80)
    print(f"{'TARGET':<18} | {'STATUS':<12} | {'LAST OBSERVED':<14} | {'FORECAST (+2.0s)':<18} | {'ERROR EST'}")
    print("-" * 80)

    for tgt_name, forecast in result.targets.items():
        status = forecast.status
        err_str = f"+/- {forecast.error_estimate:.2f}" if forecast.error_estimate is not None else "N/A"

        if not forecast.is_available:
            print(f"{tgt_name:<18} | {'UNAVAILABLE':<12} | {'None (CPU-only)':<14} | {'[None, None...]':<18} | {err_str}")
            continue

        # Last observed value from history
        last_val = forecast.raw_values[0]
        # End-of-horizon prediction
        final_pred = forecast.values[-1]

        val_str = f"{final_pred:.2f}" if final_pred is not None else "None"
        obs_str = f"{forecast.raw_values[0]:.2f}" if forecast.raw_values and forecast.raw_values[0] is not None else "N/A"

        print(f"{tgt_name:<18} | {status:<12} | {obs_str:<14} | {val_str:<18} | {err_str}")

    print("-" * 80)

    # 6. Show domain diagnostics
    if result.metadata:
        print("\n[Step 4] Domain Diagnostics & Threshold Projections:")
        for k, v in result.metadata.items():
            print(f"  - {k}: {v}")

    print("\n[SUCCESS] Module 4 predictive forecasting demo executed successfully!\n")


if __name__ == "__main__":
    main()
