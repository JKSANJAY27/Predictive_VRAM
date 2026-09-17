"""
Evaluation benchmark runner for Predictive VRAM and Network Forecasters.

Evaluates predictors on recorded or synthetic traces using chronological walk-forward validation.
Reports MAE, RMSE, Median AE, Max AE, and sample counts per target variable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prediction.evaluation import evaluate_walk_forward
from src.prediction.registry import get_predictor, list_available_predictors
from src.prediction.synthetic import SCENARIO_NAMES, SyntheticTraceGenerator
from src.state.buffer import StateBuffer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate time-series predictors on state traces.")
    parser.add_argument(
        "--predictor",
        type=str,
        default="linear_trend",
        help=f"Predictor type. Choices: {list_available_predictors()}",
    )
    parser.add_argument(
        "--trace",
        type=str,
        default=None,
        help="Path to saved StateBuffer JSON trace. If omitted, uses synthetic scenario.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="bandwidth_degradation",
        choices=list(SCENARIO_NAMES),
        help="Synthetic scenario to evaluate if --trace is not supplied.",
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=2.0,
        help="Forecast horizon in seconds (default: 2.0).",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=5,
        help="Minimum history observations before starting evaluation (default: 5).",
    )
    parser.add_argument(
        "--emulate-vram",
        action="store_true",
        help="If using synthetic trace, inject emulated VRAM values for evaluation testing.",
    )
    parser.add_argument(
        "--save-trace",
        type=str,
        default=None,
        help="Optional path to save resulting PredictionTrace JSON.",
    )

    args = parser.parse_args()

    # 1. Load or generate trace
    if args.trace:
        trace_path = Path(args.trace)
        if not trace_path.exists():
            print(f"Error: Trace file '{trace_path}' does not exist.")
            sys.exit(1)
        buf = StateBuffer.from_json_file(trace_path)
        states = buf.get_all()
        print(f"Loaded {len(states)} states from '{trace_path}'.")
    else:
        states = SyntheticTraceGenerator.generate_trace(
            scenario=args.scenario,
            num_steps=40,
            dt=0.5,
            emulate_vram=args.emulate_vram,
        )
        print(f"Generated 40 synthetic states for scenario '{args.scenario}' (emulate_vram={args.emulate_vram}).")

    # 2. Instantiate predictor
    targets = ["bandwidth_mbps", "latency_ms", "vram_free_mb", "kv_cache_bytes"]
    predictor = get_predictor(
        predictor_type=args.predictor,
        targets=targets,
        minimum_history_length=args.min_history,
    )

    print(f"\nEvaluating Predictor: '{predictor.name}' over {args.horizon}s horizon (min_history={args.min_history})...")

    # 3. Run chronological walk-forward evaluation
    trace_result = evaluate_walk_forward(
        predictor=predictor,
        trace=states,
        horizon_seconds=args.horizon,
        min_history=args.min_history,
        step_interval_seconds=0.5,
    )

    metrics_by_target = trace_result.compute_metrics()

    # 4. Print benchmark report
    print("\n" + "=" * 80)
    print(f"{'TARGET':<20} | {'SAMPLES':<8} | {'MAE':<12} | {'RMSE':<12} | {'MEDIAN AE':<12} | {'MAX AE':<12}")
    print("-" * 80)

    for tgt in targets:
        m = metrics_by_target.get(tgt)
        if not m or m["sample_count"] == 0:
            print(f"{tgt:<20} | {'0':<8} | {'UNAVAILABLE / SKIPPED':<48}")
            continue

        cnt = m["sample_count"]
        mae_str = f"{m['mae']:.3f}" if m["mae"] is not None else "N/A"
        rmse_str = f"{m['rmse']:.3f}" if m["rmse"] is not None else "N/A"
        med_str = f"{m['median_ae']:.3f}" if m["median_ae"] is not None else "N/A"
        max_str = f"{m['max_ae']:.3f}" if m["max_ae"] is not None else "N/A"

        print(f"{tgt:<20} | {cnt:<8} | {mae_str:<12} | {rmse_str:<12} | {med_str:<12} | {max_str:<12}")

    print("=" * 80)

    if args.save_trace:
        trace_result.save_json(args.save_trace)
        print(f"\nPrediction trace saved to '{args.save_trace}'.")


if __name__ == "__main__":
    main()
