"""
End-to-End Hardware Integration & 6G-Oriented Evaluation Script for Module 11.

Executes closed-loop split-inference simulations under 6G-oriented operating conditions
and benchmarks VRAM/memory validation across predictive, reactive, and static policies.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hardware.evaluation_scenarios import SIX_G_SCENARIOS
from src.hardware.factory import HardwareProviderFactory
from src.hardware.memory_validation import MemoryValidationEngine
from src.orchestration.config import OrchestrationConfig
from src.orchestration.runtime import ClosedLoopRuntime
from src.orchestration.types import OrchestrationMode


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hardware integration and 6G exploratory scenarios.")
    parser.add_argument("--provider", choices=["auto", "cpu", "cuda", "mock"], default="auto", help="Hardware provider type.")
    parser.add_argument("--steps", type=int, default=16, help="Simulation steps per scenario.")
    parser.add_argument("--output", type=str, default="results/hardware_integration/summary.json", help="Output JSON path.")
    args = parser.parse_args()

    print("=" * 80)
    print(" HARDWARE INTEGRATION & 6G-ORIENTED EVALUATION HARNESS (MODULE 11)")
    print("=" * 80)
    print(f"Selected Provider : {args.provider}")
    print(f"Simulation Steps  : {args.steps}")

    # 1. Evaluate memory validation engine
    print("\n[Phase 1] Evaluating VRAM & Memory Predictability...")
    provider = HardwareProviderFactory.create_provider(args.provider)
    mem_engine = MemoryValidationEngine(provider)
    mem_report = mem_engine.evaluate_generation_run(num_tokens=args.steps)
    print(f"  Device Type               : {mem_report.device_type}")
    print(f"  Peak Allocated Memory     : {mem_report.peak_allocated_mb:.1f} MB")
    print(f"  Mean Sampling Overhead    : {mem_report.mean_sampling_overhead_ms:.3f} ms")
    print(f"  Prediction Error (MAE)    : {mem_report.prediction_mae_mb:.2f} MB")
    print(f"  Warning Lead Time         : {mem_report.warning_lead_time_tokens} tokens")

    # 2. Run 6G-oriented scenarios across runtime modes
    print("\n[Phase 2] Evaluating 6G-Oriented Operational Scenarios in ClosedLoopRuntime...")
    scenario_names = list(SIX_G_SCENARIOS.keys())
    modes = [OrchestrationMode.STATIC, OrchestrationMode.REACTIVE, OrchestrationMode.PREDICTIVE]

    results_matrix = []

    for sc_name in scenario_names:
        print(f"\n  Scenario: {sc_name}")
        for m in modes:
            cfg = OrchestrationConfig(
                mode=m,
                scenario_name=sc_name,
                control_interval_tokens=2,
                horizon=4,
            )
            rt = ClosedLoopRuntime.create_default(config=cfg, num_layers=4, hidden_size=64)
            t0 = time.perf_counter()
            trace = rt.run_simulation(num_steps=args.steps, scenario_name=sc_name)
            dur_ms = (time.perf_counter() - t0) * 1000.0

            switches = trace.summary.total_switches if trace.summary else 0
            rollback_count = trace.summary.rolled_back_migrations if trace.summary else 0
            mean_itl = trace.summary.mean_itl_ms if trace.summary else 0.0

            print(
                f"    [{m.value.upper():10s}] Switches: {switches} | "
                f"Rollbacks: {rollback_count} | Mean ITL: {mean_itl:.2f} ms | Exec: {dur_ms:.1f} ms"
            )

            results_matrix.append({
                "scenario": sc_name,
                "mode": m.value,
                "steps": args.steps,
                "switches": switches,
                "rollbacks": rollback_count,
                "mean_itl_ms": round(mean_itl, 3),
                "duration_ms": round(dur_ms, 2),
            })

    output_data = {
        "memory_validation": mem_report.to_dict(),
        "scenario_evaluations": results_matrix,
    }

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print(f"\n[SUCCESS] Full integration evaluation complete. Results written to: {out_p.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
