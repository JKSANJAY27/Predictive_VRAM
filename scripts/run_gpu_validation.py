"""
CLI Script: Real-GPU Validation & Memory Profiling (Module 13).

Runs phased memory profiling (Idle, Model Load, Warmup, Prefill, Autoregressive Generation),
evaluates KV-cache scaling, audits false-safe / false-unsafe feasibility safety, and
executes a structured benchmark workload matrix.

Usage:
    uv run python scripts/run_gpu_validation.py --config configs/research/gpu_validation.yaml
    uv run python scripts/run_gpu_validation.py --device auto --prompt-length 32 --tokens 16
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
import yaml
import torch
from transformers import AutoConfig, AutoTokenizer, GPT2Config, GPT2LMHeadModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.research.empirical_classes import EvidenceQualityDescriptor, ExperimentClass
from src.research.gpu_validation import BenchmarkMatrixRunner, GPUValidationEngine
from src.runtime.model import LayeredTransformer
from src.runtime.partition import PartitionPlan
from src.runtime.tier import Tier, TierId

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("run_gpu_validation")


def build_default_model(device: str = "cpu") -> Tuple[LayeredTransformer, PartitionPlan, Dict[TierId, Tier]]:
    # Determine device assignment
    dev_str = device
    if dev_str == "auto":
        dev_str = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = LayeredTransformer.create_synthetic(
        num_layers=6,
        hidden_size=128,
        num_heads=4,
        vocab_size=1000,
        device=dev_str,
    )

    # 3-tier partition: 2 layers each
    plan = PartitionPlan(
        total_layers=6,
        user_device=(0, 1),
        edge_a=(2, 3),
        edge_b=(4, 5),
    )

    tiers = {
        TierId.USER_DEVICE: Tier(tier_id=TierId.USER_DEVICE, name="User Device", device=dev_str),
        TierId.EDGE_A: Tier(tier_id=TierId.EDGE_A, name="Edge Node A", device=dev_str),
        TierId.EDGE_B: Tier(tier_id=TierId.EDGE_B, name="Edge Node B", device=dev_str),
    }

    return model, plan, tiers


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-GPU Validation and Memory Feasibility Profiling.")
    parser.add_argument("--config", type=str, default="configs/research/gpu_validation.yaml", help="Path to config YAML.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Target compute device.")
    parser.add_argument("--prompt-length", type=int, default=16, help="Approximate prompt length.")
    parser.add_argument("--tokens", type=int, default=8, help="Number of tokens to generate.")
    parser.add_argument("--output-dir", type=str, default="results/research/gpu_validation", help="Output directory.")

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" REAL-GPU VALIDATION & EMPIRICAL MEMORY PROFILING (MODULE 13)")
    print("=" * 80)
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available      : {cuda_avail}")
    if cuda_avail:
        print(f"CUDA Device Name    : {torch.cuda.get_device_name(0)}")
        print(f"Device Capability   : {torch.cuda.get_device_capability(0)}")
    else:
        print("Hardware Status     : CPU-ONLY (Physical GPU validation marked NOT PERFORMED)")
    print(f"Target Device Mode  : {args.device}")
    print(f"Output Directory    : {out_dir}")
    print("=" * 80)

    # 1. Build Model and Partition Scheme
    model, plan, tiers = build_default_model(device=args.device)

    # 2. Run Phased Memory Validation
    prompt = "The future of distributed edge computing and predictive resource allocation"
    print("\n[Phase 1] Executing Phased Memory Validation...")
    report = GPUValidationEngine.run_phased_memory_validation(
        model=model,
        partition_plan=plan,
        tiers=tiers,
        prompt=prompt,
        max_new_tokens=args.tokens,
    )

    print(f"  Hardware Status              : {report.hardware_status}")
    print(f"  Device Name                  : {report.device_name}")
    print(f"  Peak Allocated VRAM (MB)     : {report.peak_allocated_mb} (DataSource={'MEASURED' if cuda_avail else 'UNAVAILABLE'})")
    print(f"  Peak Reserved VRAM (MB)      : {report.peak_reserved_mb}")
    print(f"  Mean KV Growth / Token       : {report.mean_kv_growth_per_token_bytes} bytes/token")
    print(f"  Prediction MAE (MB)          : {report.prediction_mae_mb}")
    print(f"  False-Safe Feasibility Rate  : {report.feasibility_safety.false_safe_rate:.4f} (Count: {report.feasibility_safety.false_safe_count})")
    print(f"  Feasibility Safety Accuracy  : {report.feasibility_safety.accuracy * 100.0:.2f}%")

    report_json_path = out_dir / "gpu_validation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"[OK] Phased validation report saved to {report_json_path}")

    # 3. Run Benchmark Workload Matrix
    print("\n[Phase 2] Executing Benchmark Workload Matrix...")
    matrix_points = BenchmarkMatrixRunner.run_matrix(
        model=model,
        partition_plan=plan,
        tiers=tiers,
        prompt_lengths=[8, 32],
        generation_lengths=[4, 8],
    )
    matrix_df = BenchmarkMatrixRunner.matrix_to_dataframe(matrix_points)
    matrix_csv_path = out_dir / "workload_matrix_results.csv"
    matrix_df.to_csv(matrix_csv_path, index=False)
    print(f"[OK] Completed {len(matrix_points)} workload matrix points saved to {matrix_csv_path}")

    print("\n" + "=" * 80)
    print(" EMPIRICAL VALIDATION EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
