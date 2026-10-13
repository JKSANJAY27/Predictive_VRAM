"""
CLI Script: Verify Genuine Split Inference & Token Parity (Module 13).

Audits partition placement across devices, tracks inter-tier activation transfers,
and verifies exact token parity between partitioned execution and a monolithic
reference forward pass.

Usage:
    uv run python scripts/verify_split_inference.py --device auto --tokens 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
import torch
from transformers import GPT2Config, GPT2LMHeadModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.research.split_verification import SplitInferenceVerifier
from src.runtime.model import LayeredTransformer
from src.runtime.partition import PartitionPlan
from src.runtime.tier import Tier, TierId

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("verify_split_inference")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify genuine split inference execution and token parity.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Target compute device.")
    parser.add_argument("--tokens", type=int, default=8, help="Number of tokens to generate.")
    parser.add_argument("--output-json", type=str, default="results/research/split_verification_report.json", help="Output JSON path.")

    args = parser.parse_args()

    print("=" * 80)
    print(" GENUINE SPLIT INFERENCE & TOKEN PARITY VERIFIER (MODULE 13)")
    print("=" * 80)
    dev_str = args.device
    if dev_str == "auto":
        dev_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Target Device       : {dev_str}")
    print(f"CUDA Available      : {torch.cuda.is_available()}")
    print("=" * 80)

    # 1. Build Layered Transformer
    model = LayeredTransformer.create_synthetic(
        num_layers=6,
        hidden_size=128,
        num_heads=4,
        vocab_size=1000,
        device=dev_str,
    )

    # 2. Plan 3-tier split
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

    # 3. Verify Split Inference Execution & Parity
    report = SplitInferenceVerifier.verify_plan(
        model=model,
        partition_plan=plan,
        tiers=tiers,
        max_new_tokens=args.tokens,
    )

    print(f"Device Placement Mode       : {report.device_placement_mode}")
    print(f"Active Tiers                : {report.active_tiers}")
    print(f"Total Transfers Executed    : {len(report.transfers)}")
    print(f"Total Transfer Bytes        : {report.total_transfer_bytes} bytes")
    print(f"Token Parity Verified       : {report.token_parity_verified}")
    print(f"Split Generated Tokens      : {report.generated_tokens_split}")
    print(f"Monolithic Generated Tokens : {report.generated_tokens_monolithic}")
    if not report.token_parity_verified:
        print(f"[WARNING] {report.parity_mismatch_details}")

    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"[OK] Verification report saved to {out_p}")
    print("=" * 80)


if __name__ == "__main__":
    main()
