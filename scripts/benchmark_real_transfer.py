"""
Real Network Transfer Benchmark Utility for Module 11.

Spawns a local authenticated TransferServer and evaluates end-to-end framed TCP
tensor transfers with SHA-256 integrity verification, byte accounting, and timing.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hardware.transport import TCPNetworkTransferProvider, TransferServer
from src.runtime.tier import TierId


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark framed TCP tensor transfers.")
    parser.add_argument("--tensor-dim", type=int, default=512, help="Tensor dimension (creates [dim, dim] float32 tensor).")
    parser.add_argument("--trials", type=int, default=5, help="Number of transfer trials.")
    parser.add_argument("--output", type=str, default=None, help="Output file path (JSON or CSV).")
    args = parser.parse_args()

    print("=" * 80)
    print(" REAL TCP TENSOR TRANSFER BENCHMARK (MODULE 11)")
    print("=" * 80)

    # 1. Start local ephemeral authenticated server
    auth_secret = b"benchmark_session_token_12345"
    server = TransferServer(host="127.0.0.1", port=0, auth_key=auth_secret)
    assigned_port = server.start()
    print(f"Spawned TransferServer on 127.0.0.1:{assigned_port}")

    # 2. Configure client provider
    client = TCPNetworkTransferProvider(
        endpoint_host="127.0.0.1",
        endpoint_port=assigned_port,
        auth_key=auth_secret,
        timeout_seconds=5.0,
    )

    # 3. Create test tensor: [dim, dim] float32
    test_tensor = torch.randn(args.tensor_dim, args.tensor_dim, dtype=torch.float32)
    raw_tensor_bytes = test_tensor.numel() * test_tensor.element_size()
    print(f"Test Tensor Shape: {list(test_tensor.shape)} ({raw_tensor_bytes / 1024:.1f} KB)")
    print(f"Running {args.trials} Transfer Trials...")

    raw_measurements = []
    durations_ms = []
    throughputs_mbps = []
    failures = 0

    try:
        for t_idx in range(args.trials):
            try:
                moved, bytes_wire, dur_ms, prov = client.transfer_tensor(
                    tensor=test_tensor,
                    source_tier=TierId.USER_DEVICE,
                    destination_tier=TierId.EDGE_A,
                    target_device="cpu",
                )
                durations_ms.append(dur_ms)
                # Compute effective transfer speed in Mbps
                effective_mbps = (bytes_wire * 8) / (max(1e-4, dur_ms / 1000.0) * 1e6)
                throughputs_mbps.append(effective_mbps)

                raw_measurements.append({
                    "trial": t_idx,
                    "bytes_transferred": bytes_wire,
                    "duration_ms": round(dur_ms, 3),
                    "throughput_mbps": round(effective_mbps, 3),
                    "provenance": prov.value,
                    "status": "success",
                })
            except Exception as e:
                failures += 1
                raw_measurements.append({
                    "trial": t_idx,
                    "error": str(e),
                    "status": "failure",
                })
    finally:
        server.stop()
        print("Server shutdown complete.")

    summary = {
        "trials": args.trials,
        "successes": len(durations_ms),
        "failures": failures,
        "bytes_per_transfer": raw_tensor_bytes,
        "mean_duration_ms": round(statistics.mean(durations_ms), 3) if durations_ms else None,
        "std_duration_ms": round(statistics.stdev(durations_ms), 3) if len(durations_ms) > 1 else 0.0,
        "mean_throughput_mbps": round(statistics.mean(throughputs_mbps), 3) if throughputs_mbps else None,
        "server_received_frames": server.received_frames_count,
    }

    print("\nBenchmark Summary:")
    print(f"  Transfers Completed : {summary['successes']}/{args.trials}")
    if durations_ms:
        print(f"  Mean Duration       : {summary['mean_duration_ms']} ms (std={summary['std_duration_ms']} ms)")
        print(f"  Effective Goodput   : {summary['mean_throughput_mbps']} Mbps")

    report = {
        "configuration": {
            "tensor_dim": args.tensor_dim,
            "trials": args.trials,
        },
        "summary": summary,
        "raw_measurements": raw_measurements,
    }

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix.lower() == ".csv":
            with open(out_p, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["trial", "bytes", "duration_ms", "throughput_mbps", "status"])
                for r in raw_measurements:
                    writer.writerow([
                        r.get("trial"),
                        r.get("bytes_transferred"),
                        r.get("duration_ms"),
                        r.get("throughput_mbps"),
                        r.get("status"),
                    ])
        else:
            out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[INFO] Saved results to {out_p.resolve()}")

    print("=" * 80)


if __name__ == "__main__":
    main()
