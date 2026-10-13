"""
Network Measurement Benchmark Utility for Module 11.

Benchmarks round-trip latency, jitter, packet loss, and throughput against
real or simulated network endpoints with complete raw data capture.
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

from src.hardware.network_measurement import MockNetworkProbe, SocketNetworkProbe


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark network RTT, jitter, and throughput.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Target host address.")
    parser.add_argument("--port", type=int, default=8000, help="Target port.")
    parser.add_argument("--trials", type=int, default=5, help="Number of measurement trials.")
    parser.add_argument("--mock", action="store_true", help="Use mock probe for offline environments.")
    parser.add_argument("--output", type=str, default=None, help="Output file path (JSON or CSV).")
    args = parser.parse_args()

    print("=" * 80)
    print(" NETWORK MEASUREMENT BENCHMARK (MODULE 11)")
    print("=" * 80)
    print(f"Target Endpoint : {args.host}:{args.port}")
    print(f"Trial Count     : {args.trials}")
    print(f"Probe Backend   : {'MockNetworkProbe' if args.mock else 'SocketNetworkProbe'}")

    probe = MockNetworkProbe() if args.mock else SocketNetworkProbe()

    raw_results = []
    rtts = []
    throughputs = []
    failures = 0

    for i in range(args.trials):
        res = probe.measure_rtt(args.host, args.port, count=3, timeout_s=1.0)
        raw_results.append(res.to_dict())
        if res.rtt_ms.value is not None:
            rtts.append(res.rtt_ms.value)
        else:
            failures += 1

    print(f"\nExecution Complete: {len(rtts)} succeeded, {failures} failed.")
    summary = {
        "trials": args.trials,
        "success_count": len(rtts),
        "failure_count": failures,
        "mean_rtt_ms": round(statistics.mean(rtts), 3) if rtts else None,
        "std_rtt_ms": round(statistics.stdev(rtts), 3) if len(rtts) > 1 else 0.0,
        "min_rtt_ms": round(min(rtts), 3) if rtts else None,
        "max_rtt_ms": round(max(rtts), 3) if rtts else None,
    }

    if rtts:
        print(f"  Mean RTT : {summary['mean_rtt_ms']} ms")
        print(f"  Min RTT  : {summary['min_rtt_ms']} ms")
        print(f"  Max RTT  : {summary['max_rtt_ms']} ms")
        print(f"  Std Dev  : {summary['std_rtt_ms']} ms")

    report = {
        "configuration": {
            "host": args.host,
            "port": args.port,
            "trials": args.trials,
            "mock": args.mock,
        },
        "summary": summary,
        "raw_measurements": raw_results,
    }

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix.lower() == ".csv":
            with open(out_p, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["trial", "rtt_ms", "jitter_ms", "loss_ratio", "conn_state"])
                for idx, r in enumerate(raw_results):
                    writer.writerow([
                        idx,
                        r["rtt_ms"]["value"],
                        r["jitter_ms"]["value"],
                        r["packet_loss"]["value"],
                        r["connection_state"],
                    ])
        else:
            out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[INFO] Saved results to {out_p.resolve()}")

    print("=" * 80)


if __name__ == "__main__":
    main()
