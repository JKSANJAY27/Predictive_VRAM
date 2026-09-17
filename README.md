# Predictive VRAM & Network-Aware Dynamic Split Inference for Edge LLMs

[![Research Prototype](https://img.shields.io/badge/Status-Module_1_Completed-brightgreen.svg)](#)
[![Python Version](https://img.shields.io/badge/Python-3.12-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0+cpu-orange.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

A research prototype exploring adaptive split-computing of autoregressive Large Language Models (LLMs) across a three-tier execution hierarchy:
1. **User Device** (Mobile / IoT / Edge Client)
2. **Edge Node A** (Local Aggregator / Near Edge Server)
3. **Edge Node B** (Regional Edge Server / Cloud-Edge Gateway)

---

## 1. Project Purpose & Research Contribution

The central research problem addressed by this project is:
> **How can a runtime mechanism predict near-future network and GPU-memory conditions and proactively adapt the partition of an autoregressive LLM across a user device and edge servers while accounting for the cost and stability impact of repartitioning?**

Rather than treating dynamic partitioning as a purely reactive problem or claiming novelty in static split computing, our intended contribution focuses on **predictive, cost-aware runtime control** that jointly considers:
- Short-horizon wireless channel and latency forecasting
- Short-horizon VRAM pressure & KV-cache growth projection
- Migration / weight-switching overhead ($P_{\text{switch}}$)
- Partition stability and oscillation avoidance (cooldown / dwell-time constraints)

---

## 2. Current Implementation Status: Module 1

**Module Name:** `Distributed Split-Inference Baseline`  
**Current State:** Fully implemented, verified, and benchmarked.

This module delivers the fundamental distributed inference foundation:
- **Three-Tier Topology (`Tier`, `TierId`)**: Formal representation of `UserDevice`, `EdgeA`, and `EdgeB`.
- **Partition Plan Abstraction (`PartitionPlan`, `TransferBoundary`)**: Validated contiguous layer partitioning across tiers, supporting monolithic, 2-tier, and 3-tier splits with strict error checking.
- **Layered Transformer Decomposition (`LayeredTransformer`)**: Deconstructed autoregressive transformer blocks with support for block-range forward execution and KV-cache tracking.
- **Transfer Boundary Tracking (`TransferManager`, `TransferRecord`)**: Explicit, observable inter-tier tensor communication tracking transfer count, tensor shapes, and byte volumes.
- **Autoregressive Distributed Executor (`DistributedInferenceExecutor`)**: Prefill and token-by-token decoding across arbitrary tier boundaries with state preservation.
- **Deterministic Parity Guarantee**: Validated byte-for-byte and token-for-token equality between local monolithic execution and distributed split execution under deterministic settings.

> **Important Boundary Notice:**  
> Module 1 implements **only** the baseline distributed inference runtime. Forecasting, cost modeling, adaptive controllers, hysteresis, and migration managers are intentionally deferred to future modules.

---

## 3. Architecture Overview

```
[Prompt] ──> [Tier: UserDevice] ──> [Transfer Boundary 1] ──> [Tier: EdgeNodeA] ──> [Transfer Boundary 2] ──> [Tier: EdgeNodeB]
                Layers 0..3             (Hidden States)           Layers 4..7             (Hidden States)           Layers 8..11
                + Embeddings                                                                                       + Head & Norm
                     │                                         │                                         │
                 [KV Cache 0..3]                           [KV Cache 4..7]                           [KV Cache 8..11]
```

### Module Directory Structure
```
d:/Sanjay/B.Tech CSE/vram/
├── configs/
│   ├── baseline.yaml         # Execution parameters, tier configs, and split presets
│   └── model.yaml            # Target model specs (GPT-2, 12 layers, 124M params)
├── docs/
│   └── environment.md        # Hardware and environment inspection report
├── scripts/
│   └── run_baseline.py       # End-to-end baseline runner and parity verification
├── src/
│   ├── config/
│   │   └── settings.py       # YAML parser and configuration management
│   ├── runtime/
│   │   ├── executor.py       # Distributed autoregressive generation executor
│   │   ├── model.py          # LayeredTransformer layer wrapper
│   │   ├── partition.py      # PartitionPlan validation & transfer boundaries
│   │   ├── tier.py           # Tier abstractions and metadata
│   │   └── transfer.py       # TransferManager and TransferRecord telemetry
│   └── utils/
│       └── logging.py        # Structured console logging
└── tests/
    ├── conftest.py           # Synthetic models and shared fixtures
    ├── test_partition.py     # Partition validation tests (gaps, overlaps, bounds)
    ├── test_runtime.py       # Integration tests & deterministic parity checks
    └── test_transfer.py      # Tensor transfer telemetry tests
```

---

## 4. Hardware Environment & Known Limitations

Documented in detail in `docs/environment.md`:
- **Host CPU:** Intel Core i5-4200U (2 Cores, 4 Threads @ 1.60GHz)
- **Host RAM:** 8 GB Total visible (~1.55 GB free physical memory)
- **Host GPU:** Integrated Intel HD Graphics (No discrete CUDA GPU)
- **PyTorch:** CPU build (`2.10.0+cpu`)

### Implications:
- The system supports both **Multi-GPU / Multi-Node** and **Single-Node Emulated Multi-Tier** configurations. On this host, tiers run as separate execution contexts on CPU with explicit boundary tracking and tensor movement.
- Model selection is pinned to `gpt2` (124M parameters) to fit comfortably within the ~1.5 GB free RAM threshold without memory thrashing.
- For lightning-fast CI and unit tests, a synthetic 4-layer transformer is used, ensuring tests finish in under 2 seconds without external network dependencies.

---

## 5. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.12.4)
- Git

### Setup Steps
```bash
# Clone the repository
git clone https://github.com/JKSANJAY27/Predictive_VRAM.git
cd Predictive_VRAM

# Install dependencies
pip install torch transformers pyyaml pytest
```

---

## 6. Running the Baseline & Tests

### Run Unit and Integration Tests
```bash
pytest -v tests
```

### Run the Distributed Split-Inference Baseline
```bash
python scripts/run_baseline.py
```

Options:
- `--split-mode compare_all`: Runs Monolithic, Two-Tier, and Three-Tier consecutively, comparing token parity.
- `--split-mode three_tier`: Runs exclusively on the 3-tier split.
- `--prompt "Your custom prompt"`: Custom prompt input.
- `--max-new-tokens 16`: Adjust generated length.

Example output:
```text
===========================================================================
  CORRECTNESS VERIFICATION: LOCAL vs SPLIT INFERENCE
===========================================================================
Comparison: Monolithic (Fully Local) vs Two-Tier Split:
  - Token Match:      True
  - Text Match:       True
  - Status:           PASSED (Deterministic Exact Match)
Comparison: Monolithic (Fully Local) vs Three-Tier Split:
  - Token Match:      True
  - Text Match:       True
  - Status:           PASSED (Deterministic Exact Match)

[SUCCESS] Baseline split-inference verification completed with zero discrepancies!
```

---

## 7. Next Step: Module 2 Preview

The next planned module is **Module 2: Runtime Telemetry**, which will introduce the `TelemetryAgent` to collect:
- Host memory and VRAM state ($V_t, F_t$)
- Network bandwidth, latency, and packet loss ($B_t, L_t, P_t$)
- KV-cache memory usage ($K_t$) and token state ($T_t$)
- Compute load ($G_t$) and energy proxy ($E_t$)
