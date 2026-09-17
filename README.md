# Predictive VRAM & Network-Aware Dynamic Split Inference for Edge LLMs

[![Research Prototype](https://img.shields.io/badge/Status-Module_3_Completed-brightgreen.svg)](#)
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

## 2. Implementation Progress & Architecture

```
                                  PIPELINE ARCHITECTURE
                                  
+---------------------------------------------------------------------------------------+
|  Module 1: Distributed Split-Inference Baseline (src/runtime/)                        |
|  - Three-Tier Topology (UserDevice -> EdgeA -> EdgeB)                                 |
|  - Contiguous Layer Partitioning & Transfer Boundaries                                |
|  - Autoregressive Generation & DynamicCache KV-State Preservation                     |
+---------------------------------------------------------------------------------------+
                                           │
                           (Execution Hooks & Callbacks)
                                           ▼
+---------------------------------------------------------------------------------------+
|  Module 2: Runtime Telemetry Layer (src/telemetry/)                                   |
|  - Physical RAM / CPU Metrics via psutil (MEASURED)                                   |
|  - Synthetic Network Emulation (EMULATED: Bandwidth, RTT, Loss, Jitter)               |
|  - KV-Cache & Activation Transfer Telemetry                                           |
|  - Strict DataSource Provenance (MEASURED, ESTIMATED, EMULATED, UNAVAILABLE)          |
+---------------------------------------------------------------------------------------+
                                           │
                              TelemetrySnapshot Stream
                                           ▼
+---------------------------------------------------------------------------------------+
|  Module 3: Unified Runtime State & Rolling StateBuffer (src/state/)                   |
|  - Canonical State Tuple S_t: <B_t, L_t, P_t, J_t, V_t, F_t, R_t, G_t, C_t, K_t...>  |
|  - Physical Domain Validation & Missing Value Preservation (Never Impute 0)           |
|  - StateBuffer: Fixed-Capacity Ring Buffer with Chronological Causality & Replay      |
|  - FeatureExtractor: 22 Deterministic Columns, Masks & Normalization                  |
+---------------------------------------------------------------------------------------+
                                           │
                                    (Upcoming Module 4)
                                           ▼
+---------------------------------------------------------------------------------------+
|  Module 4: Predictive Forecasters (Network & VRAM Prediction)                         |
+---------------------------------------------------------------------------------------+
```

### Module Status Summary
- **Module 1 (Distributed Split-Inference Baseline):** COMPLETE (19 tests)
  - Static partition plans, 3 logical tiers, KV-cache handling, exact parity with monolithic execution.
- **Module 2 (Runtime Telemetry Layer):** COMPLETE (76 tests)
  - `DataSource` tagging, zero CUDA/GPU fabrication on CPU dev environment, configurable network emulation scenarios, `TelemetryBuffer`.
- **Module 3 (Unified Runtime State & Rolling StateBuffer):** COMPLETE (26+ tests)
  - Canonical `RuntimeState`, `StateBuffer` ring buffer with trace JSON save/load, deterministic `FeatureExtractor` (22 columns, availability and provenance masks, derived growth rates, deterministic normalization).

---

## 3. Project Directory Structure

```
d:/Sanjay/B.Tech CSE/vram/
├── configs/
│   ├── baseline.yaml         # Execution parameters, tier configs, and split presets
│   └── model.yaml            # Target model specs (GPT-2, 12 layers, 124M params)
├── docs/
│   ├── environment.md        # Hardware and environment inspection report
│   ├── telemetry.md          # Runtime telemetry layer architecture and schema
│   └── state.md              # Canonical state, buffer invariants, and feature extraction
├── scripts/
│   └── run_baseline.py       # End-to-end baseline runner and parity verification
├── src/
│   ├── config/
│   │   └── settings.py       # Configuration management
│   ├── runtime/              # Module 1: Distributed split-inference baseline
│   │   ├── executor.py       # Distributed autoregressive generation executor
│   │   ├── model.py          # LayeredTransformer layer wrapper
│   │   ├── partition.py      # PartitionPlan validation & transfer boundaries
│   │   ├── tier.py           # Tier abstractions and metadata
│   │   └── transfer.py       # TransferManager and TransferRecord telemetry
│   ├── telemetry/            # Module 2: Runtime telemetry collection & emulation
│   │   ├── buffer.py         # Rolling TelemetryBuffer
│   │   ├── collectors.py     # Memory, CPU, KV-Cache, and Network collectors
│   │   ├── network_emulation.py # Scenario profiles & latency estimators
│   │   └── types.py          # DataSource, TaggedValue, TelemetrySnapshot
│   ├── state/                # Module 3: Canonical state & rolling StateBuffer
│   │   ├── buffer.py         # StateBuffer with causality & replay
│   │   ├── features.py       # FeatureVector & FeatureExtractor
│   │   └── types.py          # RuntimeState & component state dataclasses
│   └── utils/
│       └── logging.py        # Structured logging utilities
└── tests/
    ├── conftest.py           # Synthetic models and shared fixtures
    ├── test_partition.py     # Partition validation tests
    ├── test_runtime.py       # End-to-end split execution & parity tests
    ├── test_telemetry.py     # Telemetry collectors, emulation, & buffer tests
    ├── test_transfer.py      # Transfer telemetry tests
    └── test_state.py         # RuntimeState, StateBuffer, & feature extraction tests
```

---

## 4. Hardware Environment & Constraints

Documented in detail in `docs/environment.md`:
- **Host CPU:** Intel Core i5-4200U (2 Cores, 4 Threads @ 1.60GHz)
- **Host RAM:** 8 GB Total visible (~1.55 GB free physical memory)
- **Host GPU:** Integrated Intel HD Graphics (No discrete CUDA GPU)
- **PyTorch:** CPU build (`2.10.0+cpu`)

### Strict Integrity Rules:
- GPU/VRAM metrics are **never fabricated** or reported as zero. When hardware is absent, values are explicitly tagged `DataSource.UNAVAILABLE` with value `None`.
- Network conditions are explicitly tagged `DataSource.EMULATED`.
- Test suite uses synthetic lightweight models to guarantee instant CI and 100% test execution in seconds.

---

## 5. Running the Test Suite

Execute the entire test suite across Modules 1, 2, and 3:
```bash
python -m pytest tests/ -v
```

Run Module 3 tests specifically:
```bash
python -m pytest tests/test_state.py -v
```
