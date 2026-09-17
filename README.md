# Predictive VRAM & Network-Aware Dynamic Split Inference for Edge LLMs

[![Research Prototype](https://img.shields.io/badge/Status-Module_8_Completed-brightgreen.svg)](#)
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

> **Important Boundary**: Module 6 scores candidates; Module 7 makes the keep/migrate decision; Module 8 executes the transactional migration.

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
                             Historical State Windows
                                           ▼
+---------------------------------------------------------------------------------------+
|  Module 4: Predictive Forecasting Engine (src/prediction/)                            |
|  - Common Predictor Protocol & Strongly Typed PredictionResult                        |
|  - Baselines: LastValue, LinearTrend, MovingAverage                                    |
|  - KV-Cache-Aware Memory Predictor & Time-to-Threshold Warning (tau_lead)              |
|  - Ridge-Regularized Learned Predictor (Strict Anti-Leakage Isolation)                |
|  - Physical Domain Bounds & Constraint Clipping Indicators                            |
|  - Chronological Walk-Forward Time-Series Validation & Benchmark Runner               |
+---------------------------------------------------------------------------------------+
                                           │
                           Forecasts & Capacity Auditing
                                           ▼
+---------------------------------------------------------------------------------------+
|  Module 5: Split Catalog & Feasible Candidate Plan Generation (src/partitioning/)     |
|  - Structural Plan Enumerator (Monolithic, Two-Tier, Three-Tier Contiguous Splits)    |
|  - Model Metadata & Per-Tier Parameter/KV Memory Estimator with Safety Margin         |
|  - Provenance-Aware Feasibility Evaluator (FEASIBLE, INFEASIBLE, UNKNOWN)             |
|  - CandidatePlan Factory & SplitCatalog Filtering (Exhaustive vs Feasible-Only)       |
|  - Plan Comparison & Boundary Delta Analysis for Transition Cost Inputs               |
+---------------------------------------------------------------------------------------+
                                           │
                              Candidate Plans & Forecasts
                                           ▼
+---------------------------------------------------------------------------------------+
|  Module 6: Cost Model & Candidate Scoring (src/cost/)                                 |
|  - Composite Objective: J(a) = alpha*L + beta*C + gamma*M + delta*E + epsilon*P_switch|
|  - Decomposed Latency (L_compute + L_comm + L_queue) & Separate Steady-State Comm C(a)|
|  - Multi-Horizon Memory Pressure M(a) with UNKNOWN VRAM Preservation                  |
|  - Structural Transition Penalty P_switch(a) using PlanDifference (0 for identical)  |
|  - Uniform Scaling Normalization & Ablation Presets (Reactive, Latency, No-Switch)    |
+---------------------------------------------------------------------------------------+
                                           │
                              Candidate Scores & Forecasts
                                           ▼
+---------------------------------------------------------------------------------------+
|  Module 7: Adaptive Partition Controller (src/controller/)                            |
|  - Decision Modes: Static (Baseline), Reactive (Zero Lookahead), Predictive (Horizon) |
|  - Anti-Thrashing Guardrails: Cooldown Period, Minimum Dwell Time, Hysteresis Votes   |
|  - Minimum Benefit Threshold (theta) & Expected Gain Calculation                      |
|  - Strict Safety Hierarchy: OOM Prevention & Network Disconnection Emergency Override|
|  - MigrationRequest Specification for Module 8 (Zero in-module physical movement)     |
|  - Trace Replay Simulation & ControllerHistory Metrics Accumulator                    |
+---------------------------------------------------------------------------------------+
                                           │
                              Declarative MigrationRequest
                                           ▼
+---------------------------------------------------------------------------------------+
|  Module 8: Physical Migration & Runtime State Transition (src/migration/)             |
|  - Transactional Migration Coordinator (MigrationManager) with Lock & Quiescence      |
|  - Plan Delta & Layer Transition Planner (MigrationPlanner)                           |
|  - Layer Weight & KV-Cache Transfer with In-Memory Snapshots & Rollback               |
|  - Pluggable Transfer Backends (LocalTensorTransfer & EmulatedNetworkTransfer)        |
|  - Fast Structural & Deep Forward Verification Modes (MigrationVerifier)              |
|  - Deterministic Exact Numerical Token Parity Preservation                            |
+---------------------------------------------------------------------------------------+
                                           │
                               Transactional Migration
                                           ▼
+---------------------------------------------------------------------------------------+
|  Module 9: Closed-Loop Predictive Runtime Integration (src/orchestration/)            |
|  - End-to-End Autoregressive Generation with Dynamic Re-partitioning                  |
|  - Periodic Control Cycle Execution (Telemetry -> Predict -> Score -> Migrate)        |
|  - Asynchronous & Synchronous Operational Handshake with Physical Runtime             |
|  - DynamicCache KV-State Preservation Across Partition Transitions                    |
|  - Complete RuntimeTrace Recording for All Decision Cycles and Token Latencies        |
+---------------------------------------------------------------------------------------+
                                           │
                               RuntimeTrace & Metrics
                                           ▼
+---------------------------------------------------------------------------------------+
|  Module 10: Experimental Harness, Baselines & Statistical Evaluation (src/evaluation/)|
|  - Five Strictly Specified Research Baselines (B1 Static -> B5 Predictive)            |
|  - Trace-First Environmental Fairness with Bit-Identical Replay Across Baselines      |
|  - Complete Metric Suite (Inference, Memory, Stability, Overhead, Prediction, SLO)   |
|  - Statistical Aggregation (Mean, Std, Median, 95% CI, Cohen's d, Paired Tests)       |
|  - Component Ablation Suite (A1-A8) & Multi-Parameter Sensitivity Sweeps              |
|  - Immutable Trial Results, Invariant Validation, and Open Research Dataset Export     |
+---------------------------------------------------------------------------------------+
```

### Module Status Summary
- **Module 1 (Distributed Split-Inference Baseline):** COMPLETE (19 tests)
  - Static partition plans, 3 logical tiers, KV-cache handling, exact parity with monolithic execution.
- **Module 2 (Runtime Telemetry Layer):** COMPLETE (76 tests)
  - `DataSource` tagging, zero CUDA/GPU fabrication on CPU dev environment, configurable network emulation scenarios, `TelemetryBuffer`.
- **Module 3 (Unified Runtime State & Rolling StateBuffer):** COMPLETE (32 tests)
  - Canonical `RuntimeState`, `StateBuffer` ring buffer with trace JSON save/load, deterministic `FeatureExtractor` (22 columns, availability and provenance masks, derived growth rates, deterministic normalization).
- **Module 4 (Predictive Forecasting Engine):** COMPLETE (30 tests)
  - Pluggable predictors behind common `Predictor` interface, `PredictionResult` with provenance and error estimates, VRAM `UNAVAILABLE` handling without CPU RAM substitution, walk-forward validation without data leakage.
- **Module 5 (Split Catalog & Feasible Candidate Plan Generation):** COMPLETE (82 tests)
  - Exhaustive contiguous partition enumeration across tiers, deterministic per-tier memory estimation (weights + KV-cache + safety margins), provenance-aware feasibility evaluation (`FEASIBLE`, `INFEASIBLE`, `UNKNOWN`), delta comparison vs current plan, auditable candidate tables.
- **Module 6 (Cost Model and Candidate Scoring):** COMPLETE (31 tests)
  - Multi-objective composite cost calculation $J(a)$, decomposed latency, decoupled steady-state communication vs one-time switching penalty $P_{\text{switch}}$, uniform scale normalizer, ablation support, auditable score breakdowns.
- **Module 7 (Adaptive Predictive Partition Controller):** COMPLETE (66 tests)
  - Static, reactive, and predictive decision policies; anti-thrashing guardrails (cooldown, dwell time, threshold, hysteresis); safety hierarchy with emergency OOM overrides; typed `MigrationRequest` emission for Module 8; audit history and metrics.
- **Module 8 (Physical Migration and Runtime State Transition):** COMPLETE (111 tests)
  - All-or-nothing transactional migration lifecycle; quiescence lock management; stateful layer and KV-cache delta transfer; rollback and state restoration on injected faults; Fast and Deep verification; verified token parity before/after migration.
- **Module 9 (Closed-Loop Predictive Runtime Integration):** COMPLETE (17 tests)
  - Closed-loop orchestration loop; periodic control scheduling; end-to-end autoregressive generation with live migrations; complete runtime trace logging.
- **Module 10 (Experimental Harness, Baselines, Ablations & Statistical Evaluation):** COMPLETE
  - Five research baselines (B1-B5); deterministic trace generator & bit-identical replay; full metric suite calculation; statistical aggregation with effect sizes; A1-A8 ablation suite; sensitivity sweeper; invariant validation & sanity checking; research dataset export (CSV/JSON).

---

## 3. Project Directory Structure

```
d:/Sanjay/B.Tech CSE/vram/
├── configs/
│   ├── baseline.yaml         # Execution parameters, tier configs, split presets, controller params
│   └── model.yaml            # Target model specs (GPT-2, 12 layers, 124M params)
├── docs/
│   ├── environment.md        # Hardware and environment inspection report
│   ├── telemetry.md          # Runtime telemetry layer architecture and schema
│   ├── state.md              # Canonical state, buffer invariants, and feature extraction
│   ├── prediction.md         # Predictive forecasting engine, targets, and evaluation
│   ├── partitioning.md       # Split catalog, structural enumeration, memory estimation, feasibility
│   ├── cost_model.md         # Multi-objective cost formulation, normalization, and trade-off scoring
│   ├── controller.md         # Controller decision policy, anti-thrashing, safety, and handoff
│   └── migration.md          # Physical migration, transactional execution, and rollback
├── scripts/
│   ├── run_baseline.py       # End-to-end baseline runner and parity verification
│   ├── run_prediction_demo.py # Interactive prediction demonstration on StateBuffer
│   ├── evaluate_predictor.py # Walk-forward evaluation benchmark on traces
│   ├── run_candidate_demo.py # Module 5 candidate catalog and feasibility demonstration
│   ├── run_cost_demo.py      # Module 6 cost model and multi-objective scoring demonstration
│   ├── run_controller_demo.py # Module 7 controller demonstration across modes and guards
│   ├── evaluate_controller.py # Module 7 benchmark runner across synthetic scenarios
│   ├── run_migration_demo.py # Module 8 end-to-end physical migration and token parity demo
│   └── run_migration_failure_demo.py # Module 8 failure injection and transactional rollback demo
├── src/
│   ├── config/
│   │   └── settings.py       # Configuration management
│   ├── runtime/              # Module 1: Distributed split-inference baseline
│   ├── telemetry/            # Module 2: Runtime telemetry collection & emulation
│   ├── state/                # Module 3: Canonical state & rolling StateBuffer
│   ├── prediction/           # Module 4: Predictive forecasting engine
│   ├── partitioning/         # Module 5: Split catalog & feasible candidate plan generation
│   ├── cost/                 # Module 6: Cost model and candidate scoring
│   ├── controller/           # Module 7: Adaptive predictive partition controller
│   │   ├── types.py          # ControlAction, DecisionReason, MigrationRequest, ControllerConfig
│   │   ├── safety.py         # SafetyPolicy and SafetyEvaluation
│   │   ├── stability.py      # StabilityChecker and StabilityResult (anti-thrashing)
│   │   ├── history.py        # ControllerHistory tracking and metrics aggregation
│   │   └── controller.py     # AdaptivePartitionController orchestrator
│   ├── migration/            # Module 8: Physical migration & runtime state transition
│   │   ├── types.py          # MigrationStatus, MigrationMode, MigrationConfig, MigrationResult
│   │   ├── validator.py      # MigrationValidator (idempotency, stale request, bounds checks)
│   │   ├── planner.py        # MigrationPlanner (plan delta, layer transition mapping)
│   │   ├── transfer.py       # Transfer backends (LocalTensorTransfer, EmulatedNetworkTransfer)
│   │   ├── state.py          # LayerTransferManager & KVCacheTransferManager with backup snapshots
│   │   ├── verifier.py       # MigrationVerifier (Fast structural & Deep forward pass checks)
│   │   ├── rollback.py       # RollbackManager (atomic restoration of layers and KV-cache)
│   │   ├── adapter.py        # RuntimeAdapter (facade over DistributedInferenceExecutor)
│   │   └── manager.py        # MigrationManager (transactional state machine coordinator)
│   └── utils/
│       └── logging.py        # Structured logging utilities
└── tests/
    ├── conftest.py           # Synthetic models and shared fixtures
    ├── test_partition.py     # Partition validation tests
    ├── test_runtime.py       # End-to-end split execution & parity tests
    ├── test_telemetry.py     # Telemetry collectors, emulation, & buffer tests
    ├── test_transfer.py      # Transfer telemetry tests
    ├── test_state.py         # RuntimeState, StateBuffer, & feature extraction tests
    ├── test_prediction.py    # Predictors, physical bounds, metrics, walk-forward tests
    ├── test_partitioning.py  # Module 5 catalog, memory, feasibility, comparison tests
    ├── test_cost.py          # Module 6 weights, normalization, latency, comm, memory, switching tests
    ├── test_controller.py    # Module 7 modes, threshold, dwell, cooldown, hysteresis, safety tests
    └── test_migration.py     # Module 8 lifecycle, planning, transfer, verification, rollback tests
```

---

## 4. Hardware Environment & Constraints

Documented in detail in `docs/environment.md`:
- **Host CPU:** Intel Core i5-4200U (2 Cores, 4 Threads @ 1.60GHz)
- **Host RAM:** 8 GB Total visible (~1.55 GB free physical memory)
- **Host GPU:** Integrated Intel HD Graphics (No discrete CUDA GPU)
- **PyTorch:** CPU build (`2.10.0+cpu`)

### Strict Integrity Rules:
- GPU/VRAM metrics are **never fabricated** or reported as zero. When hardware is absent, values are explicitly tagged `DataSource.UNAVAILABLE` with value `None`. CPU RAM is **never substituted** for VRAM.
- Network conditions are explicitly tagged `DataSource.EMULATED`.
- Test suite uses synthetic lightweight models and profiles to guarantee instant CI and 100% test execution in seconds.

---

## 5. Running the Test Suite & Demos

Execute the entire test suite across Modules 1 through 8 (447 tests):
```bash
python -m pytest tests/ -v
```

Run the Module 4 interactive demo:
```bash
python scripts/run_prediction_demo.py
```

Run walk-forward prediction benchmark on synthetic scenarios:
```bash
python scripts/evaluate_predictor.py --predictor linear_trend --scenario bandwidth_degradation
```

Run the Module 5 candidate catalog & feasibility demo:
```bash
python scripts/run_candidate_demo.py
```

Run the Module 6 cost model demonstration:
```bash
python scripts/run_cost_demo.py
```

Run the Module 7 controller demonstration:
```bash
python scripts/run_controller_demo.py
```

Run the Module 8 physical migration and token parity demo:
```bash
python scripts/run_migration_demo.py
```

Run the Module 8 failure injection and transactional rollback demo:
```bash
python scripts/run_migration_failure_demo.py
```
