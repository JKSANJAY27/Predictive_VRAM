# Module 13: Real-GPU Validation, Empirical Benchmarking & Research-Claim Verification

## Overview

Module 13 establishes a hardware-aware empirical validation workflow for the Predictive VRAM runtime. It enables genuine physical device profiling when CUDA-capable GPUs are present, while maintaining strict provenance invariants and honest reporting on CPU-only environments.

Crucially, Module 13:
1. **Never substitutes host RAM for VRAM**: In the absence of physical CUDA hardware, GPU memory metrics are tagged `DataSource.UNAVAILABLE` with `value=None`.
2. **Never labels simulated or mock data as measured GPU telemetry**.
3. **Classifies experiments into five non-fungible classes** that are never conflated.
4. **Verifies genuine split-inference execution**: Audits partition placement across devices, tracks activation transfers, and verifies exact token parity against a monolithic reference model.
5. **Evaluates memory feasibility safety**: Distinguishes **False-Safe** predictions (declaring an infeasible partition feasible, risking OOM) from **False-Unsafe** predictions (conservatively rejecting a feasible partition).

---

## The 5 Non-Fungible Experiment Classes

To prevent misleading claims in research publications, Module 13 establishes the `ExperimentClass` taxonomy:

| Experiment Class | Compute Device | Communication Transport | Hardware Telemetry | Status / Interpretation |
|---|---|---|---|---|
| **Class 1: Real GPU** (`CLASS_1_REAL_GPU`) | Physical CUDA GPU (`cuda:0`, `cuda:1`) | Intradevice / NVLink / PCIe | Genuine NVML / `torch.cuda` telemetry (`MEASURED`) | Confirmatory real-hardware GPU benchmark |
| **Class 2: Real Socket** (`CLASS_2_REAL_SOCKET`) | CPU or GPU | OS loopback / LAN TCP socket | Real network timing & OS socket buffers (`MEASURED`) | Real socket network benchmark |
| **Class 3: Network Emulated** (`CLASS_3_NETWORK_EMULATED`) | CPU or GPU | Simulated / shaped network delays | Measured compute, emulated channel (`EMULATED`) | Hybrid compute-in-the-loop experiment |
| **Class 4: Fully Simulated** (`CLASS_4_FULLY_SIMULATED`) | Simulated / synthetic trace | Simulated discrete event clock | Synthetic execution models | Rapid policy exploration & ablation |
| **Class 5: Mixed** (`CLASS_5_MIXED`) | Real CPU compute | Emulated network / simulated GPU | Mixed provenance | Formally partitioned, partially physical |

Experiments from different classes **must never be aggregated into a single mean or claim** without explicit methodological justification.

---

## Phase Breakdown & Validation Architecture

### 1. Hardware Readiness & Capability Audit (`scripts/inspect_hardware.py`)
Queries local compute providers and exports device profiles:
- Detects physical accelerator devices (NVIDIA GPUs, compute capability).
- Audits available VRAM telemetry (allocated, reserved, free, total) via `torch.cuda`.
- Audits Host RAM and CPU cores via `psutil`.
- Enforces `DataSource.UNAVAILABLE` fallback when CUDA is absent.

### 2. Phased Memory Profiling (`src/research/gpu_validation.py`)
Profiles device memory across distinct operational phases:
1. **Idle Baseline**: Pre-allocation baseline memory.
2. **Model Placed**: Static model weights loaded onto device.
3. **Warmup Completed**: Single forward pass to initialize framework caches.
4. **Prefill Phase**: Prompt token processing and initial activation memory.
5. **Autoregressive Generation**: Token-by-token generation measuring incremental KV-cache allocation.

### 3. Split Inference & Token Parity Audit (`src/research/split_verification.py`)
Verifies that multi-tier partitioned execution matches monolithic single-device execution:
- Audits layer assignments to physical devices (`cpu`, `cuda:0`, etc.).
- Records every cross-tier tensor transfer (tensor shape, dtype, byte count, timestamp).
- Checks KV-cache preservation across tier handoffs.
- Enforces strict token parity: ensures the generated token sequence $\tau_{\text{split}} \equiv \tau_{\text{mono}}$.

### 4. Feasibility Safety Evaluation
Evaluates predictor reliability under memory constraints:
- **True Safe ($TS$)**: Predicted feasible, actual memory $\le$ budget.
- **True Unsafe ($TU$)**: Predicted infeasible, actual memory $>$ budget.
- **False-Safe ($FS$)**: Predicted feasible, actual memory $>$ budget. *(Critical safety violation: risks Out-Of-Memory termination).*
- **False-Unsafe ($FU$)**: Predicted infeasible, actual memory $\le$ budget. *(Efficiency penalty: unnecessary offloading).*

$$\text{False-Safe Rate} = \frac{FS}{FS + TU}$$

A production edge runtime must strictly minimize the False-Safe rate to prevent catastrophic out-of-memory crashes on resource-constrained user devices.

---

## Workload Benchmark Matrix

The validation engine supports sweeping across:
- **Prompt Lengths**: e.g., 8, 16, 32, 64 tokens.
- **Generation Lengths**: e.g., 4, 8, 16, 32 tokens.
- **Precisions**: `float32`, `float16`, `bfloat16`.
- **Partition Plans**: Monolithic, 2-Tier (`user_device` + `edge_a`), 3-Tier (`user_device` + `edge_a` + `edge_b`).

---

## Command Reference

### Hardware Readiness Inspection
```bash
uv run python scripts/inspect_hardware.py --provider auto
```

### Genuine Split Inference & Token Parity Verification
```bash
uv run python scripts/verify_split_inference.py --device auto --tokens 8
```

### Phased GPU Memory Profiling & Benchmark Workload Sweep
```bash
uv run python scripts/run_gpu_validation.py --config configs/research/gpu_validation.yaml
```

### Run Full Test Suite (including GPU Validation Unit Tests)
```bash
uv run pytest tests/test_gpu_validation.py -v
```

---

## Status on Current Development Host

- **Host Environment**: Windows 11, AMD64 16 logical cores, 16 GB Host RAM, CPU-only (no discrete CUDA hardware).
- **GPU Status**: `torch.cuda.is_available() == False`.
- **Physical GPU Telemetry**: Explicitly marked `CPU_ONLY_NOT_PERFORMED`.
- **VRAM Telemetry**: Tagged `DataSource.UNAVAILABLE`, values are `None`.
- **Validation Tests**: 10 passed, 2 skipped (`TestPhysicalGPUIntegration` skipped due to lack of CUDA hardware).
