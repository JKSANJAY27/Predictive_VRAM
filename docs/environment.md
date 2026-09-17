# Hardware and Execution Environment Specification

**Project:** Predictive VRAM and Network-Aware Dynamic Split Inference for Edge LLMs  
**Status:** Baseline Development Environment Inspection  
**Date of Inspection:** September 17, 2026  

---

## 1. Operating System & Host Platform
- **OS Platform:** Microsoft Windows 10 Home Single Language (64-bit)
- **Host Architecture:** x86_64
- **Shell Environment:** PowerShell 5.1 / Windows Terminal
- **Workspace Directory:** `D:\Sanjay\B.Tech CSE\vram`
- **Git Remote:** `https://github.com/JKSANJAY27/Predictive_VRAM.git`

## 2. Processor & Memory Architecture
- **CPU:** Intel(R) Core(TM) i5-4200U CPU @ 1.60GHz
- **Cores / Threads:** 2 Physical Cores, 4 Logical Processors
- **Total Physical Memory (RAM):** ~8,302,872 KB (~7.92 GB visible)
- **Available Physical Memory:** ~1.55 GB free at baseline inspection
- **Memory Constraint Considerations:** High memory pressure environment. The runtime executor must be memory-conservative, avoiding redundant in-memory tensor copies and large model weights that exceed free physical RAM.

## 3. GPU / Accelerator Hardware & CUDA Status
- **Installed Video Controller:** Intel(R) HD Graphics Family (Integrated GPU)
- **Dedicated GPU:** None (No discrete NVIDIA GPU / AMD GPU attached)
- **NVIDIA SMI / CUDA Driver:** Not available (`nvidia-smi` command not recognized)
- **CUDA Device Count:** 0
- **PyTorch CUDA Availability:** `torch.cuda.is_available() == False`

### Implication for Research Architecture:
Because no physical discrete multi-GPU cluster exists on this local development host, the architecture must strictly support two execution modes:
1. **Multi-GPU / Multi-Device Mode (Target Deployment):**
   - Different tiers (User Device, Edge Node A, Edge Node B) map to distinct physical GPUs or networked edge nodes.
2. **Single-Node / Emulated Multi-Tier Mode (Local Development & Verification):**
   - Logical tiers (`UserDevice`, `EdgeA`, `EdgeB`) are maintained as separate execution contexts / worker abstractions with explicit layer ownership and transfer boundaries.
   - Tensor movement between tiers is mediated via an explicit `TransferManager` interface to allow realistic telemetry, latency modeling, and serialization without faking physical hardware existence.

## 4. Software Stack & Frameworks
- **Python Version:** 3.12.4
- **PyTorch Version:** `2.10.0+cpu`
- **Transformers Version:** `5.5.0`
- **Accelerate Version:** `1.14.0`
- **Pytest Version:** `9.0.2`
- **PyYAML Version:** `6.0.2`
- **Pydantic Version:** `2.13.4`
- **NumPy Version:** `1.26.4`
- **Psutil Version:** `6.1.0`

## 5. Network Interfaces
- **Ethernet:** `169.254.179.12`
- **Wi-Fi:** `192.168.1.8`
- **Loopback:** `127.0.0.1`

## 6. Selected Baseline Model
- **Model Identifier:** `gpt2` (OpenAI GPT-2 Small, 124M parameters)
- **Layers:** 12 Transformer Blocks (`wte`, `wpe`, 12 x `GPT2Block`, `ln_f`, `lm_head`)
- **Default Precision:** `torch.float32` (CPU-compatible)
- **Tier Partitioning Strategy:**
  - 3-tier split:
    - `UserDevice`: Layers 0 to 3 (4 layers)
    - `EdgeA`: Layers 4 to 7 (4 layers)
    - `EdgeB`: Layers 8 to 11 (4 layers)
  - 2-tier split:
    - `UserDevice`: Layers 0 to 5 (6 layers)
    - `EdgeA`: Layers 6 to 11 (6 layers)
  - Monolithic (All User Device):
    - `UserDevice`: Layers 0 to 11 (12 layers)
- **Synthetic Test Model Configuration:**
  - For rapid, offline CI testing and unit tests, a lightweight synthetic configuration (e.g. 3-6 layers, hidden size 64) is provided to ensure sub-second deterministic test runs without external network dependencies.
