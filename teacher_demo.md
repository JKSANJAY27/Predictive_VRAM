# Teacher & Professor Demonstration Guide
## Adaptive Split-Inference Controller for Edge Large Language Models

> **Demonstration Script & Oral Defense Protocol**  
> **Environment:** Python 3.12, PyTorch, Interactive Dashboard (`http://localhost:8000`)  
> **Tested Hardware:** AMD Radeon Host CPU (Real CPU Fallback) & Emulated NVIDIA RTX 4090  

---

## 1. Project Pitch (30-Second Summary)

Deploying Large Language Models (LLMs) on resource-constrained consumer devices is bottlenecked by limited VRAM and expanding KV-cache footprints during autoregressive generation. Naive static offloading suffers severe latency penalties when wireless channels degrade, while reactive re-partitioning causes pipeline thrashing and thrashing overhead. 

This project implements an **Adaptive Split-Inference Controller** that combines continuous telemetry sampling, multi-step predictive forecasting, and cost-optimal partition scoring with **zero-downtime, transactional state migration**. It dynamically shifts model layers and KV-cache blocks across a three-tier hierarchy (User Device, Near-Edge, Cloud-Edge) *before* memory exhaustion strikes, while actively suppressing panic migrations during transient network jitter.

---

## 2. Pre-Demo Checklist

Run this checklist 5 minutes before your professor enters:

- [ ] **Run Pre-flight Diagnostic:**
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts/preflight.ps1
  ```
  *Ensure Python 3.12+, dependencies, and port 8000 are verified.*

- [ ] **Launch Dashboard Server:**
  ```powershell
  powershell -ExecutionPolicy Bypass -File start.ps1
  ```
  *(Or run `uv run python scripts/run_dashboard.py --port 8000`)*

- [ ] **Open Browser:** Navigate to `http://localhost:8000` (or `http://127.0.0.1:8000`).
- [ ] **Expected First Screen:**
  - Status Badge: `● LIVE (SSE Connected)` (neon green)
  - Hardware Badge: `Host: CPU (psutil) | Provenance: Verified`
  - Active Partition Plan: `USER_DEVICE [0-1] | EDGE_A [2-3]`
  - Split Topology: 2 layers on User Device (Layer 0, Layer 1), 2 layers on Edge Node A (Layer 2, Layer 3), Edge Node B in Standby.
  - CUDA Panel: `CUDA unavailable — running on CPU` (clear, honest CPU fallback).

---

## 3. Dashboard Tour (What Each Panel Means)

Walk your professor through the interface in this order:

1. **Header & Provenance Badges (Top):**
   - Shows live server connection status via Server-Sent Events (SSE).
   - Shows host hardware provenance: real measured data tagged via `psutil`, adhering to strict research integrity (Section 0 Rule 2: No fake data).
2. **Demonstration Control Bar (Top-Center):**
   - Dedicated buttons for Scenarios A through E.
   - Stop and Reset controls.
   - Live Scenario Info Card: displays title, what the observer should look for, and the architectural lesson sentence.
   - Progress pill and elapsed time timer.
3. **KPI Metrics Grid (Center):**
   - **Active Partition Plan:** Current physical layer distribution.
   - **Bandwidth & RTT:** Channel conditions labeled with provenance tags (**Measured** vs **Simulated**).
   - **KV-Cache Footprint:** Monitored memory growth of the autoregressive context cache.
   - **Local Memory Pressure:** Measured memory load on the client device.
   - **Predicted Time-To-Exhaustion:** Predictive forecast lead time (color codes to neon red when below 10s critical threshold).
   - **Migrations (Success / Rollback):** Real-time tally of completed versus rolled-back transactions.
4. **Three-Tier Split Execution Topology (Centerpiece):**
   - **Tier 1 (User Device):** Client-side edge node executing embeddings and early layers.
   - **Tier 2 (Edge Node A):** Near-edge server providing compute acceleration.
   - **Tier 3 (Edge Node B):** Regional cloud-edge fallback on standby.
   - **Layer Chips:** Hover over any chip to view layer index, device placement, parameter footprint (~1.2 MB FP32), and KV state.
   - **Migration Phase Track:** Displays the real-time transactional lifecycle (`IDLE` → `EVALUATING` → `LOCKED` → `TRANSFERRING` → `COMMITTED` / `ROLLED BACK`).
5. **Hardware & GPU Telemetry Panel (Bottom Left):**
   - Real hardware introspection via `torch.cuda` and `psutil`.
   - On CPU/AMD systems, demonstrates explicit "CUDA unavailable — running on CPU" handling.
   - "Toggle Mock GPU" button allows presenting the NVIDIA RTX 4090 telemetry mode with 24 GB VRAM introspection.
6. **Controller Decision Transparency Log (Bottom Right):**
   - Scrollable, auto-following real-time decision feed.
   - Color-coded decision types:
     - `HOLD` (amber): Controller keeps active partition.
     - `MIGRATE` (neon green): Controller proactively re-partitions.
     - `ROLLBACK` (neon red): Controller catches transfer fault and restores snapshot.

---

## 4. Button-by-Button Scenario Guide

### Scenario A — Everything is Stable
- **What it does:** Simulates ideal, stable channel conditions (100 Mbps, 5 ms RTT) with flat memory pressure across all 3 tiers (`UD [0-1] | EA [2-3] | EB [4-5]`).
- **What to point at:** Point at the KPI cards (100 Mbps) and the Decision Log. Show that the controller issues `HOLD` every single step with reason `stable`.
- **What the audience should conclude:** The controller does not thrash or migrate when conditions are optimal across the distributed 3-tier pipeline.
- **The Lesson Sentence:** *"A good adaptive system should know when not to act."*
- **Duration:** ~6 seconds (10 steps).

---

### Scenario B — KV-Cache Growth Threatens Memory
- **What it does:** Autoregressive tokens are generated continuously; KV-cache expands (+24 KB/step).
- **What to point at:** 
  1. Watch **Predicted Time-To-Exhaustion** decrease step-by-step: `28.0s → 21.5s → 15.0s → 8.5s` (card turns amber, then red).
  2. Point at the **Migration Phase Track**: watch it sequence through `EVALUATING` → `LOCKED` → `TRANSFERRING` → `COMMITTED`.
  3. Watch the **Split Topology**: User Device offloads Layer 1 to Edge Node A, which offloads Layer 3 to Edge Node B (`UD [0-0] | EA [1-2] | EB [3-5]`).
  4. Local layer count changes from 2 → 1. Local memory drops from 18 MB → 13.5 MB.
- **What the audience should conclude:** The predictive controller forecasted exhaustion ahead of time and offloaded layers across edge tiers to Cloud-Edge Node B *before* an out-of-memory crash could occur.
- **The Lesson Sentence:** *"Predict a memory problem early enough to avoid it."*
- **Duration:** ~8 seconds (12 steps).

---

### Scenario C — Temporary Bandwidth Collapse
- **What it does:** Injects a sudden 8-second bandwidth collapse (100 Mbps → 2.0 Mbps, 85 ms RTT), followed by full recovery.
- **What to point at:**
  1. Point at the Bandwidth KPI plunging to `2.0 Mbps` (tagged `Simulated`).
  2. Point at the Decision Log: the controller explicitly logs `HOLD` with reason `switching_cost_exceeds_gain`.
  3. Migration count stays `0 / 0`. The controller refuses to panic-migrate because the transfer cost across a 2 Mbps link exceeds the transient gain.
  4. Bandwidth recovers to 100 Mbps; 3-tier generation continues uninterrupted (`UD [0-1] | EA [2-3] | EB [4-5]`).
- **What the audience should conclude:** Predictive hysteresis and switching-cost evaluation filter out transient channel drops that would cause reactive systems to thrash.
- **The Lesson Sentence:** *"Do not migrate just because conditions briefly worsen."*
- **Duration:** ~10 seconds (15 steps).

---

### Scenario D — Bandwidth Stays Poor
- **What it does:** Starts with 1 layer local and 5 layers remote (`UD [0-0] | EA [1-2] | EB [3-5]`). Bandwidth drops to 5.0 Mbps and *remains* low.
- **What to point at:**
  1. Notice the initial hold period (steps 1–3): the controller evaluates persistence.
  2. Once the persistence criterion is met (>3 degraded cycles), the controller triggers a migration.
  3. Layer 1 migrates from Edge Node A *back to User Device* (`UD [0-1] | EA [2-3] | EB [4-5]`).
  4. Local layer count increases (1 → 2).
- **What the audience should conclude:** When degradation is permanent, the controller adapts by selecting a local-heavy partition to minimize cross-network activation transfers.
- **The Lesson Sentence:** *"Ignore short-lived fluctuations, but adapt to persistent degradation."*
- **Duration:** ~10 seconds (14 steps).

---

### Scenario E — Migration Fails Midway (Rollback & Fault Recovery)
- **What it does:** Initiates a migration to relieve pressure, but injects a transfer fault (socket timeout / corrupted checksum) midway during state transfer to Edge Node B.
- **What to point at:**
  1. Point at the Migration Phase Track: sequences to `TRANSFERRING`, then flashes `ROLLED BACK` (neon red).
  2. Point at the Rollback counter: increments to `1`.
  3. Point at the Decision Log: `ROLLBACK (injected_transfer_fault)` with full snapshot restoration notice.
  4. Point at the Split Topology: layers remain intact (`UD [0-1] | EA [2-3] | EB [4-5]`); no layer is orphaned or duplicated.
  5. The runtime quiescence lock is released, and autoregressive generation resumes smoothly across all 3 tiers.
- **What the audience should conclude:** The migration manager is strictly transactional: failure triggers atomic rollback without crashing the inference pipeline.
- **The Lesson Sentence:** *"A safe system needs failure recovery, not just a successful migration path."*
- **Duration:** ~7 seconds (10 steps).

---

### Stop & Reset Controls
- **Stop Button:** Immediately pauses/halts an ongoing scenario cleanly mid-run.
- **Reset Button:** Restores pristine initial state (`UD [0-1] | EA [2-3] | EB [4-5]`), clears logs, resets metrics, and releases any acquired locks. Always functional.
- **Toggle Mock GPU:** Toggles the hardware panel between real host CPU fallback and an emulated NVIDIA RTX 4090 with 24 GB VRAM introspection.

---

## 5. Suggested Demo Order and Timing

### 10-Minute Presentation
| Time | Segment | Focus |
|------|---------|-------|
| 0:00–1:30 | **Pitch & Problem** | VRAM limits on edge devices, wireless channel volatility, why reactive fails |
| 1:30–3:00 | **Dashboard Tour** | 3-tier split visualization, live provenance, CPU fallback honest display |
| 3:00–5:30 | **Scenarios A & B** | Contrast stability (knowing when not to act) with proactive KV migration |
| 5:30–8:00 | **Scenarios C & D** | Contrast temporary dip (filtering jitter) with persistent degradation (local-heavy) |
| 8:00–9:30 | **Scenario E** | Transactional failure & atomic rollback guarantee |
| 9:30–10:00 | **Conclusion & Q&A** | Summary of verified results |

### 20-Minute In-Depth Academic Defense
- Include all of the above, plus:
- Run `Toggle Mock GPU` to demonstrate dual-hardware support.
- Open `tests/` and run the preflight / test suite (`uv run pytest tests/test_scenarios_deterministic.py`). Show the professor the 10-consecutive-run 100% pass rate evidence.
- Review the 14-step control cycle in `src/orchestration/cycle.py`.

---

## 6. Likely Teacher / Professor Questions & Answers

**Q1: How does your system predict memory exhaustion before it happens?**  
*A:* Module 4 utilizes a `KVCacheAwareMemoryPredictor`. It measures the per-token activation and KV-cache expansion rate (bytes/token) and performs linear trend extrapolation over a sliding window. When the projected time-to-exhaustion $T_{\text{exhaust}} \le \tau_{\text{crit}}$ (e.g., 8 seconds), the controller proactively evaluates alternative partitions.

**Q2: What prevents the controller from constantly thrashing between partitions?**  
*A:* Module 7's `StabilityChecker` enforces three mathematical constraints before permitting a switch:
1. **Gain Threshold:** The candidate plan must yield expected gain $\Delta J > \text{threshold}$ (switching cost penalty).
2. **Dwell Time:** A mandatory minimum interval between consecutive physical migrations.
3. **Hysteresis Counter:** The candidate plan must be preferred for $\ge 3$ consecutive evaluation cycles, which successfully suppresses transient noise (demonstrated in Scenario C).

**Q3: How is atomic rollback guaranteed if a network connection drops during migration?**  
*A:* Module 8 treats migration as a two-phase transactional commit. Before tensors are transferred, a memory-mapped snapshot of the layer weights and active KV-cache pointers is preserved, and a quiescence lock is acquired. If any error occurs during validation, layer transfer, or KV-cache synchronization, the catch block invokes `RollbackManager`, restores the snapshot pointers, releases the lock, and logs the abort without corrupting the token generator.

**Q4: Why does Scenario D migrate to a local-heavy partition when bandwidth is degraded?**  
*A:* In split inference, communication latency is $T_{\text{comm}} = \frac{\text{Activation Bytes}}{\text{Bandwidth}} + \text{RTT}$. When bandwidth drops from 100 Mbps to 5 Mbps, transfer latency spikes by 20×. By moving layers to the local device (`USER_DEVICE`), the model executes more computation locally, completely bypassing the degraded channel for those layers.

**Q5: What is your project's hardware behavior when an NVIDIA GPU is not present?**  
*A:* Following Section 0 Rule 3 ("No silent fallbacks"), our system never presents fake GPU metrics. It detects hardware via `torch.cuda.is_available()`. If absent, it explicitly displays `CUDA unavailable — running on CPU`, routes execution through the CPU provider, and measures real host RAM via `psutil`. For testing GPU-specific paths, our framework includes `MockCUDAHardwareProvider` with provenance explicitly tagged as `EMULATED`.

---

## 7. Troubleshooting During the Live Demo

- **Scenario looks stuck or unresponsive?**  
  Click the **Reset** button on the control bar. It releases any runtime lock and restores the baseline partition.
- **Port 8000 already in use?**  
  Run `.\end.ps1` in PowerShell to terminate any orphan dashboard instances, or run `uv run python scripts/run_dashboard.py --port 8080`.
- **Accidentally clicked the wrong button?**  
  Click **Stop**, then **Reset**, and click the intended scenario.

---

## 8. Stop / Cleanup Commands

To stop the dashboard and release background processes:
```powershell
powershell -ExecutionPolicy Bypass -File end.ps1
```
Or press `Ctrl+C` in the terminal running the server.
