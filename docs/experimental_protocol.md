# Module 10: Experimental Protocol and Reproducibility Standard

## 1. Scope and Scientific Purpose
This document establishes the experimental protocol for evaluating the **Predictive VRAM and Network-Aware Dynamic Split Inference for Edge LLMs** framework. The goal is to rigorously quantify the performance trade-offs of predictive partition control relative to static and reactive baselines without introducing methodological bias or cherry-picked evaluation regimes.

---

## 2. Research Baselines

We define five strictly specified baselines (B1–B5):

| Identifier | Name | Control Paradigm | Available Signals | Prediction Enabled |
| :--- | :--- | :--- | :--- | :---: |
| **B1** | `static` | Static Reference | None (fixed initial plan) | No |
| **B2** | `network_reactive` | Reactive Network | Instantaneous Bandwidth/RTT only | No |
| **B3** | `memory_reactive` | Reactive Memory | Instantaneous VRAM/KV headroom only | No |
| **B4** | `joint_reactive` | Joint Reactive | Instantaneous Network + VRAM | No |
| **B5** | `predictive` | Predictive Cost-Aware | Joint Network + VRAM + Multi-step Forecast | **Yes** |

---

## 3. Controlled Experimental Invariants

To eliminate confounding factors, the following invariants are strictly enforced across all comparisons:

1. **Trace-First Environmental Fairness**: For a given `(scenario_id, seed)` pair, a single `CombinedEnvironmentTrace` is generated once and replayed bit-identically across all five baselines. Environmental noise is not re-sampled per policy.
2. **Causal Monotonicity (No Future Leakage)**: Predictive forecasting at inference cycle $t$ strictly ingests state observations with timestamp $\le t$. Future telemetry samples are never visible.
3. **Reactive Policy Purity**: Reactive policies (B2–B4) do not compute, store, or trigger proactive partition switches.
4. **Static Policy Immutability**: The B1 `static` policy undergoes zero partition switches under any environmental conditions.
5. **Rollback Accounting**: Any migration that aborts or rolls back due to failure is recorded as an aborted migration and is never counted as an operational partition switch.
6. **Hardware Provenance**: All GPU/VRAM telemetry gathered on CPU-only evaluation machines is explicitly tagged with `DataSource.EMULATED`.
7. **Result Immutability**: Each trial produces a deterministic `trial_id`. Completed trial records are protected by `DuplicateRunGuard` to prevent overwriting.

---

## 4. Evaluation Scenarios

The framework evaluates across thirteen benchmark scenarios defined in `ScenarioCatalog`:
- `S1`: `stable` — baseline stationary channel and memory
- `S2`: `kv_cache_growth` — memory pressure driven by generation length
- `S3`: `gradual_bandwidth_degradation` — linear decline in transmission capacity
- `S4`: `sudden_bandwidth_drop` — step-function drop simulating handover
- `S5`: `bandwidth_oscillation` — cyclical fluctuations
- `S6`: `edge_compute_saturation` — load spikes on edge accelerator
- `S7`: `mixed_degradation` (aliased as `combined_degradation`) — canonical multi-phase stress test
- `S8`: `recovery` — degradation followed by restoration
- `S9`: `telemetry_dropout` — missing and noisy observations
- `S10`: `prediction_error` — adversarial drift testing robustness to mispredictions
- `S11`–`S13`: Multi-node heterogeneous edge cluster configurations

---

## 5. Statistical Aggregation and Reporting

All reported statistics adhere to neutral scientific standards:
- **Measures of Central Tendency**: Mean $\pm$ sample standard deviation and median with 95% bootstrap confidence intervals.
- **Tail Latency**: 95th and 99th percentile inter-token latency (ITL).
- **Effect Size**: Cohen's $d$ and nonparametric Wilcoxon signed-rank tests for paired cross-seed trials.
- **No Ranking Adjectives**: Results are presented as quantified observations (e.g. "Predictive P95 ITL was $124.5\text{ ms}$ versus $188.2\text{ ms}$ for Static") rather than value judgments ("Predictive is better").
