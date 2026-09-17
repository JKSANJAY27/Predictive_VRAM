# Unified Runtime State and Rolling StateBuffer

Module 3 converts raw observations (`TelemetrySnapshot`) from the telemetry collection layer into a clean, normalized, timestamp-ordered canonical runtime state representation (`RuntimeState`) and manages its historical temporal window in a ring buffer (`StateBuffer`).

---

## 1. Role in the Research Architecture

In the predictive split-inference framework, the runtime cannot make proactive partitioning or migration decisions directly from raw, heterogeneous, asynchronous sensor streams. It requires:

1. **A Canonical State Vector ($S_t$)**: A mathematically well-defined snapshot at discrete time step $t$ uniting network conditions, memory pressure, compute utilization, inference progression, activation transfer cost, and energy metrics.
2. **Explicit Provenance**: Rigorous distinction between measured physical data (`MEASURED`), analytically derived figures (`ESTIMATED`), externally injected test profiles (`EMULATED`), and unobservable metrics on the host hardware (`UNAVAILABLE`). Missing values are preserved as `None` and are **never imputed as zero**.
3. **Chronological Ring Buffer**: Fixed-capacity, thread-safe historical window preserving causality (rejecting out-of-order samples) and supporting trace replay.
4. **Deterministic Feature Extraction**: A fixed-order vectorization pipeline with availability and data-source masks, derived temporal rates, and deterministic parameter normalization.

```
       +---------------------------------------------+
       |   Module 2: Runtime Telemetry Layer         |
       |   (Memory, CPU, Network, KV, Activations)   |
       +---------------------------------------------+
                             |
                   TelemetrySnapshot
                             v
       +---------------------------------------------+
       |   Module 3: Unified Runtime State           |
       |                                             |
       |  RuntimeState.from_telemetry(...)           |
       |  - Physical domain validation               |
       |  - DataSource tag propagation               |
       |  - Strict causality enforcement             |
       +---------------------------------------------+
                             |
                             +------------------------+
                             |                        |
                             v                        v
                +------------------------+ +------------------------+
                |      StateBuffer       | |    FeatureExtractor    |
                | - Ring buffer (deque)  | | - Deterministic order  |
                | - Trace Save / Replay  | | - Availability mask    |
                | - Windowing queries    | | - Derived rates        |
                +------------------------+ +------------------------+
                             |                        |
                             +-----------+------------+
                                         |
                                         v
                         (Future Module 4: Prediction)
```

---

## 2. Canonical Mathematical State ($S_t$)

At each autoregressive step $t$, the system condition is captured as a composite tuple:

$$S_t = \langle B_t, L_t, P_t, J_t, V_t, F_t, R_t, G_t, C_t, K_t, \Delta K_t, T_t, A_t, E_t \rangle$$

| Component | Symbol | Description | Unit | Dev Environment Provenance |
| :--- | :--- | :--- | :--- | :--- |
| **Network** | $B_t$ | Uplink / Tier Bandwidth | Mbps | `EMULATED` (injected scenario) |
| | $L_t$ | Round-Trip Latency (RTT) | ms | `EMULATED` (injected scenario) |
| | $P_t$ | Packet Loss Rate | $[0.0, 1.0]$ | `EMULATED` (injected scenario) |
| | $J_t$ | Latency Jitter | ms | `EMULATED` (injected scenario) |
| **Memory** | $V_t$ | VRAM Allocated | MB | `UNAVAILABLE` (no CUDA GPU) |
| | $F_t$ | VRAM Free | MB | `UNAVAILABLE` (no CUDA GPU) |
| | $R_t$ | RAM Used & Available | MB | `MEASURED` (via `psutil`) |
| **Compute** | $G_t$ | GPU Utilization | % | `UNAVAILABLE` (no CUDA GPU) |
| | $C_t$ | CPU Utilization | % | `MEASURED` (via `psutil`) |
| **Inference** | $K_t$ | Total KV-Cache Size | bytes | `MEASURED` (DynamicCache) |
| | $\Delta K_t$ | KV-Cache Step Growth | bytes | `ESTIMATED` (step difference) |
| | $T_t$ | Tokens Generated & Context | count | `MEASURED` |
| | Rate | Generation Rate | tokens/s | `ESTIMATED` ($\Delta \text{tokens} / \Delta t$) |
| **Activation** | $A_t$ | Latest Tensor Size & Latency | bytes, ms | `MEASURED` & `ESTIMATED` |
| **Energy** | $E_t$ | Energy Proxy Metric | proxy units | `ESTIMATED` ($C_t \cdot \Delta t$) |

---

## 3. Data Integrity & Validation Invariants

All fields in `RuntimeState` enforce physical domain validation:
- Bandwidth ($B_t \ge 0$), Latency ($L_t \ge 0$), Jitter ($J_t \ge 0$)
- Packet Loss ($0.0 \le P_t \le 1.0$)
- RAM & VRAM ($R_t \ge 0$, $V_t \ge 0$, $F_t \ge 0$)
- Utilization ($0.0 \le C_t \le 100.0$, $0.0 \le G_t \le 100.0$)
- Generated tokens & context length ($T_t \ge 0$)
- Activation tensor size & transfer latency ($\ge 0$)
- Any violation raises an explicit `ValueError`.
- Validation is evaluated only on `MEASURED`, `ESTIMATED`, and `EMULATED` values; `UNAVAILABLE` / `None` values are permitted and preserved.

---

## 4. StateBuffer (Ring Buffer & Trace Replay)

`StateBuffer` provides a thread-safe, bounded memory structure:
- **Bounded Capacity**: Fixed `maxlen` using `collections.deque` protected by `threading.Lock`. Eviction follows strict FIFO order.
- **Chronological Ordering Enforcement**: An appended state whose timestamp is strictly less than the latest stored timestamp (or with identical timestamp but lower step index) is rejected with a `ValueError`.
- **Window Retrieval**: `get_window(size)` retrieves up to `size` recent states along with completeness flags (`actual_length`, `is_partial`) without padding or throwing errors.
- **Trace Persistence & Replay**:
  - `save_json(filepath)` exports recorded execution traces.
  - `load_json(filepath)` and `from_json_file(filepath)` reconstruct the exact sequence for offline analysis and deterministic replay.

---

## 5. FeatureExtractor & Vectorization

Downstream prediction models require fixed-dimension, deterministically ordered inputs:

### Feature Ordering (22 Columns)
```
 0: bandwidth_mbps
 1: latency_ms
 2: packet_loss
 3: jitter_ms
 4: vram_allocated_mb
 5: vram_free_mb
 6: ram_used_mb
 7: ram_available_mb
 8: gpu_utilization
 9: cpu_utilization
10: kv_cache_bytes
11: kv_cache_growth_bytes
12: generated_tokens
13: context_length
14: generation_rate
15: latest_activation_bytes
16: latest_transfer_latency_ms
17: energy_proxy
18: kv_cache_growth_rate   (derived: Delta K / Delta t)
19: memory_growth_rate     (derived: Delta R / Delta t)
20: token_growth_rate      (derived: Delta T / Delta t)
21: elapsed_time           (derived: t - t_0)
```

### Masks & Normalization
- **Availability Mask**: Binary flag ($1$ = available, $0$ = unavailable / missing).
- **Source Mask**: Exact provenance string for every feature (`measured`, `estimated`, `emulated`, `unavailable`).
- **Missing Value Handling**: Kept as `None` or explicitly converted to dense arrays using `to_dense(fill_value=0.0)`.
- **Deterministic Normalization**:
  - `normalize_min_max(values, min_vals, max_vals)`: Scaled to $[0, 1]$ using externally supplied statistics.
  - `normalize_z_score(values, mean_vals, std_vals)`: Standardized using externally supplied statistics.
  - No internal fitting or trainable weights are included in Module 3.

---

## 6. Usage Example

```python
from src.state import RuntimeState, StateBuffer, FeatureExtractor
from src.telemetry.types import TelemetrySnapshot

# 1. Initialize rolling state buffer
buffer = StateBuffer(capacity=64)

# 2. Ingest telemetry from runtime
for snapshot in telemetry_snapshots:
    prev_state = buffer.latest()
    state = RuntimeState.from_telemetry(snapshot, prev_state=prev_state)
    buffer.append(state)

# 3. Query rolling window
window = buffer.get_window(size=16)
print(f"Window size: {window['actual_length']}, is_partial: {window['is_partial']}")

# 4. Extract feature vectors with masks
vectors = FeatureExtractor.from_states(window["states"])
latest_vec = vectors[-1]
print("Latest RAM:", latest_vec.get("ram_used_mb"))
print("VRAM available:", latest_vec.is_available("vram_allocated_mb"))  # False (CPU-only)
print("Source mask:", latest_vec.get_source("ram_used_mb"))            # 'measured'

# 5. Export trace for offline replay
buffer.save_json("experiments/traces/session_01.json")
```
