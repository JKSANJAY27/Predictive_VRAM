# Predictive VRAM and Network Forecasting Engine

Module 4 implements the short-horizon time-series forecasting engine for the Predictive VRAM split-inference system.

Given recent runtime state history from `StateBuffer`, the engine answers:
> **"What will relevant network, memory, and KV-cache conditions look like over a short future horizon $\Delta$?"**

---

## 1. Research Role & Boundaries

In the overall research architecture:

```
Telemetry (Module 2) ──> RuntimeState (Module 3) ──> StateBuffer (Module 3)
                                                              │
                                                              ▼
                                                   PREDICTOR (Module 4)
                                                              │ (PredictionResult)
                                                              ▼
                                            (Future Module 5: Cost Model)
                                                              │
                                                              ▼
                                            (Future Module 7: Controller)
```

### Explicit Scope Boundaries
- **What Module 4 DOES:**
  - Forecasts near-future trajectories for network conditions ($B_t, L_t, P_t, J_t$) and memory ($V_t, R_t, K_t$).
  - Evaluates prediction errors (residuals, MAE, RMSE) and warns of abrupt shifts.
  - Computes threshold crossing lead-time ($\tau_{\text{lead}}$).
  - Performs chronological walk-forward validation with zero temporal leakage.
- **What Module 4 DOES NOT DO:**
  - It does **not** choose partitions.
  - It does **not** migrate layers or KV caches.
  - It does **not** modify active model execution.
  - It does **not** implement controller thresholds or hysteresis.

---

## 2. Hardware Constraint & Missing Data Integrity

The development machine is CPU-only (Intel Core i5-4200U, 8 GB RAM, PyTorch CPU, No CUDA GPU).

### Strict Hardware Integrity Invariants:
1. **Never Fabricate VRAM**: When VRAM is physically unavailable, target forecasts for VRAM are flagged `is_available = False`, `status = "unavailable"`, and values are populated with `None`.
2. **Never Substitute CPU RAM as VRAM**: CPU host memory ($R_t$) and GPU memory ($V_t$) represent distinct physical resource pools. They are never conflated or substituted.
3. **Explicit Synthetic/Emulated Tagging**: Synthetic traces used for controlled evaluation are explicitly labeled `DataSource.EMULATED`.

---

## 3. Forecasting Targets

| Target Identifier | Symbol | Unit | Description | Host Availability |
| :--- | :--- | :--- | :--- | :--- |
| `bandwidth_mbps` | $B_t$ | Mbps | Tier uplink transmission speed | `EMULATED` |
| `latency_ms` | $L_t$ | ms | Round-trip transmission time | `EMULATED` |
| `packet_loss` | $P_t$ | $[0.0, 1.0]$ | Transmission loss ratio | `EMULATED` |
| `jitter_ms` | $J_t$ | ms | Latency variation standard deviation | `EMULATED` |
| `vram_free_mb` | $F_t$ | MB | Remaining unallocated GPU memory | `UNAVAILABLE` |
| `vram_allocated_mb` | $V_t$ | MB | Currently allocated GPU memory | `UNAVAILABLE` |
| `ram_used_mb` | $R_t$ | MB | System host RAM used | `MEASURED` |
| `ram_available_mb` | $R_{\text{avail}}$ | MB | System host RAM available | `MEASURED` |
| `kv_cache_bytes` | $K_t$ | bytes | Cumulative transformer KV-cache volume | `MEASURED` |

---

## 4. Predictor Interface & Architecture

All predictors adhere to the common abstract interface defined in `src.prediction.base.Predictor`:

```python
class Predictor(ABC):
    def predict(
        self,
        state_window: Sequence[RuntimeState],
        horizon_seconds: float = 2.0,
        step_interval_seconds: Optional[float] = None,
    ) -> PredictionResult:
        ...
```

### Strongly Typed `PredictionResult` Schema
```python
PredictionResult(
    prediction_timestamp: float,      # Clock time at t0
    horizon_seconds: float,           # Requested forecast duration Delta
    step_interval_seconds: float,     # Discretization step dt
    forecast_timestamps: List[float], # [t0 + dt, t0 + 2dt, ...]
    targets: Dict[str, TargetForecast],
    predictor_name: str,
    input_window_length: int,
    is_valid: bool,                   # False if insufficient history
    status_message: str,
    prediction_latency_ms: float,     # Wall-clock prediction overhead
)
```

Each `TargetForecast` encapsulates:
- `values`: Forecast trajectory with physical bounds applied.
- `raw_values`: Unconstrained forecast before clipping.
- `error_estimate`: In-sample error magnitude (recent MAE of fit).
- `is_available`: Boolean indicating target observability.
- `status`: `"ok"`, `"unavailable"`, or `"insufficient_history"`.
- `constraint_applied`: Boolean flag indicating whether clipping occurred.

---

## 5. Predictor Implementations

### 5.1 Last-Value Baseline (`LastValuePredictor`)
Persistence model projecting current conditions indefinitely:
$$\hat{y}(t + \Delta) = y(t)$$

### 5.2 Linear-Trend Baseline (`LinearTrendPredictor`)
Estimates recent rate of change via least-squares linear fit on the historical window:
$$\hat{y}(t + \Delta) = y(t) + s \cdot \Delta$$
Where slope $s = \frac{\sum (t_i - \bar{t})(y_i - \bar{y})}{\sum (t_i - \bar{t})^2}$.

### 5.3 Moving-Average Baseline (`MovingAveragePredictor`)
Projects future values as the arithmetic mean of the recent $k$ points:
$$\hat{y}(t + \Delta) = \frac{1}{k} \sum_{i=1}^k y(t - k + i)$$

### 5.4 KV-Cache-Aware Memory Predictor (`KVCacheAwareMemoryPredictor`)
Jointly models memory accumulation alongside token generation:
$$\hat{V}(t + \Delta) = V_t + s_V \cdot \Delta$$
$$\hat{K}(t + \Delta) = K_t + s_K \cdot \Delta$$
Computes time-to-threshold $\tau_{\text{thresh}}$ for early memory exhaustion warning:
$$\tau_{\text{thresh}} = \frac{\text{threshold} - y_t}{s}$$

### 5.5 Learned Time-Series Predictor (`LearnedTimeSeriesPredictor`)
Ridge-regularized multivariate linear model with strict feature isolation:
$$W = (X^T X + \lambda I)^{-1} X^T Y$$
Zero future-data leakage: feature normalization statistics (mean, std) are computed strictly on training sequences.

---

## 6. Physical Constraints & Domain Bounds

All forecasts are constrained to realistic physical intervals using `apply_physical_constraints`:
- Bandwidth $\ge 0$ Mbps
- Latency $\ge 0$ ms, Jitter $\ge 0$ ms
- Packet Loss $\in [0.0, 1.0]$
- Memory (VRAM, RAM, KV-cache) $\ge 0$
- Utilization $\in [0.0, 100.0]$ %

The system preserves `raw_values` alongside clamped `values` and records `constraint_applied = True`.

---

## 7. Metrics & Walk-Forward Validation

### Chronological Walk-Forward Evaluation
Time-series validation strictly forbids random sample shuffling. `evaluate_walk_forward()` operates chronologically:
1. Feeds historical window $[t_0 \dots t_k]$ to the predictor.
2. Predicts future interval $[t_k + 1 \dots t_k + \Delta]$.
3. Advances one step, appending the new actual observation to history.
4. Compares forecasts against subsequently observed ground truth.

### Evaluation Metrics
- **MAE** (Mean Absolute Error): $\frac{1}{N} \sum |y_i - \hat{y}_i|$
- **RMSE** (Root Mean Squared Error): $\sqrt{\frac{1}{N} \sum (y_i - \hat{y}_i)^2}$
- **MAPE** (Mean Absolute Percentage Error): $\frac{100\%}{N} \sum |\frac{y_i - \hat{y}_i}{y_i}|$ (guarded when $y_i \approx 0$)
- **Prediction Lead-Time**: $\tau_{\text{lead}} = t_{\text{actual\_crossing}} - t_{\text{predicted\_crossing}}$
- **Abrupt Change Residual Detector**: Identifies sudden distribution shifts when $|e_t| > \mu_e + k \cdot \sigma_e$.

---

## 8. Synthetic Trace Scenarios

The `SyntheticTraceGenerator` provides 9 controlled, deterministic benchmark profiles:
1. `stable_memory`: Constant RAM & VRAM baseline.
2. `linear_memory_growth`: Steady +50 MB/s memory pressure.
3. `kv_cache_driven_memory_growth`: Super-linear KV expansion matching context length.
4. `stable_network`: Constant 100 Mbps uplink, 20 ms RTT.
5. `bandwidth_degradation`: Gradual drop from 100 Mbps to 10 Mbps.
6. `bandwidth_recovery`: Uplink improvement from 10 Mbps to 100 Mbps.
7. `oscillating_bandwidth`: Sinusoidal variation between 20 and 80 Mbps.
8. `latency_spike`: Sudden jump to 180 ms simulating cellular handoff.
9. `combined_degradation`: Simultaneous bandwidth collapse and memory expansion.

---

## 9. Usage Example

```python
from src.prediction import get_predictor
from src.state.buffer import StateBuffer

# 1. Instantiate predictor from registry
predictor = get_predictor(
    predictor_type="kv_aware_memory",
    targets=["bandwidth_mbps", "latency_ms", "vram_free_mb", "ram_used_mb"],
    minimum_history_length=4,
)

# 2. Extract recent state window from StateBuffer
state_window = state_buffer.get_window(size=10)["states"]

# 3. Forecast future trajectory for 2.0s horizon
result = predictor.predict(state_window, horizon_seconds=2.0)

# 4. Consume forecast
bw_forecast = result.get_target("bandwidth_mbps")
print(f"Predicted Bandwidth (+2s): {bw_forecast.values[-1]:.1f} Mbps (error: +/- {bw_forecast.error_estimate:.1f})")

vram_forecast = result.get_target("vram_free_mb")
print(f"VRAM Status: {vram_forecast.status}")  # 'unavailable' on CPU hardware
```
