# Module 7: Adaptive Predictive Partition Controller

## 1. Architectural Role & System Placement

Module 7 is the operational decision-making engine of the system. It receives candidate partition cost evaluations from Module 6, telemetry states from Module 3, and forecast trajectories from Module 4, and makes formal, auditable split-adaptation decisions.

```
Telemetry (M2) -> RuntimeState (M3) -> Predictor (M4) 
                                             |
Model Catalog (M5) -> Cost Model (M6) -> CONTROLLER (M7) -> Migration Engine (M8)
```

### Strict Separation of Concerns
- **Module 6 evaluates cost and utility**: Computes $J(a)$ without selecting or acting.
- **Module 7 decides policy**: Implements policy selection, stability checking (cooldown, hysteresis, minimum benefit threshold), and safety checks (emergency overrides). It generates a declarative `MigrationRequest` without physically executing model tensor transfers.
- **Module 8 executes migration**: Consumes `MigrationRequest` to coordinate physical model weights, KV-cache migration, and socket handoffs.

---

## 2. Decision Formulation & Algorithmic Flow

At step $t$, the controller evaluates whether the system should retain its active configuration $a_{\text{active}}$ or transition to an alternate partition $a^*$:

```
                              +----------------------------+
                              |   Incoming RuntimeState    |
                              |   & Horizon Forecast       |
                              +--------------+-------------+
                                             |
                                             v
                              +----------------------------+
                              |  1. Safety Evaluation      |
                              +--------------+-------------+
                                     |               |
                         Emergency?  |               |  OK / Warning
                                     v               v
                +-------------------------+   +----------------------------+
                | Action:                 |   |  2. Mode Check             |
                | EMERGENCY_FALLBACK      |   |  (Static / Reactive /      |
                | (Bypasses all guards)   |   |   Predictive)              |
                +-------------------------+   +--------------+-------------+
                                                             |
                                                             v
                                              +----------------------------+
                                              |  3. Cost Optimization      |
                                              |     argmin J(a)            |
                                              +--------------+-------------+
                                                             |
                                      a* == active?          |  a* != active
                                      +----------------------+--------------------+
                                      |                                           |
                                      v                                           v
                        +---------------------------+               +----------------------------+
                        | Action: KEEP              |               |  4. Stability Guards       |
                        | Reason: ACTIVE_OPTIMAL    |               |     - Cooldown check       |
                        +---------------------------+               |     - Min-benefit delta    |
                                                                    |     - Hysteresis margin    |
                                                                    +--------------+-------------+
                                                                                   |
                                                                    Passed all?    |  Failed?
                                                                    +--------------+--------------+
                                                                    |                             |
                                                                    v                             v
                                                     +----------------------------+  +----------------------------+
                                                     | Action: MIGRATE            |  | Action: HOLD or KEEP       |
                                                     | Issues MigrationRequest    |  | Reason: COOLDOWN /         |
                                                     +----------------------------+  |         MARGINAL_BENEFIT   |
                                                                                     +----------------------------+
```

---

## 3. Controller Operating Modes

The controller supports three distinct operational regimes:

### 3.1 Static Mode (`ControllerMode.STATIC`)
- Always retains the initial active partition plan ($a_{\text{active}}$).
- Ignores network variations, memory changes, or cost discrepancies.
- Serves as the fixed-partition control baseline for research comparisons.
- Only severe emergency violations can force intervention if configured.

### 3.2 Reactive Mode (`ControllerMode.REACTIVE`)
- Evaluates candidate plans using only the current `RuntimeState` telemetry snapshot.
- Does not inspect `forecast` trajectories.
- Adapts only after degradation has already manifested in physical measurements.

### 3.3 Predictive Mode (`ControllerMode.PREDICTIVE`)
- Integrates `PredictionResult` forecasting over a configurable horizon $H$.
- Scores candidate plans against predicted conditions (e.g. forecasted bandwidth collapse or rising memory footprint).
- Triggers proactive migrations before bottlenecks cause token latency spikes or out-of-memory faults.

---

## 4. Anti-Thrashing & Stability Guards

Frequent repartitioning degrades overall throughput due to migration downtime, cache invalidation, and activation pipeline flushing. Module 7 introduces a multi-tier stability policy:

### 4.1 Cooldown Period ($T_{\text{cooldown}}$)
Following any partition migration, the controller enters a mandatory cooldown window of $N_{\text{cooldown}}$ steps:
$$\Delta t_{\text{switch}} = t_{\text{current}} - t_{\text{last\_switch}} < N_{\text{cooldown}} \implies \text{HOLD}$$
No non-emergency migration is permitted during this interval.

### 4.2 Minimum Benefit Threshold ($\theta_{\text{benefit}}$)
A candidate plan $a^*$ must demonstrate an improvement over the active plan that exceeds a minimum relative fraction $\theta$:
$$\frac{J(a_{\text{active}}) - J(a^*)}{J(a_{\text{active}})} \ge \theta_{\text{benefit}}$$
Prevents switching for negligible numerical gains that do not compensate for disruption.

### 4.3 Hysteresis Margin ($\delta_{\text{hysteresis}}$)
To prevent oscillation between two near-equivalent configurations (e.g., plans $A$ and $B$), transitioning requires the new plan to outperform the prior plan by more than the hysteresis margin $\delta$:
$$J(a_{\text{active}}) - J(a^*) > \delta_{\text{hysteresis}}$$

---

## 5. Safety Hierarchy & Emergency Overrides

System survival supersedes cost optimization. The `SafetyPolicy` monitors tier health and enforces emergency intervention:

1. **Out-of-Memory (OOM) Protection**:
   If memory utilization on any participating tier exceeds the critical threshold (default $90\%$), an emergency is flagged.
2. **Network Disconnection / Unreachable Tier**:
   If effective bandwidth drops below minimum viable thresholds ($< 0.1$ Mbps) or packet loss exceeds catastrophic limits ($> 50\%$), the tier is marked unreachable.
3. **Emergency Fallback Execution**:
   When an emergency condition is detected:
   - Cooldown timers are **bypassed**.
   - Minimum benefit thresholds are **bypassed**.
   - Hysteresis margins are **bypassed**.
   - The controller executes `EMERGENCY_FALLBACK` to the designated fallback plan (e.g., local monolithic or conservative partition).

---

## 6. Migration Request Contract (Module 8 Handoff)

When `ControlAction.MIGRATE` or `ControlAction.EMERGENCY_FALLBACK` is decided, the controller emits an explicit `MigrationRequest` dataclass:

```python
@dataclass(frozen=True)
class MigrationRequest:
    source_plan_id: str
    target_plan_id: str
    layers_to_move: Dict[str, List[int]]
    estimated_overhead_ms: float
    is_emergency: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
```

Module 8 consumes this structure to initiate and coordinate physical state migration.

---

## 7. Configuration Reference (`configs/baseline.yaml`)

```yaml
controller:
  mode: "predictive"             # "static" | "reactive" | "predictive"
  prediction_horizon: 4          # Forecast steps evaluated
  cooldown_steps: 3              # Minimum steps between non-emergency migrations
  min_benefit_threshold: 0.05    # 5% relative cost improvement required
  hysteresis_margin: 0.02        # Absolute cost hysteresis barrier
  max_migrations: 10             # Safety ceiling on total migrations per session
  safety:
    max_memory_utilization: 0.90 # Emergency threshold (90% tier capacity)
    min_bandwidth_mbps: 0.5      # Minimum viable network link
    max_packet_loss: 0.20        # 20% packet loss ceiling
```

---

## 8. Verification & Test Coverage

The Module 7 suite validates:
- Deterministic decision repeatability across identical states.
- Strict enforcement of cooldown, threshold, and hysteresis rejections.
- Immediate emergency fallback triggering on simulated memory and network failures.
- Zero floating-point regressions and pure ASCII compatibility across platforms.
