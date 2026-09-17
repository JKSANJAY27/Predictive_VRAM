# Module 9: Closed-Loop Predictive Runtime Integration

## 1. Architectural Role & End-to-End Orchestration

Module 9 is the top-level orchestration and integration engine uniting Modules 1 through 8 into an autonomous, closed-loop predictive runtime for distributed edge LLM split inference.

```
       +---------------------------------------------------------------+
       |                 ClosedLoopRuntime (Module 9)                  |
       +-------------------------------+-------------------------------+
                                       |
    +----------------------------------+----------------------------------+
    |                                  |                                  |
    v                                  v                                  v
+------------------+         +--------------------+             +-------------------+
| Module 1: Core   | <=====> | ControlCycleRunner | <=========> | Module 8: Phys.   |
| Split Inference  |         | (14-Step Loop)     |             | Migration Manager |
+------------------+         +---------+----------+             +-------------------+
                                       |
           +---------------------------+---------------------------+
           |               |           |            |              |
           v               v           v            v              v
     +-----------+   +-----------+ +-------+  +-----------+  +-----------+
     | Module 2  |   | Module 3  | | Mod 4 |  | Module 5  |  | Module 6  |
     | Telemetry |   | StateBuf  | | Pred  |  | Catalog   |  | CostModel |
     +-----------+   +-----------+ +-------+  +-----------+  +-----------+
```

### Core Responsibilities
1. **Control Cycle Lifecycle**: Coordinates the discrete 14-step control cycle at token boundaries without interrupting mid-layer forward passes.
2. **Strict Policy Separation**: Guarantees identical execution conditions across three policies:
   - **STATIC**: Fixed initial partition plan; ignores telemetry changes.
   - **REACTIVE**: Dynamic optimization evaluated strictly against current instantaneous measurements.
   - **PREDICTIVE**: Proactive optimization evaluating predicted state trajectories over horizon $H$.
3. **Execution Regimes**:
   - `EXECUTION`: Real PyTorch model evaluation across local tiers.
   - `SIMULATION`: Repeatable synthetic environment dynamics (10 registered scenarios).
   - `REPLAY`: Trace replay through control cycles for counterfactual evaluation.
4. **Structured Provenance & Tracing**: Records complete `RuntimeTrace` with fine-grained timings, decision rationales, migration windows, and performance metrics.

---

## 2. The 14-Step Control Cycle

Every control opportunity executes the following discrete lifecycle sequence:

| Step | Operation | Module Responsible | Invariant / Guarantee |
|:---|:---|:---|:---|
| **1** | Collect Raw Telemetry | Module 2 (`TelemetryAgent`) | Complete provenance tagging per metric |
| **2** | State Assembly | Module 3 (`to_runtime_state`) | Canonical `RuntimeState` conversion |
| **3** | Buffer Append | Module 3 (`StateBuffer`) | Chronological order verification |
| **4** | Forecast Construction | Module 4 (`Predictor`) | **Strictly disabled** in STATIC & REACTIVE modes |
| **5** | Candidate Generation | Module 5 (`SplitCatalog`) | Layer-consistent physical candidate plans |
| **6** | Candidate Scoring | Module 6 (`CostModel`) | Composite objective $J(a)$ evaluation |
| **7** | Controller Decision | Module 7 (`Controller`) | Anti-thrashing stability and safety checks |
| **8** | Quiescence & Lock | Module 8 (`MigrationManager`) | Single-flight execution guarantee |
| **9** | Physical Migration | Module 8 (`MigrationManager`) | Weights & KV-cache migration or rollback |
| **10** | State Verification | Module 8 (`MigrationVerifier`) | Numerical and structural pipeline integrity |
| **11** | Runtime State Commit | Module 8 / Module 1 | Active plan pointer atomically updated |
| **12** | Outcome Measurement | Module 9 (`ControlCycle`) | Post-action latency and bandwidth capture |
| **13** | Lead-Time Accounting | Module 9 (`ControlCycle`) | Proactive lead-time recording |
| **14** | Cycle Completion | Module 9 (`RuntimeTrace`) | JSON-serializable audit record stored |

---

## 3. Strict Fairness & Anti-Leakage Invariants

To maintain scientific integrity when comparing policies:
1. **No Future Leakage**: In REACTIVE mode, `forecast` is explicitly forced to `None`. The cost model and controller evaluate only current observations.
2. **Identical Candidate Space**: All three policies draw candidate partitions from the same `SplitCatalog` with identical capacity bounds.
3. **Constant Objective Weights**: The same objective function weights $(\alpha, \beta, \gamma, \delta, \epsilon)$ are supplied to all three modes.
4. **Isolated Migration Execution**: Module 9 never directly touches tensor weights or KV caches; all physical state mutations are delegated strictly to `MigrationManager`.
5. **Deterministic Seeding**: PRNG seeds are fixed (`seed=42`) across all comparative scenario runs.

---

## 4. Verification & Testing

Module 9 is fully validated via automated tests in `tests/test_orchestration.py`:
- `test_closed_loop_runtime_init`: Validates dependency injection and components.
- `test_create_default`: Tests factory setup with synthetic model and tiers.
- `test_static_mode_behavior`: Asserts STATIC mode never migrates (`KEEP_CURRENT`).
- `test_reactive_mode_disables_prediction`: Confirms no forecast generated in REACTIVE.
- `test_predictive_mode_generates_forecast`: Asserts valid forecasts in PREDICTIVE.
- `test_single_control_cycle_lifecycle`: Full 14-step cycle verification.
- `test_temporal_consistency_invariant`: Verifies forecast timestamps $\ge$ observation timestamps.
- `test_scheduler_token_interval`: Validates synchronization on token boundaries.
- `test_scheduler_single_flight_lock`: Asserts rejection of concurrent control cycles.
- `test_telemetry_failure_containment`: Validates fallback on collector exceptions.
- `test_predictor_failure_fallback_to_reactive`: Ensures graceful fallback if forecast fails.
- `test_migration_failure_and_rollback_containment`: Verifies state restoration on migration error.
- `test_trace_json_serialization`: Validates round-trip trace persistence to disk.
- `test_deterministic_replay`: Ensures bitwise identical decisions upon trace replay.
- `test_invariants_across_execution_modes`: Evaluates architectural invariants.
- `test_canonical_research_scenario_comparison`: Evaluates STATIC vs REACTIVE vs PREDICTIVE.
- `test_execution_mode_full_forward_pass`: End-to-end PyTorch autoregressive decode.
