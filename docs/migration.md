# Module 8: Physical Migration and Runtime State Transition

## 1. Architectural Role & System Placement

Module 8 is the physical migration and execution engine for distributed partition transitions. It sits directly between the high-level policy decision engine (**Module 7 Controller**) and the physical execution runtime (**Module 1 Split Inference Runtime**).

```
+-----------------------------+
|   Module 7: Controller      |  --> Issues declarative MigrationRequest
+--------------+--------------+
               |
               v
+-----------------------------+
|   Module 8: Migration Engine|  --> Coordinates atomic, transactional plan migration
+--------------+--------------+
               |
               v
+-----------------------------+
|   Module 1: Runtime Core    |  --> Physically executes split inference across tiers
+-----------------------------+
```

### Strict Separation of Concerns
- **Module 7 Decides**: Evaluates candidate plans, scores costs, evaluates hysteresis and cooldown, and outputs a declarative `MigrationRequest`.
- **Module 8 Executes**: Consumes `MigrationRequest`, acquires the runtime quiescence lock, transfers model layer weights and active KV-cache tensors, updates tier partition layouts, verifies pipeline integrity, and commits or rolls back atomically.
- **Invariants**: Module 8 **never** evaluates the cost model $J(a)$, never computes $\text{argmin}$, never makes threshold decisions, and never forecasts conditions. It is purely an atomic migration executor.

---

## 2. Transactional Migration Lifecycle

Migration operates under strict all-or-nothing transactional semantics. If any stage encounters an exception or verification failure, the system initiates an immediate automated rollback to restore the pre-migration state.

```
+---------------------------------------------------------------+
|                      Migration Request                        |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
| 1. Quiescence & Lock          Acquire runtime lock            |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
| 2. Validation                 Verify active plan == src plan   |
|                               Check idempotency & bounds      |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
| 3. Planning & Delta           Compute affected layers/tiers   |
|                               Create layer & KV backup states |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
| 4. Model Weight Transfer      Transfer layer blocks between   |
|                               physical / emulated tiers       |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
| 5. KV-Cache Migration         Slice and transfer active keys  |
|                               and values for migrating layers |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
| 6. Post-Migration Verification FAST (shape/placement check) or|
|                               DEEP (activation equivalence)   |
+-------------------------------+-------------------------------+
          |                                            |
   Validation Passes                            Validation Fails
          |                                            |
          v                                            v
+-----------------------+                    +-----------------------+
| 7a. Commit            |                    | 7b. Rollback          |
|  - Update active_plan |                    |  - Restore layers     |
|  - Release lock       |                    |  - Restore KV cache   |
|  - Emit SUCCESS result|                    |  - Re-arm src plan    |
+-----------------------+                    |  - Emit ROLLED_BACK   |
                                             +-----------------------+
```

---

## 3. Core Components

### 3.1 `MigrationManager`
The central transactional coordinator. Encapsulates the complete migration state machine:
- Manages the lifecycle from quiescence lock acquisition to final commit.
- Tracks phase-level millisecond timings (`MigrationPhaseTimings`).
- Tracks physical bytes transferred per layer and per tier (`MigrationMetrics`).
- Correlates predicted vs. actual migration latency (`MigrationComparison`).

### 3.2 `RuntimeAdapter`
An abstraction layer interfacing between Module 8 and Module 1's `DistributedInferenceExecutor` and `LayeredTransformer`. It insulates the migration engine from internal execution details and provides:
- Partition plan inspection (`get_current_plan()`).
- Layer retrieval, placement, and eviction (`get_layer()`, `set_layer()`, `remove_layer()`).
- Active KV-cache discovery and swapping (`get_kv_cache()`, `set_kv_cache()`).
- Migration locking to ensure quiescence during token generation boundaries.

### 3.3 `MigrationPlanner` & `MigrationPlan`
Calculates the exact structural delta between two `PartitionPlan` instances:
- Determines which layers must be migrated, their source tier, and their destination tier.
- Identifies newly involved tiers and evacuated tiers.
- Minimizes redundant data transfers by preserving stationary layers.

### 3.4 `MigrationTransferProvider`
Pluggable transfer backends supporting diverse execution environments:
- **`LocalTensorTransfer`**: In-memory tensor cloning for single-machine or CPU simulation tests.
- **`EmulatedNetworkTransfer`**: Injects analytical bandwidth and latency delays (derived from Module 2 network telemetry) without requiring physical distributed sockets.

### 3.5 `LayerTransferManager` & `KVCacheTransferManager`
Stateful managers responsible for transferring and backing up model weights and attention cache state:
- Maintains an in-memory snapshot of layers prior to mutation.
- Slices multi-layer KV-cache tensors for migrated blocks while maintaining sequential generation context.
- Provides immediate rollback methods (`restore_backup()`) invoked during transaction aborts.

### 3.6 `MigrationVerifier`
Verifies runtime state before committing:
- **`VerificationMode.FAST`**: Verifies that all layers in the target plan are present, assigned to valid tiers, and have non-empty weights.
- **`VerificationMode.DEEP`**: Runs a synthetic forward pass through the newly configured pipeline and verifies token output consistency against the pre-migration baseline.

### 3.7 `RollbackManager`
Coordinates full state reversion if any stage fails:
1. Restores original layer placements from backup snapshots.
2. Restores pre-migration KV-cache slices.
3. Re-applies the original `PartitionPlan` to the runtime executor.
4. Returns a comprehensive `RollbackReport` detailing restoration integrity.

---

## 4. Configuration & Operational Modes

Configured via `MigrationConfig`:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `execution_mode` | `MigrationMode` | `EMULATED` | `REAL`, `EMULATED`, or `SIMULATE` (dry run). |
| `verification_mode` | `VerificationMode` | `FAST` | `FAST` (structural) or `DEEP` (forward pass). |
| `rollback_enabled` | `bool` | `True` | Whether to automatically roll back on failure. |
| `single_flight` | `bool` | `True` | Enforce one active migration at a time. |
| `emulated_bandwidth_delay` | `bool` | `False` | Simulate network transfer delay based on byte volume. |
| `lock_timeout_seconds` | `float` | `2.0` | Maximum wait time for runtime quiescence lock. |

---

## 5. Failure Injection & Resilience

Module 8 is verified against 6 deliberate failure scenarios:
1. **Preparation Stage Failure**: Aborts before transfers begin; verifies clean lock release and no state corruption.
2. **Layer Transfer Failure**: Injects an error during weight movement of a specific layer; verifies that partially migrated layers are reverted.
3. **KV-Cache Transfer Failure**: Simulates network interruption during KV-cache transfer; verifies that KV caches are restored without history loss.
4. **Post-Migration Verification Failure**: Fails deep verification; triggers immediate rollback to preserve inference correctness.
5. **Stale Request Handling**: Rejects migration requests whose `source_plan` does not match the active runtime plan.
6. **Idempotent No-Op**: Safely acknowledges requests where `source_plan == target_plan == active_plan` without redundant transfers.
