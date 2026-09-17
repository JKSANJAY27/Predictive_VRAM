# Module 5 — Split Catalog and Feasible Candidate Plan Generation

## Overview

Module 5 translates raw model metadata, per-tier capacity configuration, a current
`RuntimeState` observation, and an optional short-horizon forecast from Module 4 into
a **structured, provenance-auditable catalog of candidate `PartitionPlan` objects**.

It does **not** score, rank, select, migrate, or execute plans. It is the
information-preparation layer that Module 6 (the Cost Model and Controller) will
consume in a later stage.

---

## Position in the Research Architecture

```
Module 1  →  Distributed split-inference baseline
Module 2  →  Runtime telemetry layer
Module 3  →  RuntimeState and StateBuffer
Module 4  →  Predictive forecasting engine
Module 5  →  Split Catalog and Feasible Candidate Generation   ← THIS MODULE
Module 6  →  Cost model and partition controller  (future)
Module 7  →  Migration manager                   (future)
Module 8  →  Performance and ablation evaluation (future)
```

---

## Scope and Invariants

| Property | Value |
|---|---|
| CPU-only host | VRAM metrics are always `DataSource.UNAVAILABLE` — never fabricated |
| Feasibility on CPU | `UNKNOWN` (not `INFEASIBLE`) when memory telemetry is absent |
| Separation of concerns | Module 5 produces **candidates only** — no scoring, no selection |
| Memory estimation provenance | Always `DataSource.ESTIMATED` |
| Plan IDs | Deterministic, stable, human-readable strings |

---

## File Map

```
src/partitioning/
├── __init__.py          — Public API exports
├── metadata.py          — ModelMetadata, TierCapacity
├── plan_id.py           — Deterministic plan identifier generation
├── structural.py        — Contiguous layer split enumeration
├── memory_estimator.py  — Per-tier memory requirement projection
├── feasibility.py       — Feasibility engine (FEASIBLE / INFEASIBLE / UNKNOWN)
├── candidate.py         — CandidatePlan dataclass
├── comparison.py        — PlanDifference delta analysis
├── formatting.py        — ASCII audit table rendering
└── catalog.py           — SplitCatalog orchestrator
```

---

## Sub-Module Reference

### `ModelMetadata`

Encodes static structural dimensions of the target transformer model.

```python
from src.partitioning import ModelMetadata

# GPT-2 base synthetic default (12 layers, 124M parameters)
meta = ModelMetadata.synthetic_default(total_layers=12)

meta.total_layers    # 12
meta.hidden_size     # 768
meta.num_heads       # 12
meta.head_dim        # 64  (property: hidden_size // num_heads)
meta.dtype_bytes     # 4   (float32)
meta.per_layer_params[0]  # ~7.09M params per transformer block
```

> **Important**: `ModelMetadata` carries no runtime measurements. It is derived
> from model configuration, not from actual loaded weights.

---

### `TierCapacity`

Per-tier static configuration: device type, memory limits, layer constraints.

```python
from src.partitioning import TierCapacity
from src.runtime.tier import TierId
from src.telemetry.types import DataSource

# CPU-only tier — no memory limit configured (UNAVAILABLE provenance)
cap = TierCapacity(tier_id=TierId.USER_DEVICE, device="cpu")

# Emulated tier with 8 GB available (for experiment replay)
cap_emulated = TierCapacity(
    tier_id=TierId.EDGE_A,
    device="cpu",
    available_memory_mb=8192.0,
    memory_provenance=DataSource.EMULATED,
)
```

**Validation rules:**
- `total_memory_mb` and `available_memory_mb` must be ≥ 0 if set
- `compute_capacity_score` must be > 0
- `max_layers` must be ≥ 0 if set

---

### `enumerate_structural_plans`

Exhaustively generates all contiguous-layer `PartitionPlan` configurations.

```python
from src.partitioning import enumerate_structural_plans

plans = enumerate_structural_plans(
    total_layers=12,
    allow_monolithic=True,
    allow_two_tier=True,
    allow_three_tier=True,
)
# For 12 layers → 1 + 2×11 + C(11,2) = 78 plans
```

**Guarantees:**
- Every plan covers all layers `[0, total_layers-1]` with no gaps or overlaps
- No duplicate plan IDs in the output
- Deterministic ordering: monolithic → two-tier U→EA, U→EB → three-tier

**Plan counts for `N` layers:**
| Topology | Count |
|---|---|
| Monolithic (U only) | 1 |
| Two-tier (U→EA) | N − 1 |
| Two-tier (U→EB) | N − 1 |
| Three-tier (U→EA→EB) | (N−1)(N−2)/2 |

---

### `generate_plan_id`

Produces a stable, human-readable string ID from any `PartitionPlan`.

| Plan | ID |
|---|---|
| Monolithic UserDevice | `local` |
| UserDevice → EdgeA, cut at layer 5 | `u0-5_ea6-11` |
| UserDevice → EdgeB, cut at layer 5 | `u0-5_eb6-11` |
| Three-tier, cuts at 3 and 7 | `u0-3_ea4-7_eb8-11` |

---

### `estimate_plan_memory`

Projects memory footprint per active tier including:
- **Parameter memory**: sum of `per_layer_params × dtype_bytes`, plus embeddings on the
  first tier and `lm_head` on the last tier
- **KV-cache memory**: `2 × n_heads × head_dim × context_length × dtype_bytes` per layer
- **Safety margin**: configurable overhead buffer (default 256 MB)

```python
from src.partitioning import estimate_plan_memory, ModelMetadata
from src.runtime.partition import PartitionPlan

meta = ModelMetadata.synthetic_default(total_layers=12)
plan = PartitionPlan.two_tier(total_layers=12, cut_layer=5)

reqs = estimate_plan_memory(plan, meta, context_length=512, safety_margin_mb=256.0)
# reqs[TierId.USER_DEVICE].total_required_mb → param + kv + 256 MB
# reqs[TierId.EDGE_A].total_required_mb     → param + kv + 256 MB
# All requirements carry provenance=DataSource.ESTIMATED
```

---

### `evaluate_plan_feasibility`

Three-state feasibility engine:

| State | Meaning |
|---|---|
| `FEASIBLE` | All structural and resource constraints satisfied |
| `INFEASIBLE` | Violates structural limits or observed memory is insufficient |
| `UNKNOWN` | Required resource telemetry unavailable (CPU-only, no VRAM) |

```python
from src.partitioning import evaluate_plan_feasibility, FeasibilityStatus

s_now, s_pred, reasons, violations = evaluate_plan_feasibility(
    plan=plan,
    tier_capacities=tier_caps,
    memory_requirements=reqs,
    current_state=runtime_state,   # Optional
    forecast=prediction_result,    # Optional, from Module 4
)
```

**Evaluation order:**
1. **Structural check** — disabled tiers, `max_layers` limit
2. **Current memory check** — compares `available_memory_mb` vs `total_required_mb`
3. **Predicted memory check** — walks across forecast horizon values

**Critical provenance rule:** If a tier has `device="cpu"` and no `available_memory_mb`
is configured, the result is `UNKNOWN` — **not** `INFEASIBLE`. This correctly represents
the state of this development machine (no CUDA, no GPU VRAM).

---

### `CandidatePlan`

Frozen dataclass that bundles a `PartitionPlan` with its full feasibility assessment.

```python
from src.partitioning import CandidatePlan, FeasibilityStatus

candidate.plan_id                    # str, e.g. "u0-5_ea6-11"
candidate.partition_plan             # PartitionPlan
candidate.active_tiers               # List[TierId]
candidate.layer_assignment           # Dict[str, Tuple[int, int]]
candidate.number_of_boundaries       # int
candidate.feasibility_now            # FeasibilityStatus
candidate.feasibility_predicted      # FeasibilityStatus
candidate.is_feasible                # bool — True only if BOTH are FEASIBLE
candidate.feasibility_reasons        # List[str]
candidate.estimated_memory_requirements  # Dict[str, float] (MB)
candidate.resource_violations        # List[str]
candidate.metadata                   # Dict — delta vs current plan, context_length, etc.
```

> `CandidatePlan` is **frozen** (immutable). It carries **no** cost score, utility
> value, or migration decision. Those belong to Module 6.

---

### `compare_plans` / `PlanDifference`

Quantifies the structural distance between a currently active plan and a proposed
candidate.

```python
from src.partitioning import compare_plans

diff = compare_plans(current_plan, candidate_plan)

diff.is_identical             # bool
diff.changed_layer_count      # number of layers assigned to a different tier
diff.affected_tiers           # List[TierId] — tiers whose assignment changed
diff.boundary_shift_distance  # total displacement of transfer cut points
diff.added_boundaries         # new inter-tier communication cuts
diff.removed_boundaries       # eliminated inter-tier communication cuts
diff.boundary_count_delta     # net change in number of boundaries
```

---

### `SplitCatalog`

The primary entry point for Module 5. Orchestrates all sub-modules.

```python
from src.partitioning import SplitCatalog, ModelMetadata

catalog = SplitCatalog(
    tier_capacities=my_caps,       # Dict[TierId, TierCapacity]
    allow_monolithic=True,
    allow_two_tier=True,
    allow_three_tier=True,
    safety_margin_mb=256.0,
    default_context_length=128,
)

# Generate all candidates (with feasibility metadata attached)
candidates = catalog.generate(
    model_metadata=meta,
    current_state=runtime_state,   # Optional RuntimeState from Module 3
    forecast=prediction_result,    # Optional PredictionResult from Module 4
    current_plan=active_plan,      # Optional PartitionPlan for delta tagging
    mode="exhaustive",             # or "filtered" (removes INFEASIBLE)
    context_length=256,
)

# Retrieve only the strictly FEASIBLE subset
feasible = catalog.filter_feasible(candidates)

# Render audit table
print(catalog.format_table(candidates))
```

**`mode` options:**

| Mode | Behavior |
|---|---|
| `"exhaustive"` | Returns all structurally valid candidates with feasibility metadata |
| `"filtered"` | Excludes candidates where `feasibility_now == INFEASIBLE` or `feasibility_predicted == INFEASIBLE` |

---

### `format_candidate_table`

Renders a human-readable ASCII table of candidates for logging and debugging.

```
PLAN ID                | TIERS    | BOUNDS | NOW        | PREDICTED  | EXPLANATION
---------------------------------------------------------------------------------------------
local                  | U        | 0      | UNKNOWN    | UNKNOWN    | Tier 'user_device' memory ...
u0-0_ea1-3             | U-EA     | 1      | UNKNOWN    | UNKNOWN    | Tier 'user_device' memory ...
u0-0_ea1-2_eb3-3       | U-EA-EB  | 2      | UNKNOWN    | UNKNOWN    | Tier 'user_device' memory ...
```

---

## Integration Points

### Inputs consumed

| Source | Object | How consumed |
|---|---|---|
| Module 1 | `PartitionPlan` | Reused as the underlying representation |
| Module 2 | `DataSource` | Provenance tagging on all fields |
| Module 3 | `RuntimeState` | Feeds current memory observations to feasibility engine |
| Module 4 | `PredictionResult` | Feeds forecast horizon to predicted feasibility check |

### Outputs produced

| Object | Description |
|---|---|
| `List[CandidatePlan]` | Primary output: catalog of evaluated candidate plans |
| `PlanDifference` | Structural delta between any two plans |
| `format_candidate_table()` | ASCII audit table for logging |

---

## CPU-Only Development Constraints

The following invariants hold throughout Module 5 on this hardware:

1. **`DataSource.UNAVAILABLE` for VRAM**: No CUDA GPU is present. `TierCapacity.memory_provenance`
   defaults to `UNAVAILABLE` and no VRAM measurement is ever fabricated.

2. **`FeasibilityStatus.UNKNOWN`** (not `INFEASIBLE`) is returned when required memory
   telemetry is absent. `UNKNOWN` is a first-class, valid provenance state — not an error.

3. **Memory estimation provenance** is always `DataSource.ESTIMATED` — analytically
   derived from model config, not from loaded weights or live measurements.

4. **Emulated capacities** use `DataSource.EMULATED` to explicitly signal that
   the values were externally injected for experimentation — they do not reflect
   real physical conditions.

---

## Testing

```bash
# Module 5 tests only
python -m pytest tests/test_partitioning.py -v

# Full suite (all modules)
python -m pytest --tb=short -q
```

**Test coverage (82 tests):**

| Class | Tests |
|---|---|
| `TestModelMetadata` | 8 |
| `TestTierCapacity` | 7 |
| `TestGeneratePlanId` | 5 |
| `TestEnumerateStructuralPlans` | 10 |
| `TestEstimatePlanMemory` | 9 |
| `TestEvaluatePlanFeasibility` | 7 |
| `TestCandidatePlan` | 7 |
| `TestComparePlans` | 7 |
| `TestFormatCandidateTable` | 5 |
| `TestSplitCatalog` | 14 |
| `TestPublicApi` | 1 |
| `TestSeparationOfConcerns` | 2 |

---

## Demo Script

```bash
python scripts/run_candidate_demo.py
```

Runs end-to-end through all five sub-systems:
1. `ModelMetadata` construction
2. Structural plan enumeration (counts and sample IDs)
3. Memory estimation (param + KV-cache + safety margin, per plan)
4. Feasibility under three hardware scenarios:
   - CPU-only no limits → `UNKNOWN` (correct provenance)
   - Emulated 32 GB → all `FEASIBLE`
   - Emulated 1 MB → all `INFEASIBLE`
5. Plan delta comparison and catalog delta metadata injection
