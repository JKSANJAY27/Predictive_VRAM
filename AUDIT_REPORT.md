# AUDIT_REPORT.md — Phase 1 Discovery & Audit

> **Generated:** 2026-10-15T03:08 IST  
> **Environment:** Windows 11, Python 3.12.0, PyTorch 2.14.1+cpu, CUDA: **Not Available**, `uv` package manager  
> **Baseline tests:** 563 passed, 4 skipped (12.27s) — no failures  

---

## 1. Module Map

| Module | Path | Description |
|--------|------|-------------|
| **Runtime (M1)** | `src/runtime/` | Distributed split-inference executor, LayeredTransformer, PartitionPlan, Tier definitions, tensor transfer |
| **Telemetry (M2)** | `src/telemetry/` | Source-tagged collectors (CPU, memory, KV-cache, network emulation, activation), provenance tagging |
| **State (M3)** | `src/state/` | Canonical RuntimeState, StateBuffer ring-buffer, FeatureExtractor |
| **Prediction (M4)** | `src/prediction/` | Forecasting engine (LinearTrend, MovingAverage, LastValue, KVCacheAwareMemory, Learned), PredictionResult |
| **Partitioning (M5)** | `src/partitioning/` | Split catalog, structural enumerator, feasibility evaluator, plan_id generator, memory estimator |
| **Cost Model (M6)** | `src/cost/` | Multi-objective cost scoring (latency, communication, memory, energy, switching), normalization |
| **Controller (M7)** | `src/controller/` | AdaptivePartitionController (decide loop), SafetyPolicy, StabilityChecker (anti-thrashing), ControllerHistory |
| **Migration (M8)** | `src/migration/` | MigrationManager (transactional), RuntimeAdapter, planner, rollback, KV/layer transfer, verifier, validator |
| **Orchestration (M9)** | `src/orchestration/` | ClosedLoopRuntime, ControlCycleRunner (14-step cycle), scenarios (10 synthetic), scheduler, trace |
| **Evaluation (M10)** | `src/evaluation/` | Experiment runner, scenario catalog, metrics, visualization, export, validation |
| **Hardware (M11)** | `src/hardware/` | CUDA/CPU hardware providers, network measurement/profiles, transport, capabilities |
| **Research (M12)** | `src/research/` | Split verification, GPU validation, ablations, statistics, report generation |
| **Config** | `src/config/` | Global settings |
| **Utils** | `src/utils/` | Logging utilities |
| **Dashboard** | `src/dashboard/` | HTTP server (`server.py`) serving single-page `static/index.html` with REST API endpoints |

---

## 2. Command Inventory

| Command | Purpose | Verified |
|---------|---------|----------|
| `.\setup.ps1` / `setup.bat` | Install dependencies into `.venv` via uv | Not yet verified |
| `.\start.ps1` / `start.bat` | Launch dashboard on port 8000 (background PID tracking) | Works |
| `.\end.ps1` / `end.bat` | Stop dashboard process | Not yet verified |
| `uv run python scripts/run_dashboard.py --port 8000` | Direct dashboard launch | Works |
| `uv run python -m pytest tests/` | Run full test suite | 563 passed, 4 skipped |

**Scripts directory (28 scripts):** Various demo runners, evaluation scripts, benchmark tools.

---

## 3. Dashboard Checklist (D1-D13)

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| **D1** | Dashboard starts with one command, no console errors | **PARTIAL** | Starts OK. Console errors not yet verified via E2E. |
| **D2** | Split visualization shows layers partitioned across tiers with correct counts | **PASS** | `index.html` L552-620: UD [0-1], EA [2-3], EB [standby]. 4 layers shown. |
| **D3** | Split visualization updates live after migration | **PARTIAL** | Migration button moves Layer 1 visually. But not driven by Scenario A-E. No live streaming. |
| **D4** | Live metrics present and correct | **FAIL** | KPI cards show hardcoded values: "100 Mbps", "4.0 KB/s", "100% PASS". Only RAM comes from API. |
| **D5** | Controller decision log | **MISSING** | No decision log panel exists. |
| **D6** | Migration state visible | **PARTIAL** | Badge shows "Quiescence: Ready" but no live state streaming. |
| **D7** | Scenario buttons A-E | **MISSING** | No scenario buttons. Dashboard has migration/prediction/parity/catalog buttons only. |
| **D8** | Scenario stop/reset | **MISSING** | No Stop or Reset buttons. |
| **D9** | CUDA/GPU panel | **PARTIAL** | Tier shows "VRAM: UNAVAILABLE (CPU)" but no dedicated GPU panel. |
| **D10** | Units, labels correct | **PARTIAL** | KPI cards have units. No charts exist. |
| **D11** | Displayed values equal backend values | **FAIL** | Most KPI values are hardcoded in HTML. |
| **D12** | Layout at 1920x1080, 1440x900, 1280x720 | **NOT VERIFIED** | CSS looks reasonable but needs viewport testing. |
| **D13** | Grey neon glassmorphism theme | **PARTIAL** | Dark theme with glass effects exists. Needs adjustment to match exact spec colors. |

---

## 4. Scenario Behavior Audit

### Critical Finding: No Dashboard Scenario Integration

The dashboard does NOT implement Scenarios A-E. The backend has `src/orchestration/scenarios.py` with 10 synthetic scenarios and `ClosedLoopRuntime` that can run them, but no API connects them to the dashboard.

### Mapping spec scenarios to existing code:

| Spec | Code Scenario | Match |
|------|--------------|-------|
| **A (Stable)** | `"stable"` | Conditions match (100 Mbps, 5ms RTT) |
| **B (KV growth)** | `"kv_cache_growth"` | Network matches but KV growth mechanism needs verification |
| **C (Temp bandwidth collapse)** | None exactly — `"sudden_bandwidth_drop"` drops to 2 Mbps but never recovers | Need new scenario with 8s dip + recovery |
| **D (Persistent poor BW)** | `"gradual_bandwidth_degradation"` or `"sudden_bandwidth_drop"` | Approximate match, needs tuning to 5 Mbps |
| **E (Migration failure)** | MigrationManager has fault injection flags | Infrastructure exists, needs scenario wrapper |

---

## 5. Bugs Found

| # | Severity | Description |
|---|----------|-------------|
| B1 | **HIGH** | No Scenario A-E buttons, no Stop/Reset in dashboard |
| B2 | **HIGH** | No live streaming (SSE/WebSocket) from backend |
| B3 | **HIGH** | No controller decision log panel |
| B4 | **HIGH** | KPI values hardcoded, not from live state |
| B5 | **MEDIUM** | No dedicated CUDA/GPU panel |
| B6 | **MEDIUM** | Scenario C spec not matched by any existing scenario |
| B7 | **MEDIUM** | No migration state lifecycle streaming |
| B8 | **LOW** | Theme colors slightly off from spec |
| B9 | **LOW** | 4 tests skipped (unknown reason) |
| B10 | **INFO** | `/api/status` runtime.active_plan is hardcoded string |

---

## 6. Files Proposed for Deletion

| File | Reason | References | Decision |
|------|--------|------------|----------|
| `DEMO_AND_PRESENTATION_GUIDE.md` | Replaced by `teacher_demo.md` | Only self-references `ACADEMIC_PROJECT_REPORT` | **Delete** after teacher_demo.md written |
| `ACADEMIC_PROJECT_REPORT_FIRST_DRAFT.md` | Only referenced from DEMO_AND_PRESENTATION_GUIDE.md | 1 reference | **Needs human decision** — academic report may be valuable |

---

## 7. Conflicts Between implementation.md and Code

| # | Conflict | Resolution |
|---|----------|------------|
| C1 | Spec says delete "acad proj .md" and "demo.md" but actual filenames are `ACADEMIC_PROJECT_REPORT_FIRST_DRAFT.md` and `DEMO_AND_PRESENTATION_GUIDE.md` | Interpret as closest matches. Delete DEMO guide. Academic report needs human decision. |
| C2 | Spec describes 5 scenarios with dashboard buttons. Code has 10 different scenarios. Dashboard has zero scenario buttons. | Build 5 dashboard scenarios (A-E) mapping to spec using existing generators. |
| C3 | Spec says "HOLD/MIGRATE/ROLLBACK" but controller uses `KEEP_CURRENT/SWITCH/SAFE_FALLBACK` | Display-only mapping in UI. No controller logic change. |
| C4 | Spec says "plan ID unchanged" for Scenario A. Need plan_id comparison. | Use `generate_plan_id()` — unchanged plan_id implies unchanged layers. |

---

## 8. Summary of Required Work

### Phase 2: Fix & Complete Dashboard
1. Add SSE streaming from backend orchestration to frontend
2. Add Scenario A-E buttons with info panels (title, what to watch, lesson)
3. Add Stop/Reset buttons
4. Add controller decision log panel (scrollable, color-coded, auto-follow)
5. Make all KPI values live from backend state
6. Add CUDA/GPU dedicated panel
7. Add reconnection handling ("Disconnected — retrying")
8. Add migration state lifecycle display
9. Create new scenario generators for C (temp dip with recovery) and tune D

### Phase 3: Theme
1. Adjust CSS variables to match spec gradient/border/glow
2. Add `prefers-reduced-motion` support
3. Verify contrast ratios

### Phase 4: Cleanup & Docs
1. Delete `DEMO_AND_PRESENTATION_GUIDE.md`
2. Update README.md with scenarios, dashboard, commands
3. Write teacher_demo.md

### Phase 5: Testing
1. Add scenario integration tests (A-E assertions)
2. Add E2E tests (Playwright)
3. Create verify command

### Phase 6: Final verification & report

---

> **This audit was completed before any code changes, per Section 0 Rule 1.**
