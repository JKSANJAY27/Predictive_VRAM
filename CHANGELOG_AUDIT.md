# CHANGELOG_AUDIT.md — Predictive VRAM Controller

All modifications and enhancements made to resolve audit findings, achieve Section 2 scenario compliance, and establish 100% demo-readiness.

---

## [Phase 2 & 3] — Dashboard Completion & Grey Neon Glassmorphism Theme

### Added
- **`src/dashboard/scenario_runner.py`**:
  - `ScenarioExecutionManager`: Thread-safe, asynchronous execution engine for Scenarios A through E.
  - Deterministic execution with seeded random state (`seed=42`).
  - Native state transitions: `idle` → `evaluating` → `locked` → `transferring` → `committed` / `rolled_back`.
  - Provenance tracking per Section 0 Rule 2: Injected metrics labeled `Simulated`, physical telemetry labeled `Measured`.
  - Exact Section 2 invariant assertion checks for Scenarios A–E.
- **`src/dashboard/static/index.html`**:
  - Phase 3 Grey Neon Glassmorphism theme with centralized CSS custom properties (`--bg-gradient-start: #0d0f12`, `--bg-gradient-end: #1a1d23`, `--glass-bg: rgba(255,255,255,0.05)`, `backdrop-filter: blur(16px)`).
  - Responsive layout audited for 1920×1080, 1440×900, and 1280×720 viewports.
  - Split visualization: dynamic layer chips across 3 tiers (User Device, Edge Node A, Edge Node B), layer count badges, and hover tooltips showing parameter footprint and precision.
  - Migration Phase Track with real-time neon glow indicators.
  - Dedicated CUDA / GPU Telemetry panel displaying honest CPU fallback state (`CUDA unavailable — running on CPU`) or emulated NVIDIA RTX 4090 telemetry.
  - Live Controller Decision Transparency Log with auto-following and color-coded badges (`HOLD`, `MIGRATE`, `ROLLBACK`).
  - Fallback polling and SSE reconnection handling (`LIVE` vs `Disconnected — retrying`).
  - `@media (prefers-reduced-motion: reduce)` accessibility support.
- **`src/dashboard/server.py`**:
  - SSE real-time streaming endpoint `/api/stream`.
  - Single authoritative state endpoint `/api/state` and `/api/status`.
  - Lifecycle control endpoints `/api/scenario/start`, `/api/scenario/stop`, `/api/scenario/reset`.
  - Mock CUDA toggle endpoint `/api/toggle-mock-cuda`.
  - Graceful socket error handling (`ConnectionAbortedError`, `BrokenPipeError`).

---

## [Phase 4] — Repository Cleanup & Documentation

### Added
- **`teacher_demo.md`**:
  - Comprehensive presenter's script for teacher and professor presentations.
  - 30-second project pitch.
  - Pre-demo checklist.
  - Dashboard tour explaining every panel.
  - Button-by-button scenario guide for Scenarios A–E, Stop, Reset, and GPU toggle.
  - 10-minute and 20-minute presentation timing schedules.
  - Oral defense Q&A covering predictive vs reactive control, anti-thrashing safeguards, atomic rollback guarantees, and no-GPU hardware behavior.
- **`scripts/preflight.ps1` & `scripts/preflight.sh`**:
  - One-command pre-flight diagnostic verifying Python 3.12+, `uv`, `.venv`, dependencies, hardware, port 8000, and scenario sanity.
- **`scripts/verify.ps1` & `scripts/verify.sh`**:
  - Regression verification gate running docs link audit, deterministic scenarios (50 runs), API tests, and Playwright E2E browser tests.

### Removed
- **`DEMO_AND_PRESENTATION_GUIDE.md`**: Safely deleted after reference audit, superseded by `teacher_demo.md`.

---

## [Phase 5] — Automated Verification & Regression Tests

### Added
- **`tests/test_scenarios_deterministic.py`**:
  - Runs Scenarios A–E 10 consecutive times each (50 total runs) with seeded repeatability.
  - Verifies all Section 2 invariants (0 migrations on stable/dip, proactive migration on KV growth, local-heavy on persistent degradation, atomic rollback on transfer failure).
- **`tests/test_dashboard_api.py`**:
  - Validates `/api/state` schema, scenario start/stop/reset lifecycle, mock CUDA toggling, and catalog queries.
- **`tests/test_e2e_dashboard.py`**:
  - Playwright automated browser tests covering:
    - Zero console errors and uncaught warnings.
    - Scenarios A–E end-to-end execution.
    - Stop and Reset mid-run.
    - Back-to-back runs (A → B → C → D → E → A).
    - CUDA panel dual states (CPU fallback and Mock GPU).
    - Viewport overflow checks (1920×1080, 1440×900, 1280×720) with screenshot capture to `docs/screenshots/`.
    - Accessibility smoke tests.
