# implementation.md — Adaptive Split-Inference Controller: Dashboard Audit, Completion & Demo Readiness

> Place this file in the **repo root**. Antigravity must follow it top to bottom.
> Read **Section 0 (Strict Rules)** before touching any file. Those rules override any shortcut you are tempted to take.

---

## 0. STRICT RULES (non-negotiable)

1. **Audit before you change.** Complete Phase 1 and write `AUDIT_REPORT.md` before editing any source file.
2. **No fake data.** Every number, chart point, badge and status on the dashboard must come from the real backend state (controller, migration manager, KV-cache monitor, network monitor, CUDA/GPU monitor). No hardcoded values, no `Math.random()` in place of real metrics. If a value is simulated by design (e.g. the injected bandwidth in Scenarios C/D), the UI must label it **"Simulated"**.
3. **No silent fallbacks.** If CUDA is unavailable, the dashboard must say so ("CUDA unavailable — running on CPU"). It must never display GPU numbers that were not measured.
4. **Do not invent features.** Implement only what is described in this file. If something here conflicts with what the code does, stop, record it in `AUDIT_REPORT.md` under "Conflicts", choose the option that keeps the controller's real behavior intact, and flag it. Do not change controller logic just to make a demo look nicer.
5. **Do not break the controller.** Controller / migration / prediction logic is only modified if the audit proves a bug. Every such change needs a regression test and an entry in `CHANGELOG_AUDIT.md`.
6. **Deletions are safe deletions.** Before deleting any file: `grep -rn "<filename>"` across the repo (code, scripts, docs, CI). Fix or remove every reference first. Work on a git branch (`chore/dashboard-finalize`), delete in a dedicated commit.
7. **Tests are not optional.** A phase is not done until its tests exist, run, and pass. Never mark a task complete on the basis of "it should work". Paste the real command and its real output into the report.
8. **Never weaken a test to make it pass.** If a test fails, fix the code or, if the test is wrong, explain why in the report. Do not delete, skip (`xfail`, `.skip`), or loosen assertions without written justification.
9. **Zero console errors / warnings** in the browser during every scenario run, and zero unhandled exceptions in backend logs.
10. **Every button, toggle, slider, tab and link must be exercised** by an automated end-to-end test. A control with no test is considered broken.
11. **Report honestly.** The final report lists what passed, what failed, what was not verified, and why. "Not verified" is acceptable; claiming verification that did not happen is not.
12. **Commands in docs must be run.** Every start / stop / test command written into README or `teacher_demo.md` must be executed verbatim once from a clean shell and confirmed to work before the doc is finalized.
13. **Keep changes minimal and reviewable.** One logical change per commit, clear commit messages. No unrelated refactors.

---

## 1. PHASE 1 — Discovery & Audit (read-only)

### 1.1 Map the project
- Detect the stack (languages, frameworks, package managers, how the dashboard is served, how the backend streams updates: WebSocket / SSE / polling).
- Produce a module map: controller, predictor (KV-cache time-to-exhaustion), planner / plan scorer, safeguards, migration manager (lock, transfer, rollback), network monitor, memory monitor, CUDA/GPU monitor, scenario runner, dashboard frontend, API layer.
- List every existing start/stop/test command and how modules are wired together.
- List every markdown/doc file and every script. Flag candidates for deletion (including `acad proj .md` and `demo.md`) and anything stale or contradictory.

### 1.2 Audit the dashboard against this checklist
For each item record **PASS / FAIL / MISSING** with evidence (file path, line, screenshot or log):

| # | Check |
|---|-------|
| D1 | Dashboard starts with one documented command and loads with no console errors |
| D2 | Split visualization shows layers partitioned across tiers (device / edge / cloud or whatever tiers the code defines) with correct layer counts per tier |
| D3 | Split visualization updates live after a migration (layers visibly move; counts update) |
| D4 | Live metrics present and correct: bandwidth (Mbps), RTT (ms), KV-cache size, memory usage, predicted time-to-exhaustion, current plan ID, migration count, controller decision log |
| D5 | Controller decision log shows each decision with timestamp, inputs, chosen action (HOLD / MIGRATE / ROLLBACK) and the reason |
| D6 | Migration state is visible: idle → evaluating → locked → transferring → committed / rolled back |
| D7 | Scenario buttons A–E exist, start the correct scenario, and show scenario title, "what to watch for", and the lesson |
| D8 | Scenario can be stopped/reset and re-run cleanly (no leftover state, locks, or timers) |
| D9 | CUDA/GPU panel present with real measured data (or an explicit "unavailable" state) |
| D10 | Units, labels and axes are correct and consistent everywhere |
| D11 | Displayed values equal backend values (spot-check against API/log output) |
| D12 | Layout holds at 1920×1080, 1440×900 and 1280×720 with no overflow or clipping |
| D13 | Theme is grey neon glassmorphism (see Phase 3) |

### 1.3 Audit scenario behavior against the specification (Section 2). Run each scenario for real, record observed behavior, and mark where it deviates.

### 1.4 Deliverable
Write `AUDIT_REPORT.md` containing: module map, command inventory, checklist results, scenario results, a list of bugs found (with severity), a list of files proposed for deletion with reference-grep output, and a list of conflicts between this spec and the code. **Stop and proceed only after this file exists.**

---

## 2. SCENARIO SPECIFICATION (source of truth for expected behavior)

The five scenarios illustrate the controller's intended behavior. Each must be runnable from a dashboard button and from a CLI/test harness with deterministic seeding where possible.

### Scenario A — Everything is stable
- **Setup:** bandwidth 100 Mbps, RTT 5 ms, memory growth poses no immediate threat.
- **Expected:** controller keeps the current plan. **Zero migrations.** Decision log shows HOLD with reason "stable".
- **Lesson shown in UI:** *A good adaptive system should know when not to act.*
- **Test assertions:** migration_count == 0 at end; plan ID unchanged; all logged decisions are HOLD; layer distribution unchanged.

### Scenario B — KV-cache growth threatens memory
- **Setup:** tokens are generated continuously; KV-cache grows; predicted time-to-exhaustion decreases.
- **Expected:** controller evaluates plans with lower local memory pressure. If a better plan passes the safeguards, some layers and associated state are migrated to another tier. If no plan passes safeguards, it must HOLD and log why.
- **Lesson shown in UI:** *Predict a memory problem early enough to avoid it.*
- **Test assertions:** predicted time-to-exhaustion is monotonically non-increasing while no migration occurs; migration (if triggered) happens **before** actual exhaustion; local memory pressure after migration < before; no OOM/exhaustion event occurs; layers moved are reflected in the split visualization.

### Scenario C — Temporary bandwidth collapse
- **Setup:** bandwidth drops 100 → 2 Mbps for **8 seconds**, then recovers to 100 Mbps.
- **Expected:** the predictive controller recognizes switching costs outweigh the temporary benefit and **stays on its existing plan**. Decision log shows HOLD with a reason referencing transient degradation / switching cost vs. expected recovery.
- **Lesson shown in UI:** *Do not migrate just because conditions briefly worsen.*
- **Test assertions:** migration_count == 0; plan ID unchanged before/during/after; bandwidth trace shows the 8-second dip and recovery; decisions during the dip are HOLD.
- **Optional (only if a reactive baseline already exists in the code):** show the baseline switching twice alongside. Do not build a new baseline unless the audit finds one exists.

### Scenario D — Bandwidth stays poor
- **Setup:** bandwidth drops to 5 Mbps and **remains** low.
- **Expected:** waiting for recovery is no longer useful; after the controller's persistence criterion is met it selects a **more local-heavy partition** to reduce communication dependence.
- **Lesson shown in UI:** *Ignore short-lived fluctuations, but adapt to persistent degradation.*
- **Test assertions:** at least one migration occurs after bandwidth has stayed degraded; layers on the local tier after > before; the migration is not triggered immediately at the drop (there is a visible hold period, consistent with Scenario C logic); final plan ID differs from initial.

### Scenario E — Migration fails midway
- **Setup:** a fault is injected during state transfer.
- **Expected:** the migration manager detects the failure, restores the previous state, releases the lock, and resumes generation with the original plan.
- **Lesson shown in UI:** *A safe system needs failure recovery, not just a successful migration path.*
- **Test assertions:** failure detected and logged; state restored (checksum/shape of restored KV state and layer placement equals pre-migration snapshot); lock released (a subsequent migration attempt can acquire it); token generation resumes and continues to produce tokens; plan ID equals the original; migration state shows ROLLED BACK; no layer is left duplicated or orphaned.

---

## 3. PHASE 2 — Fix & Complete the Dashboard

Work through every FAIL/MISSING item from the audit. Required end state:

### 3.1 Data integrity
- One authoritative state object per run, served by the backend; the frontend only renders it.
- Reconnect handling: if the stream drops, the UI shows "Disconnected — retrying" and recovers without a page reload.
- Timestamps use a single clock source; charts show the last N seconds with correct axes and units.

### 3.2 Split visualization (the centerpiece)
- Show the model's layers as blocks grouped by tier, with tier names and per-tier counts.
- Animate layer movement on migration; show migration phase (evaluating / locked / transferring / committed / rolled back).
- On rollback (Scenario E) the visualization must visibly return to the original layout.
- Hover/tooltip on a layer: layer index, tier, approximate memory footprint if available from the backend.

### 3.3 Controls
- Buttons: Scenario A, B, C, D, E, Stop, Reset. Plus any existing controls found in the audit (keep them if functional; remove them if dead and document the removal).
- Each scenario button shows (in a side/info panel) the scenario title, what the viewer should watch, and the lesson sentence from Section 2.
- A button is disabled with a clear state while a scenario is running where concurrent runs are unsafe. Reset always works.
- A running indicator and elapsed time are visible.

### 3.4 CUDA / GPU panel
- Display (measured, not hardcoded): GPU name, CUDA version, driver version, GPU memory used/total, GPU utilization, and which tier/layers are executing on the GPU.
- Use the project's existing stack for GPU introspection (e.g. `torch.cuda`, `pynvml`/`nvidia-smi`); detect, do not assume.
- If no CUDA device: show a clearly styled "CUDA unavailable — CPU fallback" state. Tests must cover **both** states (mock the detector for the unavailable path).
- If the controller's memory-pressure logic can use GPU memory, the panel must reflect the same numbers the controller uses.

### 3.5 Controller transparency
- Live decision log (scrollable, auto-follow toggle, color-coded HOLD / MIGRATE / ROLLBACK).
- Summary cards: current plan, migrations (success / rolled back), time-to-exhaustion, bandwidth, RTT, memory pressure.

---

## 4. PHASE 3 — Visual Theme: Grey Neon Glassmorphism

Implement as CSS variables so the theme is centralized.

- **Background:** deep grey gradient (≈ `#0d0f12` → `#1a1d23`), subtle noise or grid optional.
- **Glass panels:** `background: rgba(255,255,255,0.05–0.08)`, `backdrop-filter: blur(14–20px)`, 1px border `rgba(255,255,255,0.12)`, rounded corners 16–20px, soft shadow.
- **Neon accents on grey:** cyan/teal for normal/stable, amber for warning/hold, magenta/red for failure/rollback, green for success. Use glow (`box-shadow`/`text-shadow`) sparingly on active elements only.
- **Typography:** clean sans-serif for UI, monospace for logs/metrics. Minimum contrast ratio 4.5:1 for body text.
- **Fallback:** if `backdrop-filter` is unsupported, panels still render legibly (solid semi-opaque background).
- **Motion:** smooth but short transitions; respect `prefers-reduced-motion`.
- No theme-related layout shifts; no text clipped at any tested viewport.

---

## 5. PHASE 4 — Repository Cleanup & Docs

### 5.1 Delete redundant files
- Delete `acad proj .md` and `demo.md` (the exact filenames as found; note the space) **after** Section 0 rule 6 is satisfied.
- Delete any other file the audit proves redundant (stale duplicates, dead scripts, old logs, unused assets). Each deletion must be justified in `AUDIT_REPORT.md`. When in doubt, **do not delete**; list it under "Needs human decision".
- Never delete: source code in use, tests, `implementation.md`, `README.md`, `teacher_demo.md`, config required to run.

### 5.2 Update remaining docs to the current state
- `README.md`: purpose, architecture/module diagram (text or mermaid), prerequisites (Python/CUDA/etc. as detected), install, **start command(s)**, **stop command(s)**, how to run each scenario, how to run tests, troubleshooting (including no-GPU case). All commands verified per Rule 12.
- Make sure module integration descriptions match the actual code: controller ↔ predictor ↔ planner ↔ safeguards ↔ migration manager ↔ monitors ↔ API ↔ dashboard.
- Remove references to deleted files everywhere.

### 5.3 Create `teacher_demo.md`
A presenter's script so the whole project can be demonstrated from scratch. Required sections:

1. **One-paragraph project pitch** (what problem, what the controller does, why it matters).
2. **Pre-demo checklist** (environment, GPU check, start command, URL, expected first screen).
3. **Dashboard tour:** every panel explained — split visualization, metric cards, charts, decision log, migration state, CUDA panel.
4. **Button-by-button guide.** For each of Scenario A, B, C, D, E, Stop, Reset (and any other control): *What it does → what to point at on screen → what the audience should conclude → the lesson sentence → approximate duration.*
5. **Suggested demo order and timing** (e.g. 10-minute and 20-minute versions).
6. **Talking points / likely teacher questions with answers** (why predictive vs reactive, how rollback is guaranteed, what the safeguards are, why local-heavy under persistent poor bandwidth, role of CUDA).
7. **Troubleshooting during the demo** (what to do if a scenario looks stuck, how to reset, how to run without GPU).
8. **Stop / cleanup commands.**

Every claim in this file must match observed dashboard behavior. After writing it, **walk through it literally** using the running dashboard and fix any mismatch.

---

## 6. PHASE 5 — Testing (mandatory, evidence required)

### 6.1 Backend / unit tests
- Predictor: time-to-exhaustion decreases with cache growth; edge cases (zero growth, huge growth).
- Planner/safeguards: a plan failing safeguards is rejected; a plan passing is accepted.
- Switching-cost vs expected-recovery logic (Scenario C vs D boundary).
- Migration manager: success path, injected-fault rollback, lock acquire/release, re-entrancy.
- CUDA detector: available and unavailable paths (mocked).

### 6.2 Scenario integration tests (A–E)
Implement every assertion listed in Section 2 as automated tests against the real controller (time may be accelerated by a clock-injection mechanism if one exists; if not, run in real time and keep the 8-second window faithful for Scenario C).

### 6.3 API / stream tests
- Endpoints return the documented schema.
- Stream emits ordered updates; reconnect works.
- Dashboard-visible values equal API values (D11).

### 6.4 End-to-end UI tests (Playwright or equivalent already in the repo; otherwise add Playwright)
For **each** control and scenario:
- Page loads with no console errors/warnings and no failed network requests.
- Click Scenario A → verify metrics update, decision log fills with HOLD, migrations stays 0, info panel shows correct title and lesson.
- Click Scenario B → verify time-to-exhaustion decreases, split visualization changes if a migration occurs, counts per tier sum to total layers before and after.
- Click Scenario C → verify bandwidth dips to ~2 Mbps for ~8 s then recovers; plan ID unchanged; no migration.
- Click Scenario D → verify hold period then migration; local layer count increases.
- Click Scenario E → verify failure indicator, ROLLED BACK state, lock released, generation resumes, layout restored.
- Stop and Reset mid-run for every scenario; then re-run the scenario and confirm a clean state (no stale data, no leftover lock).
- Run scenarios back-to-back (A→B→C→D→E→A) to catch state leakage.
- CUDA panel: shows measured values when CUDA is present; shows the unavailable state when mocked absent.
- Viewports: 1920×1080, 1440×900, 1280×720 — no overflow/clipping (assert bounding boxes inside viewport) and capture screenshots.
- Theme: assert presence of `backdrop-filter` on glass panels and computed contrast on key text.
- Accessibility smoke: buttons reachable by keyboard, visible focus state, accessible names.

### 6.5 Docs tests
- Script that extracts every fenced shell command from `README.md` and `teacher_demo.md` that is marked runnable and executes (or dry-runs) them; failures block completion.
- Link check: no references to deleted files remain (`grep` returns nothing).

### 6.6 Regression gate
Provide a single command (e.g. `make verify` or `./scripts/verify.sh`) that runs lint, unit, integration, API and E2E tests, and the docs checks, and exits non-zero on any failure. Document it in the README.

---

## 7. PHASE 6 — Final Verification & Report

Create `FINAL_REPORT.md` with:

1. Summary of what changed (files added / modified / deleted, with reasons).
2. The Section 1.2 checklist re-run with final PASS/FAIL and evidence.
3. Scenario A–E observed results vs. expected (pass/fail per assertion).
4. Full output (or log file path) of the verify command, with date and environment (OS, Python/Node versions, GPU model or "no GPU").
5. Screenshots for each scenario, the CUDA panel (present and unavailable states), and the three viewports, saved under `docs/screenshots/`.
6. Known limitations and anything **not verified**, with reasons.
7. The exact start and stop commands, confirmed from a clean shell.

### Definition of Done (all must be true)
- [ ] `AUDIT_REPORT.md` exists and was written before code changes
- [ ] All D1–D13 checks PASS (or are explicitly justified in the report)
- [ ] Scenarios A–E satisfy every Section 2 assertion via automated tests
- [ ] Every control has an E2E test; zero console errors across all runs
- [ ] CUDA panel verified in both available and unavailable states
- [ ] Grey neon glassmorphism theme applied and checked at 3 viewports
- [ ] `acad proj .md` and `demo.md` deleted with no dangling references
- [ ] `README.md` and `teacher_demo.md` updated, and every command in them executed successfully
- [ ] `verify` command passes from a clean checkout/shell
- [ ] `FINAL_REPORT.md` written honestly, including unverified items

If any item cannot be completed, stop and state precisely why in `FINAL_REPORT.md` instead of marking it done.
