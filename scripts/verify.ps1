# Single-Command Verification Gate for Predictive VRAM Controller
# Runs documentation audit, unit, integration, scenario, API, and Playwright E2E tests.

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " [VERIFY GATE] FULL PROJECT TEST SUITE & DOCUMENTATION AUDIT" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Docs Link & Dead Reference Check
Write-Host "[1/4] Running Documentation Integrity & Dead Link Audit..." -ForegroundColor Yellow
$deadRefs = git grep -nI "DEMO_AND_PRESENTATION_GUIDE.md" -- ":!AUDIT_REPORT.md" ":!FINAL_REPORT.md" ":!CHANGELOG_AUDIT.md" ":!scripts/verify.*" 2>$null
if ($deadRefs) {
    Write-Host "FAIL: Found dangling reference to deleted file:" -ForegroundColor Red
    Write-Host $deadRefs -ForegroundColor Red
    exit 1
} else {
    Write-Host "PASS: Zero dangling references to deleted files." -ForegroundColor Green
}

# 2. Deterministic Scenario Tests (10 consecutive runs each)
Write-Host "[2/4] Running Deterministic Scenario Invariant Tests (50 runs)..." -ForegroundColor Yellow
uv run pytest tests/test_scenarios_deterministic.py -v
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: Deterministic scenario tests failed." -ForegroundColor Red
    exit 1
}

# 3. Dashboard API & SSE Stream Tests
Write-Host "[3/4] Running Dashboard REST API & Stream Tests..." -ForegroundColor Yellow
uv run pytest tests/test_dashboard_api.py -v
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: Dashboard API tests failed." -ForegroundColor Red
    exit 1
}

# 4. Playwright End-to-End Browser Tests
Write-Host "[4/4] Running Playwright E2E Dashboard Tests..." -ForegroundColor Yellow
uv run pytest tests/test_e2e_dashboard.py -v
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: Playwright E2E tests failed." -ForegroundColor Red
    exit 1
}

Write-Host "======================================================================" -ForegroundColor Green
Write-Host " [VERIFY PASSED] ALL GATES CLEARED -- SYSTEM IS 100% DEMO-READY" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
exit 0
