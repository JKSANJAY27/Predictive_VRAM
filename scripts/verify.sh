#!/usr/bin/env bash
# Single-Command Verification Gate for Linux / WSL

set -e

echo "======================================================================"
echo " [VERIFY GATE] FULL PROJECT TEST SUITE & DOCUMENTATION AUDIT"
echo "======================================================================"

echo "[1/4] Running Documentation Integrity Audit..."
if git grep -nI "DEMO_AND_PRESENTATION_GUIDE.md" -- ":!AUDIT_REPORT.md" ":!FINAL_REPORT.md" ":!CHANGELOG_AUDIT.md" ":!scripts/verify.*" 2>/dev/null; then
    echo "FAIL: Dangling reference found."
    exit 1
fi
echo "PASS: Zero dangling references."

echo "[2/4] Running Deterministic Scenario Tests (50 runs)..."
uv run pytest tests/test_scenarios_deterministic.py -v

echo "[3/4] Running Dashboard REST API Tests..."
uv run pytest tests/test_dashboard_api.py -v

echo "[4/4] Running Playwright E2E Browser Tests..."
uv run pytest tests/test_e2e_dashboard.py -v

echo "======================================================================"
echo " [VERIFY PASSED] ALL GATES CLEARED -- SYSTEM IS 100% DEMO-READY"
echo "======================================================================"
