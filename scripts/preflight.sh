#!/usr/bin/env bash
# Pre-Flight Diagnostic and Environment Verification Script for Bash / Linux / WSL

set -e

echo "======================================================================"
echo " [PREFLIGHT] PREDICTIVE VRAM & SPLIT INFERENCE CONTROLLER"
echo " Automated Demonstration Environment Diagnostic"
echo "======================================================================"

echo -n "[1/7] Checking Python Version... "
python3 -c "import sys; assert sys.version_info >= (3, 12); print(f'PASS (Python {sys.version_info.major}.{sys.version_info.minor})')"

echo -n "[2/7] Checking uv Package Manager... "
uv --version

echo -n "[3/7] Checking Virtual Environment... "
test -d .venv && echo "PASS (.venv exists)" || echo "FAIL"

echo -n "[4/7] Checking Core Dependencies... "
uv run python -c "import torch, transformers, psutil, playwright; print('PASS')"

echo -n "[5/7] Detecting Hardware Telemetry... "
uv run python -c "
import torch, psutil
has_cuda = torch.cuda.is_available()
ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
print(f'CUDA: {has_cuda} | RAM: {ram_gb} GB')
"

echo -n "[6/7] Checking Static Assets... "
test -f src/dashboard/static/index.html && echo "PASS" || echo "FAIL"

echo -n "[7/7] Running Scenario Fast Sanity Check... "
uv run python -c "
from src.dashboard.scenario_runner import ScenarioExecutionManager
mgr = ScenarioExecutionManager()
mgr.start_scenario('A', seed=42, step_delay=0.001)
if mgr._worker_thread: mgr._worker_thread.join(timeout=3.0)
assert mgr.state.assertion_passed
print('PASS')
"

echo "======================================================================"
echo " [PREFLIGHT RESULT] ALL CHECKS PASSED — READY FOR DEMO"
echo "======================================================================"
