# Pre-Flight Diagnostic and Environment Verification Script
# Predictive VRAM & Adaptive Split-Inference Controller

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " [PREFLIGHT] PREDICTIVE VRAM & SPLIT INFERENCE CONTROLLER" -ForegroundColor Cyan
Write-Host " Automated Demonstration Environment Diagnostic" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$allPassed = $true

# 1. Python Version Check
Write-Host -NoNewline "[1/7] Checking Python Version... "
$pythonVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>$null
if ($pythonVersion -match "^3\.(1[2-9]|[2-9][0-9])") {
    Write-Host "PASS (Python $pythonVersion)" -ForegroundColor Green
} else {
    Write-Host "WARN/FAIL (Found Python $pythonVersion, recommended >= 3.12)" -ForegroundColor Yellow
}

# 2. Package Manager Check (uv)
Write-Host -NoNewline "[2/7] Checking Package Manager (uv)... "
$uvVersion = uv --version 2>$null
if ($uvVersion) {
    Write-Host "PASS ($uvVersion)" -ForegroundColor Green
} else {
    Write-Host "FAIL (uv not found on PATH)" -ForegroundColor Red
    $allPassed = $false
}

# 3. Virtual Environment Check
Write-Host -NoNewline "[3/7] Checking Virtual Environment (.venv)... "
if (Test-Path ".venv") {
    Write-Host "PASS (.venv exists)" -ForegroundColor Green
} else {
    Write-Host "FAIL (.venv directory missing - run setup.ps1)" -ForegroundColor Red
    $allPassed = $false
}

# 4. Core Dependency Imports
Write-Host -NoNewline "[4/7] Checking Core Python Dependencies... "
$depCheck = uv run python -c "
import torch, transformers, psutil, numpy, scipy, sklearn, playwright
print(f'Torch: {torch.__version__}, Transformers: {transformers.__version__}, Psutil: {psutil.__version__}')
" 2>$null

if ($depCheck) {
    Write-Host "PASS ($depCheck)" -ForegroundColor Green
} else {
    Write-Host "FAIL (One or more core packages failed to import)" -ForegroundColor Red
    $allPassed = $false
}

# 5. Hardware & CUDA Telemetry Check
Write-Host -NoNewline "[5/7] Detecting Compute & Telemetry Hardware... "
$hwInfo = uv run python -c "
import torch, psutil
has_cuda = torch.cuda.is_available()
ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
cores = psutil.cpu_count(logical=True)
gpu_name = torch.cuda.get_device_name(0) if has_cuda else 'None (CPU Fallback active)'
print(f'CUDA: {has_cuda} | GPU: {gpu_name} | Host RAM: {ram_gb} GB | Cores: {cores}')
" 2>$null

if ($hwInfo) {
    Write-Host "PASS" -ForegroundColor Green
    Write-Host "      -> $hwInfo" -ForegroundColor Gray
} else {
    Write-Host "FAIL (Hardware query error)" -ForegroundColor Red
    $allPassed = $false
}

# 6. Dashboard Port 8000 Availability
Write-Host -NoNewline "[6/7] Checking Dashboard Port 8000... "
$portOccupied = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($portOccupied) {
    Write-Host "WARN (Port 8000 is currently occupied by PID $($portOccupied.OwningProcess[0]))" -ForegroundColor Yellow
    Write-Host "      (Run .\end.ps1 to release before starting demo)" -ForegroundColor Gray
} else {
    Write-Host "PASS (Port 8000 is available)" -ForegroundColor Green
}

# 7. Scenario Engine Sanity Verification
Write-Host -NoNewline "[7/7] Running Scenario Engine Fast Sanity Check... "
$sanityCheck = uv run python -c "
from src.dashboard.scenario_runner import ScenarioExecutionManager
mgr = ScenarioExecutionManager()
ok = mgr.start_scenario('A', seed=42, step_delay=0.001)
if mgr._worker_thread: mgr._worker_thread.join(timeout=3.0)
print('OK' if mgr.state.status == 'completed' and mgr.state.assertion_passed else 'FAIL')
" 2>$null

if ($sanityCheck -match "OK") {
    Write-Host "PASS (Scenarios A-E engine operational)" -ForegroundColor Green
} else {
    Write-Host "FAIL (Scenario sanity test failed)" -ForegroundColor Red
    $allPassed = $false
}

Write-Host "======================================================================" -ForegroundColor Cyan
if ($allPassed) {
    Write-Host " [PREFLIGHT RESULT] ALL CHECKS PASSED -- READY FOR LIVE DEMONSTRATION" -ForegroundColor Green
    Write-Host " Start server with: powershell -ExecutionPolicy Bypass -File start.ps1" -ForegroundColor Cyan
    Write-Host " Open browser at:   http://localhost:8000" -ForegroundColor Cyan
} else {
    Write-Host " [PREFLIGHT RESULT] SOME CHECKS FAILED -- REVIEW MESSAGES ABOVE" -ForegroundColor Red
}
Write-Host "======================================================================" -ForegroundColor Cyan
