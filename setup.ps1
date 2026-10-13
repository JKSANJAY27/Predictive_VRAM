# PowerShell setup script for Predictive VRAM runtime
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " [SETUP] Initializing Predictive VRAM Environment & Dependencies...   " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Check Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python is not installed or not in your system PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.12 (or 3.10+) from https://python.org before running setup." -ForegroundColor Yellow
    exit 1
}

$pyVersion = python --version
Write-Host "[INFO] Detected $pyVersion" -ForegroundColor Gray

# 2. Prefer uv for ultra-fast setup if installed, else fallback to standard venv+pip
$hasUv = Get-Command uv -ErrorAction SilentlyContinue

if ($hasUv) {
    Write-Host "[INFO] Fast package manager 'uv' detected. Synchronizing environment..." -ForegroundColor Green
    uv sync
} else {
    Write-Host "[INFO] Using standard Python venv and pip..." -ForegroundColor Gray
    
    # Create .venv if not exists
    if (-not (Test-Path ".venv")) {
        Write-Host "[INFO] Creating virtual environment (.venv)..." -ForegroundColor Gray
        python -m venv .venv
    }

    $pipExe = Join-Path ".venv" "Scripts\pip.exe"
    $pythonExe = Join-Path ".venv" "Scripts\python.exe"

    if (-not (Test-Path $pipExe)) {
        Write-Host "[ERROR] Virtual environment pip executable not found at $pipExe" -ForegroundColor Red
        exit 1
    }

    Write-Host "[INFO] Installing dependencies from requirements.txt..." -ForegroundColor Gray
    & $pipExe install --upgrade pip
    & $pipExe install -r requirements.txt
}

# 3. Verification Check
Write-Host "[INFO] Verifying core deep learning and telemetry imports..." -ForegroundColor Gray
$verifyScript = "import torch, transformers, psutil, numpy, scipy, sklearn, yaml, pydantic; print('All core libraries verified successfully!')"

if ($hasUv) {
    uv run python -c $verifyScript
} else {
    $pythonExe = Join-Path ".venv" "Scripts\python.exe"
    & $pythonExe -c $verifyScript
}

Write-Host "======================================================================" -ForegroundColor Green
Write-Host " [SUCCESS] Environment setup complete! Ready to run.                 " -ForegroundColor Green
Write-Host "                                                                      " -ForegroundColor Green
Write-Host " To start the interactive Web Dashboard:                              " -ForegroundColor Cyan
Write-Host "   .\start.ps1  (or start.bat)                                        " -ForegroundColor Yellow
Write-Host "                                                                      " -ForegroundColor Green
Write-Host " To stop the dashboard anytime:                                       " -ForegroundColor Cyan
Write-Host "   .\end.ps1    (or end.bat)                                          " -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Green
