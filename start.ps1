# PowerShell script to start the Predictive VRAM Interactive Dashboard
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " [START] Launching Predictive VRAM & Split Inference Dashboard...    " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$PORT = 8000
$PID_FILE = Join-Path $PSScriptRoot ".dashboard.pid"

# Check if port 8000 is already in use
$existing = Get-NetTCPConnection -LocalPort $PORT -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "[WARNING] Port $PORT is already in use by process ID $($existing.OwningProcess)." -ForegroundColor Yellow
    Write-Host "[INFO] Stopping existing process to ensure clean startup..." -ForegroundColor Yellow
    Stop-Process -Id $existing.OwningProcess -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}

# Launch dashboard using uv run or python
$pythonCmd = "python"
if (Get-Command uv -ErrorAction SilentlyContinue) {
    $process = Start-Process -FilePath "uv" -ArgumentList "run", "python", "scripts/run_dashboard.py", "--port", "$PORT" -PassThru -NoNewWindow
} else {
    $process = Start-Process -FilePath "python" -ArgumentList "scripts/run_dashboard.py", "--port", "$PORT" -PassThru -NoNewWindow
}

# Save PID
$process.Id | Out-File -FilePath $PID_FILE -Encoding ascii

Write-Host "[SUCCESS] Dashboard running in background (PID: $($process.Id))" -ForegroundColor Green
Write-Host "[URL] http://localhost:$PORT" -ForegroundColor Cyan
Write-Host "To stop the dashboard anytime, run: .\end.ps1 (or .\end.bat)" -ForegroundColor Gray
Write-Host "======================================================================" -ForegroundColor Cyan
