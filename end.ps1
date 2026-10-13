# PowerShell script to cleanly stop the Predictive VRAM Dashboard
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host " [STOP] Terminating Predictive VRAM Dashboard & Background Services... " -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Yellow

$PID_FILE = Join-Path $PSScriptRoot ".dashboard.pid"
$stopped = $false

# 1. Stop by PID file
if (Test-Path $PID_FILE) {
    $procId = Get-Content $PID_FILE -ErrorAction SilentlyContinue
    if ($procId) {
        $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if ($p) {
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            Write-Host "[OK] Stopped dashboard process (PID: $procId)" -ForegroundColor Green
            $stopped = $true
        }
    }
    Remove-Item -Path $PID_FILE -Force -ErrorAction SilentlyContinue
}

# 2. Check port 8000
$conn = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($conn) {
    foreach ($c in $conn) {
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
        Write-Host "[OK] Stopped process on port 8000 (PID: $($c.OwningProcess))" -ForegroundColor Green
        $stopped = $true
    }
}

if (-not $stopped) {
    Write-Host "[INFO] No active dashboard service was found running on port 8000." -ForegroundColor Gray
}

Write-Host "[SUCCESS] All dashboard services stopped cleanly." -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Yellow
