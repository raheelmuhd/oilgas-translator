# Oil & Gas Document Translator - Windows Start Script
# Run: powershell -ExecutionPolicy Bypass -File scripts/start.ps1

Write-Host @"

 🛢️  Starting Oil & Gas Document Translator...

"@ -ForegroundColor Cyan

$ProjectDir = Split-Path -Parent $PSScriptRoot

# Start backend
Write-Host "Starting backend server..." -ForegroundColor Yellow
Start-Process -FilePath "powershell" -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$ProjectDir\backend'; .\venv\Scripts\Activate.ps1; uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
) -PassThru

Start-Sleep -Seconds 3

# Start frontend
Write-Host "Starting frontend server..." -ForegroundColor Yellow
Start-Process -FilePath "powershell" -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$ProjectDir\frontend'; npm run dev"
) -PassThru

Start-Sleep -Seconds 5

Write-Host ""
Write-Host "Application started!" -ForegroundColor Green
Write-Host ""
Write-Host "  Backend:  http://localhost:8000" -ForegroundColor Cyan
Write-Host "  Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host "  API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""

# Open browser
Start-Process "http://localhost:3000"

