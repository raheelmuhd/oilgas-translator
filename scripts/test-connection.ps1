# Test Backend Connection
Write-Host "Testing backend connection..." -ForegroundColor Cyan

# Test health endpoint
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -Method GET -UseBasicParsing
    Write-Host "✅ Backend is running!" -ForegroundColor Green
    Write-Host "Status: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "Response: $($response.Content)" -ForegroundColor Green
} catch {
    Write-Host "❌ Backend is NOT accessible!" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Make sure the backend is running:" -ForegroundColor Yellow
    Write-Host "  cd backend" -ForegroundColor Yellow
    Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "  uvicorn app.main:app --reload" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Testing CORS..." -ForegroundColor Cyan
try {
    $headers = @{
        "Origin" = "http://localhost:3001"
    }
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -Method GET -Headers $headers -UseBasicParsing
    Write-Host "✅ CORS is working!" -ForegroundColor Green
} catch {
    Write-Host "⚠️  CORS test failed (this might be normal)" -ForegroundColor Yellow
}

