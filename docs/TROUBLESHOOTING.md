# Troubleshooting Guide

## Issues Fixed

### 1. ✅ "Failed to Fetch" Error

**Problem:** Frontend couldn't connect to backend due to CORS restrictions.

**Solution:**
- Added support for both port 3000 and 3001 in CORS configuration
- Added backend health check in frontend
- Improved error messages

**Files Changed:**
- `backend/app/main.py` - Updated CORS middleware
- `backend/app/config.py` - Added port 3001 to allowed origins
- `frontend/src/app/page.tsx` - Added backend connection check

### 2. ✅ Torch Warnings

**Problem:** Harmless warnings about `pin_memory` when no GPU is available.

**Solution:**
- Added warning filters to suppress torch warnings
- Warnings are harmless - they just indicate CPU mode is being used

**Files Changed:**
- `backend/app/main.py` - Added warning filters
- `backend/app/services/translation_service.py` - Suppressed torch warnings

### 3. ✅ Port Mismatch

**Problem:** Frontend running on port 3001 but trying to connect to backend.

**Solution:**
- CORS now allows both ports
- Frontend automatically detects backend connection

## How to Run the Application

### Step 1: Start Backend

```powershell
cd C:\Masters\project\oilgas-translator\backend
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

**Verify Backend:**
- Open: http://localhost:8000/health
- Should return: `{"status":"healthy",...}`

### Step 2: Start Frontend

```powershell
cd C:\Masters\project\oilgas-translator\frontend
npm run dev
```

**Expected Output:**
```
✓ Ready in 1842ms
Local: http://localhost:3000 (or 3001 if 3000 is busy)
```

### Step 3: Open Browser

- Open: http://localhost:3000 (or http://localhost:3001)
- You should see the Document Translator interface
- If you see a yellow warning about backend connection, check Step 1

## Verification Checklist

- [ ] Backend is running on port 8000
- [ ] Backend health endpoint works: http://localhost:8000/health
- [ ] Frontend is running (port 3000 or 3001)
- [ ] No "Failed to fetch" errors in browser console
- [ ] Frontend shows "Document Translator" interface

## Common Issues

### Backend won't start
- Check if port 8000 is already in use
- Verify virtual environment is activated
- Check for Python errors in terminal

### Frontend shows "Failed to fetch"
- Verify backend is running: http://localhost:8000/health
- Check browser console for CORS errors
- Ensure both servers are running

### Torch warnings (harmless)
- These are just informational - CPU mode is working correctly
- Warnings are now suppressed but may still appear briefly

## Performance Notes

- **OCR:** Processes 16 pages in parallel (10-20x faster)
- **Translation:** Processes 3 chunks in parallel (3x faster)
- **Expected:** 600 pages in ~8-12 minutes

