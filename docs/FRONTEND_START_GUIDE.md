# Frontend Start Guide

## Quick Start

### To see the frontend working, you need to start it:

1. **Open a new terminal/PowerShell window**

2. **Navigate to frontend directory:**
   ```powershell
   cd frontend
   ```

3. **Start the development server:**
   ```powershell
   npm run dev
   ```

4. **Wait for it to start** (you'll see):
   ```
   ✓ Ready in XXXXms
   ○ Local:        http://localhost:3000
   ```

5. **Open your browser** and go to:
   ```
   http://localhost:3000
   ```

## What You'll See

Once the frontend is running, you'll see:

- **Modern UI** with drag-and-drop file upload
- **"Browse Files" button** to select files
- **Language selection** dropdowns
- **Translation provider options** (Ollama, DeepSeek, etc.)
- **Device selection** (Auto, GPU, CPU)
- **Real-time progress** when translating
- **Page counter** showing "Page X of Y (Z remaining)"
- **Download button** when translation completes

## Features to Test

1. **File Upload:**
   - Click "Browse Files" button
   - Select your PDF file
   - File name and size will appear

2. **Settings:**
   - Select source language (or leave auto)
   - Select target language (default: English)
   - Choose translation provider
   - Select device (Auto recommended)

3. **Upload & Monitor:**
   - Click "Translate Document"
   - Watch real-time progress
   - See page-by-page progress
   - Wait for completion

4. **Download:**
   - When status shows "completed"
   - Click "Download Result"
   - Get your translated document

## Troubleshooting

**Frontend won't start?**
- Make sure Node.js is installed: `node --version`
- Install dependencies: `npm install`
- Check if port 3000 is available

**Can't connect to backend?**
- Make sure backend is running on port 8000
- Check backend health: `http://localhost:8000/health`
- Frontend will show connection status

**File upload not working?**
- Check browser console (F12) for errors
- Verify backend is running
- Check file size (max 600MB)

## Quick Start Script (Windows)

You can also use the start script:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/start.ps1
```

This will start both backend and frontend automatically!
