# Optimization Fix - Ollama Queue Contention

## Problem Identified

From the test logs:
- **Parallel processing IS working** ✅
- **But performance is 23% SLOWER** ❌ (9.1 min vs 7.4 min)

### Root Cause
**Ollama queues requests instead of processing them in parallel**

When we send multiple chunks/pages simultaneously:
- Ollama receives them all
- But processes them **sequentially in a queue**
- Result: Total time = sum of all chunk times (not max)

---

## Fix Applied

### Changes Made

1. **Reduced Page Parallelism**
   - `max_concurrent_pages: 2 → 1`
   - **Why**: Avoid Ollama queue contention
   - **Impact**: Process one page at a time, but chunks within page can still be parallel

2. **Reduced Chunk Parallelism**
   - `max_concurrent_chunks: 4 → 3`
   - **Why**: Smaller batches reduce queue depth
   - **Impact**: Still get parallel speedup but less contention

3. **Optimized Chunk Size**
   - `narrative_chars_per_chunk: 4500 → 3800`
   - **Why**: Balance between fewer API calls and per-call speed
   - **Impact**: Slightly more chunks but faster per chunk

---

## Expected Performance

### Before Fix
- **9 pages**: 9.1 minutes (548s)
- **Issue**: Ollama queue contention

### After Fix
- **9 pages**: **5-6 minutes** (estimated)
- **Improvement**: 30-40% faster than current, 20-30% faster than original

### Why This Will Work
- **One page at a time**: No page-level queue contention
- **2-3 chunks per page**: Small enough batch for Ollama to handle efficiently
- **3800 char chunks**: Optimal balance (not too large, not too many chunks)

---

## Testing

To test the fix:

1. **Restart backend** (to load new settings):
   ```powershell
   # Stop current backend (Ctrl+C)
   cd backend
   uvicorn app.main:app --reload
   ```

2. **Upload test file** again

3. **Expected results**:
   - Should see: "TRANSLATING PAGES 1-1/9" (one at a time)
   - Should see: "Processing chunks 1-3 in parallel" (small batches)
   - Total time: **5-6 minutes** (vs 9.1 minutes before)

---

## Alternative: If Still Slow

If performance is still not optimal, we can:

1. **Further reduce chunk parallelism** to 2
2. **Reduce chunk size** to 3500 chars
3. **Check Ollama configuration** for true parallel processing support

---

## Summary

**Status**: ✅ **Fix applied - ready to test**

**Changes**:
- `max_concurrent_pages: 2 → 1`
- `max_concurrent_chunks: 4 → 3`
- `narrative_chars_per_chunk: 4500 → 3800`

**Expected**: 5-6 minutes (30-40% improvement)
