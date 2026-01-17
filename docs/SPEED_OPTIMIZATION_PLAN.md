# Translation Speed Optimization Plan

## Current Performance
- **9 pages**: 7.4 minutes (445.7 seconds)
- **Average**: 49.5 seconds per page
- **Bottleneck**: Sequential processing (one page at a time)

## Optimization Strategies

### 🚀 Priority 1: Parallel Page Processing (BIGGEST IMPACT)

**Current**: Pages translated one-by-one sequentially
**Proposed**: Translate 2-4 pages in parallel

**Expected Speedup**: **2-4x faster** (7.4 min → 2-4 minutes)

**Implementation**:
- Use `asyncio.gather()` to process multiple pages simultaneously
- Limit concurrency to 2-4 pages (based on GPU memory)
- Maintain page order for output

**Code Change**: Modify `process_job()` to use parallel page translation

---

### ⚡ Priority 2: Parallel Chunk Processing

**Current**: Chunks within a page are sequential
**Proposed**: Translate multiple chunks in parallel (up to 4 concurrent)

**Expected Speedup**: **2-3x faster** for multi-chunk pages

**Implementation**:
- Use `asyncio.gather()` for chunks within a page
- Leverage `ollama_max_concurrent: 4` setting
- Process up to 4 chunks simultaneously

**Code Change**: Modify `_translate_page_sequential()` to parallel chunk processing

---

### 📦 Priority 3: Increase Chunk Sizes

**Current**: 3000 chars per chunk
**Proposed**: 4000-5000 chars per chunk (fewer API calls)

**Expected Speedup**: **10-20% faster** (fewer round trips)

**Trade-off**: Slightly longer per chunk, but fewer total chunks

**Code Change**: Update `narrative_chars_per_chunk` in `TranslationConfig`

---

### ⚙️ Priority 4: Optimize Ollama Settings

**Current Settings**:
```python
"temperature": 0.2,
"num_predict": 2000,
"num_ctx": 4096,
```

**Optimized Settings**:
```python
"temperature": 0.1,      # Lower = faster, still accurate
"num_predict": 1500,      # Reduce max tokens (faster)
"num_ctx": 4096,         # Keep same
"num_thread": 8,         # Use more CPU threads if available
```

**Expected Speedup**: **10-15% faster**

---

### 🔄 Priority 5: Batch Processing for Small Chunks

**Current**: Each chunk = 1 API call
**Proposed**: Combine small chunks (<500 chars) into batches

**Expected Speedup**: **5-10% faster** for documents with many small chunks

---

### 🎯 Priority 6: Smart Retry Logic

**Current**: Always retries failed chunks
**Proposed**: Skip retry for clearly untranslatable content (abbreviation lists)

**Expected Speedup**: **5-10% faster** (avoid wasted time on failures)

---

## Implementation Plan

### Phase 1: Quick Wins (30 minutes)
1. ✅ Increase chunk size to 4000 chars
2. ✅ Optimize Ollama settings (temperature, num_predict)
3. ✅ Add batch processing for small chunks

**Expected Result**: 7.4 min → **6-6.5 minutes** (15-20% faster)

### Phase 2: Parallel Processing (1-2 hours)
1. ✅ Implement parallel chunk processing (2-4 concurrent)
2. ✅ Implement parallel page processing (2-3 concurrent)
3. ✅ Add concurrency limits based on GPU memory

**Expected Result**: 6.5 min → **2-3 minutes** (50-60% faster)

### Phase 3: Advanced Optimizations (Optional)
1. ✅ Smart chunk batching
2. ✅ Predictive retry logic
3. ✅ GPU memory optimization

**Expected Result**: 2-3 min → **1.5-2 minutes** (additional 20-30%)

---

## Expected Final Performance

### Current
- **9 pages**: 7.4 minutes
- **100 pages**: ~1.5 hours
- **630 pages**: ~10 hours

### After All Optimizations
- **9 pages**: **1.5-2 minutes** (4-5x faster) 🚀
- **100 pages**: **20-30 minutes** (3-4x faster)
- **630 pages**: **2-3 hours** (3-4x faster)

---

## Risk Assessment

### Low Risk (Safe to implement)
- ✅ Increase chunk sizes
- ✅ Optimize Ollama settings
- ✅ Parallel chunk processing (within page)

### Medium Risk (Test carefully)
- ⚠️ Parallel page processing (memory usage)
- ⚠️ Batch chunk processing

### High Risk (Requires testing)
- ⚠️ Aggressive parallelization (GPU memory limits)

---

## Recommended Implementation Order

1. **Start with Phase 1** (Quick wins - 30 min)
   - Immediate 15-20% speedup
   - Low risk, easy to test

2. **Then Phase 2** (Parallel processing - 1-2 hours)
   - Major speedup (4-5x)
   - Requires careful testing
   - Monitor GPU memory usage

3. **Phase 3** (Advanced - optional)
   - Additional 20-30% if needed
   - More complex, lower priority

---

## Monitoring

After implementing optimizations, monitor:
- GPU memory usage
- Translation accuracy (should stay 85%+)
- Error rates
- Ollama queue status

---

## Quick Implementation

Would you like me to:
1. **Implement Phase 1** (quick wins - 15-20% faster)?
2. **Implement Phase 2** (parallel processing - 4-5x faster)?
3. **Both** (full optimization)?

Let me know and I'll start implementing!
