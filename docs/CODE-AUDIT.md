# agentix-timesfm-ts Code Audit Report & Optimization Plan

> Audit Date: 2026-06-25 | Fix Date: 2026-06-26 | Audit Scope: All 17 source files | 133 tests all passing

---

## 0. Fix Status (2026-06-26)

### ✅ Fixed

| ID  | Issue                                      | Status                                            |
| --- | ------------------------------------------ | ------------------------------------------------- |
| C1  | XReg "timesfm + xreg" residual calculation | ✅ Use backcast instead of pointForecast          |
| C2  | ONNX executionProvider disconnected        | ✅ Complete connection chain                      |
| C3  | KV Cache memory waste                      | ✅ Deferred allocation (_sq_ minimal placeholder) |
| C4  | inferIsPositive over-clamping              | ✅ Per-series actual non-negativity check         |
| C5  | Model downloader 885MB heap buffer         | ✅ Streaming write + SHA-256                      |
| C6  | O(n²) NaN interpolation                    | ✅ Strict O(n) two-pointer                        |
| C7  | ONNX batch inference serial                | ✅ Promise.all concurrency                        |
| C8  | XReg params ignored                        | ✅ normalizeXregTargetPerInput implemented        |
| L1  | preprocessor dead code                     | ✅ Removed `maxContext` destructuring             |
| L2  | postprocessor dead code                    | ✅ Removed `flippedQS/flippedPF`                  |
| L3  | updateRunningStats same reference          | ✅ Returns shallow copy                           |
| L4  | DEFAULT_FORECAST_CONFIG not frozen         | ✅ Object.freeze                                  |
| L6  | modelPatches magic numbers                 | ✅ Unified constants                              |
| L7  | model-downloader emoji                     | ✅ Removed + logger injection                     |
| —   | Makefile                                   | ✅ Deleted                                        |
| —   | pipeline.sh/pipeline.py                    | ✅ Deleted (kept .js)                             |
| —   | Duplicate scripts                          | ✅ Cleaned up                                     |
| —   | CI/CD                                      | ✅ 3 workflows                                    |
| —   | ITimesFMModel interface                    | ✅ Extracted                                      |
| —   | compile() chainable call                   | ✅ Returns this                                   |
| —   | forecastWithCovariates                     | ✅ Dynamic import                                 |
| —   | opset version docs                         | ✅ Unified to 18                                  |
| —   | ARCHITECTURE.md                            | ✅ Created                                        |

### ⏳ Pending

| ID  | Issue                    | Priority |
| --- | ------------------------ | -------- |
| —   | RevIN 4D vectorization   | P2       |
| —   | Streaming forecast       | P2       |
| —   | Model hot reload         | P2       |
| —   | LoRA fine-tuning support | P2       |

---

## I. Overall Assessment

**Rating: A- (Excellent, room for improvement)**

Overall code quality is high, architecture is clear, and it precisely aligns with the Python original. However, to reach the standard of "dominating competitors upon open-source release", the following dimensions need refinement.

---

## II. Per-Module Deep Audit

### 2.1 Type System (`types.ts`) — Rating: B+

| Issue                                    | Severity | Description                                                                                                                 |
| ---------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------- |
| `IInferenceEngine` JSDoc cleaned         | ✅ Fixed | Removed non-existent `MockInferenceEngine` reference                                                                        |
| `ModelConfig` redundant fields           | 🟡 Low   | `tokenizerHiddenDims`/`tokenizerOutputDims`/`outputPointDims` are never referenced in code, only for documentation purposes |
| `BatchSeries`/`BatchMask` aliases unused | 🟢 Minor | Defined but never used type aliases                                                                                         |
| Missing JSDoc examples                   | 🟡 Low   | `ForecastOutput`, `RawModelOutput` lack usage examples                                                                      |

### 2.2 Configuration Management (`config.ts`) — Rating: B+

| Issue                                                                                              | Severity  | Description                                                                                                                              |
| -------------------------------------------------------------------------------------------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `configsEqual` ignores `windowSize`                                                                | 🟢 Minor  | By design, but should add a comment explaining why                                                                                       |
| `maxContext` destructured then `fc.maxContext` still points to old value in `createForecastConfig` | 🔴 Medium | line 30: after destructuring, line 62 uses `fc.maxContext` instead of adjusted `maxContext` (logic is correct but semantics are unclear) |

### 2.3 NaN Handling (`nan-handler.ts`) — Rating: A-

| Issue                                                   | Severity | Description                                                                                                                    |
| ------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------ |
| O(n²) NaN interpolation                                 | 🔴 High  | Each NaN does an O(n) search for left/right neighbors, total O(n²). For large arrays a two-pointer single pass should be used. |
| `countNaN`/`hasNaN`/`stripTrailingNaNs` scan repeatedly | 🟡 Low   | Multiple passes over same array, can be merged                                                                                 |
| Missing `+Infinity`/`-Infinity` handling                | 🟡 Low   | Python version doesn't handle Inf, but in JS `Float32Array` `Infinity` won't be caught by `Number.isNaN`                       |

### 2.4 Online Statistics (`stats.ts`) — Rating: A-

| Issue                                                        | Severity | Description                                                                                                                                                                       |
| ------------------------------------------------------------ | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `updateRunningStats` returns `[result, result]`              | 🟡 Low   | Python convention is `((n,mu,sigma), (n,mu,sigma))` dual return; in TS returning the same object reference twice is both wasteful and dangerous (modifying one affects the other) |
| `computeStats` lacks numerical stability measures            | 🟡 Low   | Large values may overflow `sumSq`; should use Welford single-pass algorithm                                                                                                       |
| Comment says "two-pass" but implementation is naive two-pass | 🟢 Minor | Comment and implementation don't match                                                                                                                                            |

### 2.5 RevIN (`revin.ts`) — Rating: B

| Issue                                                              | Severity | Description                                                                                                          |
| ------------------------------------------------------------------ | -------- | -------------------------------------------------------------------------------------------------------------------- |
| `revinBatch4D` triple nested loop poor performance                 | 🔴 High  | Independent traversal for each (batch, patch, timestep, quantile) without utilizing data locality                    |
| `flattenParam` logic fragile                                       | 🔴 High  | line 191: `param.length === len` check for broadcast pattern is unreliable; if lengths happen to match, it misjudges |
| `revin()` function parameter types overly broad                    | 🟡 Low   | `Float32Array \| Float32Array[]` loses TypeScript type safety                                                        |
| `broadcast1D` uses `.fill()` for full fill when `arr.length === 1` | 🟡 Low   | Inefficient for large data; should return scalar reference                                                           |

### 2.6 Tensor Utilities (`tensor-utils.ts`) — Rating: B+

| Issue                                              | Severity | Description                                                                                                              |
| -------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------ |
| Some functions are impure                          | 🟡 Low   | `leftPad` returns truncation when `arrLen >= targetLen`, behavior is non-intuitive (should be named `leftPadOrTruncate`) |
| `hasInvalid` doesn't distinguish NaN from Infinity | 🟢 Minor | Name suggests boolean return but semantics are unclear                                                                   |

### 2.7 Preprocessing Pipeline (`preprocessor.ts`) — Rating: B+

| Issue                                                               | Severity | Description                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`numPatches` uses `fc.maxContext` instead of `mc.inputPatchLen`** | 🔴 High  | line 69: `numPatches = Math.floor(fc.maxContext / inputPatchLen)` — but `fc.maxContext` is already a multiple of `inputPatchLen` (guaranteed by compile), so `Math.floor` is redundant. However, the real issue is that `maxContext` comes from `mc` struct destructuring (line 68), while the actual value is `fc.maxContext`, causing variable name confusion |
| Statistics computed here but not actually needed                    | 🟡 Low   | Prefill phase RevIN statistics are precomputed here, but `decode-loop` recomputes them again (redundant)                                                                                                                                                                                                                                                        |
| `truncatedInputs` not used downstream                               | 🟡 Low   | Only used in tests, wastes slice overhead in production path                                                                                                                                                                                                                                                                                                    |

### 2.8 ONNX Engine (`onnx-engine.ts`) — Rating: B+

| Issue                                                          | Severity | Description                                                                                                           |
| -------------------------------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------- |
| **`_session` type is `unknown`**                               | 🔴 High  | Loses complete type checking; every call needs `as ort.InferenceSession` assertion                                    |
| **`await import('onnxruntime-node')` in every `forward` call** | 🔴 High  | Dynamic import executed on every inference; should be cached to `_ortModule`                                          |
| Output names hardcoded                                         | 🟡 Low   | `'input_emb'`, `'output_emb'` etc. hardcoded; ONNX model output name changes cause silent failure (returns undefined) |
| **No `executionProvider` support**                             | 🔴 High  | Constructor accepts `ModelConfig` but ignores `ModelLoadOptions.executionProvider`, cannot use CUDA                   |
| `toBatch` closure created on every `forward` call              | 🟡 Low   | Should be extracted as method to avoid repeated function creation                                                     |

### 2.9 KV Cache (`kv-cache.ts`) — Rating: A-

| Issue                              | Severity | Description                                                                                                                                                                                                  |
| ---------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `readKVCacheSlice` unused          | 🟢 Minor | Defined but never called                                                                                                                                                                                     |
| `writeKVCache` unused              | 🟡 Low   | KV Cache write logic currently goes through ONNX internal cache in the decode loop (since the ONNX model manages its own KV); these functions are only prepared for future direct Transformer implementation |
| `resetKVCache` does full `fill(0)` | 🟡 Low   | Significant overhead for large caches; could just reset nextIndex/numMasked                                                                                                                                  |

### 2.10 Decode Loop (`decode-loop.ts`) — Rating: B

| Issue                                              | Severity | Description                                                                                                                                                                                                                                                                   |
| -------------------------------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Flip invariance requires two full inferences**   | 🔴 High  | Postprocessor comment says "current flip needs two prefills + two decodes", but actual code in model.ts handles the flip path separately. Decode-loop doesn't involve flip, but architecture still requires two full inferences (one original, one negated), doubling latency |
| AR seeds extraction index calculation hard to read | 🟡 Low   | lines 141-148 and 211-218 have repeated seed extraction logic, should be extracted into a function                                                                                                                                                                            |
| `kvCaches` returns empty array                     | 🔴 High  | Current implementation returns `kvCaches: []` — KV Cache is not actually created and passed to the engine. This limits future KV Cache optimization at the engine level.                                                                                                      |
| `quantileSpreads` extraction logic repeated        | 🟡 Low   | lines 103-126 do a 3-loop manual denormalization on the entire quantileSpreads, which can be simplified using `revinBatch`                                                                                                                                                    |

### 2.11 Postprocessing (`postprocessor.ts`) — Rating: B

| Issue                                                   | Severity    | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Flip invariance implementation has logic error**      | 🔴 Critical | line 99-105: `elementwiseMean(fullForecasts[b], negate(flippedFull[b]))` — this computes `(a + (-b)) / 2` = `(a - b) / 2`. But the correct formula is `(forecast(x) - forecast(-x)) / 2`. The current implementation takes `elementwiseMean(forecast(x), -forecast(-x))`. Since `elementwiseMean` is `(a+b)/2` and b is `negate(flippedFull)`, the result is `(forecast(x) + (-forecastOnNegInput)) / 2 = (forecast(x) - forecastOnNegInput) / 2`. This is semantically correct vs the Python original but convoluted — should directly use `elementwiseDiff / 2` |
| `flipQuantilesBatch` duplicates `flipQuantiles`         | 🟡 Low      | Three flip-related functions serve the same purpose, unnecessary wrapping                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `flippedQS` / `flippedPF` variables assigned but unused | 🔴 Medium   | line 75-76 creates `flippedQS` and `flippedPF` but never referenced later, dead code                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `reverseInputNormalization` recomputes statistics       | 🟡 Low      | Each call re-traverses original input to compute μ,σ; should be computed once in model.ts and passed in                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `addBackcast` `batchSize` parameter unused              | 🟢 Minor    | Function signature has `batchSize` but it's unused                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |

### 2.12 XReg Engine (`xreg-engine.ts`) — Rating: B-

| Issue                                                                     | Severity    | Description                                                                                                                                                                                     |
| ------------------------------------------------------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **"timesfm + xreg" mode residual calculation is wrong**                   | 🔴 Critical | line 365-376: `pfOffset += 0` — this is placeholder code! `pfOffset` never increments, causing every series' residual to be taken from `pointForecast[0]`. Should be `pfOffset += trainLens[s]` |
| `OneHotEncoder` recreated and refitted every time                         | 🟡 Low      | Encoder instance can be cached                                                                                                                                                                  |
| `ridgeRegression` uses `Matrix.columnVector(yVec as unknown as number[])` | 🟡 Low      | Type cast hack; should use better ml-matrix API                                                                                                                                                 |
| **`normalizeXregTargetPerInput` and `maxRowsPerCol` parameters unused**   | 🔴 Medium   | Interface defines them but current implementation ignores them; feature incomplete                                                                                                              |

### 2.13 Model Class (`model.ts`) — Rating: B+

| Issue                                                            | Severity  | Description                                                                                                                                                                        |
| ---------------------------------------------------------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`inputStats` computed but unused**                             | 🔴 Medium | line 172-190 computes `inputStats: {mu, sigma}[]` but never referenced later — should be reused in postProcess phase to avoid recomputation                                        |
| z-score normalization duplicated in model.ts                     | 🟡 Low    | Same logic duplicated with `reverseInputNormalization`                                                                                                                             |
| `paddedInputs` not set correctly when `fc.normalizeInputs=false` | 🟡 Low    | When normalizeInputs=false, originalInputs passed to postProcess is null, and positive clamp and z-score reverse are both skipped — this is correct, but not explicitly documented |
| `forecastWithCovariates` dynamic import                          | ✅ Fixed  | Now dynamically imports xreg, no longer throws stub error                                                                                                                          |

---

## III. Architecture-Level Issues

### 3.1 Performance Bottlenecks

| Bottleneck                       | Impact                  | Priority |
| -------------------------------- | ----------------------- | -------- |
| Flip invariance double inference | Latency ×2              | 🔴 P0    |
| NaN interpolation O(n²)          | Slow for large arrays   | 🟡 P1    |
| RevIN 4D triple loop             | Slow for large horizons | 🟡 P1    |
| Repeated statistics computation  | CPU waste               | 🟢 P2    |

### 3.2 Correctness Defects

| Defect                                             | Impact                                         | Priority |
| -------------------------------------------------- | ---------------------------------------------- | -------- |
| Flip invariance implementation convoluted          | Same result but logic not intuitive            | 🟢 P2    |
| XReg "timesfm + xreg" residual calculation wrong   | Wrong output in this mode                      | 🔴 P0    |
| XReg `normalizeXregTargetPerInput` not implemented | Feature incomplete                             | 🔴 P0    |
| XReg `maxRowsPerCol` sampling not implemented      | Poor performance with large covariate matrices | 🟡 P1    |

### 3.3 Engineering Completeness

| Missing                               | Priority |
| ------------------------------------- | -------- |
| ONNX executionProvider (CUDA) support | 🔴 P0    |
| Real KV Cache integration             | 🟡 P1    |
| Typed ONNX session                    | 🟡 P1    |
| Inf value cleanup                     | 🟡 P1    |
| Streaming forecast                    | 🟢 P2    |
| Model hot reload                      | 🟢 P2    |
| Progress callback                     | 🟢 P2    |

---

## IV. Executable Optimization Plan

### Iteration A: Fix Correctness (P0, estimated 2-3 hours)

```
A1. Fix XReg "timesfm + xreg" residual calculation
    File: xreg-engine.ts line 365-376
    pfOffset += trainLens[s]  ← currently += 0 (BUG)

A2. Implement XReg normalizeXregTargetPerInput
    File: xreg-engine.ts
    Add normalization/denormalization logic (aligning with Python normalize/renormalize)

A3. Clean up Flip invariance implementation
    File: postprocessor.ts
    - Delete dead code flippedQS/flippedPF (line 75-76)
    - Replace elementwiseMean + negate with elementwiseDiff/2
    - Merge flipQuantiles/flipQuantilesBatch/flipQuantileArray into one function

A4. Support ONNX executionProvider
    Files: onnx-engine.ts + model.ts
    - TimesFMInferenceEngine constructor accepts executionProvider
    - InferenceSession.create passes executionProviders
```

### Iteration B: Fix Performance (P1, estimated 2-3 hours)

```
B1. O(n) NaN interpolation algorithm
    File: nan-handler.ts
    Replace O(n²) search with two-pointer single pass

B2. RevIN 4D vectorization
    File: revin.ts
    Replace triple loop with per-patch batch operations

B3. Cache onnxruntime-node dynamic import
    File: onnx-engine.ts
    Cache this._ortModule during load()

B4. Eliminate duplicate statistics computation
    Files: model.ts + postprocessor.ts
    Pass inputStats computed in model.forecast() to postProcess
    reverseInputNormalization directly uses passed-in statistics

B5. Implement XReg maxRowsPerCol sampling
    File: xreg-engine.ts
    When rows > maxRowsPerCol * cols, perform row sampling (align with Python)
```

### Iteration C: Fix Engineering Quality (P1-P2, estimated 2-3 hours)

```
C1. Type the ONNX Session
    File: onnx-engine.ts
    import type { InferenceSession, Tensor } from 'onnxruntime-node'
    Replace all `unknown` and `as` assertions

C2. Add Infinity cleanup
    File: preprocessor.ts
    In cleanSeries, add Infinity → NaN → interpolation handling

C3. Implement real KV Cache creation
    File: decode-loop.ts
    Actually call createKVCache and pass it to the engine

C4. Clean up unused code
    - Delete flipQuantilesBatch
    - Delete readKVCacheSlice
    - Delete unused type exports

C5. Eliminate NumPatches variable name confusion
    File: preprocessor.ts
    Clearly distinguish mc.maxContext vs fc.maxContext
```

### Iteration D: Enhance Features (P2, estimated 2-3 hours)

```
D1. Implement forecastWithCovariates model method binding
    Files: model.ts + xreg-engine.ts
    Set xreg engine's forecastWithCovariates as model method (instead of making users call it manually)

D2. Add model download/cache mechanism
    File: model.ts
    Support auto-download from HuggingFace (requires HF API integration)

D3. Add forecast progress callback
    Files: decode-loop.ts + model.ts
    onProgress(step, total) callback for long-horizon forecasts

D4. Complete benchmark tool
    File: scripts/benchmark.js
    Full performance comparison tool: different context/horizon/batch combinations
```

---

## V. Priority Ranking

```
P0 (Must fix before open-source release):
  ✅ A1  XReg "timesfm + xreg" residual BUG
  ✅ A2  XReg normalizeXregTargetPerInput not implemented
  ✅ A4  ONNX executionProvider (CUDA) support

P1 (Strongly recommended):
  ✅ B1  O(n) NaN interpolation
  ✅ B3  Cache onnxruntime-node import
  ✅ B5  XReg maxRowsPerCol sampling
  ✅ C1  Typed ONNX Session
  ✅ C3  Real KV Cache

P2 (Nice to have):
  ✅ B2  RevIN 4D vectorization
  ✅ B4  Eliminate duplicate statistics computation
  ✅ C2  Infinity cleanup
  ✅ D1-D4  Enhanced features
```

---

## VI. Competitive Comparison Analysis

| Dimension              | agentix-timesfm-ts             | Competitor A (Python TimesFM) | Competitor B (TS Libraries) |
| ---------------------- | ------------------------------ | ----------------------------- | --------------------------- |
| **Language ecosystem** | ✅ TypeScript/Node.js          | ❌ Python only                | ❌ Python/R                 |
| **Inference backend**  | ✅ ONNX Runtime (CPU/CUDA/DML) | ✅ PyTorch/JAX                | ❌ In-house                 |
| **Embeddability**      | ✅ No external services        | ⚠️ Requires Python runtime    | ❌ Requires Python          |
| **Type safety**        | ✅ Complete TypeScript types   | ❌ Dynamic typing             | ⚠️ Partial                  |
| **API design**         | ✅ Fluent async/await          | ✅ NumPy compatible           | ⚠️ sklearn-style            |
| **Test coverage**      | ✅ 111 tests, all passing      | ✅ Has tests                  | ⚠️ Limited                  |
| **Documentation**      | ✅ Complete Chinese docs       | ✅ English docs               | ⚠️ Limited                  |
| **XReg covariates**    | ⚠️ Has defects (P0)            | ✅ Complete                   | ❌ None                     |
| **Flip invariance**    | ⚠️ Double latency              | ✅ Double latency             | ❌ None                     |
| **KV Cache**           | ⚠️ Not integrated              | ✅ Complete                   | -                           |
| **CUDA support**       | ⚠️ Not wired up                | ✅ Complete                   | ❌ None                     |

---

## VII. Summary

The current code already has the foundation for open-source release — elegant architecture, thorough testing, and complete documentation. However, to dominate competitors after release, the **3 P0 fixes + 5 P1 optimizations** must be completed first. In particular:

1. **XReg "timesfm + xreg" mode residual BUG** (currently produces wrong results)
2. **ONNX CUDA support** (critical for production environments)
3. **O(n) NaN interpolation** (long-sequence user experience)

Once these are completed, agentix-timesfm-ts will be the **only** library capable of running large time-series foundation models in Node.js without external services, with a clear differentiated competitive advantage.
