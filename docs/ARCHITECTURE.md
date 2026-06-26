# Architecture

> `agentix-timesfm-ts` — A TypeScript/Node.js implementation of Google's TimesFM 2.5 decoder-only time-series foundation model.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     @agentix-e/timesfm-cli                     │
│  Commander-based CLI: setup (model download) + forecast      │
└──────────────────────────┬──────────────────────────────────┘
                           │ uses
┌──────────────────────────▼──────────────────────────────────┐
│                   @agentix-e/timesfm-core                      │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌───────────┐ │
│  │  Model   │  │  Config  │  │ Preprocess │  │Postprocess│ │
│  │  (API)   │  │ (types)  │  │  (pipeline)│  │ (pipeline)│ │
│  └────┬─────┘  └──────────┘  └─────┬──────┘  └─────┬─────┘ │
│       │                            │               │       │
│       │    ┌───────────────────────┴───────────────┘       │
│       │    │                                                │
│       ▼    ▼                                                │
│  ┌─────────────────────────────────────────┐              │
│  │         Decode Loop                      │              │
│  │  Phase 1: Prefill (full forward pass)    │              │
│  │  Phase 2: Autoregressive decode          │              │
│  │  KV Cache: deferred (ONNX manages it)    │              │
│  └────────────────┬────────────────────────┘              │
│                   │ uses                                    │
│  ┌────────────────▼────────────────────────┐              │
│  │       TimesFMInferenceEngine             │              │
│  │  IInferenceEngine → ONNX Runtime backend  │              │
│  │  Concurrent batch inference (Promise.all) │              │
│  └─────────────────────────────────────────┘              │
│                                                             │
│  Utilities: NaN handler, RevIN, RunningStats, Tensor ops    │
│  Model Downloader: GitHub Releases → streaming fetch        │
└─────────────────────────────────────────────────────────────┘
                           │ uses (optional)
┌──────────────────────────▼──────────────────────────────────┐
│                   @agentix-e/timesfm-xreg                      │
│  Ridge regression + OneHotEncoder for exogenous covariates   │
│  Modes: "xreg + timesfm" | "timesfm + xreg"                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Design

### 1. TimesFMModel (Public API)

**File**: `packages/timesfm-core/src/model.ts`

The main entry point. Implements `ITimesFMModel` for testability and DI.

```typescript
// Lifecycle
model = TimesFMModel.fromPretrained({ modelPath, executionProvider });
model.compile(forecastConfig); // returns this (chainable)
result = model.forecast(horizon, inputs);
model.dispose();
```

**Design decisions**:

- **Private constructor + static factory**: Ensures async initialization is enforced
- **`compile()` returns `this`**: Enables method chaining
- **`ITimesFMModel` interface**: Enables testable separation of concerns
- **`forecastWithCovariates()`**: Dynamically imports `@agentix-e/timesfm-xreg` when available

### 2. Preprocessor Pipeline

**File**: `packages/timesfm-core/src/preprocessor.ts`

Transforms raw time series into model-ready tensors:

```
Raw Series → [cleanSeries] → [pad/truncate] → [patch split] → [RevIN normalize] → [mask]
    │              │                │                  │                │
    │         NaN→interp     to maxContext      inputPatchLen     (x-μ)/σ per patch
    │         Inf→NaN
```

### 3. Decode Loop

**File**: `packages/timesfm-core/src/inference/decode-loop.ts`

Two-phase autoregressive decoding:

1. **Prefill**: Single forward pass on full context → extract last output patch
2. **AR Decode**: Autoregressively generate future patches, each feeding the last median prediction as the next input seed

KV Cache is computed but **deferred** — the ONNX model manages its own internal cache, so the external `kv-cache.ts` module (marked `@experimental`) is **not used** by the current ONNX inference path. It exists as a prepared implementation for a potential future pure-TypeScript Transformer.

### 4. Postprocessor Pipeline

**File**: `packages/timesfm-core/src/postprocessor.ts`

Applies `ForecastConfig` flags in order:

```
Step 1: Assemble forecast (last patch + AR outputs) → truncate to horizon
Step 2: Flip invariance → (f(x) - f(-x)) / 2
Step 3: Continuous quantile head calibration
Step 4: Backcast extraction (if returnBackcast)
Step 5: Quantile crossing fix → monotonic constraint
Step 6: Input normalization reversal → x·σ + μ
Step 7: Positive clamping → clamp ≥ 0 (only for all-non-negative inputs)
Step 8: Split into point/quantile arrays
```

### 5. ONNX Inference Engine

**File**: `packages/timesfm-core/src/inference/onnx-engine.ts`

Implements `IInferenceEngine`:

- **Pluggable execution providers**: CPU / CUDA / DirectML
- **Concurrent batch inference**: `Promise.all` for parallel ONNX session calls
- **Fixed-shape handling**: Exported model has `[1, 16, 64]` input shape; pads variable inputs
- **Proper resource cleanup**: Calls `session.release()` on dispose

### 6. Model Downloader

**File**: `packages/timesfm-core/src/model-downloader.ts`

- **Streaming download**: Uses Node.js `fetch` reader → file `writeStream` (no 885 MB heap buffer)
- **SHA-256 integrity**: Hashes during download for verification
- **Atomic writes**: Downloads to `.tmp` then renames
- **Cache management**: Platform-aware cache directory (`XDG_CACHE_HOME`)

### 7. XReg Engine

**File**: `packages/timesfm-xreg/src/xreg-engine.ts`

Exogenous covariate support with Ridge regression:

- **"xreg + timesfm"**: Fit covariates → forecast residuals with TimesFM → combine
- **"timesfm + xreg"**: Forecast with TimesFM → fit covariates on residuals (using backcast) → combine
- **OneHotEncoder**: Pure-TypeScript scikit-learn compatible encoder

---

## Data Flow

```
User Data (Float32Array[])
    │
    ▼
[Model.forecast()]
    │
    ├─→ [preprocess()]
    │       ├─ cleanSeries (NaN handling)
    │       ├─ leftPad / truncate → maxContext
    │       ├─ patch split → [B, P, inputPatchLen]
    │       ├─ running stats → μ, σ per patch
    │       └─ RevIN normalize → (x-μ)/σ
    │
    ├─→ [decode()]
    │       ├─ Prefill: engine.forward() → outputTimeSeries
    │       ├─ RevIN denormalize
    │       └─ AR loop: self-feed median → engine.forward()
    │
    ├─→ [flip decode] (if forceFlipInvariance)
    │       └─ Same as above on negated inputs
    │
    └─→ [postProcess()]
            ├─ Assemble full forecast
            ├─ Flip invariance: (f(x)-f(-x))/2
            ├─ Continuous quantile head
            ├─ Quantile crossing fix
            ├─ Reverse normalization
            ├─ Positive clamping (conditional)
            └─ Split → { pointForecast, quantileForecast }
```

---

## Type System

```
ForecastConfig   — mutable config (validated by compile())
ModelConfig      — frozen, read-only (TIMESFM_25_CONFIG singleton)
ForecastOutput   — { pointForecast, quantileForecast, backcast? }
ITimesFMModel    — public interface for dependency injection
IInferenceEngine — pluggable backend (ONNX)
```

---

## Key Design Principles

1. **Functional core, imperative shell**: All utility functions are pure; only `TimesFMModel` manages state
2. **Interface-based abstraction**: `IInferenceEngine` decouples the model from ONNX Runtime
3. **Self-describing models**: `model-descriptor.json` is the single source of truth for architecture constants — `fromPretrained()` resolves `ModelConfig` via `resolveModelConfig()` from the descriptor, falling back to `TIMESFM_25_CONFIG` only when no descriptor is present
4. **Zero-dependency core**: Only `onnxruntime-node` (dynamic import); all tensor math is hand-rolled
5. **Python parity**: Every source file cites the corresponding Python source for cross-verification
6. **Progressive disclosure**: Public API exports both high-level (`TimesFMModel`) and advanced (`decode`, `preprocess`, `revin`) APIs

---

## Package Sizes

| Package                   | Code Size | Dependencies                              |
| ------------------------- | --------- | ----------------------------------------- |
| `@agentix-e/timesfm-core` | ~100 KB   | `onnxruntime-node` (dynamic)              |
| `@agentix-e/timesfm-xreg` | ~30 KB    | `ml-matrix`                               |
| `@agentix-e/timesfm-cli`  | ~15 KB    | `commander`, `csv-parse`, `csv-stringify` |

Model weights (885 MB) are downloaded separately from GitHub Releases.
