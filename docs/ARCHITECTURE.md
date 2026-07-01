# Architecture

> `timesfm-ts` — A TypeScript/Node.js implementation of Google's TimesFM 2.5 decoder-only time-series foundation model.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 @agentix-e/timesfm-cli                        │
│  Commander-based CLI: setup (model download) + forecast      │
└──────────────────────────┬──────────────────────────────────┘
                           │ uses
┌──────────────────────────▼──────────────────────────────────┐
│               @agentix-e/timesfm-core                         │
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
│  └────────────────┬────────────────────────┘              │
│                   │ uses                                    │
│  ┌────────────────▼────────────────────────┐              │
│  │   IInferenceEngine (abstraction)         │              │
│  │   load() / forward() / dispose()         │              │
│  │   Implementations: ⇓                     │              │
│  └─────────────────────────────────────────┘              │
│                                                             │
│  Utilities: NaN handler, RevIN, RunningStats, Tensor ops    │
│  Model Downloader: GitHub Releases → streaming fetch        │
└─────────────────────────────────────────────────────────────┘
                           │ implements IInferenceEngine
┌──────────────────────────▼──────────────────────────────────┐
│               @agentix-e/timesfm-node                         │
│  ┌─────────────────────────────────────────┐              │
│  │       TimesFMNodeEngine                   │              │
│  │  ONNX Runtime backend (onnxruntime-node)  │              │
│  │  Concurrent batch inference (Promise.all) │              │
│  └─────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
                           │ implements IInferenceEngine
┌──────────────────────────▼──────────────────────────────────┐
│               @agentix-e/timesfm-web                          │
│  ┌─────────────────────────────────────────┐              │
│  │    TimesFMWebInferenceEngine              │              │
│  │  WASM/WebGPU backend (onnxruntime-web)    │              │
│  └─────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
                           │ uses (optional)
┌──────────────────────────▼──────────────────────────────────┐
│               @agentix-e/timesfm-xreg                         │
│  Ridge regression + OneHotEncoder for exogenous covariates   │
│  Modes: "xreg + timesfm" | "timesfm + xreg"                 │
└─────────────────────────────────────────────────────────────┘
                           │ uses (optional)
┌──────────────────────────▼──────────────────────────────────┐
│           @agentix-e/timesfm-hierarchical                     │
│  Hierarchical reconciliation: bottom-up, top-down, MinT     │
│  Components: reconciliation, summing-matrix, orchestration  │
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

The ONNX model manages its own internal KV cache — no external cache module is needed.

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

### 5. Node.js Inference Engine

**File**: `packages/timesfm-node/src/node-engine.ts`

Implements `IInferenceEngine` via `onnxruntime-node`:

- **Pluggable execution providers**: CPU / CUDA / DirectML
- **Concurrent batch inference**: `Promise.all` for parallel ONNX session calls
- **Fixed-shape handling**: Exported model has `[1, 16, 64]` input shape; pads variable inputs
- **Proper resource cleanup**: Calls `session.release()` on dispose

The `IInferenceEngine` interface in `@agentix-e/timesfm-core` keeps the core
ONNX-free, enabling:

- `@agentix-e/timesfm-node` — Node.js engine (onnxruntime-node)
- `@agentix-e/timesfm-web` — Browser engine (onnxruntime-web via WASM/WebGPU)

`TimesFMModel.fromPretrained()` dynamically imports `@agentix-e/timesfm-node`
when no custom engine is provided, making the dependency truly optional for
browser users.

### 6. Model Downloader

**File**: `packages/timesfm-core/src/model-downloader.ts`

- **Streaming download**: Uses Node.js `fetch` reader → file `writeStream` (no 885 MB heap buffer)
- **Proxy support**: Environment variables (`TIMESFM_PROXY_URL/USERNAME/PASSWORD`, `HTTPS_PROXY`) or programmatic `DownloadOptions.proxy` with username/password. Uses undici `ProxyAgent` for clean proxy handling without global environment mutations
- **SHA-256 integrity**: Hashes the extracted ONNX file after download for verification
- **Cache management**: Platform-aware cache directory (`XDG_CACHE_HOME`)
- **Cross-platform extraction**: Tries `unzip`, `7z`, and PowerShell `Expand-Archive` backends
- **Error hierarchy**: `ProxyAuthError` (HTTP 407), `DownloadError`, `ChecksumMismatchError`

### 7. XReg Engine

**File**: `packages/timesfm-xreg/src/xreg-engine.ts`

Exogenous covariate support with Ridge regression:

- **"xreg + timesfm"**: Fit covariates → forecast residuals with TimesFM → combine
- **"timesfm + xreg"**: Forecast with TimesFM → fit covariates on residuals (using backcast) → combine
- **OneHotEncoder**: Pure-TypeScript scikit-learn compatible encoder

### 8. Web Inference Engine

**File**: `packages/timesfm-web/src/web-engine.ts`

Browser-compatible inference via onnxruntime-web:

- **Pluggable backends**: WebGPU (fastest), WASM (universal), WebGL (legacy)
- **Automatic fallback**: Tries providers in order until one succeeds
- **Model loading**: Accepts URL strings (fetched) or pre-loaded ArrayBuffers
- **Implements `IInferenceEngine`**: Can be injected into `TimesFMModel.fromPretrained()` for browser use
- **CDN WASM**: Falls back to jsdelivr CDN when onnxruntime-web WASM binary is not locally resolvable

### 9. Hierarchical Reconciliation Engine

**Package**: `@agentix-e/timesfm-hierarchical`

Hierarchical time series reconciliation with multiple strategies:

| File                | Description                                               |
| ------------------- | --------------------------------------------------------- |
| `reconciliation.ts` | Optimal reconciliation (bottom-up, top-down, MinT)        |
| `summing-matrix.ts` | S matrix construction for hierarchy definition            |
| `hierarchical.ts`   | Orchestration engine managing the reconciliation pipeline |
| `types.ts`          | Type definitions for hierarchical structures              |

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

| Package                           | Code Size | Dependencies                                                                                                               |
| --------------------------------- | --------- | -------------------------------------------------------------------------------------------------------------------------- |
| `@agentix-e/timesfm-core`         | ~100 KB   | _Zero runtime dependencies_ (pure TypeScript)                                                                              |
| `@agentix-e/timesfm-node`         | ~15 KB    | `@agentix-e/timesfm-core`, `onnxruntime-node`                                                                              |
| `@agentix-e/timesfm-xreg`         | ~30 KB    | `@agentix-e/timesfm-core`, `ml-matrix`                                                                                     |
| `@agentix-e/timesfm-cli`          | ~15 KB    | `@agentix-e/timesfm-core`, `@agentix-e/timesfm-node`, `@agentix-e/timesfm-xreg`, `commander`, `csv-parse`, `csv-stringify` |
| `@agentix-e/timesfm-web`          | ~10 KB    | `@agentix-e/timesfm-core`, `onnxruntime-web` (peer)                                                                        |
| `@agentix-e/timesfm-hierarchical` | ~30 KB    | `@agentix-e/timesfm-core`, `ml-matrix`                                                                                     |

Model weights (885 MB) are downloaded separately from GitHub Releases.
