# timesfm-ts Usage Documentation

> Node.js/TypeScript TimesFM 2.5 — Zero-Shot Time Series Forecasting

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Getting the Model](#2-getting-the-model)
3. [API Usage](#3-api-usage)
4. [CLI Tools](#4-cli-tools)
5. [Configuration Reference](#5-configuration-reference)
6. [Output Description](#6-output-description)
7. [Model Update](#7-model-update)
8. [Troubleshooting](#8-troubleshooting)
9. [Performance Guide](#9-performance-guide)

---

## 1. Quick Start

### Installation

```bash
git clone https://github.com/AgentiX-E/timesfm-ts.git
cd timesfm-ts
pnpm install
pnpm build
```

### Minimal Example (requires ONNX model file, see Section 2)

```typescript
import { TimesFMModel, createForecastConfig } from '@agentix-e/timesfm-core';

// 1. Load model
const model = await TimesFMModel.fromPretrained({
  modelPath: './models/timesfm-2.5.onnx', // Required
});

// 2. Compile (set forecast parameters)
model.compile(
  createForecastConfig({
    maxContext: 1024,
    maxHorizon: 256,
  }),
);

// 3. Forecast
const { pointForecast, quantileForecast } = await model.forecast(
  24, // Forecast next 24 steps
  [new Float32Array([1, 2, 3 /* ... your data */])],
);

// 4. Use results
console.log('Point forecast:', Array.from(pointForecast[0]));
console.log('80% CI lower bound:', Array.from(quantileForecast[0][1]));
console.log('80% CI upper bound:', Array.from(quantileForecast[0][9]));

// 5. Release resources
await model.dispose();
```

> **For full API, configuration, CLI usage** → see subsequent sections of this document.
> **For model export and automation pipeline** → see [MODEL-UPDATE.md](MODEL-UPDATE.md).

---

## 2. Getting the Model

### 2.1 One-Click Full Pipeline (Recommended)

```bash
# Node.js — One-click: Download → Export → Validate → Test → Benchmark
pnpm run pipeline

# Or pure Node.js
node scripts/pipeline.js
```

> See [MODEL-UPDATE.md](MODEL-UPDATE.md) for details.

### 2.2 Manual Export

```bash
pip install "timesfm[torch]" onnx onnxruntime torch
```

**Step 2: Run the export script**

```bash
# Basic usage — auto-download TimesFM 2.5 200M and export
python scripts/export-onnx.py --output models/timesfm-2.5.onnx

# Specify a version
python scripts/export-onnx.py \
  --model google/timesfm-2.5-200m-pytorch \
  --output models/timesfm-2.5.onnx

# Skip validation (faster)
python scripts/export-onnx.py \
  --output models/timesfm-2.5.onnx \
  --skip-validation
```

**Export flow**:

```
HuggingFace Hub              Local Disk
┌──────────────────┐        ┌──────────────────┐
│ google/           │ pip    │ models/           │
│ timesfm-2.5-      │──────→│ timesfm-2.5.onnx  │
│ 200m-pytorch      │ export │ (~928 MB)          │
│ (safetensors)     │        │                   │
└──────────────────┘        └──────────────────┘
      ~800 MB                   ~928 MB
```

**Step 3: Validate the model**

```bash
# Use Node.js model checker
node packages/timesfm-core/scripts/check-model.js \
  --model models/timesfm-2.5.onnx \
  --bench
```

Expected output:

```
============================================================
  timesfm-ts — ONNX Model Checker
============================================================
Model file:
  Size:   ~928 MB
  Status: ✅ Size matches TimesFM 2.5 200M (~928 MB)

ONNX Runtime:
  Load time: ~5000 ms
  Status:    ✅ Loaded successfully
============================================================
  ✅ Model ready for use with timesfm-ts
============================================================
```

### 2.3 Model File Specifications

| Property | Value                                                          |
| -------- | -------------------------------------------------------------- |
| Filename | `timesfm-2.5.onnx`                                             |
| Size     | ~928 MB                                                        |
| Format   | ONNX (opset 18)                                                |
| Input    | `inputs: [batch, patches, 64]` (float32)                       |
| Output 1 | `input_emb: [batch, patches, 1280]`                            |
| Output 2 | `output_emb: [batch, patches, 1280]`                           |
| Output 3 | `output_ts: [batch, patches, 1280]` (128 steps×10 quantiles)   |
| Output 4 | `output_qs: [batch, patches, 10240]` (1024 steps×10 quantiles) |
| Backend  | ONNX Runtime: CPU / CUDA / DirectML                            |

### 2.4 Hardware Requirements

| Component      | Minimum      | Recommended       |
| -------------- | ------------ | ----------------- |
| RAM            | 4 GB         | 8 GB+             |
| Disk           | 1 GB free    | SSD               |
| GPU (Optional) | 2 GB VRAM    | 4 GB+ VRAM        |
| CPU Mode       | ✅ Available | Slower but viable |

---

## 3. API Usage

### 3.1 TimesFMModel

Core class through which all operations are performed.

```typescript
import { TimesFMModel } from '@agentix-e/timesfm-core';
```

#### `fromPretrained(options)`

Load a pretrained model.

```typescript
const model = await TimesFMModel.fromPretrained({
  modelPath: './models/timesfm-2.5.onnx', // Required
  executionProvider: 'cpu', // Optional: 'cpu' | 'cuda' | 'dml'
});
```

#### `compile(config)`

Compile forecast configuration. Must be called before `forecast()`.

```typescript
import { createForecastConfig } from '@agentix-e/timesfm-core';

model.compile(
  createForecastConfig({
    maxContext: 1024,
    maxHorizon: 256,
  }),
);
```

#### `forecast(horizon, inputs)`

Execute forecast.

```typescript
const result = await model.forecast(
  24, // Forecast steps (must be ≤ maxHorizon)
  [
    new Float32Array([1.0, 2.0, 3.0 /* ... */]), // Series 1
    new Float32Array([10.0, 20.0 /* ... */]), // Series 2
  ],
);

// result.pointForecast[0] → Float32Array(24) — Series 1 median forecast
// result.pointForecast[1] → Float32Array(24) — Series 2 median forecast
// result.quantileForecast[0][1] → Float32Array(24) — Series 1 q10
// result.quantileForecast[0][9] → Float32Array(24) — Series 1 q90
```

#### `dispose()`

Release ONNX Runtime resources and GPU memory.

```typescript
await model.dispose();
```

### 3.2 Full Example: Single Series Forecast

```typescript
import { TimesFMModel, createForecastConfig } from '@agentix-e/timesfm-core';

async function main() {
  // Load
  const model = await TimesFMModel.fromPretrained({
    modelPath: './models/timesfm-2.5.onnx',
  });

  // Compile
  model.compile(
    createForecastConfig({
      maxContext: 1024,
      maxHorizon: 256,
      normalizeInputs: true,
      useContinuousQuantileHead: true,
      forceFlipInvariance: true,
      inferIsPositive: true,
      fixQuantileCrossing: true,
    }),
  );

  // Prepare data — arbitrary length, may contain NaN
  const data = new Float32Array([100, 102, 105, NaN, 110, 108, 112 /* ... */]);

  // Forecast
  const { pointForecast, quantileForecast } = await model.forecast(24, [data]);

  // Use
  console.log('Forecast (median):', Array.from(pointForecast[0]));
  console.log('80% CI Lower:  ', Array.from(quantileForecast[0][1]));
  console.log('80% CI Upper:  ', Array.from(quantileForecast[0][9]));

  // Cleanup
  await model.dispose();
}

main();
```

### 3.3 Batch Forecasting

```typescript
// Forecast multiple series at once
const salesData = new Float32Array(/* ... */);
const revenueData = new Float32Array(/* ... */);
const usersData = new Float32Array(/* ... */);

const { pointForecast, quantileForecast } = await model.forecast(30, [
  salesData,
  revenueData,
  usersData,
]);

// pointForecast[0] → Sales forecast (30 steps)
// pointForecast[1] → Revenue forecast (30 steps)
// pointForecast[2] → Users forecast (30 steps)
```

### 3.4 Evaluating Forecast Accuracy

```typescript
// Hold out the last H actual values as test set
const H = 24;
const train = actualData.slice(0, -H);
const test = actualData.slice(-H);

const { pointForecast, quantileForecast } = await model.forecast(H, [train]);
const pred = pointForecast[0];

// Compute metrics
let mae = 0,
  rmse = 0,
  withinCI = 0;
for (let i = 0; i < H; i++) {
  mae += Math.abs(test[i] - pred[i]);
  rmse += (test[i] - pred[i]) ** 2;
  if (test[i] >= quantileForecast[0][1][i] && test[i] <= quantileForecast[0][9][i]) {
    withinCI++;
  }
}
mae /= H;
rmse = Math.sqrt(rmse / H);

console.log(`MAE: ${mae.toFixed(2)}`);
console.log(`RMSE: ${rmse.toFixed(2)}`);
console.log(`80% CI Coverage: ${((withinCI / H) * 100).toFixed(1)}%`);
```

### 3.5 Covariate Forecasting (XReg)

Requires `@agentix-e/timesfm-xreg`:

```typescript
import { TimesFMModel, createForecastConfig } from '@agentix-e/timesfm-core';
import { forecastWithCovariates } from '@agentix-e/timesfm-xreg';

const model = await TimesFMModel.fromPretrained({
  modelPath: './models/timesfm-2.5.onnx',
});
model.compile(
  createForecastConfig({
    maxContext: 512,
    maxHorizon: 128,
  }),
);

const result = await forecastWithCovariates(model, {
  inputs: [salesData],
  // Dynamic numerical covariates — length = context + horizon
  dynamicNumericalCovariates: {
    temperature: [new Float32Array(/* length = context+H */)],
    price: [new Float32Array(/* length = context+H */)],
  },
  // Dynamic categorical covariates
  dynamicCategoricalCovariates: {
    dayOfWeek: [
      [
        /* length = context+H */
      ],
    ],
  },
  // Static covariates — one value per series
  staticCategoricalCovariates: {
    storeType: ['flagship'],
  },
  xregMode: 'xreg + timesfm', // Or 'timesfm + xreg'
  ridge: 0.1, // Ridge regularization
});
```

---

## 4. CLI Tools

### Basic Usage

```bash
# Forecast from CSV
node packages/timesfm-cli/dist/cli.js forecast \
  --model ./models/timesfm-2.5.onnx \
  --horizon 24 \
  input.csv

# Specify output file
node packages/timesfm-cli/dist/cli.js forecast \
  --model ./models/timesfm-2.5.onnx \
  --horizon 52 \
  --output forecasts.csv \
  input.csv

# JSON format output
node packages/timesfm-cli/dist/cli.js forecast \
  --model ./models/timesfm-2.5.onnx \
  --horizon 24 \
  --output-format json \
  --output forecasts.json \
  input.csv

# Specify columns to forecast and date column
node packages/timesfm-cli/dist/cli.js forecast \
  --model ./models/timesfm-2.5.onnx \
  --horizon 24 \
  --date-col date \
  --value-cols sales,revenue \
  input.csv
```

### CLI Parameters

| Parameter                       | Required | Description                        |
| ------------------------------- | -------- | ---------------------------------- |
| `-m, --model <path>`            | ✅       | ONNX model file path               |
| `-H, --horizon <n>`             | ✅       | Number of forecast steps           |
| `-d, --date-col <name>`         | ❌       | Date column name (default: date)   |
| `-v, --value-cols <names>`      | ❌       | Comma-separated value column names |
| `-o, --output <path>`           | ❌       | Output file path                   |
| `--output-format <fmt>`         | ❌       | Output format: csv or json         |
| `--context <n>`                 | ❌       | Max context length (default: 1024) |
| `--no-normalize`                | ❌       | Disable input normalization        |
| `--no-flip-invariance`          | ❌       | Disable flip invariance            |
| `--no-positive`                 | ❌       | Disable non-negative constraint    |
| `--no-fix-quantile-crossing`    | ❌       | Disable quantile crossing fix      |
| `--no-continuous-quantile-head` | ❌       | Disable continuous quantile head   |

### CSV Input Format

```csv
date,value
2024-01-01,100
2024-01-02,102
2024-01-03,105
2024-01-04,
2024-01-05,110
```

- Supports missing values (empty or NaN) → automatic linear interpolation
- First column defaults to date column (configurable)
- Remaining numeric columns are forecasted individually

### CSV Output Format

```csv
series_id,horizon_step,point_forecast,q10,q50,q90
value,1,112.3,108.1,112.3,116.5
value,2,114.1,109.3,114.1,118.9
...
```

### JSON Output Format

```json
{
  "model": "timesfm-2.5",
  "horizon": 24,
  "series": {
    "value": {
      "point_forecast": [112.3, 114.1, ...],
      "lower_80": [108.1, 109.3, ...],
      "upper_80": [116.5, 118.9, ...],
      "quantiles": {
        "q10": [...], "q20": [...], /* ... */ "q90": [...]
      }
    }
  }
}
```

---

## 5. Configuration Reference

### ForecastConfig Full Parameters

```typescript
interface ForecastConfig {
  maxContext: number; // Max context length (auto-padded to multiple of 32)
  maxHorizon: number; // Max forecast length (auto-padded to multiple of 128)
  normalizeInputs: boolean; // Input z-score normalization (recommended)
  perCoreBatchSize: number; // Per-core batch size (default 1)
  useContinuousQuantileHead: boolean; // Continuous quantile head (recommended)
  forceFlipInvariance: boolean; // Flip invariance (recommended)
  inferIsPositive: boolean; // Non-negative constraint (default on)
  fixQuantileCrossing: boolean; // Quantile crossing fix (recommended)
  returnBackcast: boolean; // Return backcast (needed for covariates)
}
```

### Recommended Configuration

```typescript
// Production
createForecastConfig({
  maxContext: 1024,
  maxHorizon: 256,
  normalizeInputs: true,
  perCoreBatchSize: 1,
  useContinuousQuantileHead: true,
  forceFlipInvariance: true,
  inferIsPositive: true,
  fixQuantileCrossing: true,
  returnBackcast: false,
});
```

### Context Length Recommendations

| Scenario                | maxContext | Notes                    |
| ----------------------- | ---------- | ------------------------ |
| Quick prototyping       | 128        | Second-level forecast    |
| Daily data (~1 year)    | 256-512    | Good balance             |
| Daily data (~2-3 years) | 512-1024   | Standard production      |
| High-frequency data     | 1024-4096  | More accurate but slower |
| Extreme                 | 4096-16384 | TimesFM 2.5 upper limit  |

---

## 6. Output Description

### Output Shapes

```
pointForecast:    [numSeries, horizon]           — Median forecast
quantileForecast: [numSeries, horizon, 10]       — Full quantile distribution
```

### Quantile Indices

| Index | Meaning  | Usage                                   |
| ----- | -------- | --------------------------------------- |
| 0     | **Mean** | Average forecast                        |
| 1     | **Q10**  | 80% prediction interval lower bound     |
| 2     | Q20      | 60% lower                               |
| 3     | Q30      | 40% lower                               |
| 4     | Q40      | 20% lower                               |
| **5** | **Q50**  | **Median = pointForecast**              |
| 6     | Q60      | 20% upper                               |
| 7     | Q70      | 40% upper                               |
| 8     | Q80      | 60% upper                               |
| 9     | **Q90**  | **80% prediction interval upper bound** |

### Useful Constants

```typescript
import { QUANTILE_INDICES } from '@agentix-e/timesfm-core';

// QUANTILE_INDICES.MEAN → 0
// QUANTILE_INDICES.Q10  → 1
// QUANTILE_INDICES.Q50  → 5
// QUANTILE_INDICES.Q90  → 9
```

---

## 7. Model Update

### Check Current TimesFM Latest Version

TimesFM models are published on HuggingFace:
https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6

### Update to Latest Model

```bash
# Re-export latest version
python scripts/export-onnx.py \
  --model google/timesfm-2.5-200m-pytorch \
  --output models/timesfm-2.5.onnx \
  --skip-validation

# Verify
node packages/timesfm-core/scripts/check-model.js \
  --model models/timesfm-2.5.onnx \
  --bench
```

### Verify Model Compatibility

```bash
# Run all tests to confirm compatibility
cd packages/timesfm-core
pnpm test
```

If all tests pass (especially model tests), the new model is compatible with current code.

---

## 8. Troubleshooting

### Model Load Failure

```
Error: ONNX engine not loaded. Call load() first.
```

**Solution**:

1. Verify model file exists and is ≥ 800MB
2. Run `python scripts/export-onnx.py --validate-only --output <path>`
3. Check that `onnxruntime-node` is installed

### Out of Memory (OOM)

```
JavaScript heap out of memory
```

**Solution**:

```typescript
// Reduce batch size
model.compile(
  createForecastConfig({
    perCoreBatchSize: 1, // Minimum batch
  }),
);

// Or use smaller context
model.compile(
  createForecastConfig({
    maxContext: 256, // Reduce to 256
  }),
);
```

### NaN Forecast Results

**Cause**: Input data may contain abnormal extreme values.

**Solution**:

```typescript
// Preprocess data
function cleanInput(arr: Float32Array): Float32Array {
  // Clip extreme values
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  const std = Math.sqrt(arr.reduce((a, b) => a + (b - mean) ** 2, 0) / arr.length);
  const result = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    result[i] = Math.max(-1e6, Math.min(1e6, arr[i]));
  }
  return result;
}
```

### ONNX Runtime Not Found

```
Cannot find module 'onnxruntime-node'
```

**Solution**:

```bash
cd packages/timesfm-core
pnpm install
```

---

## 9. Performance Guide

### Expected Inference Speed

| Hardware       | Context=256 | Context=1024 | Context=4096 |
| -------------- | ----------- | ------------ | ------------ |
| CPU (32 cores) | 0.5-2s      | 2-8s         | 8-30s        |
| CPU (8 cores)  | 1-4s        | 4-15s        | 15-60s       |
| GPU (8GB)      | 0.1-0.5s    | 0.3-2s       | 1-5s         |
| GPU (24GB)     | 0.05-0.2s   | 0.1-1s       | 0.5-3s       |

### Optimization Suggestions

1. **Reduce context**: `maxContext: 256` instead of 1024, 3-5x speedup
2. **Increase batch**: `perCoreBatchSize: 32` to improve throughput
3. **Disable flip invariance**: `forceFlipInvariance: false` to halve inference load
4. **Disable continuous quantile head**: `useContinuousQuantileHead: false` for slight speedup
5. **Use GPU**: `executionProvider: 'cuda'` for 5-20x speedup

### Memory Optimization

```typescript
// Process large batches in chunks
const CHUNK = 50;
for (let i = 0; i < allInputs.length; i += CHUNK) {
  const chunk = allInputs.slice(i, i + CHUNK);
  const { pointForecast, quantileForecast } = await model.forecast(H, chunk);
  // Save chunk results...
}
```

---

## Architecture Overview

```
┌─────────────────────────────────────────┐
│              timesfm-ts          │
├─────────────────────────────────────────┤
│                                          │
│  User Data (Float32Array[])              │
│      │                                   │
│      ▼                                   │
│  ┌──────────┐    ┌───────────────┐      │
│  │Preprocess│───→│TimesFMInfer.. │      │
│  │          │    │ (ONNX Runtime) │      │
│  │•NaN clnup│    │               │      │
│  │•Patch    │    │ C++ native inf│      │
│  │•RevIN    │    │ CPU/CUDA/DML  │      │
│  │•Welford  │    └───────────────┘      │
│  └──────────┘           │               │
│                         ▼               │
│                    ┌──────────┐         │
│                    │DecodeLoop│         │
│                    │          │         │
│                    │•Prefill  │         │
│                    │•AR Decode│         │
│                    └──────────┘         │
│                         │               │
│                         ▼               │
│                    ┌──────────┐         │
│                    │Postprocess│        │
│                    │          │         │
│                    │•Flip     │         │
│                    │•Quantile │         │
│                    │•Crossing │         │
│                    │•Positive │         │
│                    └──────────┘         │
│                         │               │
│                         ▼               │
│  Forecast Output (pointForecast + quantile)│
│                                          │
└─────────────────────────────────────────┘
```

---

## References

- **Project Repository**: https://github.com/AgentiX-E/timesfm-ts
- **TimesFM Paper**: https://arxiv.org/abs/2310.10688
- **HuggingFace Models**: https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6
- **ONNX Runtime**: https://onnxruntime.ai/
