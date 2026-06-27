# @agentix-e/timesfm-core

Node.js / TypeScript implementation of Google Research's **TimesFM 2.5** — a 200M-parameter decoder-only transformer for zero-shot time-series forecasting. No training, no Python, no external services.

[![npm](https://img.shields.io/npm/v/@agentix-e/timesfm-core?color=blue)](https://www.npmjs.com/package/@agentix-e/timesfm-core)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../../LICENSE)

📚 [API Documentation](https://agentix-e.github.io/agentix-timesfm-ts/api/modules/timesfm_core.html) · 📊 [Benchmark](https://agentix-e.github.io/agentix-timesfm-ts/benchmark/) · 📈 [Coverage](https://agentix-e.github.io/agentix-timesfm-ts/coverage/) · 💻 [Source](https://github.com/AgentiX-E/agentix-timesfm-ts)

```bash
npm install @agentix-e/timesfm-core
```

## Quick Start

```typescript
import { TimesFMModel, createForecastConfig } from '@agentix-e/timesfm-core';

// 1. Load the model (requires a TimesFM 2.5 ONNX file — see below)
const model = await TimesFMModel.fromPretrained({
  modelPath: './timesfm-2.5.onnx',
});

// 2. Configure forecast behaviour
model.compile(
  createForecastConfig({
    maxContext: 1024,
    maxHorizon: 256,
  }),
);

// 3. Forecast — inputs can be any length, with NaN gaps
const { pointForecast, quantileForecast } = await model.forecast(24, [
  new Float32Array([100, 102, 105, NaN, 110, 108 /* ... */]),
]);

// 4. Use results
console.log('Median    :', Array.from(pointForecast[0]));
console.log('80% CI low:', Array.from(quantileForecast[0][1])); // Q10
console.log('80% CI hi :', Array.from(quantileForecast[0][9])); // Q90

// 5. Clean up
await model.dispose();
```

## Getting the Model

The model file (~885 MB) is NOT included in the npm package. You have several options:

| Method            | Command                                                                                                  |
| ----------------- | -------------------------------------------------------------------------------------------------------- |
| **Auto-download** | `modelPath = await downloadModel()` — fetches from GitHub Releases                                       |
| **CLI**           | `npx @agentix-e/timesfm-cli setup`                                                                       |
| **Manual export** | `python scripts/export-onnx.py` from the [project repo](https://github.com/AgentiX-E/agentix-timesfm-ts) |

```typescript
import { downloadModel } from '@agentix-e/timesfm-core';
const modelPath = await downloadModel(); // → ~/.cache/agentix-timesfm-ts/timesfm-2.5.onnx
```

## API Reference

### TimesFMModel

The main entry point. Implements the full inference pipeline.

```typescript
const model = await TimesFMModel.fromPretrained({
  modelPath: './timesfm-2.5.onnx',
  executionProvider: 'cpu', // 'cpu' | 'cuda' | 'dml'
});

model.compile(createForecastConfig({ maxContext: 1024, maxHorizon: 256 }));

const result = await model.forecast(horizon, inputs, {
  signal: abortController.signal, // optional AbortSignal
  onProgress: (e) => console.log(e), // optional progress callback
});

await model.dispose();
```

### ForecastConfig

| Parameter                   | Type    | Default | Description                                                |
| --------------------------- | ------- | ------- | ---------------------------------------------------------- |
| `maxContext`                | number  | 1024    | Maximum context length (rounded to 32×)                    |
| `maxHorizon`                | number  | 256     | Maximum forecast horizon (rounded to 128×)                 |
| `normalizeInputs`           | boolean | true    | Z-score normalise each series before inference             |
| `forceFlipInvariance`       | boolean | true    | Ensure `f(−x) = −f(x)` — trades 2× latency for calibration |
| `useContinuousQuantileHead` | boolean | true    | Finer prediction intervals at longer horizons              |
| `fixQuantileCrossing`       | boolean | true    | Enforce monotonic quantiles: q10 ≤ q20 ≤ … ≤ q90           |
| `inferIsPositive`           | boolean | true    | Clamp forecasts ≥ 0 when input is all-non-negative         |
| `returnBackcast`            | boolean | false   | Return model's reconstruction of historical data           |
| `perCoreBatchSize`          | number  | 1       | Sequences processed per batch                              |

### ForecastOutput

| Field              | Shape                      | Description                                             |
| ------------------ | -------------------------- | ------------------------------------------------------- |
| `pointForecast`    | `[numSeries, horizon]`     | Median forecast per series                              |
| `quantileForecast` | `[numSeries, horizon, 10]` | Full quantile distribution per series                   |
| `backcast`         | `[numSeries, contextLen]`  | Historical reconstruction (when `returnBackcast: true`) |

Quantile indices (use `QUANTILE_INDICES`):

```
0 = mean   1 = Q10   2 = Q20   3 = Q30   4 = Q40
5 = Q50    6 = Q60   7 = Q70   8 = Q80   9 = Q90
```

### Quantile Helpers

```typescript
import { getQuantile, getPredictionInterval, QUANTILE_INDICES } from '@agentix-e/timesfm-core';

const q10 = getQuantile(result, 0, QUANTILE_INDICES.Q10);
const { lower, upper } = getPredictionInterval(result, 0, 0.8);
```

### Evaluation Metrics

```typescript
import {
  mae,
  rmse,
  mape,
  smape,
  mase,
  r2Score,
  picCoverage,
  piWidth,
} from '@agentix-e/timesfm-core';

const error = mae(actualValues, pointForecast[0]);
const coverage = picCoverage(actualValues, quantileForecast[0][1], quantileForecast[0][9]);
```

| Function                            | Description                                  |
| ----------------------------------- | -------------------------------------------- |
| `mae(actual, predicted)`            | Mean Absolute Error                          |
| `rmse(actual, predicted)`           | Root Mean Square Error                       |
| `mape(actual, predicted)`           | Mean Absolute Percentage Error               |
| `smape(actual, predicted)`          | Symmetric MAPE                               |
| `mase(actual, predicted, naive)`    | Mean Absolute Scaled Error vs naive baseline |
| `r2Score(actual, predicted)`        | R² coefficient of determination              |
| `picCoverage(actual, lower, upper)` | Prediction Interval Coverage                 |
| `piWidth(lower, upper)`             | Mean Prediction Interval Width               |

### Cancellation & Progress

```typescript
const ac = new AbortController();

const result = await model.forecast(256, inputs, {
  signal: ac.signal,
  onProgress: (event) => {
    // event: { phase, step, total, batchIndex?, totalBatches? }
    // phase: 'preprocess' | 'prefill' | 'decode' | 'flip' | 'postprocess'
  },
});

ac.abort(); // cancels a running forecast
```

### Model Downloader

```typescript
import { downloadModel, isModelCached, getCachedModelPath } from '@agentix-e/timesfm-core';

const path = await downloadModel({
  dest: './my-model.onnx', // custom path
  force: true, // re-download
  onProgress: (received, total, speed) => {
    /* MB */
  },
});
```

## System Requirements

| Component      | Minimum   | Recommended  |
| -------------- | --------- | ------------ |
| Node.js        | ≥ 20      | ≥ 22         |
| RAM            | 4 GB      | 8 GB+        |
| Disk (model)   | 1 GB free | SSD          |
| GPU (optional) | 2 GB VRAM | 4 GB+ (CUDA) |

## License

Apache 2.0 · Based on [Google Research TimesFM](https://github.com/google-research/timesfm)
