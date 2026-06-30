# @agentix-e/timesfm-core

> Core inference engine for TimesFM — zero-shot time series forecasting powered by ONNX Runtime.

[![npm](https://img.shields.io/npm/v/@agentix-e/timesfm-core?color=blue)](https://www.npmjs.com/package/@agentix-e/timesfm-core)
[![API Docs](https://img.shields.io/badge/docs-TypeDoc-blue)](https://agentix-e.github.io/timesfm-ts/api/modules/timesfm-core.html)

## Overview

`@agentix-e/timesfm-core` is the heart of timesfm-ts — a production-grade Node.js/TypeScript implementation of Google Research's TimesFM 2.5 (200M parameter decoder-only transformer). It provides zero-shot univariate time-series forecasting with calibrated prediction intervals, no training required.

### Architecture

```
Raw Series → [NaN Handler] → [Pad/Truncate] → [Patch Split] → [RevIN Norm]
→ [ONNX Runtime] → [RevIN Denorm] → [Flip Invariance] → [Quantile Calibration]
→ Forecasts
```

## Installation

```bash
npm install @agentix-e/timesfm-core
```

Requires Node.js ≥ 20.

## Quick Start

```typescript
import { TimesFMModel, downloadModel, createForecastConfig } from '@agentix-e/timesfm-core';

// Auto-download model (~885 MB, first time only, cached thereafter)
const modelPath = await downloadModel();

const model = await TimesFMModel.fromPretrained({ modelPath });
model.compile(createForecastConfig({ maxContext: 1024, maxHorizon: 256 }));

const { pointForecast, quantileForecast } = await model.forecast(24, [
  new Float32Array([1, 2, 3 /* ... */]),
]);

console.log(pointForecast); // Shape: [1, 24]
console.log(quantileForecast); // Shape: [1, 10, 24]

await model.dispose();
```

## API Documentation

📚 **Full API reference**: [agentix-e.github.io/timesfm-ts/api/modules/timesfm-core.html](https://agentix-e.github.io/timesfm-ts/api/modules/timesfm-core.html)

Key exports:

- `TimesFMModel` — Main model class (`fromPretrained`, `compile`, `forecast`, `forecastWithCovariates`)
- `downloadModel` / `defaultModelPath` / `isModelCached` — Model download & cache management
- `createForecastConfig` / `validateAndNormalizeConfig` — Configuration builder
- `TimesFMInferenceEngine` — ONNX Runtime inference engine
- `preprocess` / `postProcess` — Preprocessing & postprocessing pipelines
- `decode` — Autoregressive decode loop
- `mae`, `rmse`, `mape`, `smape`, `mase`, `r2Score` — Evaluation metrics
- Utility exports: `cleanSeries`, `stripLeadingNaNs`, `linearInterpolateNaNs`, `computeStats`, `revin`, `revinBatch`

## Model Download

```typescript
import { downloadModel } from '@agentix-e/timesfm-core';

// Default: ~/.cache/timesfm-ts/timesfm-2.5.onnx
const path = await downloadModel();

// With proxy (corporate network)
const path = await downloadModel({
  proxy: { url: 'http://proxy.company.com:8080', username: 'user', password: 'pass' },
  onProgress: (received, total, speed) => console.log(`${received}/${total} MB @ ${speed} MB/s`),
});
```

Proxy can also be configured via environment variables:

- `TIMESFM_PROXY_URL` / `TIMESFM_PROXY_USERNAME` / `TIMESFM_PROXY_PASSWORD`
- `TIMESFM_PROXY_PASSWORD_FILE` — read password from a file (Docker/K8s secrets)
- Standard `HTTPS_PROXY` / `HTTP_PROXY`

## License

Apache 2.0
