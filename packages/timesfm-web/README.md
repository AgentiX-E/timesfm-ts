# @agentix-e/timesfm-web

> Browser inference engine for TimesFM — zero-shot forecasting via WebAssembly, WebGPU, or WebGL.

[![npm](https://img.shields.io/npm/v/@agentix-e/timesfm-web?color=orange)](https://www.npmjs.com/package/@agentix-e/timesfm-web)
[![API Docs](https://img.shields.io/badge/docs-TypeDoc-blue)](https://agentix-e.github.io/timesfm-ts/api/modules/timesfm-web.html)

## Overview

`@agentix-e/timesfm-web` enables TimesFM inference directly in the browser using `onnxruntime-web`. It supports multiple backends (WASM, WebGPU, WebGL) and loads the 885 MB ONNX model via `fetch()` with streaming.

### Backend Support

| Backend | Browser Support        | Performance |
| ------- | ---------------------- | ----------- |
| WebGPU  | Chrome 113+, Edge 113+ | 🚀 Fastest  |
| WASM    | All modern browsers    | ✅ Default  |
| WebGL   | Fallback               | ⚠️ Slow     |

> **Memory note**: The 885 MB model must fit in browser WASM memory (~4 GB limit).

## Installation

```bash
npm install @agentix-e/timesfm-web onnxruntime-web
```

`onnxruntime-web` is a peer dependency — install it alongside.

## Quick Start

```typescript
import { TimesFMModel, createForecastConfig, TIMESFM_25_CONFIG } from '@agentix-e/timesfm-core';
import { TimesFMWebInferenceEngine, loadModelFromUrl } from '@agentix-e/timesfm-web';

// 1. Download the model via fetch() with progress tracking
const { buffer } = await loadModelFromUrl('https://cdn.example.com/timesfm-2.5.onnx', {
  onProgress: (received, total) => console.log(`${received}/${total}`),
});

// 2. Create web engine and load model
const engine = new TimesFMWebInferenceEngine(TIMESFM_25_CONFIG);
await engine.load(buffer);

// 3. Create model with injected web engine (DI pattern)
const model = await TimesFMModel.fromPretrained({
  modelPath: '/models/timesfm-2.5.onnx',
  engine,
});

// 4. Compile and forecast
model.compile(createForecastConfig({ maxContext: 1024, maxHorizon: 256 }));
const result = await model.forecast(24, [
  new Float32Array([
    /* historical values */
  ]),
]);

console.log(result.pointForecast); // Shape: [1, 24]
console.log(result.quantileForecast); // Shape: [1, 10, 24]

await model.dispose();
```

## API Documentation

📚 **Full API reference**: [agentix-e.github.io/timesfm-ts/api/modules/timesfm-web.html](https://agentix-e.github.io/timesfm-ts/api/modules/timesfm-web.html)

Key exports:

- `TimesFMWebInferenceEngine` — Browser inference engine implementing `IInferenceEngine` (WebGPU / WASM / WebGL)
- `loadModelFromUrl` — Fetch-based model loader with streaming progress callback
- `checkModelAvailability` — HEAD request check for model availability
- `WebEngineLogger` — Custom logger interface for browser engine diagnostics

## License

Apache 2.0
