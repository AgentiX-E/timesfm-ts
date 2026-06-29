# @agentix-e/timesfm-web

> Browser inference engine for TimesFM — zero-shot forecasting via WebAssembly, WebGPU, or WebGL.

[![npm](https://img.shields.io/npm/v/@agentix-e/timesfm-web?color=orange)](https://www.npmjs.com/package/@agentix-e/timesfm-web)
[![API Docs](https://img.shields.io/badge/docs-TypeDoc-blue)](https://agentix-e.github.io/agentix-timesfm-ts/api/)

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
import { TimesFMWebEngine } from '@agentix-e/timesfm-web';
import { createForecastConfig } from '@agentix-e/timesfm-core';

const engine = new TimesFMWebEngine();

// Load model from URL (streaming fetch)
await engine.load('https://cdn.example.com/timesfm-2.5.onnx');

const fc = createForecastConfig({ maxContext: 1024, maxHorizon: 256 });

const output = await engine.forecast(
  [
    new Float32Array([
      /* historical values */
    ]),
  ],
  fc,
);
```

## API Documentation

📚 **Full API reference**: [agentix-e.github.io/agentix-timesfm-ts/api/](https://agentix-e.github.io/agentix-timesfm-ts/api/)

Key exports:

- `TimesFMWebEngine` — Browser inference engine (load, forward, dispose)
- `loadWebModel` — Fetch-based model loader with progress tracking

## License

Apache 2.0
