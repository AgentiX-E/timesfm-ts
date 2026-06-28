# @agentix-e/timesfm-web

> 🌐 **TimesFM in the browser** — zero-install time-series forecasting via WebAssembly.

Runs the Google Research TimesFM 2.5 200M model directly in the browser using [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web). Supports **WebGPU** (fastest), **WASM** (universal), and **WebGL** (legacy) backends.

📚 [API Documentation](https://agentix-e.github.io/agentix-timesfm-ts/api/modules/timesfm_web.html) · 📊 [Benchmark](https://agentix-e.github.io/agentix-timesfm-ts/benchmark/) · 📈 [Coverage](https://agentix-e.github.io/agentix-timesfm-ts/coverage/) · 💻 [Source](https://github.com/AgentiX-E/agentix-timesfm-ts)

## Installation

```bash
npm install @agentix-e/timesfm-core @agentix-e/timesfm-web onnxruntime-web
```

> **Note**: `onnxruntime-web` must be installed as a peer dependency. The bundled WASM files (~3 MB) are loaded at runtime.

## Quick Start

```typescript
import { TimesFMModel, createForecastConfig, TIMESFM_25_CONFIG } from '@agentix-e/timesfm-core';
import { TimesFMWebInferenceEngine, loadModelFromUrl } from '@agentix-e/timesfm-web';

// 1. Download model with progress tracking
const { buffer } = await loadModelFromUrl('/models/timesfm-2.5.onnx', {
  onProgress: (received, total) => {
    const pct = ((received / total) * 100).toFixed(0);
    console.log(`Downloading model... ${pct}%`);
  },
});

// 2. Create web engine (auto-detects best backend: WebGPU → WASM)
const engine = new TimesFMWebInferenceEngine(TIMESFM_25_CONFIG);
await engine.load(buffer);

// 3. Create model with the web engine injected
const model = await TimesFMModel.fromPretrained({
  modelPath: '/models/timesfm-2.5.onnx',
  engine,
});

// 4. Forecast
model.compile(
  createForecastConfig({
    maxContext: 512,
    maxHorizon: 128,
  }),
);

const { pointForecast } = await model.forecast(24, [inputData]);
console.log('Forecast:', pointForecast[0]);

// 5. Clean up
await model.dispose();
```

## Execution Providers

| Provider      | Speed                          | Browser Support        |
| ------------- | ------------------------------ | ---------------------- |
| **WebGPU** 🚀 | Fastest (near-native)          | Chrome 113+, Edge 113+ |
| **WASM** ⚡   | Good (2-5× slower than native) | All modern browsers    |
| **WebGL** 🐢  | Legacy                         | Older browsers         |

The engine auto-detects the best available provider. You can override:

```typescript
const engine = new TimesFMWebInferenceEngine(config, ['wasm']); // force WASM
```

## API

### `checkModelAvailability(url)`

HEAD request to check if a model exists at a URL. Returns the Content-Length in bytes, or `null` if inaccessible.

```typescript
const sizeBytes = await checkModelAvailability('https://cdn.example.com/model.onnx');
if (sizeBytes !== null) console.log(`Model is ${(sizeBytes / 1e6).toFixed(0)} MB`);
```

### `TimesFMWebInferenceEngine`

Implements `IInferenceEngine` from `@agentix-e/timesfm-core`.

- `constructor(config: ModelConfig, providers?: Array<'webgpu' | 'wasm' | 'webgl'>)`
- `load(modelPath: string | ArrayBuffer): Promise<void>`
- `forward(inputs: Float32Array[], masks: Uint8Array[]): Promise<RawModelOutput>`
- `dispose(): Promise<void>`
- `isLoaded(): boolean`

### `loadModelFromUrl(url, options?)`

Downloads a model from a URL with progress tracking and cancellation.

- `url: string` — URL of the ONNX model
- `options.onProgress?: (received: number, total: number) => void`
- `options.signal?: AbortSignal`
- Returns `Promise<{ buffer: ArrayBuffer, sizeBytes: number, contentType: string }>`

## Bundle Size

The package itself is ~5 KB gzipped. The `onnxruntime-web` WASM binary (~3 MB) is loaded asynchronously at runtime and cached by the browser.

## Important Notes

- The 885 MB ONNX model must be downloaded once before first use. Consider hosting it on a CDN or static file server.
- WebGPU provides the best performance but requires a compatible browser and GPU.
- Model warmup: the first `forecast()` call triggers WASM JIT compilation and may be slower.
- Worker threads: For non-blocking inference in the UI thread, load the model in a Web Worker.
