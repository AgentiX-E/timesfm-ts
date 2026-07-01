# @agentix-e/timesfm-node

Node.js ONNX Runtime inference engine for TimesFM — zero-shot time series forecasting.

[![npm](https://img.shields.io/npm/v/@agentix-e/timesfm-node?color=blue)](https://www.npmjs.com/package/@agentix-e/timesfm-node)
[![API Docs](https://img.shields.io/badge/docs-TypeDoc-blue)](https://agentix-e.github.io/timesfm-ts/api/classes/timesfm_node.TimesFMNodeEngine.html)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../../LICENSE)

📚 [API Documentation](https://agentix-e.github.io/timesfm-ts/api/classes/timesfm_node.TimesFMNodeEngine.html) · 📊 [Benchmark](https://agentix-e.github.io/timesfm-ts/benchmark/) · 📈 [Coverage](https://agentix-e.github.io/timesfm-ts/coverage/) · 💻 [Source](https://github.com/AgentiX-E/timesfm-ts)

## Overview

`@agentix-e/timesfm-node` provides the default Node.js inference engine for TimesFM.
It implements the `IInferenceEngine` interface from `@agentix-e/timesfm-core` using
[onnxruntime-node](https://www.npmjs.com/package/onnxruntime-node).

This package is automatically loaded by `TimesFMModel.fromPretrained()` when no
custom engine is provided. Browser users who inject `TimesFMWebInferenceEngine`
(from `@agentix-e/timesfm-web`) never install this package or the native ONNX
Runtime addon.

```bash
npm install @agentix-e/timesfm-node
```

## Usage

```typescript
import { TimesFMModel, createForecastConfig } from '@agentix-e/timesfm-core';
// The model automatically loads timesfm-node when no engine is provided
const model = await TimesFMModel.fromPretrained({
  modelPath: './timesfm-2.5.onnx',
});
```

**Explicit engine creation:**

```typescript
import { TimesFMNodeEngine, createDefaultEngine } from '@agentix-e/timesfm-node';

// Option A: Direct construction
const engine = new TimesFMNodeEngine(TIMESFM_25_CONFIG, {
  executionProvider: 'cpu', // 'cpu' | 'cuda' | 'dml'
  intraOpNumThreads: 4,
});

// Option B: Factory (used internally by TimesFMModel)
const engine = createDefaultEngine(config, { executionProvider: 'cpu' });

const model = await TimesFMModel.fromPretrained({
  modelPath: './timesfm-2.5.onnx',
  engine,
});
```

## Exports

| Export                | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| `TimesFMNodeEngine`   | `IInferenceEngine` implementation backed by `onnxruntime-node` |
| `createDefaultEngine` | Factory function for creating the default engine               |

## Execution Providers

| Provider | Description                                                |
| -------- | ---------------------------------------------------------- |
| `cpu`    | CPUExecutionProvider (default)                             |
| `cuda`   | CUDAExecutionProvider (requires NVIDIA GPU + CUDA drivers) |
| `dml`    | DmlExecutionProvider (requires DirectML on Windows)        |

## System Requirements

- Node.js ≥ 22
- 4+ GB RAM
- 1 GB disk space for the ONNX model (~885 MB)
- CUDA 12+ (optional, for GPU inference)
