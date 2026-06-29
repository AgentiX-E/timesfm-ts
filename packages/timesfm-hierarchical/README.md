# @agentix-e/timesfm-hierarchical

> Hierarchical time series reconciliation engine — bottom-up, OLS, WLS, and MinT forecast reconciliation.

[![npm](https://img.shields.io/npm/v/@agentix-e/timesfm-hierarchical?color=blue)](https://www.npmjs.com/package/@agentix-e/timesfm-hierarchical)
[![API Docs](https://img.shields.io/badge/docs-TypeDoc-blue)](https://agentix-e.github.io/agentix-timesfm-ts/api/modules/timesfm-hierarchical.html)

## Overview

`@agentix-e/timesfm-hierarchical` provides forecast reconciliation for hierarchical time series structures. When forecasting at multiple levels of a hierarchy (e.g., product → category → total), base forecasts at each level are mathematically inconsistent — the sum of child forecasts doesn't equal the parent forecast. This package reconciles them via optimal combination methods.

### Reconciliation Strategies

| Strategy   | Description                                                     |
| ---------- | --------------------------------------------------------------- |
| Bottom-Up  | Forecast at bottom level, aggregate up                          |
| OLS        | Ordinary Least Squares — minimizes $\|\tilde{y} - S\beta\|^2$   |
| WLS        | Weighted Least Squares — weights by forecast variance           |
| MinT (Cov) | Minimum Trace — minimizes reconciliation error using covariance |

## Installation

```bash
npm install @agentix-e/timesfm-hierarchical
```

Requires `@agentix-e/timesfm-core` (peer dependency).

## Quick Start

```typescript
import { TimesFMModel, downloadModel, createForecastConfig } from '@agentix-e/timesfm-core';
import { reconcileForecast, buildSummingMatrix } from '@agentix-e/timesfm-hierarchical';

const model = await TimesFMModel.fromPretrained({
  modelPath: await downloadModel(),
});
model.compile(createForecastConfig({ maxContext: 512, maxHorizon: 128 }));

// Define hierarchy: 3-level tree (Total → {West, East} → 4 stores)
const result = await reconcileForecast(model, {
  hierarchy: {
    nodes: [
      { id: 'total', parentId: null },
      { id: 'west', parentId: 'total' },
      { id: 'east', parentId: 'total' },
      { id: 's1', parentId: 'west' },
      { id: 's2', parentId: 'west' },
      { id: 's3', parentId: 'east' },
      { id: 's4', parentId: 'east' },
    ],
  },
  inputs: {
    total: totalSeries,
    west: westSeries,
    east: eastSeries,
    s1: store1Series,
    s2: store2Series,
    s3: store3Series,
    s4: store4Series,
  },
  horizon: 24,
  reconcile: { strategy: 'mint' },
});

// Coherent reconciled forecasts
console.log(result.reconciled.total.pointForecast); // Aggregate
console.log(result.reconciled.s1.pointForecast); // Bottom-level
console.log(result.reconciled.total.quantileForecast); // Full quantile bands

// Verify coherence: total ≈ west + east
assertCoherent(result.reconciled.total, result.reconciled.west, result.reconciled.east);
```

## API Documentation

📚 **Full API reference**: [agentix-e.github.io/agentix-timesfm-ts/api/modules/timesfm-hierarchical.html](https://agentix-e.github.io/agentix-timesfm-ts/api/modules/timesfm-hierarchical.html)

Key exports:

- `reconcileForecast` — Main entry: forecasts all nodes via TimesFM then reconciles
- `buildSummingMatrix` — Constructs the hierarchical summing matrix S (m×n)
- `reconcileBaseForecasts` — Low-level reconciliation given base forecasts + summing matrix
- `computeProjectionMatrix` — Computes the optimal projection matrix P for a given strategy

## License

Apache 2.0
