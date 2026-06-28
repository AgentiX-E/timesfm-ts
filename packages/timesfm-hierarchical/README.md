# @agentix-e/timesfm-hierarchical

Hierarchical time series reconciliation for TimesFM — ensures aggregate forecasts are mathematically consistent across all levels of a tree-structured hierarchy.

[![npm](https://img.shields.io/npm/v/@agentix-e/timesfm-hierarchical?color=blue)](https://www.npmjs.com/package/@agentix-e/timesfm-hierarchical)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../../LICENSE)

📚 [API Documentation](https://agentix-e.github.io/agentix-timesfm-ts/api/modules/_agentix_e_timesfm_hierarchical.html) · 💻 [Source](https://github.com/AgentiX-E/agentix-timesfm-ts)

```bash
npm install @agentix-e/timesfm-hierarchical
```

Requires `@agentix-e/timesfm-core` as a peer dependency.

## Why Reconciliation?

When you forecast each node in a hierarchy independently, the bottom-up sum rarely matches the top-level forecast. Reconciliation transforms the independent base forecasts into a **coherent** set where:

```
Total = RegionA + RegionB = Store1 + Store2 + Store3 + Store4
```

This guarantees decision-makers see consistent numbers at every aggregation level.

## Quick Start

```typescript
import { TimesFMModel, createForecastConfig } from '@agentix-e/timesfm-core';
import { reconcileForecast } from '@agentix-e/timesfm-hierarchical';

const model = await TimesFMModel.fromPretrained({ modelPath: './timesfm-2.5.onnx' });
model.compile(createForecastConfig({ maxContext: 512, maxHorizon: 128 }));

const result = await reconcileForecast(model, {
  // Define the hierarchy tree
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
  // Historical data for every node
  inputs: {
    total: totalData,
    west: westData,
    east: eastData,
    s1: store1Data,
    s2: store2Data,
    s3: store3Data,
    s4: store4Data,
  },
  horizon: 24,
  reconcile: { strategy: 'mint' },
});

// Access per-node reconciled forecasts
const totalForecast = result.reconciled.total.pointForecast;
const store1CI = result.reconciled.s1.quantileForecast; // [10 quantiles]
```

## Reconciliation Strategies

| Strategy           | Algorithm                       | Speed  | Accuracy | Use Case                                           |
| ------------------ | ------------------------------- | ------ | -------- | -------------------------------------------------- |
| `'bu'` — Bottom-Up | P = [0 \| I_n]                  | ⚡⚡⚡ | ★★☆☆☆    | Fastest; uses only bottom-level forecasts          |
| `'ols'` — OLS      | P = (SᵀS)⁻¹Sᵀ                   | ⚡⚡   | ★★★☆☆    | Simple, ignores error correlation                  |
| `'wls'` — WLS      | P = (SᵀW⁻¹S)⁻¹SᵀW⁻¹, W diagonal | ⚡⚡   | ★★★★☆    | Weights each level by residual variance            |
| `'mint'` — MinT ⭐ | P = (SᵀW⁻¹S)⁻¹SᵀW⁻¹, W full cov | ⚡     | ★★★★★    | Best theoretical properties (default, recommended) |

## Parameters

| Parameter                      | Type                               | Default  | Description                                   |
| ------------------------------ | ---------------------------------- | -------- | --------------------------------------------- |
| `hierarchy`                    | `HierarchyDefinition`              | required | Tree topology                                 |
| `inputs`                       | `Record<string, Float32Array>`     | required | Historical series per node                    |
| `horizon`                      | `number`                           | required | Forecast horizon                              |
| `reconcile.strategy`           | `'bu' \| 'ols' \| 'wls' \| 'mint'` | `'mint'` | Reconciliation strategy                       |
| `reconcile.residualCovariance` | `number[][]`                       | auto     | Residual covariance (m × m) for WLS/MinT      |
| `reconcile.ridge`              | `number`                           | `1e-6`   | Ridge regularisation for matrix invertibility |

## Standalone Utilities

```typescript
import {
  buildSummingMatrix,
  reconcileBaseForecasts,
  computeProjectionMatrix,
} from '@agentix-e/timesfm-hierarchical';

// Build the S-matrix from a hierarchy definition
const summing = buildSummingMatrix(hierarchy);
// summing.S → m × n matrix
// summing.allNodeIds → topologically sorted node ids
// summing.bottomNodeIds → leaf node ids

// Reconcile pre-computed base forecasts (no model required)
const result = reconcileBaseForecasts(summing, baseForecasts, { strategy: 'ols' });
// result.reconciled → coherent forecasts per node
// result.projectionMatrix → P for diagnostics
```

## Mathematical Reference

Reconciled forecasts: **ŷ_h = S · P · ŷ_b**

- S — m × n summing matrix (buildSummingMatrix)
- P — n × m projection matrix (computeProjectionMatrix)
- ŷ_b — m-length stacked base forecast vector

Hyndman, R.J., Ahmed, R.A., Athanasopoulos, G., & Shang, H.L. (2011).
"Optimal combination forecasts for hierarchical time series."
_Computational Statistics & Data Analysis_, 55(9), 2579–2589.

## License

Apache 2.0
