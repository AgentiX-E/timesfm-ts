# @agentix-e/timesfm-hierarchical

> Hierarchical time series reconciliation engine — bottom-up, top-down, OLS, WLS, and MinT forecast reconciliation.

[![npm](https://img.shields.io/npm/v/@agentix-e/timesfm-hierarchical?color=blue)](https://www.npmjs.com/package/@agentix-e/timesfm-hierarchical)
[![API Docs](https://img.shields.io/badge/docs-TypeDoc-blue)](https://agentix-e.github.io/agentix-timesfm-ts/api/)

## Overview

`@agentix-e/timesfm-hierarchical` provides forecast reconciliation for hierarchical time series structures. When forecasting at multiple levels of a hierarchy (e.g., product → category → total), base forecasts at each level are mathematically inconsistent — the sum of child forecasts doesn't equal the parent forecast. This package reconciles them via optimal combination methods.

### Reconciliation Strategies

| Strategy   | Description                                                     |
| ---------- | --------------------------------------------------------------- | --- | ------------------ | --- | --- |
| Bottom-Up  | Forecast at bottom level, aggregate up                          |
| Top-Down   | Forecast at top level, disaggregate down by proportions         |
| OLS        | Ordinary Least Squares — minimizes $                            |     | \tilde{y} - S\beta |     | ^2$ |
| WLS        | Weighted Least Squares — weights by forecast variance           |
| MinT (Cov) | Minimum Trace — minimizes reconciliation error using covariance |

## Installation

```bash
npm install @agentix-e/timesfm-hierarchical
```

Requires `@agentix-e/timesfm-core` (peer dependency).

## Quick Start

```typescript
import { HierarchicalEngine, SummingMatrix } from '@agentix-e/timesfm-hierarchical';

// Define hierarchy: 3 bottom-level, 2 middle, 1 top
const S = SummingMatrix.fromTree([
  { name: 'Total', children: ['A', 'B'] },
  { name: 'A', children: ['A1', 'A2'] },
  { name: 'B', children: ['B1'] },
]);

const engine = new HierarchicalEngine(S);

// Base forecasts at each level (6 nodes total)
const baseForecasts = new Float32Array([
  /* 6 forecasts */
]);

// Reconcile via MinT
const reconciled = engine.reconcile(baseForecasts, 'mint_cov');
```

## API Documentation

📚 **Full API reference**: [agentix-e.github.io/agentix-timesfm-ts/api/](https://agentix-e.github.io/agentix-timesfm-ts/api/)

Key exports:

- `HierarchicalEngine` — Reconciliation engine with all strategies
- `SummingMatrix` — Hierarchical summing matrix builder
- `reconcileForecasts` — Low-level reconciliation function

## License

Apache 2.0
