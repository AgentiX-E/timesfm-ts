# @agentix-e/timesfm-xreg

> Covariate regression extension for TimesFM — Ridge regression + OneHot encoding for exogenous variables.

[![npm](https://img.shields.io/npm/v/@agentix-e/timesfm-xreg?color=blue)](https://www.npmjs.com/package/@agentix-e/timesfm-xreg)
[![API Docs](https://img.shields.io/badge/docs-TypeDoc-blue)](https://agentix-e.github.io/agentix-timesfm-ts/api/)

## Overview

`@agentix-e/timesfm-xreg` extends the TimesFM forecasting pipeline with exogenous covariate support. It provides a scikit-learn-compatible Ridge regression engine and OneHot encoder for categorical variables, used in the `forecastWithCovariates()` workflow.

### Capabilities

- **Dynamic numerical covariates** — Time-varying features (e.g., weather, promotions, events)
- **Static numerical covariates** — Per-series constant features (e.g., store ID, location)
- **Categorical covariates** — OneHot-encoded categorical variables with scikit-learn compatibility
- **Ridge regression** — L2-regularized linear regression for covariate modeling
- **XReg + TimesFM hybrid** — Combine covariate regression with TimesFM residuals

## Installation

```bash
npm install @agentix-e/timesfm-xreg
```

Requires `@agentix-e/timesfm-core` (peer dependency).

## Quick Start

```typescript
import { TimesFMModel, downloadModel, createForecastConfig } from '@agentix-e/timesfm-core';

const model = await TimesFMModel.fromPretrained({
  modelPath: await downloadModel(),
});
model.compile(createForecastConfig({ maxContext: 512, maxHorizon: 128 }));

const result = await model.forecastWithCovariates({
  inputs: [new Float32Array(/* historical values */)],
  dynamicNumericalCovariates: {
    temperature: [new Float32Array(/* future temperature values */)],
  },
  staticNumericalCovariates: {
    value: [new Float32Array([42])],
  },
  xregMode: 'xreg + timesfm',
});
```

## API Documentation

📚 **Full API reference**: [agentix-e.github.io/agentix-timesfm-ts/api/](https://agentix-e.github.io/agentix-timesfm-ts/api/)

Key exports:

- `forecastWithCovariates` — Main entry point for covariate-aware forecasting
- `XRegEngine` — Ridge regression engine with design matrix construction
- `OneHotEncoder` — Scikit-learn-compatible OneHot encoder

## License

Apache 2.0
