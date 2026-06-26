# @agentix-e/timesfm-xreg

Exogenous covariates (XReg) extension for TimesFM — boost forecast accuracy by incorporating external variables: weather, holidays, prices, promotions, or any domain-specific signals.

```bash
npm install @agentix-e/timesfm-xreg
```

Requires `@agentix-e/timesfm-core` as a peer dependency.

## Quick Start

```typescript
import { TimesFMModel, createForecastConfig } from '@agentix-e/timesfm-core';
import { forecastWithCovariates } from '@agentix-e/timesfm-xreg';

const model = await TimesFMModel.fromPretrained({ modelPath: './timesfm-2.5.onnx' });
model.compile(createForecastConfig({ maxContext: 512, maxHorizon: 128 }));

const result = await forecastWithCovariates(model, {
  // The target time series
  inputs: [salesData],

  // Numerical variables that change over time — length = context + horizon
  dynamicNumericalCovariates: {
    temperature: [new Float32Array(640)], // 512 context + 128 horizon
    price: [new Float32Array(640)],
  },

  // Categorical variables that change over time
  dynamicCategoricalCovariates: {
    dayOfWeek: [['Mon', 'Tue', 'Wed' /* ... length = 640 */]],
    isHoliday: [[0, 0, 1 /* ... */]],
  },

  // Static variables — one value per series
  staticNumericalCovariates: {
    storeSize: [1500],
  },
  staticCategoricalCovariates: {
    storeType: ['flagship'],
  },

  xregMode: 'xreg + timesfm',
  ridge: 0.1,
  normalizeXregTargetPerInput: true,
  maxRowsPerCol: 100,
});

// result.pointForecast[0]  — final forecast (TimesFM residual + XReg contribution)
// result.xregOutputs[0]    — pure covariate contribution
// result.quantileForecast   — quantile bands adjusted by XReg offset
```

## Covariate Types

| Type                           | Length                         | Example                      |
| ------------------------------ | ------------------------------ | ---------------------------- |
| `dynamicNumericalCovariates`   | `context + horizon` per series | Temperature, price, ad spend |
| `dynamicCategoricalCovariates` | `context + horizon` per series | Day of week, promotion flag  |
| `staticNumericalCovariates`    | 1 per series                   | Store area, location lat/lng |
| `staticCategoricalCovariates`  | 1 per series                   | Region, category, tier       |

## Modes

| Mode             | Algorithm                                                                      | Best for                                                                    |
| ---------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| `xreg + timesfm` | Fit linear model on targets → forecast **residuals** with TimesFM → combine    | Covariates explain most of the signal (e.g. promotions driving sales)       |
| `timesfm + xreg` | Forecast with TimesFM → fit linear model on residuals (via backcast) → combine | TimesFM already captures the main pattern; covariates correct the remainder |

## Parameters

| Parameter                     | Type                                   | Default            | Description                                                  |
| ----------------------------- | -------------------------------------- | ------------------ | ------------------------------------------------------------ |
| `inputs`                      | `Float32Array[]`                       | required           | Target time series                                           |
| `xregMode`                    | `'xreg + timesfm' \| 'timesfm + xreg'` | `'xreg + timesfm'` | Fitting order                                                |
| `ridge`                       | `number`                               | `0`                | L2 regularisation strength for Ridge regression              |
| `normalizeXregTargetPerInput` | `boolean`                              | `false`            | Z-score normalise each series before fitting                 |
| `maxRowsPerCol`               | `number`                               | `0` (unlimited)    | Sub-sample rows to `maxRowsPerCol × cols` for memory control |

## Standalone OneHotEncoder

```typescript
import { OneHotEncoder } from '@agentix-e/timesfm-xreg';

const encoder = new OneHotEncoder({ drop: 'first', handleUnknown: 'ignore' });
encoder.fit(['a', 'b', 'c', 'a']);
const encoded = encoder.transform(['b', 'd']); // [[0, 1], [0, 0]]
```

Scikit-learn compatible. Supports `handleUnknown: 'ignore' | 'error'`.

## License

Apache 2.0
