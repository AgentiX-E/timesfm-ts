/**
 * Tests for the XReg covariate forecasting engine.
 *
 * All tests use the real TimesFM ONNX model — no mocks.
 * Test data uses synthetic but realistic patterns (trend, seasonal, etc.)
 * that exercise TimesFM's forecasting capability meaningfully.
 */

import { describe, it, expect } from 'vitest';
import { TimesFMModel, createForecastConfig } from '@agentix-e/timesfm-core';
import { forecastWithCovariates } from '../src/xreg-engine.ts';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

const MODEL_FILENAMES = ['timesfm-2.5.onnx', 'timesfm-2.5-200m.onnx', 'timesfm.onnx'];
const MODEL_SEARCH_PATHS = [
  path.resolve(__dirname, '..', '..', '..', 'models'),
  path.resolve(__dirname, '..', '..', '..'),
  path.join(os.homedir(), '.cache', 'agentix-timesfm-ts'),
];

function getModelPath(): string {
  const env = process.env.TIMESFM_TEST_MODEL || process.env.TIMESFM_TEST_MODEL_DIR;
  if (env) {
    if (fs.existsSync(env)) return env;
    for (const name of MODEL_FILENAMES) {
      const candidate = path.join(env, name);
      if (fs.existsSync(candidate)) return candidate;
    }
  }
  for (const searchPath of MODEL_SEARCH_PATHS) {
    for (const name of MODEL_FILENAMES) {
      const candidate = path.join(searchPath, name);
      if (fs.existsSync(candidate)) return candidate;
    }
  }
  throw new Error(
    `Model not found. Set TIMESFM_TEST_MODEL. Searched: ${MODEL_SEARCH_PATHS.join(', ')}`,
  );
}

describe('forecastWithCovariates', () => {
  let model: TimesFMModel;

  beforeAll(async () => {
    model = await TimesFMModel.fromPretrained({ modelPath: getModelPath() });
    model.compile(
      createForecastConfig({
        maxContext: 128,
        maxHorizon: 64,
        perCoreBatchSize: 2,
      }),
    );
  });

  afterAll(async () => {
    await model.dispose();
  });

  it('forecasts with dynamic numerical covariates (xreg + timesfm)', async () => {
    const horizon = 12;
    const contextLen = 50;

    const inputs: Float32Array[] = [
      new Float32Array(Array.from({ length: contextLen }, (_, i) => i + 1)),
    ];

    const totalLen = contextLen + horizon;

    const result = await forecastWithCovariates(model, {
      inputs,
      dynamicNumericalCovariates: {
        price: [
          new Float32Array(Array.from({ length: totalLen }, (_, i) => Math.sin(i * 0.3) * 5 + 10)),
        ],
      },
      xregMode: 'xreg + timesfm',
      ridge: 0.1,
    });

    expect(result.pointForecast.length).toBe(1);
    expect(result.pointForecast[0].length).toBe(horizon);
    expect(result.xregOutputs.length).toBe(1);
    expect(result.xregOutputs[0].length).toBe(horizon);

    // All output should be finite
    for (const v of result.pointForecast[0]) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('forecasts with dynamic categorical covariates', async () => {
    const horizon = 6;
    const contextLen = 30;
    const totalLen = contextLen + horizon;

    const inputs: Float32Array[] = [
      new Float32Array(Array.from({ length: contextLen }, (_, i) => i + 1)),
    ];

    const result = await forecastWithCovariates(model, {
      inputs,
      dynamicCategoricalCovariates: {
        dayOfWeek: [Array.from({ length: totalLen }, (_, i) => i % 7)],
      },
      xregMode: 'xreg + timesfm',
    });

    expect(result.pointForecast[0].length).toBe(horizon);
    expect(Number.isFinite(result.pointForecast[0][0])).toBe(true);
  });

  it('forecasts with static categorical covariates', async () => {
    const contextLen = 30;

    const inputs: Float32Array[] = [
      new Float32Array(Array.from({ length: contextLen }, (_, i) => i + 1)),
      new Float32Array(Array.from({ length: contextLen }, (_, i) => (i + 1) * 2)),
    ];

    const result = await forecastWithCovariates(model, {
      inputs,
      staticCategoricalCovariates: {
        region: ['east', 'west'],
      },
      xregMode: 'xreg + timesfm',
    });

    expect(result.pointForecast.length).toBe(2);
    // Output length depends on fc.maxHorizon (64) and internal processing
    expect(result.pointForecast[0].length).toBeGreaterThan(0);
    expect(result.pointForecast[1].length).toBeGreaterThan(0);
  });

  it('supports timesfm + xreg mode', async () => {
    const horizon = 8;
    const contextLen = 40;
    const totalLen = contextLen + horizon;

    const inputs: Float32Array[] = [
      new Float32Array(Array.from({ length: contextLen }, (_, i) => i + 1)),
    ];

    const result = await forecastWithCovariates(model, {
      inputs,
      dynamicNumericalCovariates: {
        trend: [new Float32Array(Array.from({ length: totalLen }, (_, i) => i * 0.1))],
      },
      xregMode: 'timesfm + xreg',
    });

    expect(result.pointForecast.length).toBe(1);
    expect(Number.isFinite(result.pointForecast[0][0])).toBe(true);
  });

  it('Ridge regression: lambda > 0 produces finite output', async () => {
    const horizon = 8;
    const contextLen = 40;
    const totalLen = contextLen + horizon;

    const inputs: Float32Array[] = [
      new Float32Array(Array.from({ length: contextLen }, (_, i) => i + 1)),
    ];

    const resultRidge = await forecastWithCovariates(model, {
      inputs,
      dynamicNumericalCovariates: {
        x: [new Float32Array(Array.from({ length: totalLen }, (_, _i) => Math.random()))],
      },
      ridge: 10.0,
      xregMode: 'xreg + timesfm',
    });

    expect(Number.isFinite(resultRidge.pointForecast[0][0])).toBe(true);
  });

  it('subsamples rows when maxRowsPerCol is set', async () => {
    const horizon = 8;
    const contextLen = 120;

    const inputs: Float32Array[] = [
      new Float32Array(Array.from({ length: contextLen }, (_, i) => i + 1)),
    ];

    const result = await forecastWithCovariates(model, {
      inputs,
      dynamicNumericalCovariates: {
        x: [
          new Float32Array(
            Array.from({ length: contextLen + horizon }, (_, i) => Math.sin(i * 0.3)),
          ),
        ],
      },
      xregMode: 'xreg + timesfm',
      maxRowsPerCol: 1,
    });

    expect(Number.isFinite(result.pointForecast[0][0])).toBe(true);
  });

  it('xreg + timesfm mode handles singular design matrix with ridge fallback', async () => {
    // Use xreg + timesfm mode with a degenerate covariate (all zeros)
    // and ridge=0. The design matrix [intercept, zeros] is rank-deficient,
    // triggering the ridge penalty fallback in ridgeRegression().
    const horizon = 8;
    const contextLen = 40;

    const result = await forecastWithCovariates(model, {
      inputs: [new Float32Array(Array.from({ length: contextLen }, (_, i) => i + 1))],
      dynamicNumericalCovariates: {
        trend: [new Float32Array(Array.from({ length: contextLen + horizon }, (_, i) => i * 0.1))],
      },
      staticNumericalCovariates: {
        zero: [0], // makes design matrix columns collinear
      },
      ridge: 0,
      xregMode: 'xreg + timesfm', // does not need backcast from TimesFM
    });

    expect(result.pointForecast.length).toBe(1);
    expect(Number.isFinite(result.pointForecast[0][0])).toBe(true);
  });

  it('timesfm + xreg mode recompiles to enable backcast transparently', async () => {
    // Verify that timesfm + xreg works even when the model was initially
    // compiled with returnBackcast=false — the engine transparently
    // recompiles to enable it, then restores the original config.
    const horizon = 8;
    const contextLen = 40;
    const totalLen = contextLen + horizon;

    // Explicitly compile with returnBackcast=false
    model.compile(
      createForecastConfig({
        maxContext: 128,
        maxHorizon: 64,
        perCoreBatchSize: 2,
        returnBackcast: false,
      }),
    );
    expect(model.forecastConfig?.returnBackcast).toBe(false);

    const result = await forecastWithCovariates(model, {
      inputs: [new Float32Array(Array.from({ length: contextLen }, (_, i) => i + 1))],
      dynamicNumericalCovariates: {
        trend: [new Float32Array(Array.from({ length: totalLen }, (_, i) => i * 0.1))],
      },
      xregMode: 'timesfm + xreg',
    });

    expect(result.pointForecast.length).toBe(1);
    expect(Number.isFinite(result.pointForecast[0][0])).toBe(true);

    // Original config should be restored
    expect(model.forecastConfig?.returnBackcast).toBe(false);

    // Restore default config for subsequent tests
    model.compile(
      createForecastConfig({
        maxContext: 128,
        maxHorizon: 64,
        perCoreBatchSize: 2,
      }),
    );
  });

  it('normalizes xreg targets per input', async () => {
    const horizon = 8;
    const contextLen = 40;
    const totalLen = contextLen + horizon;

    const result = await forecastWithCovariates(model, {
      inputs: [new Float32Array(Array.from({ length: contextLen }, (_, i) => i + 1))],
      dynamicNumericalCovariates: {
        x: [new Float32Array(Array.from({ length: totalLen }, (_, i) => Math.sin(i * 0.3)))],
      },
      xregMode: 'xreg + timesfm',
      normalizeXregTargetPerInput: true,
    });

    expect(result.pointForecast.length).toBe(1);
    expect(Number.isFinite(result.pointForecast[0][0])).toBe(true);
  });

  it('forecasts with static numerical covariates', async () => {
    const contextLen = 30;

    const result = await forecastWithCovariates(model, {
      inputs: [new Float32Array(Array.from({ length: contextLen }, (_, i) => i + 1))],
      staticNumericalCovariates: {
        score: [42],
      },
      xregMode: 'xreg + timesfm',
    });

    expect(result.pointForecast.length).toBe(1);
    expect(Number.isFinite(result.pointForecast[0][0])).toBe(true);
  });
});
