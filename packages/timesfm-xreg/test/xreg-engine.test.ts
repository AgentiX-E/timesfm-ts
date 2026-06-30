/**
 * Tests for the XReg covariate forecasting engine.
 *
 * All tests use the real TimesFM ONNX model — no mocks.
 * Test data uses synthetic but realistic patterns (trend, seasonal, etc.)
 * that exercise TimesFM's forecasting capability meaningfully.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
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

function getModelPath(): string | null {
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
  return null;
}

const MODEL_PATH = getModelPath();

describe('forecastWithCovariates', () => {
  if (!MODEL_PATH) {
    it.skip('all covariate tests require ONNX model — skipping', () => {});
    return;
  }

  let model: TimesFMModel;

  beforeAll(async () => {
    model = await TimesFMModel.fromPretrained({ modelPath: MODEL_PATH });
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

  it('timesfm + xreg mode uses configOverrides (no global recompile)', async () => {
    // Verify that timesfm + xreg works even when the model was initially
    // compiled with returnBackcast=false — the engine transparently
    // uses configOverrides to request backcast, then the stored config
    // is left unchanged (no race condition window).
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

    // Stored config must be untouched — configOverrides are per-call only
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

  // -------------------------------------------------------------------
  // Edge path coverage
  // -------------------------------------------------------------------

  it('works with no covariates at all (xreg acts as intercept-only)', async () => {
    // When no covariates are supplied the design matrix has only the intercept column.
    // testLens falls back to fc.maxHorizon (line 361 in xreg-engine.ts).
    const contextLen = 30;

    const result = await forecastWithCovariates(model, {
      inputs: [new Float32Array(Array.from({ length: contextLen }, (_, i) => i + 1))],
      xregMode: 'xreg + timesfm',
    });

    expect(result.pointForecast.length).toBe(1);
    expect(result.pointForecast[0].length).toBeGreaterThan(0);
    expect(Number.isFinite(result.pointForecast[0][0])).toBe(true);
    // xregOutputs should exist (intercept-only linear fit)
    expect(result.xregOutputs.length).toBe(1);
  });

  it('handles all-constant dynamic covariates (std near zero branch)', async () => {
    // A covariate with zero variance triggers the safeStd=1 branch at line 102
    // (z-score normalization with near-zero std should not divide by zero).
    const horizon = 8;
    const contextLen = 40;
    const totalLen = contextLen + horizon;

    const result = await forecastWithCovariates(model, {
      inputs: [new Float32Array(Array.from({ length: contextLen }, (_, i) => i + 1))],
      dynamicNumericalCovariates: {
        constantCov: [new Float32Array(totalLen).fill(5)], // all same value → std = 0
      },
      xregMode: 'xreg + timesfm',
    });

    expect(result.pointForecast.length).toBe(1);
    expect(Number.isFinite(result.pointForecast[0][0])).toBe(true);
  });

  it('handles combined dynamic, static, and categorical covariates', async () => {
    // Exercise buildDesignMatrices with all covariate types simultaneously.
    const horizon = 8;
    const contextLen = 40;
    const totalLen = contextLen + horizon;

    const result = await forecastWithCovariates(model, {
      inputs: [new Float32Array(Array.from({ length: contextLen }, (_, i) => i + 1))],
      dynamicNumericalCovariates: {
        trend: [new Float32Array(Array.from({ length: totalLen }, (_, i) => i * 0.1))],
      },
      dynamicCategoricalCovariates: {
        season: [Array.from({ length: totalLen }, (_, i) => i % 4)],
      },
      staticNumericalCovariates: {
        score: [42],
      },
      staticCategoricalCovariates: {
        region: ['east'],
      },
      xregMode: 'xreg + timesfm',
    });

    expect(result.pointForecast.length).toBe(1);
    expect(Number.isFinite(result.pointForecast[0][0])).toBe(true);
  });

  it('timesfm + xreg mode works with multiple series', async () => {
    // Test the timesfm + xreg path with a batch of multiple series.
    const horizon = 8;
    const contextLen = 40;
    const totalLen = contextLen + horizon;

    const inputs: Float32Array[] = [
      new Float32Array(Array.from({ length: contextLen }, (_, i) => i + 1)),
      new Float32Array(Array.from({ length: contextLen }, (_, i) => (i + 1) * 10)),
    ];

    const result = await forecastWithCovariates(model, {
      inputs,
      dynamicNumericalCovariates: {
        trend: [
          new Float32Array(Array.from({ length: totalLen }, (_, i) => i * 0.1)),
          new Float32Array(Array.from({ length: totalLen }, (_, i) => Math.sin(i * 0.3))),
        ],
      },
      xregMode: 'timesfm + xreg',
    });

    expect(result.pointForecast.length).toBe(2);
    expect(result.pointForecast[0].length).toBeGreaterThan(0);
    expect(result.pointForecast[1].length).toBeGreaterThan(0);
    expect(Number.isFinite(result.pointForecast[0][0])).toBe(true);
    expect(Number.isFinite(result.pointForecast[1][0])).toBe(true);
  });

  it('defaults xregMode to "xreg + timesfm" when omitted', async () => {
    // Cover the ?? 'xreg + timesfm' default branch at line 337.
    const contextLen = 30;

    const result = await forecastWithCovariates(model, {
      inputs: [new Float32Array(Array.from({ length: contextLen }, (_, i) => i + 1))],
      dynamicNumericalCovariates: {
        price: [new Float32Array(Array.from({ length: contextLen + 8 }, (_, i) => i + 1))],
      },
    });

    expect(result.pointForecast.length).toBe(1);
    expect(Number.isFinite(result.pointForecast[0][0])).toBe(true);
  });

  it('throws when model is not compiled', async () => {
    // Cover line 335: if (!fc) throw new Error(...)
    const rawModel = await TimesFMModel.fromPretrained({ modelPath: getModelPath() });
    // Do NOT compile — forecastConfig is null
    expect(rawModel.forecastConfig).toBeNull();

    await expect(
      forecastWithCovariates(rawModel, {
        inputs: [new Float32Array([1, 2, 3])],
      }),
    ).rejects.toThrow('Model not compiled');

    await rawModel.dispose();
  });

  // ── Covariate validation: NaN / Infinity ─────────────────────────────────

  it('rejects NaN in dynamic numerical covariates', async () => {
    const nanArr = new Float32Array([1, 2, NaN, 4]);
    await expect(
      forecastWithCovariates(model, {
        inputs: [new Float32Array([1, 2, 3, 4, 5])],
        dynamicNumericalCovariates: { bad: [nanArr] },
        xregMode: 'xreg + timesfm',
      }),
    ).rejects.toThrow(/must be finite/);
  });

  it('rejects Infinity in dynamic numerical covariates', async () => {
    const infArr = new Float32Array([1, 2, Infinity]);
    await expect(
      forecastWithCovariates(model, {
        inputs: [new Float32Array([1, 2, 3, 4, 5])],
        dynamicNumericalCovariates: { bad: [infArr] },
        xregMode: 'xreg + timesfm',
      }),
    ).rejects.toThrow(/must be finite/);
  });

  it('rejects NaN in static numerical covariates', async () => {
    await expect(
      forecastWithCovariates(model, {
        inputs: [new Float32Array([1, 2, 3, 4, 5])],
        staticNumericalCovariates: { bad: [NaN] },
        xregMode: 'xreg + timesfm',
      }),
    ).rejects.toThrow(/must be finite/);
  });

  it('rejects Infinity in static numerical covariates', async () => {
    await expect(
      forecastWithCovariates(model, {
        inputs: [new Float32Array([1, 2, 3, 4, 5])],
        staticNumericalCovariates: { bad: [Infinity] },
        xregMode: 'xreg + timesfm',
      }),
    ).rejects.toThrow(/must be finite/);
  });
});
