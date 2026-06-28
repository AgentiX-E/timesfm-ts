/**
 * XReg (Exogenous Regression) engine for TimesFM covariate forecasting.
 *
 * Mirrors the Python `BatchedInContextXRegLinear` in utils/xreg_lib.py.
 *
 * Supports two modes:
 *   - "xreg + timesfm": Fit covariates → forecast residuals with TimesFM → combine.
 *   - "timesfm + xreg": Forecast with TimesFM → fit covariates on residuals → combine.
 *
 * Uses `ml-matrix` for linear algebra (Ridge regression via normal equations).
 */

import { Matrix, solve } from 'ml-matrix';
import { OneHotEncoder, type Category } from './one-hot-encoder';
import type { TimesFMModel, ForecastOutput } from '@agentix-e/timesfm-core';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TOL = 1e-6;

/**
 * Wrap `Matrix.columnVector` to accept `Float32Array`.
 *
 * `ml-matrix` only accepts `number[]` — this wrapper makes the
 * conversion explicit and avoids `as unknown as number[]` casts.
 */
function columnVector(arr: Float32Array): Matrix {
  return Matrix.columnVector(Array.from(arr));
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type XRegMode = 'xreg + timesfm' | 'timesfm + xreg';

export interface CovariateForecastParams {
  inputs: Float32Array[];
  dynamicNumericalCovariates?: Record<string, Float32Array[]>;
  dynamicCategoricalCovariates?: Record<string, Category[][]>;
  staticNumericalCovariates?: Record<string, number[]>;
  staticCategoricalCovariates?: Record<string, Category[]>;
  xregMode?: XRegMode;
  normalizeXregTargetPerInput?: boolean;
  ridge?: number;
  maxRowsPerCol?: number;
}

export interface CovariateForecastOutput extends ForecastOutput {
  /** The pure xreg contribution for each series. */
  xregOutputs: Float32Array[];
}

// ---------------------------------------------------------------------------
// Covariate matrix builder
// ---------------------------------------------------------------------------

interface DesignMatrices {
  /** Design matrix for training (context). */
  xTrain: Matrix;
  /** Design matrix for test (horizon). */
  xTest: Matrix;
}

function buildDesignMatrices(
  params: CovariateForecastParams,
  trainLens: number[],
  testLens: number[],
  inputLens: number[],
): DesignMatrices {
  const numSeries = params.inputs.length;
  const totalTrain = trainLens.reduce((a, b) => a + b, 0);
  const totalTest = testLens.reduce((a, b) => a + b, 0);

  const trainBlocks: number[][] = [];
  const testBlocks: number[][] = [];

  // ---- Dynamic numerical covariates ----
  if (params.dynamicNumericalCovariates) {
    for (const name of Object.keys(params.dynamicNumericalCovariates).sort()) {
      const covs = params.dynamicNumericalCovariates[name];
      const trainCol: number[] = [];
      const testCol: number[] = [];

      for (let s = 0; s < numSeries; s++) {
        const cov = covs[s];
        for (let t = inputLens[s] - trainLens[s]; t < inputLens[s]; t++) {
          trainCol.push(t < cov.length ? cov[t] : 0);
        }
        for (let t = inputLens[s]; t < inputLens[s] + testLens[s] && t < cov.length; t++) {
          testCol.push(cov[t]);
        }
      }

      // Standardize (z-score) using training data
      const trainMean = trainCol.reduce((a, b) => a + b, 0) / trainCol.length;
      const trainStd = Math.sqrt(
        trainCol.reduce((a, b) => a + (b - trainMean) ** 2, 0) / trainCol.length,
      );
      const safeStd = trainStd < 1e-6 ? 1 : trainStd;

      trainBlocks.push(trainCol.map((v) => (v - trainMean) / safeStd));
      testBlocks.push(testCol.map((v) => (v - trainMean) / safeStd));
    }
  }

  // ---- Static numerical covariates ----
  if (params.staticNumericalCovariates) {
    for (const name of Object.keys(params.staticNumericalCovariates).sort()) {
      const covs = params.staticNumericalCovariates[name];
      const trainCol: number[] = [];
      const testCol: number[] = [];

      for (let s = 0; s < numSeries; s++) {
        for (let t = 0; t < trainLens[s]; t++) trainCol.push(covs[s]);
        for (let t = 0; t < testLens[s]; t++) testCol.push(covs[s]);
      }

      trainBlocks.push(trainCol);
      testBlocks.push(testCol);
    }
  }

  // ---- Dynamic categorical covariates ----
  if (params.dynamicCategoricalCovariates) {
    for (const name of Object.keys(params.dynamicCategoricalCovariates).sort()) {
      const covs = params.dynamicCategoricalCovariates[name];
      const encoder = new OneHotEncoder({ drop: 'first', handleUnknown: 'ignore' });

      // Collect all categories for fitting
      const allCats: Category[] = [];
      for (let s = 0; s < numSeries; s++) {
        for (const v of covs[s]) allCats.push(v);
      }
      encoder.fit(allCats);

      // Transform
      const trainRows: number[][] = [];
      const testRows: number[][] = [];

      for (let s = 0; s < numSeries; s++) {
        const cov = covs[s];
        for (let t = inputLens[s] - trainLens[s]; t < inputLens[s]; t++) {
          trainRows.push(encoder.transform([cov[t]])[0]);
        }
        for (let t = inputLens[s]; t < inputLens[s] + testLens[s] && t < cov.length; t++) {
          testRows.push(encoder.transform([cov[t]])[0]);
        }
      }

      // Append columns
      for (let c = 0; c < encoder.numColumns; c++) {
        trainBlocks.push(trainRows.map((r) => r[c] ?? 0));
        testBlocks.push(testRows.map((r) => r[c] ?? 0));
      }
    }
  }

  // ---- Static categorical covariates ----
  if (params.staticCategoricalCovariates) {
    for (const name of Object.keys(params.staticCategoricalCovariates).sort()) {
      const covs = params.staticCategoricalCovariates[name];
      const encoder = new OneHotEncoder({ drop: 'first', handleUnknown: 'ignore' });
      encoder.fit(covs as Category[]);

      const trainRows: number[][] = [];
      const testRows: number[][] = [];

      for (let s = 0; s < numSeries; s++) {
        const encoded = encoder.transform([covs[s]])[0];
        for (let t = 0; t < trainLens[s]; t++) trainRows.push(encoded);
        for (let t = 0; t < testLens[s]; t++) testRows.push(encoded);
      }

      for (let c = 0; c < encoder.numColumns; c++) {
        trainBlocks.push(trainRows.map((r) => r[c] ?? 0));
        testBlocks.push(testRows.map((r) => r[c] ?? 0));
      }
    }
  }

  // ---- Transpose: blocks are rows, we need columns → build as [row, col] matrix ----

  // Train matrix: rows = totalTrain, cols = intercept + features
  const numCols = trainBlocks.length + 1; // +1 for intercept
  const xTrainMat = new Matrix(totalTrain, numCols);
  const xTestMat = new Matrix(totalTest, numCols);

  for (let r = 0; r < totalTrain; r++) {
    xTrainMat.set(r, 0, 1.0); // intercept
    for (let c = 0; c < trainBlocks.length; c++) {
      xTrainMat.set(r, c + 1, trainBlocks[c][r] ?? 0);
    }
  }

  for (let r = 0; r < totalTest; r++) {
    xTestMat.set(r, 0, 1.0); // intercept
    for (let c = 0; c < testBlocks.length; c++) {
      xTestMat.set(r, c + 1, testBlocks[c][r] ?? 0);
    }
  }

  return { xTrain: xTrainMat, xTest: xTestMat };
}

// ---------------------------------------------------------------------------
// Ridge regression via normal equations
// ---------------------------------------------------------------------------

/**
 * Solve Ridge regression: β = (XᵀX + λI)⁻¹ Xᵀy
 *
 * @param maxRowsPerCol  If > 0, subsample rows to at most maxRowsPerCol * cols
 *                       before fitting (for large design matrices).
 */
function ridgeRegression(
  x: Matrix,
  y: Float32Array,
  ridge: number,
  maxRowsPerCol: number = 0,
): Float32Array {
  let xMat = x;
  let yVec = y;

  // Subsample if too many rows
  if (maxRowsPerCol > 0) {
    const nrows = x.rows;
    const ncols = x.columns;
    const maxRows = ncols * maxRowsPerCol;
    if (nrows > maxRows) {
      // Deterministic reservoir-like subset: take evenly spaced rows.
      // When step < 1, avoid duplicates by capping at nrows.
      const step = nrows / maxRows;
      const selRows: number[] = [];
      const seen = new Set<number>();
      for (let i = 0; i < maxRows && selRows.length < maxRows; i++) {
        const idx = Math.floor(i * step);
        if (!seen.has(idx) && idx < nrows) {
          seen.add(idx);
          selRows.push(idx);
        }
      }
      const effectiveRows = selRows.length;
      const subX = new Matrix(effectiveRows, ncols);
      const subY = new Float32Array(effectiveRows);
      for (let i = 0; i < effectiveRows; i++) {
        const src = selRows[i];
        for (let c = 0; c < ncols; c++) subX.set(i, c, xMat.get(src, c));
        subY[i] = yVec[src];
      }
      xMat = subX;
      yVec = subY;
    }
  }

  const n = xMat.rows;
  const p = xMat.columns;

  // XᵀX
  const xtx = xMat.transpose().mmul(xMat);

  // Add ridge penalty
  if (ridge > 0) {
    for (let i = 0; i < p; i++) {
      xtx.set(i, i, xtx.get(i, i) + ridge);
    }
  }

  // Xᵀy
  const yFlat = new Float32Array(n);
  for (let i = 0; i < n; i++) yFlat[i] = yVec[i] ?? 0;
  const xty = xMat.transpose().mmul(columnVector(yFlat));

  // Solve
  try {
    const beta = solve(xtx, xty);
    const result = new Float32Array(p);
    for (let i = 0; i < p; i++) result[i] = beta.get(i, 0);
    return result;
  } catch {
    // Singular matrix fallback: increase ridge penalty with recursion guard
    if (ridge >= 100) {
      throw new Error(
        `Ridge regression failed: matrix remains singular after increasing ridge to ${ridge}. ` +
          `Check for multicollinear or constant covariates.`,
      );
    }
    return ridgeRegression(x, y, ridge + 0.01, maxRowsPerCol);
  }
}

// ---------------------------------------------------------------------------
// Main XReg forecast function
// ---------------------------------------------------------------------------

/** Per-series normalization: (x - μ) / σ */
function normalizeXregTargets(batch: Float32Array[]): {
  normalized: Float32Array[];
  stats: { mu: number; sigma: number }[];
} {
  const stats = batch.map((x) => {
    // Numerically-stable two-pass variance (matches stats.ts::computeStats).
    // The one-pass E[X²] − E[X]² formula suffers catastrophic cancellation
    // when values are large relative to their variance.
    let sum = 0,
      n = 0;
    for (let i = 0; i < x.length; i++) {
      const v = x[i];
      if (!Number.isFinite(v)) continue;
      sum += v;
      n++;
    }
    const mu = n > 0 ? sum / n : 0;

    // Second pass: squared deviations from the computed mean
    let varSum = 0;
    for (let i = 0; i < x.length; i++) {
      const v = x[i];
      if (!Number.isFinite(v)) continue;
      const diff = v - mu;
      varSum += diff * diff;
    }
    const variance = n > 0 ? Math.max(0, varSum / n) : 0;
    const sigma = Math.sqrt(variance);
    return { mu, sigma: sigma < TOL ? 1 : sigma };
  });
  const normalized = batch.map((x, i) => {
    const r = new Float32Array(x.length);
    for (let j = 0; j < x.length; j++) r[j] = (x[j] - stats[i].mu) / stats[i].sigma;
    return r;
  });
  return { normalized, stats };
}

/** Per-series denormalization: x * σ + μ */
function renormalizeXregOutputs(
  batch: Float32Array[],
  stats: { mu: number; sigma: number }[],
): Float32Array[] {
  return batch.map((x, i) => {
    const r = new Float32Array(x.length);
    for (let j = 0; j < x.length; j++) r[j] = x[j] * stats[i].sigma + stats[i].mu;
    return r;
  });
}

/**
 * Run TimesFM forecast with exogenous covariates.
 *
 * @param model  A compiled TimesFMModel instance.
 * @param params  Covariate forecast parameters.
 */
export async function forecastWithCovariates(
  model: TimesFMModel,
  params: CovariateForecastParams,
): Promise<CovariateForecastOutput> {
  const fc = model.forecastConfig;
  if (!fc) throw new Error('Model not compiled. Call compile() first.');

  const xregMode: XRegMode = params.xregMode ?? 'xreg + timesfm';
  const ridge = params.ridge ?? 0;
  const numSeries = params.inputs.length;

  if (numSeries === 0) {
    throw new Error('At least one input series is required.');
  }

  // Validate covariate array lengths match inputs
  if (params.dynamicNumericalCovariates) {
    for (const [name, covs] of Object.entries(params.dynamicNumericalCovariates)) {
      if (covs.length !== numSeries) {
        throw new Error(
          `Dynamic numerical covariate "${name}" has ${covs.length} entries but ${numSeries} input series were provided.`,
        );
      }
    }
  }
  if (params.dynamicCategoricalCovariates) {
    for (const [name, covs] of Object.entries(params.dynamicCategoricalCovariates)) {
      if (covs.length !== numSeries) {
        throw new Error(
          `Dynamic categorical covariate "${name}" has ${covs.length} entries but ${numSeries} input series were provided.`,
        );
      }
    }
  }
  if (params.staticNumericalCovariates) {
    for (const [name, covs] of Object.entries(params.staticNumericalCovariates)) {
      if (covs.length !== numSeries) {
        throw new Error(
          `Static numerical covariate "${name}" has ${covs.length} entries but ${numSeries} input series were provided.`,
        );
      }
    }
  }
  if (params.staticCategoricalCovariates) {
    for (const [name, covs] of Object.entries(params.staticCategoricalCovariates)) {
      if (covs.length !== numSeries) {
        throw new Error(
          `Static categorical covariate "${name}" has ${covs.length} entries but ${numSeries} input series were provided.`,
        );
      }
    }
  }

  // Validate covariate value finiteness — NaN/Infinity in covariates would
  // cause Ridge regression to produce NaN coefficients with no clear error.
  function validateDim2Finiteness(covName: string, covType: string, series: Float32Array[]): void {
    for (let s = 0; s < series.length; s++) {
      for (let i = 0; i < series[s].length; i++) {
        if (!Number.isFinite(series[s][i])) {
          throw new Error(
            `${covType} covariate "${covName}" series[${s}][${i}] = ${series[s][i]} (must be finite). ` +
              `Clean data before calling forecastXReg().`,
          );
        }
      }
    }
  }

  function validateDim1Finiteness(covName: string, covType: string, series: number[]): void {
    for (let s = 0; s < series.length; s++) {
      if (!Number.isFinite(series[s])) {
        throw new Error(
          `${covType} covariate "${covName}"[${s}] = ${series[s]} (must be finite). ` +
            `Clean data before calling forecastXReg().`,
        );
      }
    }
  }

  if (params.dynamicNumericalCovariates) {
    for (const [name, covs] of Object.entries(params.dynamicNumericalCovariates)) {
      validateDim2Finiteness(name, 'Dynamic numerical', covs);
    }
  }
  if (params.staticNumericalCovariates) {
    for (const [name, covs] of Object.entries(params.staticNumericalCovariates)) {
      validateDim1Finiteness(name, 'Static numerical', covs);
    }
  }

  // ---- Track lengths ----
  const inputLens = params.inputs.map((s) => s.length);
  const trainLens: number[] = [];
  const testLens: number[] = [];

  for (let s = 0; s < numSeries; s++) {
    if (xregMode === 'timesfm + xreg') {
      // Don't use first patch for model fitting
      trainLens.push(Math.max(0, inputLens[s] - model.modelConfig.inputPatchLen));
    } else {
      trainLens.push(inputLens[s]);
    }

    if (params.dynamicNumericalCovariates) {
      const firstCov = Object.values(params.dynamicNumericalCovariates)[0];
      testLens.push(firstCov[s].length - inputLens[s]);
    } else if (params.dynamicCategoricalCovariates) {
      const firstCov = Object.values(params.dynamicCategoricalCovariates)[0];
      testLens.push(firstCov[s].length - inputLens[s]);
    } else {
      testLens.push(fc.maxHorizon);
    }
  }

  // ---- Build design matrices ----
  const { xTrain, xTest } = buildDesignMatrices(params, trainLens, testLens, inputLens);

  // ---- Fit ----
  const doNormalize = params.normalizeXregTargetPerInput ?? false;

  if (xregMode === 'xreg + timesfm') {
    // Step 1: Fit linear model on targets
    let targets: Float32Array[] = params.inputs.map((s, i) => {
      const start = inputLens[i] - trainLens[i];
      return new Float32Array(s.slice(start, inputLens[i]));
    });

    // Per-series normalization
    let xregStats: { mu: number; sigma: number }[] | null = null;
    if (doNormalize) {
      const result = normalizeXregTargets(targets);
      targets = result.normalized;
      xregStats = result.stats;
    }

    // Flatten targets
    const flatY = new Float32Array(targets.reduce((sum, t) => sum + t.length, 0));
    let offset = 0;
    for (const t of targets) {
      flatY.set(t, offset);
      offset += t.length;
    }

    const beta = ridgeRegression(xTrain, flatY, ridge, params.maxRowsPerCol ?? 0);

    // Predict on context and horizon
    const yHatContext = xTrain.mmul(columnVector(beta));
    let yHatTest = xTest.mmul(columnVector(beta));

    // Step 2: Compute residuals
    const residuals: Float32Array[] = [];
    let trainOffset = 0;
    for (let s = 0; s < numSeries; s++) {
      const residual = new Float32Array(trainLens[s]);
      for (let t = 0; t < trainLens[s]; t++) {
        residual[t] = targets[s][t] - yHatContext.get(trainOffset + t, 0);
      }
      residuals.push(residual);
      trainOffset += trainLens[s];
    }

    // Step 3: Forecast residuals with TimesFM
    const tsResult = await model.forecast(fc.maxHorizon, residuals);

    // If normalized, renormalize the XReg outputs back to original scale
    if (doNormalize && xregStats) {
      // Extract yHatTest as per-series arrays
      const yHatSeries: Float32Array[] = [];
      let to = 0;
      for (let s = 0; s < numSeries; s++) {
        const arr = new Float32Array(testLens[s]);
        for (let t = 0; t < testLens[s]; t++) arr[t] = yHatTest.get(to + t, 0);
        yHatSeries.push(arr);
        to += testLens[s];
      }
      const renormalized = renormalizeXregOutputs(yHatSeries, xregStats);
      // Rebuild flat yHatTest
      const flatRenorm = new Float32Array(renormalized.reduce((sum, r) => sum + r.length, 0));
      let ro = 0;
      for (const r of renormalized) {
        flatRenorm.set(r, ro);
        ro += r.length;
      }
      yHatTest = columnVector(flatRenorm);
    }

    // Step 4: Combine
    const xregOutputs: Float32Array[] = [];
    let testOffset = 0;
    for (let s = 0; s < numSeries; s++) {
      const xreg = new Float32Array(testLens[s]);
      for (let t = 0; t < testLens[s]; t++) {
        xreg[t] = yHatTest.get(testOffset + t, 0);
      }
      xregOutputs.push(xreg);
      testOffset += testLens[s];
    }

    // Combine: forecast + xreg
    return {
      pointForecast: tsResult.pointForecast.map((pf, i) => {
        const result = new Float32Array(Math.min(pf.length, testLens[i]));
        for (let t = 0; t < result.length; t++) {
          /* v8 ignore next — xregOutputs[i][t] is always a number from linear regression */
          result[t] = pf[t] + (xregOutputs[i][t] ?? 0);
        }
        return result;
      }),
      quantileForecast: tsResult.quantileForecast.map((qf, i) => {
        return qf.map((q) => {
          const result = new Float32Array(Math.min(q.length, testLens[i]));
          for (let t = 0; t < result.length; t++) {
            /* v8 ignore next — xregOutputs[i][t] is always a number from linear regression */
            result[t] = q[t] + (xregOutputs[i][t] ?? 0);
          }
          return result;
        });
      }),
      xregOutputs,
    };
  } else {
    // "timesfm + xreg" mode
    //
    // 1. Forecast with TimesFM (request backcast via configOverrides — no
    //    global state mutation, avoiding race conditions on concurrent calls)
    // 2. Compute residuals = target - backcast
    // 3. Fit linear model on residuals
    // 4. Combine: final = pointForecast + xreg_prediction

    // Request backcast inline via configOverrides (caller's compiled config is never mutated).
    const tsResult = await model.forecast(fc.maxHorizon, params.inputs, {
      configOverrides: { returnBackcast: true },
    });

    // Step 2: Compute residuals using backcast (historical reconstruction)
    const targets: Float32Array[] = params.inputs.map((s, i) => {
      const start = inputLens[i] - trainLens[i];
      return new Float32Array(s.slice(start, inputLens[i]));
    });

    // Use the backcast portion matching the training window
    const residuals: Float32Array[] = [];
    const backcasts = tsResult.backcast;
    /* v8 ignore start — defensive: unreachable when configOverrides.requestBackcast is true */
    if (!backcasts) {
      throw new Error(
        'timesfm + xreg mode requires backcast. Ensure the model was compiled with returnBackcast=true.',
      );
    }
    /* v8 ignore stop */

    for (let s = 0; s < numSeries; s++) {
      const residual = new Float32Array(trainLens[s]);
      const contextLen = backcasts[s].length;
      const backcastOffset = contextLen - trainLens[s];

      for (let t = 0; t < trainLens[s]; t++) {
        residual[t] = targets[s][t] - backcasts[s][backcastOffset + t];
      }
      residuals.push(residual);
    }

    // Step 3: Fit linear model on residuals
    const flatY = new Float32Array(residuals.reduce((sum, r) => sum + r.length, 0));
    let offset = 0;
    for (const r of residuals) {
      flatY.set(r, offset);
      offset += r.length;
    }

    const beta = ridgeRegression(xTrain, flatY, ridge, params.maxRowsPerCol ?? 0);
    const yHatTest = xTest.mmul(columnVector(beta));

    // Step 4: Combine
    const xregOutputs: Float32Array[] = [];
    let testOffset = 0;
    for (let s = 0; s < numSeries; s++) {
      const xreg = new Float32Array(testLens[s]);
      for (let t = 0; t < testLens[s]; t++) {
        xreg[t] = yHatTest.get(testOffset + t, 0);
      }
      xregOutputs.push(xreg);
      testOffset += testLens[s];
    }

    return {
      pointForecast: tsResult.pointForecast.map((pf, i) => {
        const result = new Float32Array(Math.min(pf.length, testLens[i]));
        for (let t = 0; t < result.length; t++) {
          /* v8 ignore next — xregOutputs[i][t] is always a number from linear regression */
          result[t] = pf[t] + (xregOutputs[i][t] ?? 0);
        }
        return result;
      }),
      quantileForecast: tsResult.quantileForecast.map((qf, i) => {
        return qf.map((q) => {
          const result = new Float32Array(Math.min(q.length, testLens[i]));
          for (let t = 0; t < result.length; t++) {
            /* v8 ignore next — xregOutputs[i][t] is always a number from linear regression */
            result[t] = q[t] + (xregOutputs[i][t] ?? 0);
          }
          return result;
        });
      }),
      xregOutputs,
    };
  }
}
