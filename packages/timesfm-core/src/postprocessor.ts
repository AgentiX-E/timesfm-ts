/**
 * Output post-processing pipeline for TimesFM.
 *
 * Applies all ForecastConfig flags to the raw decode output:
 *   1. Assemble full forecast (last output patch + AR outputs, truncate to horizon)
 *   2. Flip invariance enforcement
 *   3. Continuous quantile head calibration
 *   4. Return backcast (if requested)
 *   5. Quantile crossing fix
 *   6. Input z-score normalization / denormalization reversal
 *   7. Positive-value clamping
 *   8. Split into point and quantile output arrays
 *
 * Mirrors the logic in `compile()` / `_compiled_decode()` in
 * timesfm_2p5_torch.py.
 */

import {
  TIMESFM_25_CONFIG,
  type ForecastConfig,
  type ModelConfig,
  type ForecastOutput,
} from './types';
import { elementwiseMean, negate, clipMin } from './utils/tensor-utils';
import type { DecodeResult } from './inference/decode-loop';

// ---------------------------------------------------------------------------
// Main post-processing entry point
// ---------------------------------------------------------------------------

/**
 * Apply all post-processing steps to model decode output.
 *
 * @param decodeResult   Raw output from the decode loop.
 * @param horizon        Requested forecast horizon.
 * @param fc             Forecast configuration.
 * @param mc             Model configuration.
 * @param inputStats     Pre-computed {mu, sigma} per batch element for z-score reversal.
 *                       Only used when fc.normalizeInputs is true.
 * @param flipDecode     Decode result for negated inputs (if forceFlipInvariance).
 */
export function postProcess(
  decodeResult: DecodeResult,
  horizon: number,
  fc: ForecastConfig,
  mc: ModelConfig = TIMESFM_25_CONFIG,
  inputStats: { mu: number; sigma: number }[] | null = null,
  flipDecode: DecodeResult | null = null,
  isPositiveFlags: boolean[] | null = null,
): ForecastOutput {
  const batchSize = decodeResult.pfOutputs.length;
  const { pfOutputs, quantileSpreads, arOutputs } = decodeResult;

  // ---- Step 1: Assemble full forecast ----
  let fullForecasts: Float32Array[] = [];

  for (let b = 0; b < batchSize; b++) {
    // Take the last output patch
    const perPatch = mc.outputPatchLen * mc.numQuantiles;
    const lastPatch = pfOutputs[b].slice(pfOutputs[b].length - perPatch);

    // Concatenate with AR outputs
    let full = new Float32Array(lastPatch);
    if (arOutputs) {
      const arFlat = arOutputs[b];
      const combined = new Float32Array(full.length + arFlat.length);
      combined.set(full, 0);
      combined.set(arFlat, full.length);
      full = combined;
    }

    // Truncate to horizon
    full = full.slice(0, horizon * mc.numQuantiles);
    fullForecasts.push(full);
  }

  // ---- Step 2: Flip invariance ----
  // Formula: forecast_final = (forecast(x) - forecast(-x)) / 2
  // This guarantees f(-x) = -f(x) as a mathematical invariant.
  if (fc.forceFlipInvariance && flipDecode) {
    const { pfOutputs: flipPf, arOutputs: flipAr } = flipDecode;

    // Build the flipped forecast from the negated-input decode
    for (let b = 0; b < batchSize; b++) {
      const perPatch = mc.outputPatchLen * mc.numQuantiles;
      const lastFlipPf = flipPf[b].slice(flipPf[b].length - perPatch);
      const flippedLast = flipQuantileArray(lastFlipPf, mc.numQuantiles);

      let combined = new Float32Array(flippedLast);
      if (flipAr) {
        const flippedAr = flipQuantileArray(flipAr[b], mc.numQuantiles);
        const tmp = new Float32Array(flippedLast.length + flippedAr.length);
        tmp.set(flippedLast, 0);
        tmp.set(flippedAr, flippedLast.length);
        combined = tmp;
      }
      const flippedFull = combined.slice(0, horizon * mc.numQuantiles);

      // (forecast(x) - forecast(-x)) / 2
      fullForecasts[b] = elementwiseMean(fullForecasts[b], negate(flippedFull));
    }
  }

  // ---- Step 3: Continuous quantile head ----
  if (fc.useContinuousQuantileHead) {
    fullForecasts = applyContinuousQuantileHead(fullForecasts, quantileSpreads, horizon, mc);
  }

  // ---- Step 4: Return backcast (if requested) ----
  // Compute backcast from pfOutputs BEFORE Step 1 assembles the forecast.
  // Backcast = model's reconstruction of historical context (all but last output patch)
  let backcastOutputs: Float32Array[] | undefined;
  if (fc.returnBackcast) {
    backcastOutputs = pfOutputs.map((pf) => {
      const perPatch = mc.outputPatchLen * mc.numQuantiles;
      const numPatches = Math.floor(pf.length / perPatch);
      if (numPatches < 2) return new Float32Array(0);
      // Extract all but the last output patch, taking inputPatchLen per patch
      const backcastLen = (numPatches - 1) * mc.inputPatchLen * mc.numQuantiles;
      const backcast = new Float32Array(backcastLen);
      for (let p = 0; p < numPatches - 1; p++) {
        const patchStart = p * perPatch;
        for (let i = 0; i < mc.inputPatchLen; i++) {
          for (let q = 0; q < mc.numQuantiles; q++) {
            const srcIdx = patchStart + i * mc.numQuantiles + q;
            const dstIdx = (p * mc.inputPatchLen + i) * mc.numQuantiles + q;
            backcast[dstIdx] = pf[srcIdx];
          }
        }
      }
      return backcast;
    });

    // Apply the same z-score reversal to backcast if inputs were normalized
    if (fc.normalizeInputs && inputStats) {
      backcastOutputs = backcastOutputs.map((bc, b) => {
        const { mu, sigma } = inputStats[b] ?? { mu: 0, sigma: 1 };
        const safeSigma = sigma < 1e-6 ? 1.0 : sigma;
        const result = new Float32Array(bc.length);
        for (let i = 0; i < bc.length; i++) {
          result[i] = bc[i] * safeSigma + mu;
        }
        return result;
      });
    }
  }

  // ---- Step 5: Fix quantile crossing ----
  if (fc.fixQuantileCrossing) {
    fullForecasts = fullForecasts.map((f) => fixQuantileCrossing(f, mc.numQuantiles));
  }

  // ---- Step 6: Input normalization reversal ----
  if (fc.normalizeInputs && inputStats) {
    fullForecasts = reverseInputNormalization(fullForecasts, inputStats);
  }

  // ---- Step 7: Positive clamping ----
  // Only clamp series whose raw input was all ≥ 0 (matching Python reference behavior).
  // This is determined by model.ts which passes the per-series isPositiveFlags.
  if (fc.inferIsPositive && isPositiveFlags) {
    fullForecasts = fullForecasts.map((f, b) => {
      return isPositiveFlags[b] ? clipMin(f, 0) : f;
    });
  }

  // ---- Step 8: Split into point and quantile outputs ----
  const pointForecast: Float32Array[] = [];
  const quantileForecast: Float32Array[][] = [];

  for (let b = 0; b < batchSize; b++) {
    const qLen = horizon;
    const numQ = mc.numQuantiles;
    const pointArr = new Float32Array(qLen);
    const quantArr: Float32Array[] = [];

    for (let q = 0; q < numQ; q++) {
      quantArr.push(new Float32Array(qLen));
    }

    for (let h = 0; h < horizon; h++) {
      for (let q = 0; q < numQ; q++) {
        const val = fullForecasts[b][h * numQ + q];
        quantArr[q][h] = Number.isFinite(val) ? val : 0;
      }
      pointArr[h] = quantArr[mc.decodeIndex][h];
    }

    pointForecast.push(pointArr);
    quantileForecast.push(quantArr);
  }

  return { pointForecast, quantileForecast, backcast: backcastOutputs };
}

// ---------------------------------------------------------------------------
// Flip invariance helpers
// ---------------------------------------------------------------------------

/**
 * Flip the ordering of quantiles (excluding the mean at index 0):
 * [mean, q10, q20, ..., q90] → [mean, q90, q80, ..., q10]
 *
 * @param arr        The flat quantile array to flip.
 * @param numQuantiles  Number of quantiles per step (10 for TimesFM 2.5).
 * @param inPlace    If true, writes the result back into `arr` instead of allocating.
 *                   Default false (safe, allocates new array).
 */
export function flipQuantileArray(
  arr: Float32Array,
  numQuantiles: number,
  inPlace = false,
): Float32Array {
  const numSteps = Math.floor(arr.length / numQuantiles);
  const result = inPlace ? arr : new Float32Array(arr.length);
  // Copy the mean (index 0) for each step if not in-place
  if (!inPlace) {
    for (let t = 0; t < numSteps; t++) {
      result[t * numQuantiles] = arr[t * numQuantiles];
    }
  }

  for (let t = 0; t < numSteps; t++) {
    const base = t * numQuantiles;
    // Swap quantiles 1↔9, 2↔8, 3↔7, 4↔6; mean stays at 0
    for (let q = 1; q < numQuantiles; q++) {
      const dst = base + q;
      const src = base + numQuantiles - q;
      // In-place: only copy when src > dst to avoid overwriting
      if (inPlace && src > dst) {
        const tmp = arr[dst];
        result[dst] = arr[src];
        result[src] = tmp;
      } else if (!inPlace) {
        result[dst] = arr[src];
      }
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Continuous quantile head
// ---------------------------------------------------------------------------

/**
 * Replace the fixed-bucket quantiles with the continuous quantile head's
 * calibrated values for quantiles 1-4 and 6-9.
 *
 * q_new = quantile_spread[q] - quantile_spread[5] + full_forecast[5]
 */
export function applyContinuousQuantileHead(
  fullForecasts: Float32Array[],
  quantileSpreads: Float32Array[],
  horizon: number,
  mc: ModelConfig,
): Float32Array[] {
  return fullForecasts.map((ff, b) => {
    const qs = quantileSpreads[b];
    const result = new Float32Array(ff.length);
    const numSteps = Math.floor(ff.length / mc.numQuantiles);

    for (let h = 0; h < Math.min(numSteps, horizon); h++) {
      const base = h * mc.numQuantiles;

      // Mean stays
      result[base] = ff[base];

      // Lower quantiles: 1-4
      for (let q = 1; q <= 4; q++) {
        const qsIdx = h * mc.numQuantiles + q;
        const spreadVal = qsIdx < qs.length ? qs[qsIdx] : 0;
        const medianIdx = h * mc.numQuantiles + 5;
        const medianSpread = medianIdx < qs.length ? qs[medianIdx] : 0;
        result[base + q] = spreadVal - medianSpread + ff[base + 5];
      }

      // Median stays
      result[base + 5] = ff[base + 5];

      // Upper quantiles: 6-9
      for (let q = 6; q <= 9; q++) {
        const qsIdx = h * mc.numQuantiles + q;
        const spreadVal = qsIdx < qs.length ? qs[qsIdx] : 0;
        const medianIdx = h * mc.numQuantiles + 5;
        const medianSpread = medianIdx < qs.length ? qs[medianIdx] : 0;
        result[base + q] = spreadVal - medianSpread + ff[base + 5];
      }
    }

    // Copy remaining values
    for (let i = numSteps * mc.numQuantiles; i < ff.length; i++) {
      result[i] = ff[i];
    }

    return result;
  });
}

// ---------------------------------------------------------------------------
// Quantile crossing fix
// ---------------------------------------------------------------------------

/**
 * Ensure monotonicity: q10 ≤ q20 ≤ … ≤ q90.
 *
 * For lower quantiles (1→4): if q[i] > q[i+1], set q[i] = q[i+1].
 * For upper quantiles (6→9): if q[i] < q[i-1], set q[i] = q[i-1].
 * Median (5) and mean (0) are not modified.
 */
export function fixQuantileCrossing(arr: Float32Array, numQuantiles: number): Float32Array {
  const result = new Float32Array(arr);
  const numSteps = Math.floor(arr.length / numQuantiles);

  for (let t = 0; t < numSteps; t++) {
    const base = t * numQuantiles;

    // Lower quantiles: ensure q[i] ≤ q[i+1]
    for (let q = 4; q >= 1; q--) {
      if (result[base + q] > result[base + q + 1]) {
        result[base + q] = result[base + q + 1];
      }
    }

    // Upper quantiles: ensure q[i] ≥ q[i-1]
    for (let q = 6; q <= 9; q++) {
      if (result[base + q] < result[base + q - 1]) {
        result[base + q] = result[base + q - 1];
      }
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Input z-score normalization reversal
// ---------------------------------------------------------------------------

/**
 * Reverse the effect of `normalizeInputs` using pre-computed statistics
 * from the original (pre-normalized) inputs.
 */
export function reverseInputNormalization(
  forecasts: Float32Array[],
  stats: { mu: number; sigma: number }[],
): Float32Array[] {
  return forecasts.map((ff, b) => {
    const { mu, sigma } = stats[b] ?? { mu: 0, sigma: 1 };
    const safeSigma = sigma < 1e-6 ? 1.0 : sigma;
    const result = new Float32Array(ff.length);
    for (let i = 0; i < ff.length; i++) {
      result[i] = ff[i] * safeSigma + mu;
    }
    return result;
  });
}
