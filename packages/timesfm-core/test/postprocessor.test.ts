/**
 * Unit tests for postprocessor.ts — flip invariance, quantile crossing,
 * positive clamping, continuous quantile head, and integration tests for
 * the main `postProcess` pipeline.
 *
 * All functions are imported from the REAL production source.  No inline
 * reimplementations — these tests validate shipped code, not algorithm spec.
 */

import { describe, it, expect } from 'vitest';
import {
  // Helpers exported from postprocessor.ts
  flipQuantileArray,
  fixQuantileCrossing,
  applyContinuousQuantileHead,
  reverseInputNormalization,
  postProcess,
  // Utilities
  clipMin,
  elementwiseMean,
  negate,
  // Configuration & types
  TIMESFM_25_CONFIG,
  DEFAULT_FORECAST_CONFIG,
  type ForecastConfig,
} from '@agentix-e/timesfm-core';

// ---------------------------------------------------------------------------
// Convenience helpers for constructing test data
// ---------------------------------------------------------------------------

/** Standard TimesFM 2.5 quantile count (mean + q10..q90). */
const NUM_Q = TIMESFM_25_CONFIG.numQuantiles; // 10

/**
 * Build a simple stacked quantile array for one timestep.
 * Values: [mean, q10, q20, q30, q40, q50, q60, q70, q80, q90].
 */
function makeOneStep(values: number[]): Float32Array {
  return new Float32Array(values);
}

/**
 * Build a minimal DecodeResult for one batch element with no AR outputs.
 * Each batch element gets `timesteps * numQuantiles` values filled by `factory`.
 */
function makeDecodeResult(
  timesteps: number,
  factory: (b: number, t: number, q: number) => number,
  batchSize = 1,
): DecodeResult {
  const pfOutputs: Float32Array[] = [];
  const quantileSpreads: Float32Array[] = [];
  for (let b = 0; b < batchSize; b++) {
    const len = timesteps * NUM_Q;
    const pf = new Float32Array(len);
    const qs = new Float32Array(len);
    for (let t = 0; t < timesteps; t++) {
      for (let q = 0; q < NUM_Q; q++) {
        const idx = t * NUM_Q + q;
        pf[idx] = factory(b, t, q);
        qs[idx] = factory(b, t, q) * 0.1; // small spread
      }
    }
    pfOutputs.push(pf);
    quantileSpreads.push(qs);
  }
  return { pfOutputs, quantileSpreads, arOutputs: null };
}

// ===================================================================
// flipQuantileArray
// ===================================================================

describe('postprocessor — flipQuantileArray', () => {
  it('preserves mean at index 0', () => {
    const arr = makeOneStep([5.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
    const flipped = flipQuantileArray(arr, NUM_Q);
    expect(flipped[0]).toBe(5.0);
  });

  it('reverses quantile order for standard 10-quantile layout', () => {
    const arr = makeOneStep([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const flipped = flipQuantileArray(arr, NUM_Q);
    // Index 1 becomes original index 9
    expect(flipped[1]).toBe(9);
    // Index 9 becomes original index 1
    expect(flipped[NUM_Q - 1]).toBe(1);
  });

  it('handles multi-step arrays independently', () => {
    const numSteps = 3;
    const arr = new Float32Array(numSteps * NUM_Q);
    for (let t = 0; t < numSteps; t++) {
      for (let q = 0; q < NUM_Q; q++) {
        arr[t * NUM_Q + q] = t * 100 + q;
      }
    }

    const flipped = flipQuantileArray(arr, NUM_Q);
    for (let t = 0; t < numSteps; t++) {
      const base = t * NUM_Q;
      expect(flipped[base]).toBe(t * 100); // mean unchanged
      expect(flipped[base + 1]).toBe(t * 100 + 9); // q90 → q10
      expect(flipped[base + 9]).toBe(t * 100 + 1); // q10 → q90
    }
  });

  it('works with non-standard quantile count (3 quantiles: mean + q1 + q2)', () => {
    // 2 time steps × 3 quantiles each
    const NQ = 3;
    const arr = new Float32Array([0, 1, 2, 10, 11, 12]);
    const flipped = flipQuantileArray(arr, NQ);
    // Step 0: mean=0 stays, q1/q2 swapped
    expect(flipped[0]).toBe(0); // mean
    expect(flipped[1]).toBe(2); // was index 2
    expect(flipped[2]).toBe(1); // was index 1
    // Step 1
    expect(flipped[3]).toBe(10); // mean
    expect(flipped[4]).toBe(12); // was index 5
    expect(flipped[5]).toBe(11); // was index 4
  });

  it('verify flip + negate pattern used in production code', () => {
    // The production code does: elementwiseMean(forecastX, negate(flippedForecastNegX))
    // flippedForecastNegX = flipQuantileArray(forecastNegX, numQuantiles)
    // Then negated. This simulates (forecast(x) - forecast(-x)) / 2.
    const arr = makeOneStep([5, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const flipped = flipQuantileArray(arr, NUM_Q);
    const negated = negate(flipped);

    // Mean: 5 stays as 5, then negated → -5
    expect(negated[0]).toBe(-5);
    // q10 (index 1) was 1, flipped maps q→q reversed: index 1 gets 9, negated → -9
    expect(negated[1]).toBe(-9);
    // q90 (index 9) was 9, flipped gets 1, negated → -1
    expect(negated[9]).toBe(-1);
  });
});

// ===================================================================
// fixQuantileCrossing
// ===================================================================

describe('postprocessor — fixQuantileCrossing', () => {
  it('preserves already-monotonic quantiles', () => {
    const arr = makeOneStep([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const fixed = fixQuantileCrossing(arr, NUM_Q);
    expect(Array.from(fixed)).toEqual(Array.from(arr));
  });

  it('fixes lower-quantile crossing (right-to-left enforcement)', () => {
    const arr = makeOneStep([0, 5, 3, 4, 5, 6, 7, 8, 9, 10]);
    const fixed = fixQuantileCrossing(arr, NUM_Q);
    // q10 (index 1) should be ≤ q20 (index 2) after fix
    expect(fixed[1]).toBeLessThanOrEqual(fixed[2]);
    // All lower quantiles should be monotonic
    for (let q = 1; q < 4; q++) {
      expect(fixed[q]).toBeLessThanOrEqual(fixed[q + 1]);
    }
  });

  it('fixes upper-quantile crossing', () => {
    const arr = makeOneStep([0, 1, 2, 3, 4, 5, 3, 7, 8, 9]);
    const fixed = fixQuantileCrossing(arr, NUM_Q);
    // q60 (index 6, value 3) should be ≥ q50 (index 5, value 5) after fix
    expect(fixed[6]).toBeGreaterThanOrEqual(fixed[5]);
  });

  it('handles multi-step with crossing at specific timesteps', () => {
    const numSteps = 2;
    const arr = new Float32Array(numSteps * NUM_Q);
    // Step 0: monotonic
    for (let q = 0; q < NUM_Q; q++) arr[q] = q;
    // Step 1: reversed (will have crossing)
    for (let q = 0; q < NUM_Q; q++) arr[NUM_Q + q] = 9 - q;

    const fixed = fixQuantileCrossing(arr, NUM_Q);
    // Step 0: still monotonic
    for (let q = 1; q < NUM_Q - 1; q++) {
      expect(fixed[q]).toBeLessThanOrEqual(fixed[q + 1]);
    }
    // Step 1: fixed to be monotonic
    for (let q = 1; q < NUM_Q - 1; q++) {
      expect(fixed[NUM_Q + q]).toBeLessThanOrEqual(fixed[NUM_Q + q + 1]);
    }
  });

  it('handles all-equal quantiles without error', () => {
    const arr = makeOneStep([3, 3, 3, 3, 3, 3, 3, 3, 3, 3]);
    const fixed = fixQuantileCrossing(arr, NUM_Q);
    // All values should remain 3
    for (let q = 0; q < NUM_Q; q++) {
      expect(fixed[q]).toBe(3);
    }
  });

  it('does not modify mean (index 0) or median (index 5)', () => {
    const arr = makeOneStep([99, 5, 3, 4, 5, 88, 3, 7, 8, 9]);
    const fixed = fixQuantileCrossing(arr, NUM_Q);
    expect(fixed[0]).toBe(99); // mean untouched
    expect(fixed[5]).toBe(88); // median untouched
  });
});

// ===================================================================
// Positive clamping (clipMin from tensor-utils)
// ===================================================================

describe('postprocessor — positive clamping', () => {
  it('clamps negative values to zero', () => {
    const arr = new Float32Array([1, -2, 3, -4, 0]);
    const clamped = clipMin(arr, 0);
    expect(Array.from(clamped)).toEqual([1, 0, 3, 0, 0]);
  });

  it('preserves all-positive array', () => {
    const arr = new Float32Array([1, 2, 3]);
    const clamped = clipMin(arr, 0);
    expect(Array.from(clamped)).toEqual([1, 2, 3]);
  });

  it('handles empty array', () => {
    const arr = new Float32Array(0);
    const clamped = clipMin(arr, 0);
    expect(clamped.length).toBe(0);
  });
});

// ===================================================================
// Flip invariance formula (elementwiseMean + negate from tensor-utils)
// ===================================================================

describe('postprocessor — flip invariance formula', () => {
  it('(forecast(x) - forecast(-x)) / 2 yields zero for symmetric function', () => {
    const forecastX = new Float32Array([10, 20, 30]);
    const forecastNegX = new Float32Array([10, 20, 30]);

    const flipApplied = elementwiseMean(forecastX, negate(forecastNegX));
    expect(Array.from(flipApplied)).toEqual([0, 0, 0]);
  });

  it('handles anti-symmetric output correctly', () => {
    const forecastX = new Float32Array([10, 20, 30]);
    const forecastNegX = new Float32Array([-10, -20, -30]);

    const flipApplied = elementwiseMean(forecastX, negate(forecastNegX));
    expect(Array.from(flipApplied)).toEqual([10, 20, 30]);
  });

  it('produces correct result for generic asymmetric case', () => {
    // (forecast(x) - forecast(-x)) / 2 = ( [10, 20, 30] - [2, 8, 14] ) / 2
    //                               = [8, 12, 16] / 2 = [4, 6, 8]
    const forecastX = new Float32Array([10, 20, 30]);
    const forecastNegX = new Float32Array([2, 8, 14]);

    const flipApplied = elementwiseMean(forecastX, negate(forecastNegX));
    expect(Array.from(flipApplied)).toEqual([4, 6, 8]);
  });
});

// ===================================================================
// applyContinuousQuantileHead
// ===================================================================

describe('postprocessor — applyContinuousQuantileHead', () => {
  it('uses fallback=0 when quantile spread is empty (covers medianSpread fallback)', () => {
    const ff = makeOneStep([5, 5, 5, 5, 5, 5, 5, 5, 5, 5]);
    const qs = new Float32Array(0); // empty

    const result = applyContinuousQuantileHead([ff], [qs], 1, TIMESFM_25_CONFIG);

    // Mean stays 5, median stays 5, all quantiles become 0 - 0 + 5 = 5
    expect(result[0][0]).toBe(5); // mean
    expect(result[0][5]).toBe(5); // median
    for (let q = 1; q <= 4; q++) expect(result[0][q]).toBe(5);
    for (let q = 6; q <= 9; q++) expect(result[0][q]).toBe(5);
  });

  it('copies remaining values when ff.length is not a multiple of numQuantiles', () => {
    const ff = new Float32Array(15);
    for (let i = 0; i < 15; i++) ff[i] = i + 1;
    const qs = new Float32Array(NUM_Q);
    for (let i = 0; i < NUM_Q; i++) qs[i] = i;

    const result = applyContinuousQuantileHead([ff], [qs], 1, TIMESFM_25_CONFIG);

    expect(result[0][10]).toBe(11);
    expect(result[0][11]).toBe(12);
    expect(result[0][12]).toBe(13);
    expect(result[0][13]).toBe(14);
    expect(result[0][14]).toBe(15);
  });

  it('applies canonical TimesFM 2.5 shape with non-zero quantile spreads', () => {
    // 2 timesteps × 10 quantiles using TIMESFM_25_CONFIG
    const ff = new Float32Array(20);
    for (let i = 0; i < 20; i++) ff[i] = 10 + i;
    // Quantile spreads: all 1.0
    const qs = new Float32Array(20);
    qs.fill(1);

    const result = applyContinuousQuantileHead([ff], [qs], 2, TIMESFM_25_CONFIG);

    // Mean stays
    expect(result[0][0]).toBe(10);
    expect(result[0][10]).toBe(20);

    // For quantiles 1-4: q_new = spreadVal - medianSpread + ff[median]
    // spreadVal = 1, medianSpread = 1 (since all spreads = 1)
    // So q_new = 1 - 1 + ff[median] = ff[median]
    // This means the quantiles should match the median for each timestep
    const medianStep0 = ff[5]; // 15
    const medianStep1 = ff[15]; // 25
    for (let q = 1; q <= 4; q++) {
      expect(result[0][q]).toBeCloseTo(medianStep0, 5);
      expect(result[0][10 + q]).toBeCloseTo(medianStep1, 5);
    }
    // Median stays
    expect(result[0][5]).toBe(medianStep0);
    expect(result[0][15]).toBe(medianStep1);
    // Upper quantiles also = median
    for (let q = 6; q <= 9; q++) {
      expect(result[0][q]).toBeCloseTo(medianStep0, 5);
      expect(result[0][10 + q]).toBeCloseTo(medianStep1, 5);
    }
  });

  it('correctly shifts quantiles relative to median when spreads differ', () => {
    // Single timestep, 10 quantiles
    const ff = makeOneStep([0, 1, 2, 3, 4, 10, 6, 7, 8, 9]);

    // Quantile spreads: q[i] spread = i (0..9)
    const qs = new Float32Array(NUM_Q);
    for (let i = 0; i < NUM_Q; i++) qs[i] = i;

    const result = applyContinuousQuantileHead([ff], [qs], 1, TIMESFM_25_CONFIG);

    // Mean stays
    expect(result[0][0]).toBe(0);

    // For q=1 (q10): q_new = qs[1] - qs[5] + ff[5] = 1 - 5 + 10 = 6
    expect(result[0][1]).toBeCloseTo(6, 5);
    // For q=4 (q40): q_new = 4 - 5 + 10 = 9
    expect(result[0][4]).toBeCloseTo(9, 5);

    // Median stays
    expect(result[0][5]).toBe(10);

    // For q=6 (q60): q_new = 6 - 5 + 10 = 11
    expect(result[0][6]).toBeCloseTo(11, 5);
    // For q=9 (q90): q_new = 9 - 5 + 10 = 14
    expect(result[0][9]).toBeCloseTo(14, 5);
  });
});

// ===================================================================
// reverseInputNormalization
// ===================================================================

describe('postprocessor — reverseInputNormalization', () => {
  it('reverses z-score normalization correctly', () => {
    const ff = new Float32Array([0, 1, -1, 0.5]);
    const stats = [{ mu: 100, sigma: 10 }];

    const result = reverseInputNormalization([ff], stats);
    // z=0 → 100, z=1 → 110, z=-1 → 90, z=0.5 → 105
    expect(Array.from(result[0])).toEqual([100, 110, 90, 105]);
  });

  it('uses sigma=1.0 fallback when sigma is near zero', () => {
    const ff = new Float32Array([0, 1, -1]);
    const stats = [{ mu: 50, sigma: 1e-7 }]; // effectively zero

    const result = reverseInputNormalization([ff], stats);
    expect(Array.from(result[0])).toEqual([50, 51, 49]);
  });

  it('handles multiple batch elements independently', () => {
    const ff0 = new Float32Array([0, 1]);
    const ff1 = new Float32Array([0, -1]);
    const stats = [
      { mu: 10, sigma: 2 },
      { mu: 100, sigma: 5 },
    ];

    const result = reverseInputNormalization([ff0, ff1], stats);
    expect(Array.from(result[0])).toEqual([10, 12]); // 0*2+10=10, 1*2+10=12
    expect(Array.from(result[1])).toEqual([100, 95]); // 0*5+100=100, -1*5+100=95
  });
});

// ===================================================================
// postProcess — integration tests
// ===================================================================

describe('postprocessor — postProcess (main entry point)', () => {
  const horizon = 2;
  const fc: ForecastConfig = { ...DEFAULT_FORECAST_CONFIG };
  const mc = TIMESFM_25_CONFIG;

  it('produces point and quantile forecasts for a simple decode result', () => {
    // 1 batch element, 2 timesteps (== horizon), monotonic values
    const dr = makeDecodeResult(horizon, (_b, t, q) => t * 100 + q);

    const output = postProcess(dr, horizon, fc, mc, null, null, null);

    expect(output.pointForecast).toHaveLength(1);
    expect(output.pointForecast[0]).toHaveLength(horizon);
    expect(output.quantileForecast).toHaveLength(1);
    expect(output.quantileForecast[0]).toHaveLength(NUM_Q);
    expect(output.quantileForecast[0][0]).toHaveLength(horizon);
    // backcast should be undefined (returnBackcast is false by default in DEFAULT_FORECAST_CONFIG)
    expect(output.backcast).toBeUndefined();
  });

  it('returns backcast when returnBackcast is enabled', () => {
    const fcWithBackcast: ForecastConfig = { ...DEFAULT_FORECAST_CONFIG, returnBackcast: true };
    // 4 timesteps with outputPatchLen=128 means numPatches=1, not enough for backcast
    // Use 132 timesteps to get at least 2 patches (ceil(132/128)=2)
    const timesteps = mc.outputPatchLen + 4; // 132
    const dr = makeDecodeResult(timesteps, (_b, t, q) => t * 100 + q);

    const output = postProcess(dr, horizon, fcWithBackcast, mc, null, null, null);

    expect(output.backcast).toBeDefined();
    expect(output.backcast).toHaveLength(1);
  });

  it('applies quantile crossing fix when enabled', () => {
    const fcCrossing: ForecastConfig = { ...DEFAULT_FORECAST_CONFIG, fixQuantileCrossing: true };
    // Create crossed data: q10 > q20
    const pf = new Float32Array(20); // 2 timesteps × 10 quantiles
    for (let t = 0; t < 2; t++) {
      const base = t * NUM_Q;
      for (let q = 0; q < NUM_Q; q++) {
        pf[base + q] = t * 100 + (NUM_Q - 1 - q); // reversed per step
      }
    }
    const dr: DecodeResult = {
      pfOutputs: [pf],
      quantileSpreads: [new Float32Array(pf.length)],
      arOutputs: null,
    };

    const output = postProcess(dr, horizon, fcCrossing, mc, null, null, null);

    // Point forecast should be the median (index 5) at each horizon step
    // After crossing fix, quantiles are monotonic
    const qf = output.quantileForecast[0];
    for (let h = 0; h < horizon; h++) {
      for (let q = 1; q < NUM_Q - 1; q++) {
        expect(qf[q][h]).toBeLessThanOrEqual(qf[q + 1][h]);
      }
    }
  });

  it('applies input normalization reversal when normalizeInputs is true', () => {
    const fcNorm: ForecastConfig = { ...DEFAULT_FORECAST_CONFIG, normalizeInputs: true };
    const dr = makeDecodeResult(horizon, (_b, t, q) => t * 10 + q);
    const stats = [{ mu: 50, sigma: 2 }];

    const outputNoNorm = postProcess(dr, horizon, fcNorm, mc, null, null, null);
    const outputWithNorm = postProcess(dr, horizon, fcNorm, mc, stats, null, null);

    // With normalization reversal, values should be scaled up by sigma=2 and shifted by mu=50
    // timestep 0, median (index 5): pf[5] = 5, reversed: 5*2+50 = 60
    expect(outputWithNorm.pointForecast[0][0]).not.toEqual(outputNoNorm.pointForecast[0][0]);
    expect(outputWithNorm.pointForecast[0][0]).toBe(60); // 5*2 + 50
  });

  it('applies positive clamping when inferIsPositive is true and flag is set', () => {
    const fcPos: ForecastConfig = { ...DEFAULT_FORECAST_CONFIG, inferIsPositive: true };
    // Create data with negative values
    const pf = new Float32Array(20);
    for (let t = 0; t < 2; t++) {
      const base = t * NUM_Q;
      for (let q = 0; q < NUM_Q; q++) {
        pf[base + q] = (t === 1) ? -10 - q : 10 + q; // second step is negative
      }
    }
    const dr: DecodeResult = {
      pfOutputs: [pf],
      quantileSpreads: [new Float32Array(pf.length)],
      arOutputs: null,
    };

    const output = postProcess(dr, horizon, fcPos, mc, null, null, [true]);

    // Step 1 (horizon index 1) should be clamped to >= 0
    for (let q = 0; q < NUM_Q; q++) {
      expect(output.quantileForecast[0][q][1]).toBeGreaterThanOrEqual(0);
    }
  });

  it('does not clamp when isPositive flag is false', () => {
    const fcPos: ForecastConfig = { ...DEFAULT_FORECAST_CONFIG, inferIsPositive: true };
    const pf = new Float32Array(20);
    for (let t = 0; t < 2; t++) {
      const base = t * NUM_Q;
      for (let q = 0; q < NUM_Q; q++) {
        pf[base + q] = (t === 1) ? -10 - q : 10 + q;
      }
    }
    const dr: DecodeResult = {
      pfOutputs: [pf],
      quantileSpreads: [new Float32Array(pf.length)],
      arOutputs: null,
    };

    const output = postProcess(dr, horizon, fcPos, mc, null, null, [false]);

    // Step 1 should still be negative (not clamped)
    expect(output.quantileForecast[0][0][1]).toBeLessThan(0);
  });

  it('handles empty decode result gracefully', () => {
    const dr: DecodeResult = {
      pfOutputs: [],
      quantileSpreads: [],
      arOutputs: null,
    };

    const output = postProcess(dr, 0, fc, mc, null, null, null);

    expect(output.pointForecast).toHaveLength(0);
    expect(output.quantileForecast).toHaveLength(0);
  });

  it('truncates forecast to horizon length', () => {
    // Give more timesteps than horizon — should be truncated
    const timesteps = horizon + 3; // 5 steps, horizon=2
    const dr = makeDecodeResult(timesteps, (_b, t, q) => t * 100 + q);

    const output = postProcess(dr, horizon, fc, mc, null, null, null);

    expect(output.pointForecast[0]).toHaveLength(horizon);
    for (let q = 0; q < NUM_Q; q++) {
      expect(output.quantileForecast[0][q]).toHaveLength(horizon);
    }
  });

  it('preserves median as point forecast', () => {
    const dr = makeDecodeResult(horizon, (_b, t, q) => (t * 100) + q);
    const output = postProcess(dr, horizon, fc, mc, null, null, null);

    // Point forecast should be median (index 5) at each step
    for (let h = 0; h < horizon; h++) {
      expect(output.pointForecast[0][h]).toBe(h * 100 + 5);
    }
  });

  it('handles multiple batch elements', () => {
    const batchSize = 3;
    const dr = makeDecodeResult(horizon, (b, t, q) => b * 1000 + t * 100 + q, batchSize);

    const output = postProcess(dr, horizon, fc, mc, null, null, null);

    expect(output.pointForecast).toHaveLength(batchSize);
    expect(output.quantileForecast).toHaveLength(batchSize);
    for (let b = 0; b < batchSize; b++) {
      expect(output.pointForecast[b]).toHaveLength(horizon);
    }
  });
});
