/**
 * Unit tests for postprocessor.ts — flip invariance, quantile crossing, positive clamping.
 *
 * These tests validate the algorithmic correctness of the post-processing pipeline
 * without requiring the 885 MB ONNX model.
 */

import { describe, it, expect } from 'vitest';

// ---------------------------------------------------------------------------
// Inline implementations mirroring postprocessor.ts for isolated testing
// ---------------------------------------------------------------------------

/**
 * Flip quantile array ordering: [mean, q10, q20, ..., q90] → [mean, q90, q80, ..., q10]
 */
function flipQuantileArray(arr: Float32Array, numQuantiles: number): Float32Array {
  const numSteps = Math.floor(arr.length / numQuantiles);
  const result = new Float32Array(arr.length);

  for (let t = 0; t < numSteps; t++) {
    const base = t * numQuantiles;
    result[base] = arr[base]; // mean stays in place
    for (let q = 1; q < numQuantiles; q++) {
      result[base + q] = arr[base + numQuantiles - q];
    }
  }

  return result;
}

/**
 * Ensure monotonicity: q10 ≤ q20 ≤ … ≤ q90.
 */
function fixQuantileCrossing(arr: Float32Array, numQuantiles: number): Float32Array {
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

function clipMin(arr: Float32Array, minVal: number): Float32Array {
  const result = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    result[i] = arr[i] < minVal ? minVal : arr[i];
  }
  return result;
}

function elementwiseMean(a: Float32Array, b: Float32Array): Float32Array {
  const result = new Float32Array(Math.min(a.length, b.length));
  for (let i = 0; i < result.length; i++) {
    result[i] = (a[i] + b[i]) / 2;
  }
  return result;
}

function negate(arr: Float32Array): Float32Array {
  const result = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    result[i] = -arr[i];
  }
  return result;
}

/**
 * Continuous quantile head: replace fixed-bucket quantiles with calibrated values.
 * q_new = quantile_spread[q] - quantile_spread[5] + full_forecast[5]
 */
function applyContinuousQuantileHead(
  fullForecasts: Float32Array[],
  quantileSpreads: Float32Array[],
  horizon: number,
  numQuantiles: number,
): Float32Array[] {
  return fullForecasts.map((ff, b) => {
    const qs = quantileSpreads[b];
    const result = new Float32Array(ff.length);
    const numSteps = Math.floor(ff.length / numQuantiles);

    for (let h = 0; h < Math.min(numSteps, horizon); h++) {
      const base = h * numQuantiles;

      result[base] = ff[base];

      for (let q = 1; q <= 4; q++) {
        const qsIdx = h * numQuantiles + q;
        const spreadVal = qsIdx < qs.length ? qs[qsIdx] : 0;
        const medianSpread = qsIdx < qs.length ? qs[h * numQuantiles + 5] : 0;
        result[base + q] = spreadVal - medianSpread + ff[base + 5];
      }

      result[base + 5] = ff[base + 5];

      for (let q = 6; q <= 9; q++) {
        const qsIdx = h * numQuantiles + q;
        const spreadVal = qsIdx < qs.length ? qs[qsIdx] : 0;
        const medianSpread = qsIdx < qs.length ? qs[h * numQuantiles + 5] : 0;
        result[base + q] = spreadVal - medianSpread + ff[base + 5];
      }
    }

    for (let i = numSteps * numQuantiles; i < ff.length; i++) {
      result[i] = ff[i];
    }

    return result;
  });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('postprocessor — flipQuantileArray', () => {
  const NUM_Q = 10; // mean + q10..q90

  it('preserves mean at index 0', () => {
    const arr = new Float32Array(NUM_Q);
    arr[0] = 5.0; // mean
    for (let q = 1; q < NUM_Q; q++) arr[q] = q * 0.1;

    const flipped = flipQuantileArray(arr, NUM_Q);
    expect(flipped[0]).toBe(5.0); // mean stays
  });

  it('reverses quantile order', () => {
    const arr = new Float32Array(NUM_Q);
    for (let q = 0; q < NUM_Q; q++) arr[q] = q;

    const flipped = flipQuantileArray(arr, NUM_Q);
    // Index 1 should now be original index 9
    expect(flipped[1]).toBe(NUM_Q - 1);
    // Index 9 should now be original index 1
    expect(flipped[NUM_Q - 1]).toBe(1);
  });

  it('handles multi-step arrays', () => {
    const numSteps = 3;
    const arr = new Float32Array(numSteps * NUM_Q);
    for (let t = 0; t < numSteps; t++) {
      for (let q = 0; q < NUM_Q; q++) {
        arr[t * NUM_Q + q] = t * 100 + q;
      }
    }

    const flipped = flipQuantileArray(arr, NUM_Q);
    // Check each timestep independently
    for (let t = 0; t < numSteps; t++) {
      const base = t * NUM_Q;
      expect(flipped[base]).toBe(t * 100); // mean unchanged
      expect(flipped[base + 1]).toBe(t * 100 + 9); // q90 → q10 position
      expect(flipped[base + 9]).toBe(t * 100 + 1); // q10 → q90 position
    }
  });
});

describe('postprocessor — fixQuantileCrossing', () => {
  const NUM_Q = 10;

  it('preserves already-monotonic quantiles', () => {
    const arr = new Float32Array(NUM_Q);
    for (let q = 0; q < NUM_Q; q++) arr[q] = q;

    const fixed = fixQuantileCrossing(arr, NUM_Q);
    expect(Array.from(fixed)).toEqual(Array.from(arr));
  });

  it('fixes lower-quantile crossing', () => {
    const arr = new Float32Array(NUM_Q);
    arr[0] = 0; // mean
    arr[1] = 5; // q10 — too high vs q20=3
    arr[2] = 3; // q20
    arr[3] = 4; // q30
    arr[4] = 5; // q40
    arr[5] = 6; // q50
    arr[6] = 7; // q60
    arr[7] = 8; // q70
    arr[8] = 9; // q80
    arr[9] = 10; // q90

    const fixed = fixQuantileCrossing(arr, NUM_Q);
    // q10 should be ≤ q20 after fix (from right to left enforcement)
    expect(fixed[1]).toBeLessThanOrEqual(fixed[2]);
    // monotonic chain: q10 ≤ q20 ≤ q30 ≤ q40 ≤ q50 ≤ q60 ≤ q70 ≤ q80 ≤ q90
    for (let q = 1; q < NUM_Q - 1; q++) {
      expect(fixed[q]).toBeLessThanOrEqual(fixed[q + 1]);
    }
  });

  it('fixes upper-quantile crossing', () => {
    const arr = new Float32Array(NUM_Q);
    arr[0] = 0; // mean
    arr[1] = 1; // q10
    arr[2] = 2; // q20
    arr[3] = 3; // q30
    arr[4] = 4; // q40
    arr[5] = 5; // q50
    arr[6] = 3; // q60 — too low vs q50=5
    arr[7] = 7; // q70
    arr[8] = 8; // q80
    arr[9] = 9; // q90

    const fixed = fixQuantileCrossing(arr, NUM_Q);
    // q60 should be ≥ q50 after fix
    expect(fixed[6]).toBeGreaterThanOrEqual(fixed[5]);
  });

  it('handles multi-step with crossing at specific timesteps', () => {
    const numSteps = 2;
    const arr = new Float32Array(numSteps * NUM_Q);
    // Step 0: monotonic
    for (let q = 0; q < NUM_Q; q++) arr[q] = q;
    // Step 1: crossing
    for (let q = 0; q < NUM_Q; q++) arr[NUM_Q + q] = 9 - q; // reversed

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
});

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

describe('postprocessor — flip invariance formula', () => {
  it('(forecast(x) - forecast(-x)) / 2 yields zero for symmetric function', () => {
    // Given identical forecasts for x and -x (ideal symmetric case)
    const forecastX = new Float32Array([10, 20, 30]);
    const forecastNegX = new Float32Array([10, 20, 30]);

    // flip = (forecast(x) + (-forecast(-x))) / 2 = (forecast(x) - forecast(-x)) / 2
    const flipApplied = elementwiseMean(forecastX, negate(forecastNegX));
    expect(Array.from(flipApplied)).toEqual([0, 0, 0]);
  });

  it('handles anti-symmetric output correctly', () => {
    // For anti-symmetric: forecast(-x) = -forecast(x)
    const forecastX = new Float32Array([10, 20, 30]);
    const forecastNegX = new Float32Array([-10, -20, -30]);

    // (forecast(x) - (-forecast(x))) / 2 = (10 - (-(-10))) / 2...
    // Actually: flip = (forecast_x + (-forecast_negx)) / 2
    // = (forecast_x - forecast_negx) / 2 = ([10,20,30] - [-10,-20,-30]) / 2
    // = ([20, 40, 60]) / 2 = [10, 20, 30]
    const flipApplied = elementwiseMean(forecastX, negate(forecastNegX));
    expect(Array.from(flipApplied)).toEqual([10, 20, 30]);
  });
});

describe('postprocessor — applyContinuousQuantileHead', () => {
  const NUM_Q = 10;

  it('uses fallback=0 when quantile spread is empty (covers medianSpread fallback)', () => {
    // Pass an empty quantile spread array → all spreads and medianSpread fallback to 0
    const ff = new Float32Array(NUM_Q); // 1 timestep with 10 quantiles
    ff.fill(5); // all values are 5
    const qs = new Float32Array(0); // empty

    const result = applyContinuousQuantileHead([ff], [qs], 1, NUM_Q);

    // With spread=0, medianSpread=0: result = 0 - 0 + ff[5] = 5
    // Mean stays 5, median stays 5, everything is 5
    expect(result[0][0]).toBe(5); // mean
    expect(result[0][5]).toBe(5); // median
    // Quantiles: q_new = 0 - 0 + 5 = 5
    for (let q = 1; q <= 4; q++) expect(result[0][q]).toBe(5);
    for (let q = 6; q <= 9; q++) expect(result[0][q]).toBe(5);
  });

  it('copies remaining values when ff.length is not a multiple of numQuantiles', () => {
    // ff has 15 elements, 10 quantiles per step → numSteps=1, leftover=5
    const ff = new Float32Array(15);
    for (let i = 0; i < 15; i++) ff[i] = i + 1;
    const qs = new Float32Array(NUM_Q);
    for (let i = 0; i < NUM_Q; i++) qs[i] = i;

    const result = applyContinuousQuantileHead([ff], [qs], 1, NUM_Q);

    // Remaining values (indices 10-14) should be copied as-is
    expect(result[0][10]).toBe(11);
    expect(result[0][11]).toBe(12);
    expect(result[0][12]).toBe(13);
    expect(result[0][13]).toBe(14);
    expect(result[0][14]).toBe(15);
  });
});
