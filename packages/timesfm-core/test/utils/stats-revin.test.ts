/**
 * Tests for Welford online statistics and RevIN normalization.
 *
 * Mirrors the Python tests in tests/test_torch_utils.py.
 */

import { describe, it, expect } from 'vitest';
import { createRunningStats, updateRunningStats, computeStats } from '../../src/utils/stats.ts';
import { revin, revinBatch, revinBatch4D } from '../../src/utils/revin.ts';

// ---------------------------------------------------------------------------
// updateRunningStats
// ---------------------------------------------------------------------------

describe('updateRunningStats', () => {
  it('single batch matches numpy ground truth', () => {
    const stats = createRunningStats();
    const values = new Float32Array([1, 2, 3, 4, 5]);
    const mask = new Uint8Array(5); // all zeros = all valid

    const [result] = updateRunningStats(stats, values, mask);

    expect(result.n).toBe(5);
    expect(result.mu).toBeCloseTo(3, 5);
    expect(result.sigma).toBeCloseTo(Math.sqrt(2), 5); // pop std of [1,2,3,4,5]
  });

  it('incremental accumulation matches full computation', () => {
    const batch1 = new Float32Array([1, 2, 3]);
    const batch2 = new Float32Array([4, 5, 6]);
    const mask = new Uint8Array(3);

    // Full
    const fullValues = new Float32Array([1, 2, 3, 4, 5, 6]);
    const fullMask = new Uint8Array(6);
    const [fullResult] = updateRunningStats(createRunningStats(), fullValues, fullMask);

    // Incremental
    const [step1] = updateRunningStats(createRunningStats(), batch1, mask);
    const [step2] = updateRunningStats(step1, batch2, mask);

    expect(step2.n).toBe(fullResult.n);
    expect(step2.mu).toBeCloseTo(fullResult.mu, 5);
    expect(step2.sigma).toBeCloseTo(fullResult.sigma, 5);
  });

  it('masked elements are completely ignored', () => {
    const stats = createRunningStats();
    const values = new Float32Array([0, 10, 20]);
    const mask = new Uint8Array([1, 0, 0]); // first element masked

    const [result] = updateRunningStats(stats, values, mask);

    expect(result.n).toBe(2);
    expect(result.mu).toBeCloseTo(15, 5);
    expect(result.sigma).toBeCloseTo(5, 5);
  });

  it('all-masked input returns zero stats without NaN', () => {
    const stats = createRunningStats();
    const values = new Float32Array([99, 99, 99]);
    const mask = new Uint8Array([1, 1, 1]);

    const [result] = updateRunningStats(stats, values, mask);

    expect(result.n).toBe(0);
    expect(result.mu).toBe(0);
    expect(result.sigma).toBe(0);
  });

  it('constant input yields zero sigma', () => {
    const stats = createRunningStats();
    const values = new Float32Array([7, 7, 7, 7]);
    const mask = new Uint8Array(4);

    const [result] = updateRunningStats(stats, values, mask);

    expect(result.mu).toBeCloseTo(7, 5);
    expect(result.sigma).toBeCloseTo(0, 5);
  });

  it('sigma is always non-negative', () => {
    const stats = createRunningStats();
    const values = new Float32Array([1, 2, 1, 2, 1, 2]);
    const mask = new Uint8Array(6);

    const [result] = updateRunningStats(stats, values, mask);
    expect(result.sigma).toBeGreaterThanOrEqual(0);
  });

  it('skips NaN values to prevent running stats corruption', () => {
    const stats = createRunningStats();
    // [1, NaN, 2, 3] → valid: 1, 2, 3 → mean=2, std≈0.816
    const values = new Float32Array([1, NaN, 2, 3]);
    const mask = new Uint8Array(4);

    const [result] = updateRunningStats(stats, values, mask);
    expect(Number.isFinite(result.mu)).toBe(true);
    expect(result.mu).toBeCloseTo(2, 5);
    expect(Number.isFinite(result.sigma)).toBe(true);
  });

  it('all NaN yields original stats unchanged', () => {
    const stats = createRunningStats();
    const values = new Float32Array([NaN, NaN, NaN]);
    const mask = new Uint8Array(3);

    const [result] = updateRunningStats(stats, values, mask);
    // All values skipped → stats unchanged
    expect(result.n).toBe(0);
    expect(result.mu).toBe(0);
    expect(result.sigma).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// computeStats
// ---------------------------------------------------------------------------

describe('computeStats', () => {
  it('computes mean and std', () => {
    const values = new Float32Array([1, 2, 3, 4, 5]);
    const { mean, std } = computeStats(values);
    expect(mean).toBeCloseTo(3, 5);
    expect(std).toBeCloseTo(Math.sqrt(2), 5);
  });

  it('respects mask', () => {
    const values = new Float32Array([99, 1, 2, 3]);
    const mask = new Uint8Array([1, 0, 0, 0]);
    const { mean } = computeStats(values, mask);
    expect(mean).toBeCloseTo(2, 5);
  });

  it('handles empty data', () => {
    const { mean, std } = computeStats(new Float32Array(0));
    expect(mean).toBe(0);
    expect(std).toBe(0);
  });

  it('skips NaN values to prevent data corruption', () => {
    // Without NaN skipping, a single NaN would make mean=NaN, std=NaN
    const values = new Float32Array([1, NaN, 3, 4, 5]);
    const { mean, std } = computeStats(values);
    expect(Number.isFinite(mean)).toBe(true);
    expect(mean).toBeCloseTo(3.25, 5);
    expect(Number.isFinite(std)).toBe(true);
  });

  it('skips Infinity values', () => {
    const values = new Float32Array([1, Infinity, 3, 4, 5]);
    const { mean, std } = computeStats(values);
    expect(Number.isFinite(mean)).toBe(true);
    expect(mean).toBeCloseTo(3.25, 5);
    expect(Number.isFinite(std)).toBe(true);
  });

  it('all NaN values returns zero stats', () => {
    const values = new Float32Array([NaN, NaN, NaN]);
    const { mean, std } = computeStats(values);
    expect(mean).toBe(0);
    expect(std).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// revin
// ---------------------------------------------------------------------------

describe('revin', () => {
  it('forward then reverse is identity (round-trip)', () => {
    const x = new Float32Array([10, 20, 30]);
    const mu = new Float32Array([20]);
    const sigma = new Float32Array([10]);

    const normed = revin(x, mu, sigma, false);
    const recovered = revin(normed, mu, sigma, true);

    for (let i = 0; i < 3; i++) {
      expect(recovered[i]).toBeCloseTo(x[i], 5);
    }
  });

  it('forward produces correct normalization', () => {
    const x = new Float32Array([10, 20, 30]);
    const mu = new Float32Array([20]);
    const sigma = new Float32Array([10]);

    const normed = revin(x, mu, sigma, false);

    expect(normed[0]).toBeCloseTo(-1, 5);
    expect(normed[1]).toBeCloseTo(0, 5);
    expect(normed[2]).toBeCloseTo(1, 5);
  });

  it('reverse produces correct denormalization', () => {
    const normed = new Float32Array([-1, 0, 1]);
    const mu = new Float32Array([20]);
    const sigma = new Float32Array([10]);

    const recovered = revin(normed, mu, sigma, true);

    expect(recovered[0]).toBeCloseTo(10, 5);
    expect(recovered[1]).toBeCloseTo(20, 5);
    expect(recovered[2]).toBeCloseTo(30, 5);
  });

  it('zero sigma does not produce NaN', () => {
    const x = new Float32Array([5, 5, 5]);
    const mu = new Float32Array([5]);
    const sigma = new Float32Array([0]);

    const normed = revin(x, mu, sigma, false);

    for (let i = 0; i < 3; i++) {
      expect(Number.isFinite(normed[i])).toBe(true);
    }
  });

  it('near-zero sigma guarded by tolerance', () => {
    const x = new Float32Array([1, 2, 3]);
    const mu = new Float32Array([2]);
    const sigma = new Float32Array([5e-7]); // below 1e-6 threshold

    const normed = revin(x, mu, sigma, false);

    // With sigma replaced by 1.0: result = x - mu
    expect(normed[0]).toBeCloseTo(-1, 5);
    expect(normed[1]).toBeCloseTo(0, 5);
    expect(normed[2]).toBeCloseTo(1, 5);
  });

  it('handles negative values correctly', () => {
    const x = new Float32Array([-10, -5, 0, 5, 10]);
    const mu = new Float32Array([0]);
    const sigma = new Float32Array([7.07]);

    const normed = revin(x, mu, sigma, false);
    const recovered = revin(normed, mu, sigma, true);

    for (let i = 0; i < 5; i++) {
      expect(recovered[i]).toBeCloseTo(x[i], 2);
    }
  });
});

// ---------------------------------------------------------------------------
// revinBatch
// ---------------------------------------------------------------------------

describe('revinBatch', () => {
  it('round-trip with batched 2D input', () => {
    const batch = 2;
    const patches = 4;
    const patchLen = 32;

    // Generate random data
    const values: Float32Array[] = [];
    const mu: Float32Array[] = [];
    const sigma: Float32Array[] = [];

    for (let b = 0; b < batch; b++) {
      const arr = new Float32Array(patches * patchLen);
      for (let i = 0; i < arr.length; i++) arr[i] = Math.random() * 100;
      values.push(arr);
      for (let p = 0; p < patches; p++) {
        mu.push(new Float32Array([Math.random() * 10]));
        sigma.push(new Float32Array([Math.random() * 5 + 1]));
      }
    }

    const normed = revinBatch(values, mu, sigma, false, patches, patchLen);
    const recovered = revinBatch(normed, mu, sigma, true, patches, patchLen);

    for (let b = 0; b < batch; b++) {
      for (let i = 0; i < values[b].length; i++) {
        expect(recovered[b][i]).toBeCloseTo(values[b][i], 3);
      }
    }
  });
});

// ---------------------------------------------------------------------------
// revinBatch4D
// ---------------------------------------------------------------------------

describe('revinBatch4D', () => {
  it('round-trip with quantile-shaped 4D output', () => {
    const batchSize = 1;
    const numPatches = 2;
    const patchLen = 2;
    const numQuantiles = 10;
    const perBatch = numPatches * patchLen * numQuantiles;

    // Generate input values
    const values: Float32Array[] = [];
    const mu: Float32Array[] = [];
    const sigma: Float32Array[] = [];

    const arr = new Float32Array(batchSize * perBatch);
    for (let i = 0; i < perBatch; i++) arr[i] = (i + 1) * 0.5;
    values.push(arr);

    for (let p = 0; p < numPatches; p++) {
      mu.push(new Float32Array([10 + p]));
      sigma.push(new Float32Array([2]));
    }

    const normed = revinBatch4D(values, mu, sigma, false, numPatches, patchLen, numQuantiles);
    const recovered = revinBatch4D(normed, mu, sigma, true, numPatches, patchLen, numQuantiles);

    for (let i = 0; i < perBatch; i++) {
      expect(recovered[0][i]).toBeCloseTo(values[0][i], 5);
    }
  });
});

// ---------------------------------------------------------------------------
// revin — flattenParam + broadcast1D edge cases
// ---------------------------------------------------------------------------

describe('revin — flattenParam / broadcast1D edge cases', () => {
  it('handles mu/sigma as arrays of scalar Float32Arrays (per-element)', () => {
    // This triggers flattenParam with param as Float32Array[] where param.length === len
    const x = new Float32Array([10, 20, 30]);
    const muArr: Float32Array[] = [
      new Float32Array([5]),
      new Float32Array([15]),
      new Float32Array([25]),
    ];
    const sigmaArr: Float32Array[] = [
      new Float32Array([5]),
      new Float32Array([5]),
      new Float32Array([5]),
    ];

    const normed = revin(x, muArr, sigmaArr, false);
    // element 0: (10 - 5) / 5 = 1
    // element 1: (20 - 15) / 5 = 1
    // element 2: (30 - 25) / 5 = 1
    expect(normed[0]).toBeCloseTo(1, 5);
    expect(normed[1]).toBeCloseTo(1, 5);
    expect(normed[2]).toBeCloseTo(1, 5);
  });

  it('handles mu/sigma as arrays with per-patch broadcasting', () => {
    // param.length < len → ratio = len / param.length, per-patch repeat
    const x = new Float32Array([1, 2, 3, 4]);
    const muArr: Float32Array[] = [new Float32Array([10]), new Float32Array([20])];
    const sigmaArr: Float32Array[] = [new Float32Array([1]), new Float32Array([1])];

    const normed = revin(x, muArr, sigmaArr, false);
    // ratio = 4/2 = 2, so each mu applied to 2 consecutive elements
    // element 0: (1 - 10) / 1 = -9
    // element 1: (2 - 10) / 1 = -8
    expect(normed[0]).toBeCloseTo(-9, 5);
    expect(normed[1]).toBeCloseTo(-8, 5);
    // element 2: (3 - 20) / 1 = -17
    // element 3: (4 - 20) / 1 = -16
    expect(normed[2]).toBeCloseTo(-17, 5);
    expect(normed[3]).toBeCloseTo(-16, 5);
  });

  it('handles mu as Float32Array with length > 1 (broadcast1D direct pass-through)', () => {
    // broadcast1D returns arr as-is when arr.length !== 1
    const x = new Float32Array([10, 20, 30]);
    const mu = new Float32Array([5, 15, 25]); // length 3, same as x
    const sigma = new Float32Array([5, 5, 5]);

    const normed = revin(x, mu, sigma, false);
    expect(normed[0]).toBeCloseTo(1, 5);
    expect(normed[1]).toBeCloseTo(1, 5);
    expect(normed[2]).toBeCloseTo(1, 5);
  });
});

// ---------------------------------------------------------------------------
// revinBatch — per-batch mode and safe sigma edge cases
// ---------------------------------------------------------------------------

describe('revinBatch — per-batch mu/sigma and safe sigma', () => {
  it('uses per-batch broadcasting when mu.length === batchSize (not per-patch)', () => {
    // Line 141 in revin.ts: `const perPatch = mu.length === batchSize * numPatches;`
    // When mu.length === batchSize, perPatch=false → same mu/sigma for all patches
    const batch = 2;
    const patches = 3;
    const patchLen = 4;

    const values: Float32Array[] = [];
    const mu: Float32Array[] = [];
    const sigma: Float32Array[] = [];

    for (let b = 0; b < batch; b++) {
      const arr = new Float32Array(patches * patchLen);
      for (let i = 0; i < arr.length; i++) arr[i] = b * 100 + i;
      values.push(arr);
      mu.push(new Float32Array([50 + b * 100]));
      sigma.push(new Float32Array([2]));
    }

    const normed = revinBatch(values, mu, sigma, false, patches, patchLen);
    const recovered = revinBatch(normed, mu, sigma, true, patches, patchLen);

    for (let b = 0; b < batch; b++) {
      for (let i = 0; i < values[b].length; i++) {
        expect(recovered[b][i]).toBeCloseTo(values[b][i], 3);
      }
    }
  });

  it('handles near-zero sigma via safe sigma fallback (forward pass)', () => {
    // Line 106 in revin.ts: `const safeS = s < TOLERANCE ? 1.0 : s;`
    const values: Float32Array[] = [new Float32Array([10, 20, 30, 40])];
    const mu: Float32Array[] = [new Float32Array([25])];
    const sigma: Float32Array[] = [new Float32Array([0])]; // zero sigma triggers safeS

    const normed = revinBatch(values, mu, sigma, false, 1, 4);
    // safeS=1, so normed[i] = (value[i] - 25) / 1
    expect(normed[0][0]).toBeCloseTo(-15, 5);
    expect(normed[0][1]).toBeCloseTo(-5, 5);
    expect(normed[0][2]).toBeCloseTo(5, 5);
    expect(normed[0][3]).toBeCloseTo(15, 5);
  });
});

// ---------------------------------------------------------------------------
// revinBatch4D — per-batch mode and safe sigma edge cases
// ---------------------------------------------------------------------------

describe('revinBatch4D — per-batch mu/sigma and safe sigma', () => {
  it('uses per-batch broadcasting when mu.length === batchSize', () => {
    // Line 141, 152 in revin.ts: perPatch=false path in 4D mode
    const batchSize = 2;
    const numPatches = 1;
    const patchLen = 2;
    const numQuantiles = 3;
    const perBatch = numPatches * patchLen * numQuantiles; // 1*2*3 = 6

    const values: Float32Array[] = [];
    const mu: Float32Array[] = [];
    const sigma: Float32Array[] = [];

    for (let b = 0; b < batchSize; b++) {
      const arr = new Float32Array(perBatch);
      for (let i = 0; i < perBatch; i++) arr[i] = b * 10 + i + 1;
      values.push(arr);
      mu.push(new Float32Array([b * 10 + 3]));
      sigma.push(new Float32Array([2]));
    }

    const normed = revinBatch4D(values, mu, sigma, false, numPatches, patchLen, numQuantiles);
    const recovered = revinBatch4D(normed, mu, sigma, true, numPatches, patchLen, numQuantiles);

    for (let b = 0; b < batchSize; b++) {
      for (let i = 0; i < perBatch; i++) {
        expect(recovered[b][i]).toBeCloseTo(values[b][i], 5);
      }
    }
  });

  it('handles near-zero sigma via safe sigma fallback in 4D (forward pass)', () => {
    // Line 159 in revin.ts: `const safeS = s < TOLERANCE ? 1.0 : s;` in 4D mode
    const values: Float32Array[] = [new Float32Array([1, 2, 3, 4, 5, 6])]; // 1 batch, 1 patch × 2 len × 3 q
    const mu: Float32Array[] = [new Float32Array([3])];
    const sigma: Float32Array[] = [new Float32Array([0])]; // zero sigma triggers safeS

    const normed = revinBatch4D(values, mu, sigma, false, 1, 2, 3);
    // safeS=1, so each value = (val - 3) / 1
    expect(normed[0][0]).toBeCloseTo(-2, 5);
    expect(normed[0][1]).toBeCloseTo(-1, 5);
    expect(normed[0][2]).toBeCloseTo(0, 5);
    expect(normed[0][3]).toBeCloseTo(1, 5);
    expect(normed[0][4]).toBeCloseTo(2, 5);
    expect(normed[0][5]).toBeCloseTo(3, 5);
  });
});
