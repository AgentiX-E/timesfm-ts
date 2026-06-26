/**
 * Tests for time-series forecast evaluation metrics.
 */
import { describe, it, expect } from 'vitest';
import {
  mae,
  rmse,
  mape,
  smape,
  mase,
  r2Score,
  picCoverage,
  piWidth,
} from '../../src/helpers/metrics';

describe('mae', () => {
  it('returns 0 for perfect prediction', () => {
    expect(mae(new Float32Array([1, 2, 3]), new Float32Array([1, 2, 3]))).toBe(0);
  });

  it('computes correct error', () => {
    expect(mae(new Float32Array([1, 2, 3]), new Float32Array([2, 3, 4]))).toBeCloseTo(1, 10);
  });

  it('skips non-finite values', () => {
    expect(mae(new Float32Array([1, NaN, 3]), new Float32Array([2, 3, 4]))).toBeCloseTo(1, 10);
  });

  it('returns 0 for empty arrays', () => {
    expect(mae(new Float32Array(0), new Float32Array(0))).toBe(0);
  });

  it('throws on length mismatch', () => {
    expect(() => mae(new Float32Array([1, 2]), new Float32Array([1]))).toThrow(RangeError);
  });
});

describe('rmse', () => {
  it('returns 0 for perfect prediction', () => {
    expect(rmse(new Float32Array([1, 2, 3]), new Float32Array([1, 2, 3]))).toBe(0);
  });

  it('RMSE > MAE for outliers', () => {
    const actual = new Float32Array([1, 1, 1, 1, 10]);
    const pred = new Float32Array([1, 1, 1, 1, 1]);
    const errMae = mae(actual, pred);
    const errRmse = rmse(actual, pred);
    expect(errRmse).toBeGreaterThan(errMae);
  });
});

describe('mape', () => {
  it('returns 0 for perfect prediction', () => {
    expect(mape(new Float32Array([10, 20]), new Float32Array([10, 20]))).toBe(0);
  });

  it('handles zero actual values', () => {
    const result = mape(new Float32Array([0, 10, 20]), new Float32Array([1, 11, 19]));
    expect(Number.isFinite(result)).toBe(true);
    expect(result).toBeGreaterThan(0);
  });
});

describe('smape', () => {
  it('is symmetric', () => {
    const a = new Float32Array([1, 2, 3]);
    const b = new Float32Array([2, 3, 4]);
    expect(smape(a, b)).toBeCloseTo(smape(b, a), 10);
  });

  it('range is [0, 200]', () => {
    const result = smape(new Float32Array([1, 100]), new Float32Array([100, 1]));
    expect(result).toBeGreaterThanOrEqual(0);
    expect(result).toBeLessThanOrEqual(200);
  });
});

describe('mase', () => {
  it('returns < 1 when model beats naive', () => {
    const actual = new Float32Array([2, 3, 4, 5, 6]);
    const modelPred = new Float32Array([2.1, 3.1, 4.1, 5.1, 6.1]);
    const naivePred = new Float32Array([1, 2, 3, 4, 5]); // worse
    expect(mase(actual, modelPred, naivePred)).toBeLessThan(1);
  });

  it('returns 1 when model equals naive', () => {
    const vals = new Float32Array([1, 2, 3]);
    expect(mase(vals, vals, vals)).toBe(1);
  });
});

describe('r2Score', () => {
  it('returns 1 for perfect fit', () => {
    const vals = new Float32Array([1, 2, 3]);
    expect(r2Score(vals, vals)).toBeCloseTo(1, 10);
  });

  it('returns ~0 for mean baseline', () => {
    const actual = new Float32Array([1, 2, 3, 4, 5]);
    const mean = 3;
    const pred = new Float32Array([mean, mean, mean, mean, mean]);
    expect(r2Score(actual, pred)).toBeCloseTo(0, 5);
  });

  it('returns 0 for empty', () => {
    expect(r2Score(new Float32Array(0), new Float32Array(0))).toBe(0);
  });
});

describe('picCoverage', () => {
  it('returns 1 when all values covered', () => {
    const actual = new Float32Array([1, 2, 3]);
    const lower = new Float32Array([0, 0, 0]);
    const upper = new Float32Array([4, 4, 4]);
    expect(picCoverage(actual, lower, upper)).toBeCloseTo(1, 10);
  });

  it('returns 0 when none covered', () => {
    const actual = new Float32Array([1, 2, 3]);
    const lower = new Float32Array([5, 5, 5]);
    const upper = new Float32Array([6, 6, 6]);
    expect(picCoverage(actual, lower, upper)).toBe(0);
  });
});

describe('piWidth', () => {
  it('returns mean interval width', () => {
    const lower = new Float32Array([0, 1, 2]);
    const upper = new Float32Array([2, 3, 4]);
    expect(piWidth(lower, upper)).toBeCloseTo(2, 10);
  });
});
