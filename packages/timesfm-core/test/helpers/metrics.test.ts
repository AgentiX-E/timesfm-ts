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

  it('returns 0 when all values are NaN', () => {
    const actual = new Float32Array([NaN, NaN]);
    const predicted = new Float32Array([NaN, NaN]);
    expect(mae(actual, predicted)).toBe(0);
  });

  it('skips elements where predicted is NaN while actual is finite', () => {
    // Predicted has NaN at index 0; only |3 - 4| = 1 contributes
    const actual = new Float32Array([0, 3]);
    const predicted = new Float32Array([NaN, 4]);
    expect(mae(actual, predicted)).toBeCloseTo(1, 10);
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

  it('throws RangeError on length mismatch', () => {
    expect(() => rmse(new Float32Array([1, 2, 3]), new Float32Array([1, 2]))).toThrow(RangeError);
  });

  it('returns 0 for empty arrays', () => {
    expect(rmse(new Float32Array(0), new Float32Array(0))).toBe(0);
  });

  it('skips NaN values and computes correct RMSE', () => {
    // First pair (0, 10) has NaN in predicted, skipped.
    // Remaining: (|2-5|=3, |3-6|=3) → sqrt((9+9)/2) = sqrt(9) = 3
    const actual = new Float32Array([0, 2, 3]);
    const predicted = new Float32Array([NaN, 5, 6]);
    expect(rmse(actual, predicted)).toBeCloseTo(3, 10);
  });

  it('returns 0 when all values are NaN', () => {
    const actual = new Float32Array([NaN, NaN, NaN]);
    const predicted = new Float32Array([NaN, NaN, NaN]);
    expect(rmse(actual, predicted)).toBe(0);
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

  it('throws RangeError on length mismatch', () => {
    expect(() => mape(new Float32Array([1, 2]), new Float32Array([1, 2, 3]))).toThrow(RangeError);
  });

  it('returns 0 for empty arrays', () => {
    expect(mape(new Float32Array(0), new Float32Array(0))).toBe(0);
  });

  it('returns 0 when all actual values are below 1e-10 threshold', () => {
    // Every |actual_i| < 1e-10 → all skipped → count=0 → 0
    const actual = new Float32Array([0, 1e-11, -1e-11, 0]);
    const predicted = new Float32Array([5, 3, 2, 7]);
    expect(mape(actual, predicted)).toBe(0);
  });

  it('skips NaN values in actual and predicted', () => {
    // NaN in actual skipped, NaN in predicted skipped.
    // Only |(30-33)/30| = 0.1 contributes → (0.1/1)*100 = 10
    const actual = new Float32Array([NaN, 10, 30]);
    const predicted = new Float32Array([5, NaN, 33]);
    expect(mape(actual, predicted)).toBeCloseTo(10, 10);
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

  it('throws RangeError on length mismatch', () => {
    expect(() => smape(new Float32Array([1, 2]), new Float32Array([3]))).toThrow(RangeError);
  });

  it('returns 0 for empty arrays', () => {
    expect(smape(new Float32Array(0), new Float32Array(0))).toBe(0);
  });

  it('skips pairs where denominator (|a|+|p|) ≤ 1e-10', () => {
    // First pair: |0|+|0| = 0 ≤ 1e-10 → skipped
    // Second pair: |0|+|5e-11| = 5e-11 ≤ 1e-10 → skipped
    // Third pair: |1|+|2| = 3 > 1e-10 → contributes
    // SMAPE = 2*|1-2|/3 = 2/3, times 100 → 200/3 ≈ 66.6667
    const actual = new Float32Array([0, 0, 1]);
    const predicted = new Float32Array([0, 5e-11, 2]);
    expect(smape(actual, predicted)).toBeCloseTo(200 / 3, 4);
  });

  it('returns 0 when all values are NaN', () => {
    const actual = new Float32Array([NaN, NaN]);
    const predicted = new Float32Array([NaN, NaN]);
    expect(smape(actual, predicted)).toBe(0);
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

  it('returns 1 when both model and naive are perfect (naiveMAE < 1e-10 and modelMAE < 1e-10)', () => {
    const actual = new Float32Array([5, 5, 5]);
    const modelPred = new Float32Array([5, 5, 5]);
    const naivePred = new Float32Array([5, 5, 5]);
    expect(mase(actual, modelPred, naivePred)).toBe(1);
  });

  it('returns MAX_SAFE_INTEGER when naive is perfect but model has error', () => {
    const actual = new Float32Array([5, 5, 5]);
    const modelPred = new Float32Array([6, 6, 6]); // MAE = 1
    const naivePred = new Float32Array([5, 5, 5]); // MAE = 0 (perfect)
    expect(mase(actual, modelPred, naivePred)).toBe(Number.MAX_SAFE_INTEGER);
  });

  it('returns > 1 when model is worse than naive', () => {
    // model MAE: (|1-5|+|2-5|+|3-5|)/3 = (4+3+2)/3 = 3
    // naive MAE: (|1-1.1|+|2-2.1|+|3-3.1|)/3 = 0.1
    // MASE = 3 / 0.1 = 30 > 1
    const actual = new Float32Array([1, 2, 3]);
    const modelPred = new Float32Array([5, 5, 5]);
    const naivePred = new Float32Array([1.1, 2.1, 3.1]);
    expect(mase(actual, modelPred, naivePred)).toBeGreaterThan(1);
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

  it('throws RangeError on length mismatch', () => {
    expect(() => r2Score(new Float32Array([1, 2]), new Float32Array([3]))).toThrow(RangeError);
  });

  it('returns 0 when all actual values are NaN (n = 0)', () => {
    const actual = new Float32Array([NaN, NaN]);
    const predicted = new Float32Array([1, 2]);
    expect(r2Score(actual, predicted)).toBe(0);
  });

  it('returns 1 when ssTot < 1e-10 (constant actual values)', () => {
    // All actual values are identical → ssTot ≈ 0 → returns 1
    const actual = new Float32Array([7, 7, 7, 7]);
    const predicted = new Float32Array([7, 7, 7, 7]);
    expect(r2Score(actual, predicted)).toBeCloseTo(1, 10);
  });

  it('returns negative R² when prediction is worse than mean baseline', () => {
    // actual=[1,2,3,4,5], mean=3, ssTot=10
    // predicted=[10,10,10,10,10], ssRes=255, R²=1-255/10=-24.5
    const actual = new Float32Array([1, 2, 3, 4, 5]);
    const predicted = new Float32Array([10, 10, 10, 10, 10]);
    const result = r2Score(actual, predicted);
    expect(result).toBeLessThan(0);
    expect(result).toBeCloseTo(-24.5, 5);
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

  it('throws RangeError on length mismatch', () => {
    expect(() =>
      picCoverage(new Float32Array([1, 2]), new Float32Array([0]), new Float32Array([3, 4])),
    ).toThrow(RangeError);
  });

  it('returns 0 for empty arrays', () => {
    expect(picCoverage(new Float32Array(0), new Float32Array(0), new Float32Array(0))).toBe(0);
  });

  it('returns partial coverage when some values are in interval and some out', () => {
    // actual=[1,2,3,4,5], lower=[0,0,6,0,0], upper=[3,3,8,3,3]
    // Covered: 1∈[0,3]✓, 2∈[0,3]✓, 3∈[6,8]✗, 4∈[0,3]✗, 5∈[0,3]✗
    // coverage = 2/5 = 0.4
    const actual = new Float32Array([1, 2, 3, 4, 5]);
    const lower = new Float32Array([0, 0, 6, 0, 0]);
    const upper = new Float32Array([3, 3, 8, 3, 3]);
    expect(picCoverage(actual, lower, upper)).toBeCloseTo(0.4, 10);
  });

  it('skips non-finite values in actual, lower, or upper', () => {
    // actual[1] is NaN → skipped; lower[2] is Infinity → skipped
    // Only first pair (1∈[0,4]) contributes → 1/1 = 1
    const actual = new Float32Array([1, NaN, 3]);
    const lower = new Float32Array([0, 0, Infinity]);
    const upper = new Float32Array([4, 4, 4]);
    expect(picCoverage(actual, lower, upper)).toBe(1);
  });
});

describe('piWidth', () => {
  it('returns mean interval width', () => {
    const lower = new Float32Array([0, 1, 2]);
    const upper = new Float32Array([2, 3, 4]);
    expect(piWidth(lower, upper)).toBeCloseTo(2, 10);
  });

  it('throws RangeError on length mismatch', () => {
    expect(() => piWidth(new Float32Array([0, 1]), new Float32Array([2]))).toThrow(RangeError);
  });

  it('returns 0 for empty arrays', () => {
    expect(piWidth(new Float32Array(0), new Float32Array(0))).toBe(0);
  });

  it('skips non-finite values and computes mean from valid pairs', () => {
    // lower=[0, NaN, 2, Infinity], upper=[2, 3, 4, 5]
    // Only indices 0 and 2 are both finite:
    // (2-0)=2, (4-2)=2 → (2+2)/2 = 2
    const lower = new Float32Array([0, NaN, 2, Infinity]);
    const upper = new Float32Array([2, 3, 4, 5]);
    expect(piWidth(lower, upper)).toBeCloseTo(2, 10);
  });
});
