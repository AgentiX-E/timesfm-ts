/**
 * Tests for quantile helper functions.
 */
import { describe, it, expect } from 'vitest';
import { getQuantile, getPredictionInterval } from '../../src/helpers/quantile';
import { QUANTILE_INDICES } from '../../src/types';

function makeMockOutput() {
  const pf = new Float32Array([1, 2, 3]);
  // quantileForecast[seriesIdx][quantileIdx] = Float32Array
  const qf: Float32Array[][] = [
    Array.from({ length: 10 }, (_, q) => new Float32Array([(q + 1) * 10])),
  ];
  return { pointForecast: [pf], quantileForecast: qf };
}

describe('getQuantile', () => {
  const output = makeMockOutput();

  it('extracts Q50 correctly', () => {
    const q50 = getQuantile(output, 0, QUANTILE_INDICES.Q50);
    expect(q50).toBe(output.quantileForecast[0][5]);
  });

  it('extracts Q10 correctly', () => {
    const q10 = getQuantile(output, 0, QUANTILE_INDICES.Q10);
    expect(q10).toBe(output.quantileForecast[0][1]);
  });

  it('throws on out-of-range seriesIndex', () => {
    expect(() => getQuantile(output, 99, 0)).toThrow(RangeError);
  });

  it('throws on negative seriesIndex', () => {
    expect(() => getQuantile(output, -1, 0)).toThrow(RangeError);
  });

  it('throws on out-of-range quantileIndex', () => {
    expect(() => getQuantile(output, 0, 99)).toThrow(RangeError);
  });

  it('returns the same reference (zero-copy)', () => {
    const ref = getQuantile(output, 0, 0);
    ref[0] = 999;
    expect(output.quantileForecast[0][0][0]).toBe(999);
  });
});

describe('getPredictionInterval', () => {
  const output = makeMockOutput();

  it('returns Q10 and Q90 for 80% CI', () => {
    const { lower, upper } = getPredictionInterval(output, 0, 0.8);
    expect(lower).toBe(output.quantileForecast[0][QUANTILE_INDICES.Q10]);
    expect(upper).toBe(output.quantileForecast[0][QUANTILE_INDICES.Q90]);
  });

  it('throws on unsupported confidence level', () => {
    expect(() => getPredictionInterval(output, 0, 0.99 as 0.8 | 0.9 | 0.95)).toThrow(RangeError);
  });
});
