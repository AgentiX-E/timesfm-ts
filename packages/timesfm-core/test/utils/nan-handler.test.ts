/**
 * Tests for NaN handling utilities.
 *
 * Mirrors the Python test conventions in tests/test_base_utils.py.
 */

import { describe, it, expect } from 'vitest';
import {
  stripLeadingNaNs,
  linearInterpolateNaNs,
  cleanSeries,
  hasNaN,
  countNaN,
  stripTrailingNaNs,
  replaceInfWithNaN,
} from '../../src/utils/nan-handler.ts';

// ---------------------------------------------------------------------------
// stripLeadingNaNs
// ---------------------------------------------------------------------------

describe('stripLeadingNaNs', () => {
  it('strips leading NaN values', () => {
    const input = new Float32Array([NaN, NaN, 1, 2, 3]);
    const result = stripLeadingNaNs(input);
    expect(Array.from(result)).toEqual([1, 2, 3]);
  });

  it('returns empty array for all-NaN input', () => {
    const input = new Float32Array([NaN, NaN, NaN]);
    const result = stripLeadingNaNs(input);
    expect(result.length).toBe(0);
  });

  it('returns same array if no leading NaN', () => {
    const input = new Float32Array([1, 2, 3]);
    const result = stripLeadingNaNs(input);
    expect(Array.from(result)).toEqual([1, 2, 3]);
  });

  it('preserves internal NaN values', () => {
    const input = new Float32Array([NaN, 1, NaN, 3]);
    const result = stripLeadingNaNs(input);
    expect(result.length).toBe(3);
    expect(result[0]).toBe(1);
    expect(Number.isNaN(result[1])).toBe(true);
    expect(result[2]).toBe(3);
  });

  it('handles empty array', () => {
    const result = stripLeadingNaNs(new Float32Array(0));
    expect(result.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// linearInterpolateNaNs
// ---------------------------------------------------------------------------

describe('linearInterpolateNaNs', () => {
  it('returns original array when no NaN present', () => {
    const input = new Float32Array([1, 2, 3, 4, 5]);
    const result = linearInterpolateNaNs(input);
    expect(Array.from(result)).toEqual([1, 2, 3, 4, 5]);
  });

  it('interpolates internal NaN values', () => {
    const input = new Float32Array([1, NaN, 3]);
    const result = linearInterpolateNaNs(input);
    expect(result[1]).toBeCloseTo(2, 5);
  });

  it('interpolates multiple consecutive NaN values', () => {
    const input = new Float32Array([1, NaN, NaN, NaN, 5]);
    const result = linearInterpolateNaNs(input);
    expect(result[1]).toBeCloseTo(2, 5);
    expect(result[2]).toBeCloseTo(3, 5);
    expect(result[3]).toBeCloseTo(4, 5);
    expect(result[0]).toBe(1);
    expect(result[4]).toBe(5);
  });

  it('extrapolates trailing NaN using last valid value', () => {
    const input = new Float32Array([1, 2, 3, NaN, NaN]);
    const result = linearInterpolateNaNs(input);
    expect(result[3]).toBe(3);
    expect(result[4]).toBe(3);
  });

  it('extrapolates leading NaN using first valid value', () => {
    const input = new Float32Array([NaN, NaN, 3, 4]);
    const result = linearInterpolateNaNs(input);
    expect(result[0]).toBe(3);
    expect(result[1]).toBe(3);
    expect(result[2]).toBe(3);
    expect(result[3]).toBe(4);
  });

  it('returns all zeros for all-NaN input', () => {
    const input = new Float32Array([NaN, NaN, NaN]);
    const result = linearInterpolateNaNs(input);
    expect(Array.from(result)).toEqual([0, 0, 0]);
  });

  it('fills NaN with single valid value', () => {
    const input = new Float32Array([NaN, 5, NaN]);
    const result = linearInterpolateNaNs(input);
    expect(result[0]).toBe(5);
    expect(result[1]).toBe(5);
    expect(result[2]).toBe(5);
  });

  it('handles large array efficiently', () => {
    const input = new Float32Array(10000);
    input.fill(NaN);
    input[0] = 1;
    input[9999] = 100;
    const start = Date.now();
    const result = linearInterpolateNaNs(input);
    const elapsed = Date.now() - start;
    expect(elapsed).toBeLessThan(3000); // O(n) — generous threshold for CI/VM environments
    expect(result[0]).toBe(1);
    expect(result[9999]).toBe(100);
  });
});

// ---------------------------------------------------------------------------
// cleanSeries
// ---------------------------------------------------------------------------

describe('cleanSeries', () => {
  it('handles leading, internal, and trailing NaN', () => {
    const input = new Float32Array([NaN, NaN, 1, NaN, 3, NaN]);
    const result = cleanSeries(input);
    // Leading stripped: [1, NaN, 3, NaN]
    // Trailing stripped: [1, NaN, 3]
    // Internal interpolated: [1, 2, 3]
    expect(result.length).toBe(3);
    expect(Array.from(result)).toEqual([1, 2, 3]);
  });

  it('returns empty for all-NaN', () => {
    expect(cleanSeries(new Float32Array([NaN, NaN])).length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

describe('hasNaN', () => {
  it('detects NaN', () => {
    expect(hasNaN(new Float32Array([1, NaN, 3]))).toBe(true);
  });

  it('returns false for clean array', () => {
    expect(hasNaN(new Float32Array([1, 2, 3]))).toBe(false);
  });
});

describe('countNaN', () => {
  it('counts NaN values', () => {
    expect(countNaN(new Float32Array([1, NaN, NaN, 4]))).toBe(2);
  });
});

describe('stripTrailingNaNs', () => {
  it('removes trailing NaN', () => {
    const input = new Float32Array([1, 2, NaN, NaN]);
    const result = stripTrailingNaNs(input);
    expect(Array.from(result)).toEqual([1, 2]);
  });

  it('returns same array if no trailing NaN', () => {
    const input = new Float32Array([1, 2, 3]);
    expect(Array.from(stripTrailingNaNs(input))).toEqual([1, 2, 3]);
  });
});

describe('replaceInfWithNaN', () => {
  it('replaces Infinity with NaN', () => {
    const input = new Float32Array([1, Infinity, -Infinity, 2]);
    const result = replaceInfWithNaN(input);
    expect(result[0]).toBe(1);
    expect(Number.isNaN(result[1])).toBe(true);
    expect(Number.isNaN(result[2])).toBe(true);
    expect(result[3]).toBe(2);
  });

  it('preserves finite values and existing NaN', () => {
    const input = new Float32Array([NaN, 1, Infinity, 3]);
    const result = replaceInfWithNaN(input);
    expect(Number.isNaN(result[0])).toBe(true);
    expect(result[1]).toBe(1);
    expect(Number.isNaN(result[2])).toBe(true);
    expect(result[3]).toBe(3);
  });
});
