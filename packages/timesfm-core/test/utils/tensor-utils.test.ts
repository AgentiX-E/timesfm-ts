/**
 * Tests for tensor utility functions.
 */

import { describe, it, expect } from 'vitest';
import {
  reshape2D,
  reshape3D,
  leftPad,
  concat,
  stack,
  sliceEach,
  takeLast,
  clipMin,
  clipMax,
  elementwiseMean,
  elementwiseDiff,
  negate,
  mean,
  std,
  allNonNegative,
  hasInvalid,
} from '../../src/utils/tensor-utils.ts';

describe('reshape2D', () => {
  it('reshapes flat array into rows', () => {
    const flat = new Float32Array([1, 2, 3, 4, 5, 6]);
    const result = reshape2D(flat, 2, 3);
    expect(result.length).toBe(2);
    expect(Array.from(result[0])).toEqual([1, 2, 3]);
    expect(Array.from(result[1])).toEqual([4, 5, 6]);
  });
});

describe('reshape3D', () => {
  it('reshapes flat array into 3D tensor', () => {
    // 2 x 2 x 3 = 12 elements
    const flat = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const result = reshape3D(flat, 2, 2, 3);
    expect(result.length).toBe(2);
    expect(result[0].length).toBe(2);
    expect(result[0][0].length).toBe(3);
    expect(Array.from(result[0][0])).toEqual([1, 2, 3]);
    expect(Array.from(result[0][1])).toEqual([4, 5, 6]);
    expect(Array.from(result[1][0])).toEqual([7, 8, 9]);
    expect(Array.from(result[1][1])).toEqual([10, 11, 12]);
  });
});

describe('leftPad', () => {
  it('pads shorter array', () => {
    const { padded, mask } = leftPad(new Float32Array([1, 2, 3]), 5);
    expect(Array.from(padded)).toEqual([0, 0, 1, 2, 3]);
    expect(Array.from(mask)).toEqual([1, 1, 0, 0, 0]);
  });

  it('does not pad longer array', () => {
    const { padded, mask } = leftPad(new Float32Array([1, 2, 3, 4, 5]), 3);
    expect(Array.from(padded)).toEqual([3, 4, 5]);
    expect(Array.from(mask)).toEqual([0, 0, 0]);
  });

  it('handles exact length', () => {
    const { padded, mask } = leftPad(new Float32Array([1, 2, 3]), 3);
    expect(Array.from(padded)).toEqual([1, 2, 3]);
    expect(Array.from(mask)).toEqual([0, 0, 0]);
  });
});

describe('concat', () => {
  it('concatenates arrays', () => {
    const result = concat([
      new Float32Array([1, 2]),
      new Float32Array([3, 4]),
      new Float32Array([5]),
    ]);
    expect(Array.from(result)).toEqual([1, 2, 3, 4, 5]);
  });
});

describe('stack', () => {
  it('stacks arrays as rows', () => {
    const result = stack([
      new Float32Array([1, 2]),
      new Float32Array([3, 4]),
      new Float32Array([5, 6]),
    ]);
    expect(Array.from(result)).toEqual([1, 2, 3, 4, 5, 6]);
  });
});

describe('sliceEach', () => {
  it('slices start:end from each array', () => {
    const result = sliceEach(
      [new Float32Array([1, 2, 3, 4]), new Float32Array([5, 6, 7, 8])],
      1,
      3,
    );
    expect(Array.from(result[0])).toEqual([2, 3]);
    expect(Array.from(result[1])).toEqual([6, 7]);
  });
});

describe('takeLast', () => {
  it('takes last N elements', () => {
    const result = takeLast([new Float32Array([1, 2, 3, 4, 5]), new Float32Array([10, 20])], 3);
    expect(Array.from(result[0])).toEqual([3, 4, 5]);
    expect(Array.from(result[1])).toEqual([10, 20]);
  });
});

describe('clipMin', () => {
  it('clips values below minimum', () => {
    const result = clipMin(new Float32Array([-5, 0, 5, 10]), 0);
    expect(Array.from(result)).toEqual([0, 0, 5, 10]);
  });
});

describe('clipMax', () => {
  it('clips values above maximum', () => {
    const result = clipMax(new Float32Array([-5, 0, 5, 10]), 5);
    expect(Array.from(result)).toEqual([-5, 0, 5, 5]);
  });
});

describe('elementwiseMean', () => {
  it('computes element-wise mean', () => {
    const a = new Float32Array([1, 2, 3]);
    const b = new Float32Array([3, 4, 5]);
    const result = elementwiseMean(a, b);
    expect(Array.from(result)).toEqual([2, 3, 4]);
  });
});

describe('elementwiseDiff', () => {
  it('computes element-wise difference', () => {
    const a = new Float32Array([5, 4, 3]);
    const b = new Float32Array([1, 2, 3]);
    const result = elementwiseDiff(a, b);
    expect(Array.from(result)).toEqual([4, 2, 0]);
  });
});

describe('negate', () => {
  it('negates all elements', () => {
    const result = negate(new Float32Array([1, -2, 0]));
    expect(Array.from(result)).toEqual([-1, 2, -0]);
  });
});

describe('mean', () => {
  it('computes mean', () => {
    expect(mean(new Float32Array([1, 2, 3, 4, 5]))).toBeCloseTo(3, 5);
  });

  it('returns 0 for empty array', () => {
    expect(mean(new Float32Array(0))).toBe(0);
  });
});

describe('std', () => {
  it('computes population std', () => {
    expect(std(new Float32Array([1, 2, 3, 4, 5]))).toBeCloseTo(Math.sqrt(2), 5);
  });

  it('returns 0 for single element', () => {
    expect(std(new Float32Array([5]))).toBe(0);
  });
});

describe('allNonNegative', () => {
  it('returns true for all positive', () => {
    expect(allNonNegative(new Float32Array([1, 2, 3]))).toBe(true);
  });

  it('returns false if any negative', () => {
    expect(allNonNegative(new Float32Array([1, -2, 3]))).toBe(false);
  });

  it('returns true for zeros', () => {
    expect(allNonNegative(new Float32Array([0, 0]))).toBe(true);
  });
});

describe('hasInvalid', () => {
  it('returns true for NaN', () => {
    expect(hasInvalid(new Float32Array([1, NaN, 3]))).toBe(true);
  });

  it('returns true for Infinity', () => {
    expect(hasInvalid(new Float32Array([1, Infinity, 3]))).toBe(true);
  });

  it('returns false for finite values', () => {
    expect(hasInvalid(new Float32Array([1, 2, 3]))).toBe(false);
  });
});
