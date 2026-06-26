/**
 * Tests for the OneHotEncoder.
 */

import { describe, it, expect } from 'vitest';
import { OneHotEncoder } from '../src/one-hot-encoder.ts';

describe('OneHotEncoder', () => {
  describe('with drop=first', () => {
    it('encodes categories correctly', () => {
      const encoder = new OneHotEncoder({ drop: 'first' });
      const encoded = encoder.fitTransform(['a', 'b', 'c', 'a']);

      // 3 categories → 2 columns after drop
      // 'a' = first → dropped, encoded as [0, 0]
      // 'b' = [1, 0]
      // 'c' = [0, 1]
      expect(encoded.length).toBe(4);
      expect(encoded[0]).toEqual([0, 0]); // a
      expect(encoded[1]).toEqual([1, 0]); // b
      expect(encoded[2]).toEqual([0, 1]); // c
      expect(encoded[3]).toEqual([0, 0]); // a (again)
    });

    it('handles single category', () => {
      const encoder = new OneHotEncoder({ drop: 'first' });
      const encoded = encoder.fitTransform(['x', 'x', 'x']);

      // 1 category, dropped → 0 columns
      expect(encoder.numColumns).toBe(0);
      for (const row of encoded) {
        expect(row.length).toBe(0);
      }
    });

    it('handles unknown categories with ignore', () => {
      const encoder = new OneHotEncoder({ drop: 'first', handleUnknown: 'ignore' });
      encoder.fit(['a', 'b']);
      const encoded = encoder.transform(['c', 'a']);

      // 'c' unknown → [0] (all zeros)
      // 'a' → [0] (first category, dropped)
      expect(encoded[0]).toEqual([0]);
      expect(encoded[1]).toEqual([0]);
    });

    it('throws on unknown with error', () => {
      const encoder = new OneHotEncoder({ drop: 'first', handleUnknown: 'error' });
      encoder.fit(['a', 'b']);
      expect(() => encoder.transform(['c'])).toThrow('Unknown category');
    });
  });

  describe('without drop', () => {
    it('encodes all categories', () => {
      const encoder = new OneHotEncoder({ drop: null });
      const encoded = encoder.fitTransform(['a', 'b', 'c']);

      expect(encoder.numColumns).toBe(3);
      expect(encoded[0]).toEqual([1, 0, 0]); // a
      expect(encoded[1]).toEqual([0, 1, 0]); // b
      expect(encoded[2]).toEqual([0, 0, 1]); // c
    });
  });

  it('handles numeric categories', () => {
    const encoder = new OneHotEncoder({ drop: 'first' });
    const encoded = encoder.fitTransform([1, 2, 3, 1, 2]);

    expect(encoded.length).toBe(5);
    expect(encoded[0]).toEqual([0, 0]); // 1 (dropped)
    expect(encoded[1]).toEqual([1, 0]); // 2
    expect(encoded[2]).toEqual([0, 1]); // 3
  });

  it('throws if transform called before fit', () => {
    const encoder = new OneHotEncoder();
    expect(() => encoder.transform(['a'])).toThrow('not fitted');
  });

  it('numColumns returns 0 before fit', () => {
    const encoder = new OneHotEncoder();
    expect(encoder.numColumns).toBe(0);
  });
});
