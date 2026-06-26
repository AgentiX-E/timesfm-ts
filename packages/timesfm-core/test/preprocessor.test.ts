/**
 * Tests for the preprocessing pipeline.
 */

import { describe, it, expect } from 'vitest';
import { preprocess } from '../src/preprocessor.ts';
import { createForecastConfig } from '../src/config.ts';
import { TIMESFM_25_CONFIG } from '../src/types.ts';

describe('preprocess', () => {
  const fc = createForecastConfig({ maxContext: 256, maxHorizon: 128 });
  const mc = TIMESFM_25_CONFIG;

  it('produces correctly shaped output', () => {
    const inputs = [new Float32Array(Array.from({ length: 100 }, (_, i) => i + 1))];

    const result = preprocess(inputs, fc, mc);

    expect(result.batchSize).toBe(1);
    expect(result.numPatches).toBe(fc.maxContext / mc.inputPatchLen); // 8
    expect(result.patchedInputs.length).toBe(1);
    expect(result.patchedInputs[0].length).toBe(fc.maxContext); // 256
    expect(result.patchedMasks.length).toBe(1);
    expect(result.patchedMasks[0].length).toBe(fc.maxContext);
    expect(result.contextMu.length).toBe(result.numPatches);
    expect(result.contextSigma.length).toBe(result.numPatches);
    expect(result.lastStats.length).toBe(1);
    expect(result.cleanedInputs.length).toBe(1);
  });

  it('handles multiple series of different lengths', () => {
    const inputs = [
      new Float32Array(Array.from({ length: 50 }, (_, i) => i + 1)),
      new Float32Array(Array.from({ length: 200 }, (_, i) => (i + 1) * 10)),
      new Float32Array(Array.from({ length: 500 }, (_, i) => (i + 1) * 0.1)),
    ];

    const result = preprocess(inputs, fc, mc);

    expect(result.batchSize).toBe(3);
    // All should be padded to maxContext
    for (let b = 0; b < 3; b++) {
      expect(result.patchedInputs[b].length).toBe(fc.maxContext);
    }
  });

  it('handles NaN in input (linear interpolation)', () => {
    const input = new Float32Array([1, NaN, 3, NaN, 5]);
    const result = preprocess([input], fc, mc);

    // Input should have been cleaned before patching
    const cleaned = result.cleanedInputs[0];
    // After cleanSeries: all NaNs should be resolved
    expect(cleaned.length).toBe(5);
    for (let i = 0; i < cleaned.length; i++) {
      expect(Number.isFinite(cleaned[i])).toBe(true);
    }
  });

  it('handles leading NaN', () => {
    const input = new Float32Array([NaN, NaN, 1, 2, 3]);
    const result = preprocess([input], fc, mc);

    const cleaned = result.cleanedInputs[0];
    expect(cleaned.length).toBe(3);
    expect(Array.from(cleaned)).toEqual([1, 2, 3]);
  });

  it('series longer than maxContext are truncated (keeps last points)', () => {
    const input = new Float32Array(Array.from({ length: 600 }, (_, i) => i + 1));
    const result = preprocess([input], fc, mc);

    const truncated = result.truncatedInputs[0];
    // Should keep last maxContext=256 points
    expect(truncated.length).toBe(fc.maxContext);
    // First value should be 600 - 256 + 1 = 345
    expect(truncated[0]).toBeCloseTo(600 - 256 + 1, 0);
  });

  it('zero-length input does not crash', () => {
    const inputs = [new Float32Array(0)];
    const result = preprocess(inputs, fc, mc);
    expect(result.batchSize).toBe(1);
  });

  it('mask is correct for short input', () => {
    // Input = 3 points, maxContext = 256, patchLen = 32
    // Should have 256 - 3 = 253 leading padding with mask=1
    const input = new Float32Array([1, 2, 3]);
    const result = preprocess([input], fc, mc);

    const mask = result.patchedMasks[0];
    // First 253 should be 1 (padding)
    let padCount = 0;
    for (let i = 0; i < fc.maxContext - 3; i++) {
      if (mask[i] === 1) padCount++;
    }
    expect(padCount).toBe(fc.maxContext - 3);

    // Last 3 should be 0 (valid)
    for (let i = fc.maxContext - 3; i < fc.maxContext; i++) {
      expect(mask[i]).toBe(0);
    }
  });
});
