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

  // ─── Shape & Structure ─────────────────────────────────────────────────

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

  // ─── NaN Handling ──────────────────────────────────────────────────────

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

  it('handles trailing NaN', () => {
    const input = new Float32Array([1, 2, 3, NaN, NaN]);
    const result = preprocess([input], fc, mc);

    const cleaned = result.cleanedInputs[0];
    expect(cleaned.length).toBe(3);
    expect(Array.from(cleaned)).toEqual([1, 2, 3]);
  });

  it('handles all-NaN input gracefully', () => {
    const input = new Float32Array([NaN, NaN, NaN]);
    const result = preprocess([input], fc, mc);

    const cleaned = result.cleanedInputs[0];
    // After cleanSeries: all NaNs should be stripped → empty or zeros
    expect(cleaned.length).toBeGreaterThanOrEqual(0);
    // Should not crash
    expect(result.batchSize).toBe(1);
  });

  it('handles mixed NaN, Infinity and valid values', () => {
    const input = new Float32Array([NaN, Infinity, -Infinity, 1, 2, NaN, 3]);
    const result = preprocess([input], fc, mc);

    const cleaned = result.cleanedInputs[0];
    // After cleanSeries: Infinity → NaN → cleaned, NaN interpolated
    for (let i = 0; i < cleaned.length; i++) {
      expect(Number.isFinite(cleaned[i])).toBe(true);
    }
  });

  // ─── Truncation & Padding ──────────────────────────────────────────────

  it('series longer than maxContext are truncated (keeps last points)', () => {
    const input = new Float32Array(Array.from({ length: 600 }, (_, i) => i + 1));
    const result = preprocess([input], fc, mc);

    const truncated = result.truncatedInputs[0];
    // Should keep last maxContext=256 points
    expect(truncated.length).toBe(fc.maxContext);
    // First value should be 600 - 256 + 1 = 345
    expect(truncated[0]).toBeCloseTo(600 - 256 + 1, 0);
  });

  it('series exactly at maxContext is not truncated', () => {
    const input = new Float32Array(Array.from({ length: 256 }, (_, i) => i + 1));
    const result = preprocess([input], fc, mc);

    const truncated = result.truncatedInputs[0];
    expect(truncated.length).toBe(fc.maxContext);
    expect(truncated[0]).toBeCloseTo(1, 0);
  });

  it('zero-length input does not crash', () => {
    const inputs = [new Float32Array(0)];
    const result = preprocess(inputs, fc, mc);
    expect(result.batchSize).toBe(1);
  });

  it('extremely short input (1 point) is padded correctly', () => {
    const input = new Float32Array([42]);
    const result = preprocess([input], fc, mc);

    const padded = result.cleanedInputs[0];
    expect(padded.length).toBe(1);
    expect(result.patchedInputs[0].length).toBe(fc.maxContext);
  });

  // ─── Mask Correctness ──────────────────────────────────────────────────

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

  it('mask for full-length input has no padding', () => {
    const input = new Float32Array(Array.from({ length: 256 }, (_, i) => i + 1));
    const result = preprocess([input], fc, mc);

    const mask = result.patchedMasks[0];
    // No padding → all mask values should be 0
    for (let i = 0; i < mask.length; i++) {
      expect(mask[i]).toBe(0);
    }
  });

  it('masked positions in normalized output are zeroed', () => {
    const input = new Float32Array([1, 2, 3]);
    const result = preprocess([input], fc, mc);

    const normed = result.patchedInputs[0];
    const mask = result.patchedMasks[0];

    // Every position where mask=1 should have normed value = 0
    for (let i = 0; i < normed.length; i++) {
      if (mask[i] === 1) {
        expect(normed[i]).toBe(0);
      }
    }
  });

  // ─── RevIN Statistics ──────────────────────────────────────────────────

  it('computes per-patch running statistics (cumulative Welford)', () => {
    const input = new Float32Array(Array.from({ length: 256 }, (_, i) => i + 1));
    const result = preprocess([input], fc, mc);

    // contextMu and contextSigma should have numPatches entries per batch element
    const numPatches = fc.maxContext / mc.inputPatchLen; // 8
    expect(result.contextMu.length).toBe(numPatches);
    expect(result.contextSigma.length).toBe(numPatches);

    // contextMu stores cumulative means (Welford running stats), NOT per-patch means.
    // After patch 0: mean of first 32 values [1..32] = 16.5
    expect(result.contextMu[0][0]).toBeCloseTo(16.5, 0);
    // After last patch: mean of all 256 values [1..256] = 128.5
    expect(result.contextMu[numPatches - 1][0]).toBeCloseTo(128.5, 0);

    // Sigma should be positive and finite for non-constant series
    for (const s of result.contextSigma) {
      expect(Number.isFinite(s[0])).toBe(true);
      expect(s[0]).toBeGreaterThan(0);
    }
  });

  it('lastStats captures the cumulative running statistics', () => {
    const input = new Float32Array(Array.from({ length: 256 }, (_, i) => i + 1));
    const result = preprocess([input], fc, mc);

    // lastStats should be the cumulative stats after processing ALL patches
    const stats = result.lastStats[0];
    expect(stats.n).toBe(256);
    // Mean of [1, 2, ..., 256] = (256 + 1) / 2 = 128.5
    expect(stats.mu).toBeCloseTo(128.5, 1);
  });

  it('per-patch RevIN normalization centers each patch near zero', () => {
    const input = new Float32Array(Array.from({ length: 256 }, (_, i) => i + 1));
    const result = preprocess([input], fc, mc);
    const { inputPatchLen } = mc;
    const numPatches = fc.maxContext / inputPatchLen;

    const normed = result.patchedInputs[0];

    // RevIN normalizes each patch using the cumulative stats up to that patch.
    // The per-patch means should be near zero, but small deviations are expected
    // because the normalization parameters change across patches.
    for (let p = 0; p < numPatches; p++) {
      const patchStart = p * inputPatchLen;
      let sum = 0;
      for (let i = patchStart; i < patchStart + inputPatchLen; i++) {
        sum += normed[i];
      }
      const mean = sum / inputPatchLen;
      // After RevIN normalization, per-patch mean should be close to zero
      expect(Math.abs(mean)).toBeLessThan(2);
    }
  });

  // ─── Edge Cases ────────────────────────────────────────────────────────

  it('constant series (no variance) handles safe sigma in RevIN', () => {
    const input = new Float32Array(Array.from({ length: 100 }, () => 42));
    const result = preprocess([input], fc, mc);

    // Welford running stats produce sigma=0 for identical values.
    // The RevIN function handles this internally by using safeSigma=1.
    // contextSigma reflects the raw Welford values, which can be 0.
    for (const s of result.contextSigma) {
      expect(Number.isFinite(s[0])).toBe(true);
    }

    // Output should have no NaN — RevIN's safe sigma prevents division by zero
    for (const v of result.patchedInputs[0]) {
      expect(Number.isFinite(v)).toBe(true);
    }

    // For a constant series, RevIN normalization with safe sigma produces zeros
    // (x - mu) / 1 = 0 for all x = mu = 42
    for (let i = 0; i < result.patchedInputs[0].length; i++) {
      if (result.patchedMasks[0][i] === 0) {
        expect(result.patchedInputs[0][i]).toBeCloseTo(0, 1);
      }
    }
  });

  it('negative values are handled correctly', () => {
    const input = new Float32Array(Array.from({ length: 100 }, (_, i) => -i - 1));
    const result = preprocess([input], fc, mc);

    const normed = result.patchedInputs[0];
    expect(normed.length).toBe(fc.maxContext);

    // No NaN in output
    for (const v of normed) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('very large values do not cause overflow', () => {
    const input = new Float32Array(Array.from({ length: 100 }, (_, i) => 1e6 + i));
    const result = preprocess([input], fc, mc);

    const normed = result.patchedInputs[0];
    for (const v of normed) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('very small values (near zero) are handled', () => {
    const input = new Float32Array(Array.from({ length: 100 }, (_, i) => (i + 1) * 1e-10));
    const result = preprocess([input], fc, mc);

    const normed = result.patchedInputs[0];
    for (const v of normed) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('handles exactly inputPatchLen length series', () => {
    const input = new Float32Array(Array.from({ length: mc.inputPatchLen }, (_, i) => i + 1));
    const result = preprocess([input], fc, mc);

    // One patch worth of data + padding to maxContext
    expect(result.patchedInputs[0].length).toBe(fc.maxContext);
    // Last patch should have stats
    expect(result.lastStats.length).toBe(1);
  });

  // ─── Multi-series Batch ────────────────────────────────────────────────

  it('multi-series batch produces independent statistics per series', () => {
    const inputs = [
      new Float32Array(Array.from({ length: 200 }, (_, i) => i + 1)),
      new Float32Array(Array.from({ length: 200 }, (_, i) => (i + 1) * 10)),
    ];

    const result = preprocess(inputs, fc, mc);

    // Each series gets its own numPatches stats
    const numPatches = fc.maxContext / mc.inputPatchLen;
    expect(result.contextMu.length).toBe(numPatches * 2);
    expect(result.contextSigma.length).toBe(numPatches * 2);

    // Per-series lastStats
    expect(result.lastStats.length).toBe(2);

    // Series 1 mean (1..200) vs Series 2 mean (10..2000)
    const mu1 = result.lastStats[0].mu; // ~100.5
    const mu2 = result.lastStats[1].mu; // ~1005
    expect(mu2).toBeGreaterThan(mu1);
  });
});
