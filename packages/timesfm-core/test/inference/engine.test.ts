/**
 * Tests for the ONNX inference engine.
 *
 * Uses the real TimesFM 2.5 200M ONNX model via ONNX Runtime.
 * Tests include realistic input data patterns for meaningful signal validation.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { TimesFMInferenceEngine } from '../../src/inference/onnx-engine';
import { TIMESFM_25_CONFIG } from '../../src/types';
import { getTestModelPath } from '../helpers';

const MODEL_PATH = getTestModelPath();
const mc = TIMESFM_25_CONFIG;

// The real exported ONNX model has fixed shape [1, MODEL_PATCHES, tokenizerInputDims]
const MODEL_BATCH = 1;
const MODEL_PATCHES = mc.exportedPatches;

import { mulberry32 } from '../test-fixtures';

// ---------------------------------------------------------------------------
// Helpers: generate realistic input patterns
// ---------------------------------------------------------------------------

/** Generate a single patched input with sinusoidal signal + light noise. */
function generateSineInput(numPatches: number): { input: Float32Array; mask: Uint8Array } {
  const rng = mulberry32(42);
  const patchLen = mc.inputPatchLen;
  const totalLen = numPatches * patchLen;
  const input = new Float32Array(totalLen);
  const mask = new Uint8Array(totalLen); // all valid

  let offset = 0;
  for (let p = 0; p < numPatches; p++) {
    for (let i = 0; i < patchLen; i++) {
      input[offset + i] = Math.sin((offset + i) * 0.15) * 20 + 50 + (rng() - 0.5) * 2;
    }
    offset += patchLen;
  }
  return { input, mask };
}

/** Generate input with linear trend + noise (business-metric-like). */
function generateTrendInput(numPatches: number): { input: Float32Array; mask: Uint8Array } {
  const rng = mulberry32(99);
  const patchLen = mc.inputPatchLen;
  const totalLen = numPatches * patchLen;
  const input = new Float32Array(totalLen);
  const mask = new Uint8Array(totalLen);

  for (let i = 0; i < totalLen; i++) {
    input[i] = 100 + i * 0.5 + Math.sin(i * 0.3) * 10 + (rng() - 0.5) * 4;
  }
  return { input, mask };
}

/** Generate all-zeros input (edge case). */
function generateZeroInput(numPatches: number): { input: Float32Array; mask: Uint8Array } {
  const patchLen = mc.inputPatchLen;
  const totalLen = numPatches * patchLen;
  return {
    input: new Float32Array(totalLen),
    mask: new Uint8Array(totalLen),
  };
}

if (!MODEL_PATH) {
  console.warn('[engine.test] ONNX model not found — all engine tests will be skipped.');
}

// Use conditional describe to skip when model is absent (maintains CI/local parity)
const engineDescribe = MODEL_PATH ? describe : describe.skip;

engineDescribe('TimesFMInferenceEngine', () => {
  let engine: TimesFMInferenceEngine;

  beforeAll(async () => {
    engine = new TimesFMInferenceEngine();
    await engine.load(MODEL_PATH);
  }, 60000);

  afterAll(async () => {
    await engine.dispose();
  });

  it('loads and reports loaded state', () => {
    expect(engine.isLoaded()).toBe(true);
  });

  it('produces correct output shapes with zero input', async () => {
    const { input, mask } = generateZeroInput(MODEL_PATCHES);
    const output = await engine.forward([input], [mask]);

    // Input embeddings: [batch, numPatches * modelDims]
    expect(output.inputEmbeddings.length).toBe(MODEL_BATCH);
    expect(output.inputEmbeddings[0].length).toBe(MODEL_PATCHES * mc.modelDims);

    // Output time series: [batch, numPatches * outputPatchLen * numQuantiles]
    expect(output.outputTimeSeries.length).toBe(MODEL_BATCH);
    expect(output.outputTimeSeries[0].length).toBe(
      MODEL_PATCHES * mc.outputPatchLen * mc.numQuantiles,
    );

    // Quantile spread: [batch, numPatches * outputQuantileLen * numQuantiles]
    expect(output.outputQuantileSpread.length).toBe(MODEL_BATCH);
    expect(output.outputQuantileSpread[0].length).toBe(
      MODEL_PATCHES * mc.outputQuantileLen * mc.numQuantiles,
    );
  });

  it('produces correct output shapes with sine input', async () => {
    const { input, mask } = generateSineInput(MODEL_PATCHES);
    const output = await engine.forward([input], [mask]);

    expect(output.outputTimeSeries.length).toBe(MODEL_BATCH);
    expect(output.outputTimeSeries[0].length).toBe(
      MODEL_PATCHES * mc.outputPatchLen * mc.numQuantiles,
    );

    // Output should contain meaningful values (not all zeros)
    const ts = output.outputTimeSeries[0];
    let nonZeroCount = 0;
    for (let i = 0; i < ts.length; i++) {
      if (Math.abs(ts[i]) > 1e-6) nonZeroCount++;
    }
    expect(nonZeroCount).toBeGreaterThan(0);
  });

  it('produces deterministic output (same input → same output)', async () => {
    const { input, mask } = generateSineInput(MODEL_PATCHES);

    const out1 = await engine.forward([input], [mask]);
    const out2 = await engine.forward([input], [mask]);

    // Compare outputs — must be bitwise identical
    for (let i = 0; i < out1.outputTimeSeries[0].length; i++) {
      expect(out1.outputTimeSeries[0][i]).toBe(out2.outputTimeSeries[0][i]);
    }
  });

  it('different inputs produce different outputs', async () => {
    const { input: input1, mask: mask1 } = generateSineInput(MODEL_PATCHES);
    const { input: input2, mask: mask2 } = generateTrendInput(MODEL_PATCHES);

    const out1 = await engine.forward([input1], [mask1]);
    const out2 = await engine.forward([input2], [mask2]);

    // At least some values should differ between outputs
    let diffCount = 0;
    for (let i = 0; i < out1.outputTimeSeries[0].length; i++) {
      if (out1.outputTimeSeries[0][i] !== out2.outputTimeSeries[0][i]) {
        diffCount++;
      }
    }
    expect(diffCount).toBeGreaterThan(0);
  });

  it('produces output with reasonable value range for sine input', async () => {
    const { input, mask } = generateSineInput(MODEL_PATCHES);
    const output = await engine.forward([input], [mask]);
    const ts = output.outputTimeSeries[0];

    // Values should be finite and within a reasonable range
    // (TimesFM 2.5 produces internally normalized outputs)
    for (const v of ts) {
      expect(Number.isFinite(v)).toBe(true);
      expect(Math.abs(v)).toBeLessThan(1e6); // not insanely large
    }
  });

  it('disposes correctly', async () => {
    const eng = new TimesFMInferenceEngine();
    await eng.load(MODEL_PATH);
    await eng.dispose();
    expect(eng.isLoaded()).toBe(false);
  });

  it('dispose handles release failure gracefully', async () => {
    const eng = new TimesFMInferenceEngine();
    await eng.load(MODEL_PATH);
    // Monkey-patch private session to make release throw
    const session = (eng as any)._session;
    if (session) {
      session.release = () => Promise.reject(new Error('mock release failure'));
    }
    // Should not throw even when release fails
    await eng.dispose();
    expect(eng.isLoaded()).toBe(false);
  });

  it('throws if forward called before load', async () => {
    const eng = new TimesFMInferenceEngine();
    await expect(eng.forward([new Float32Array(32)], [new Uint8Array(32)])).rejects.toThrow(
      'not loaded',
    );
  });

  // ── Decode-loop integration with real model ─────────────────────────
  // These tests verify the actual pretrained TimesFM model's autoregressive
  // decode behavior end-to-end — no mocked engine, no synthetic outputs.
  // They complement the algorithmic unit tests in decode-loop.test.ts
  // (which use DeterministicInferenceEngine for precise call-count spying).

  it('decode-loop: multi-step AR produces finite output', async () => {
    // Use the real decode() function with the pretrained model.
    // A horizon of 384 with outputPatchLen=128 means 2 AR decode steps
    // (numDecodeSteps = floor((384-1)/128) = 2).
    const { decode } = await import('../../src/inference/decode-loop');
    const { preprocess } = await import('../../src/preprocessor');
    const { createForecastConfig } = await import('../../src/config');

    const fc = createForecastConfig({ maxContext: 128, maxHorizon: 384 });
    const data = new Float32Array(Array.from({ length: 80 }, (_, i) => i + 1));

    const preprocessed = preprocess([data], fc, mc);
    const result = await decode(
      engine,
      preprocessed.patchedInputs,
      preprocessed.patchedMasks,
      preprocessed.contextMu,
      preprocessed.contextSigma,
      preprocessed.lastStats,
      fc.maxHorizon,
      fc,
      mc,
    );

    // AR outputs should exist (2 decode steps → arOutputs populated)
    expect(result.arOutputs).not.toBeNull();
    expect(result.arOutputs!.length).toBe(1); // 1 batch element
    const arFlat = result.arOutputs![0];
    // Each AR step produces outputPatchLen * numQuantiles = 128 * 10 = 1280 floats
    expect(arFlat.length).toBe(mc.outputPatchLen * mc.numQuantiles * 2);

    // All values must be finite
    for (let i = 0; i < arFlat.length; i++) {
      expect(Number.isFinite(arFlat[i])).toBe(true);
    }
  });

  it('decode-loop: zero-step AR (horizon ≤ outputPatchLen) returns null arOutputs', async () => {
    const { decode } = await import('../../src/inference/decode-loop');
    const { preprocess } = await import('../../src/preprocessor');
    const { createForecastConfig } = await import('../../src/config');

    // horizon=128 == outputPatchLen → 0 AR decode steps
    const fc = createForecastConfig({ maxContext: 128, maxHorizon: 128 });
    const data = new Float32Array(Array.from({ length: 80 }, (_, i) => i + 1));

    const preprocessed = preprocess([data], fc, mc);
    const result = await decode(
      engine,
      preprocessed.patchedInputs,
      preprocessed.patchedMasks,
      preprocessed.contextMu,
      preprocessed.contextSigma,
      preprocessed.lastStats,
      fc.maxHorizon,
      fc,
      mc,
    );

    // 0 decode steps → arOutputs is null
    expect(result.arOutputs).toBeNull();
    // pfOutputs should still be valid
    expect(result.pfOutputs.length).toBe(1);
    expect(result.pfOutputs[0].length).toBeGreaterThan(0);
  });

  it('reports default execution provider as CPU', () => {
    const eng = new TimesFMInferenceEngine();
    expect(eng.executionProvider).toBe('CPUExecutionProvider');
  });

  it('accepts custom execution provider', () => {
    const eng = new TimesFMInferenceEngine(TIMESFM_25_CONFIG, { executionProvider: 'cuda' });
    expect(eng.executionProvider).toBe('CUDAExecutionProvider');
  });
});
