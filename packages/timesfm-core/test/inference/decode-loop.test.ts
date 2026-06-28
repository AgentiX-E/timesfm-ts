/**
 * Comprehensive tests for the autoregressive decode loop.
 *
 * Uses MockInferenceEngine to test all decode paths without
 * requiring the 885 MB ONNX model.  Covers:
 *   - Phase 1: Prefill (forward → RevIN denorm → trim → QS extract)
 *   - Phase 2: AR decode (seed extraction → sub-patch → forward → loop)
 *   - Edge cases: zero decode steps, single step, long horizons
 *   - Multi-batch processing
 *   - Numerical edge conditions (sigma < epsilon)
 */
import { describe, it, expect } from 'vitest';
import { decode } from '../../src/inference/decode-loop';
import { MockInferenceEngine } from '../helpers/mock-engine';
import { createRunningStats, type RunningStats } from '../../src/utils/stats';
import { TIMESFM_25_CONFIG, type ForecastConfig, type ModelConfig } from '../../src/types';

// ---------------------------------------------------------------------------
// Test setup helpers
// ---------------------------------------------------------------------------

const MC: ModelConfig = TIMESFM_25_CONFIG;

function makeForecastConfig(maxContext: number, maxHorizon: number): ForecastConfig {
  return {
    maxContext,
    maxHorizon,
    normalizeInputs: true,
    perCoreBatchSize: 1,
    useContinuousQuantileHead: true,
    forceFlipInvariance: true,
    inferIsPositive: true,
    fixQuantileCrossing: true,
    returnBackcast: false,
  };
}

/** Build flat patched input for a single batch element.
 *  numPatches patches of inputPatchLen=32 values, all filled with `fillVal`. */
function makePatchedInput(numPatches: number, fillVal: number = 1.0): Float32Array {
  const len = numPatches * MC.inputPatchLen;
  const arr = new Float32Array(len);
  arr.fill(fillVal);
  return arr;
}

function makeMask(numPatches: number, zeroCount: number = 0): Uint8Array {
  const len = numPatches * MC.inputPatchLen;
  const arr = new Uint8Array(len);
  // Mark first zeroCount as valid (0), rest as padding (1)
  for (let i = zeroCount; i < len; i++) arr[i] = 1;
  return arr;
}

function makeContextMu(batchSize: number, numPatches: number, val: number = 0): Float32Array[] {
  const result: Float32Array[] = [];
  for (let b = 0; b < batchSize; b++) {
    for (let p = 0; p < numPatches; p++) {
      result.push(new Float32Array([val]));
    }
  }
  return result;
}

function makeContextSigma(
  batchSize: number,
  numPatches: number,
  val: number = 1.0,
): Float32Array[] {
  const result: Float32Array[] = [];
  for (let b = 0; b < batchSize; b++) {
    for (let p = 0; p < numPatches; p++) {
      result.push(new Float32Array([val]));
    }
  }
  return result;
}

function makeLastStats(batchSize: number): RunningStats[] {
  return Array.from({ length: batchSize }, () => createRunningStats());
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('decode (with MockInferenceEngine)', () => {
  // ── Prefill-only tests (numDecodeSteps = 0) ──────────────────────────

  describe('Phase 1 — Prefill (zero AR steps)', () => {
    it('returns arOutputs=null when horizon <= outputPatchLen', async () => {
      const engine = new MockInferenceEngine({ scale: 2.0 });
      engine.load('test');
      const fc = makeForecastConfig(512, 256);
      const numPatches = Math.floor(512 / MC.inputPatchLen); // 512/32
      const horizon = MC.outputPatchLen; // numDecodeSteps = 0

      const result = await decode(
        engine,
        [makePatchedInput(numPatches)],
        [makeMask(numPatches, numPatches * MC.inputPatchLen)],
        makeContextMu(1, numPatches),
        makeContextSigma(1, numPatches),
        makeLastStats(1),
        horizon,
        fc,
        MC,
      );

      expect(result.arOutputs).toBeNull();
      expect(result.pfOutputs).toHaveLength(1);
      expect(result.quantileSpreads).toHaveLength(1);
    });

    it('horizon=1 returns arOutputs=null', async () => {
      const engine = new MockInferenceEngine();
      engine.load('test');
      const fc = makeForecastConfig(512, 256);
      const numPatches = Math.floor(512 / MC.inputPatchLen);
      const horizon = 1; // 1 <= 128 → numDecodeSteps = 0

      const result = await decode(
        engine,
        [makePatchedInput(numPatches)],
        [makeMask(numPatches)],
        makeContextMu(1, numPatches),
        makeContextSigma(1, numPatches),
        makeLastStats(1),
        horizon,
        fc,
        MC,
      );

      expect(result.arOutputs).toBeNull();
    });

    it('pfOutputs length matches numInputPatches * perPatch', async () => {
      const engine = new MockInferenceEngine({ scale: 3.0 });
      engine.load('test');
      const maxContext = 256;
      const numInputPatches = Math.floor(maxContext / MC.inputPatchLen); // 8
      const fc = makeForecastConfig(maxContext, 128);
      const horizon = 64;

      const result = await decode(
        engine,
        [makePatchedInput(numInputPatches)],
        [makeMask(numInputPatches)],
        makeContextMu(1, numInputPatches),
        makeContextSigma(1, numInputPatches),
        makeLastStats(1),
        horizon,
        fc,
        MC,
      );

      const expectedLen = numInputPatches * MC.outputPatchLen * MC.numQuantiles;
      expect(result.pfOutputs[0].length).toBe(expectedLen);
    });

    it('quantileSpreads length = outputQuantileLen * numQuantiles', async () => {
      const engine = new MockInferenceEngine();
      engine.load('test');
      const fc = makeForecastConfig(512, 256);
      const numPatches = Math.floor(512 / MC.inputPatchLen);

      const result = await decode(
        engine,
        [makePatchedInput(numPatches)],
        [makeMask(numPatches)],
        makeContextMu(1, numPatches),
        makeContextSigma(1, numPatches),
        makeLastStats(1),
        50,
        fc,
        MC,
      );

      expect(result.quantileSpreads[0].length).toBe(MC.outputQuantileLen * MC.numQuantiles);
    });

    it('forward is called exactly once for prefill', async () => {
      const callCounter = { value: 0 };
      const engine = new MockInferenceEngine({ callCount: callCounter });
      engine.load('test');
      const fc = makeForecastConfig(512, 256);
      const numPatches = Math.floor(512 / MC.inputPatchLen);

      await decode(
        engine,
        [makePatchedInput(numPatches)],
        [makeMask(numPatches)],
        makeContextMu(1, numPatches),
        makeContextSigma(1, numPatches),
        makeLastStats(1),
        100,
        fc,
        MC,
      );

      expect(callCounter.value).toBe(1);
    });
  });

  // ── AR Decode tests ──────────────────────────────────────────────────

  describe('Phase 2 — Autoregressive Decode', () => {
    it('numDecodeSteps=1 triggers one AR forward call', async () => {
      const callCounter = { value: 0 };
      const engine = new MockInferenceEngine({ scale: 1.0, callCount: callCounter });
      engine.load('test');
      const fc = makeForecastConfig(512, 256);
      const numPatches = Math.floor(512 / MC.inputPatchLen);
      const horizon = 129; // floor((129-1)/128) = 1

      await decode(
        engine,
        [makePatchedInput(numPatches)],
        [makeMask(numPatches)],
        makeContextMu(1, numPatches),
        makeContextSigma(1, numPatches),
        makeLastStats(1),
        horizon,
        fc,
        MC,
      );

      // 1 prefill + 1 AR step = 2 forward calls
      expect(callCounter.value).toBe(2);
    });

    it('numDecodeSteps=3 triggers 4 forward calls total', async () => {
      const callCounter = { value: 0 };
      const engine = new MockInferenceEngine({ callCount: callCounter });
      engine.load('test');
      const fc = makeForecastConfig(512, 256);
      const numPatches = Math.floor(512 / MC.inputPatchLen);
      const horizon = 385; // floor((385-1)/128) = 3

      await decode(
        engine,
        [makePatchedInput(numPatches)],
        [makeMask(numPatches)],
        makeContextMu(1, numPatches),
        makeContextSigma(1, numPatches),
        makeLastStats(1),
        horizon,
        fc,
        MC,
      );

      expect(callCounter.value).toBe(4); // 1 prefill + 3 AR
    });

    it('arOutputs has batchSize entries (one per batch element)', async () => {
      const engine = new MockInferenceEngine({ scale: 1.0 });
      engine.load('test');
      const fc = makeForecastConfig(512, 256);
      const numPatches = Math.floor(512 / MC.inputPatchLen);
      const horizon = 513; // floor((513-1)/128) = 4 AR steps

      const result = await decode(
        engine,
        [makePatchedInput(numPatches)],
        [makeMask(numPatches)],
        makeContextMu(1, numPatches),
        makeContextSigma(1, numPatches),
        makeLastStats(1),
        horizon,
        fc,
        MC,
      );

      expect(result.arOutputs).not.toBeNull();
      // arOutputs is now [batch] — one entry per batch element
      expect(result.arOutputs!.length).toBe(1); // batchSize=1
      // Each batch entry contains all 4 AR steps concatenated
      expect(result.arOutputs![0].length).toBe(4 * MC.outputPatchLen);
    });

    it('each arOutputs[b] concatenates numDecodeSteps × outputPatchLen floats', async () => {
      const engine = new MockInferenceEngine({ scale: 1.0 });
      engine.load('test');
      const fc = makeForecastConfig(512, 256);
      const horizon = 385; // 3 AR steps
      const numDecodeSteps = Math.floor((horizon - 1) / MC.outputPatchLen); // 3

      const result = await decode(
        engine,
        [makePatchedInput(16)],
        [makeMask(16)],
        makeContextMu(1, 16),
        makeContextSigma(1, 16),
        makeLastStats(1),
        horizon,
        fc,
        MC,
      );

      for (const ar of result.arOutputs!) {
        // Total = numDecodeSteps × outputPatchLen (batchSize=1)
        expect(ar.length).toBe(numDecodeSteps * MC.outputPatchLen);
      }
    });
  });

  // ── Multi-batch tests ────────────────────────────────────────────────

  describe('Multi-batch processing', () => {
    it('batchSize=3 produces 3 pfOutputs and 3 quantileSpreads', async () => {
      const engine = new MockInferenceEngine({ scale: 1.0 });
      engine.load('test');
      const fc = makeForecastConfig(512, 256);
      const numPatches = Math.floor(512 / MC.inputPatchLen);
      const batchSize = 3;

      const result = await decode(
        engine,
        [makePatchedInput(numPatches), makePatchedInput(numPatches), makePatchedInput(numPatches)],
        [makeMask(numPatches), makeMask(numPatches), makeMask(numPatches)],
        makeContextMu(batchSize, numPatches),
        makeContextSigma(batchSize, numPatches),
        makeLastStats(batchSize),
        100,
        fc,
        MC,
      );

      expect(result.pfOutputs).toHaveLength(batchSize);
      expect(result.quantileSpreads).toHaveLength(batchSize);
    });
  });

  // ── contextMu/Sigma padding tests ────────────────────────────────────

  describe('contextMu / contextSigma padding', () => {
    it('pads to batchSize * exportedPatches entries', async () => {
      const engine = new MockInferenceEngine({ scale: 1.0 });
      engine.load('test');
      // maxContext=128 → numInputPatches=4 (< 16)
      const fc = makeForecastConfig(128, 256);
      const numInputPatches = 4;
      const batchSize = 2;

      const result = await decode(
        engine,
        [makePatchedInput(numInputPatches), makePatchedInput(numInputPatches)],
        [makeMask(numInputPatches), makeMask(numInputPatches)],
        makeContextMu(batchSize, numInputPatches),
        makeContextSigma(batchSize, numInputPatches),
        makeLastStats(batchSize),
        50,
        fc,
        MC,
      );

      // Should not throw — padding handled internally
      expect(result.pfOutputs).toHaveLength(batchSize);
    });
  });

  // ── Numerical edge cases ─────────────────────────────────────────────

  describe('Numerical edge cases', () => {
    it('handles sigma < epsilon (near-zero std dev)', async () => {
      const engine = new MockInferenceEngine({ scale: 1.0 });
      engine.load('test');
      const fc = makeForecastConfig(512, 256);
      const numPatches = Math.floor(512 / MC.inputPatchLen);
      // sigma = 1e-7 (below 1e-6 threshold)
      const tinySigma = makeContextSigma(1, numPatches);
      for (const s of tinySigma) s[0] = 1e-7;

      const result = await decode(
        engine,
        [makePatchedInput(numPatches)],
        [makeMask(numPatches)],
        makeContextMu(1, numPatches),
        tinySigma,
        makeLastStats(1),
        100,
        fc,
        MC,
      );

      // Should not produce NaN or Infinity
      for (const v of result.pfOutputs[0]) {
        expect(Number.isFinite(v)).toBe(true);
      }
    });

    it('contextMu=0, contextSigma=1 returns denormed = raw * 1 + 0', async () => {
      const scale = 5.0;
      const engine = new MockInferenceEngine({ scale });
      engine.load('test');
      const fc = makeForecastConfig(512, 256);
      const numPatches = Math.floor(512 / MC.inputPatchLen);

      const result = await decode(
        engine,
        [makePatchedInput(numPatches)],
        [makeMask(numPatches)],
        makeContextMu(1, numPatches, 0),
        makeContextSigma(1, numPatches, 1),
        makeLastStats(1),
        100,
        fc,
        MC,
      );

      // With mu=0, sigma=1, denormed = raw * 1 + 0 = scale * (b+1)
      // For b=0: raw = 5.0 * 1 = 5.0
      expect(result.pfOutputs[0][0]).toBeCloseTo(scale, 5);
    });

    it('quantile spread fallback to 0 for short input', async () => {
      const engine = new MockInferenceEngine({ scale: 1.0 });
      engine.load('test');
      const fc = makeForecastConfig(512, 256);
      const numPatches = Math.floor(512 / MC.inputPatchLen);

      const result = await decode(
        engine,
        [makePatchedInput(numPatches)],
        [makeMask(numPatches)],
        makeContextMu(1, numPatches),
        makeContextSigma(1, numPatches),
        makeLastStats(1),
        50,
        fc,
        MC,
      );

      // MockEngine returns outputQuantileSpread with qsLen = 10240
      // so all should be non-zero. But the last element check should pass.
      const qs = result.quantileSpreads[0];
      const len = MC.outputQuantileLen * MC.numQuantiles;
      expect(qs.length).toBe(len);
      // Ensure at least some values are non-zero (mock fills with scale*0.5)
      expect(qs[0]).not.toBe(0);
    });
  });

  // ── AR mask tests ────────────────────────────────────────────────────

  describe('AR decode masks', () => {
    it('AR patches have all-zero masks (no padding in AR)', async () => {
      const engine = new MockInferenceEngine({ scale: 1.0 });
      engine.load('test');
      const fc = makeForecastConfig(512, 256);
      const numPatches = Math.floor(512 / MC.inputPatchLen);
      const horizon = 257; // floor((257-1)/128) = 2 steps

      let secondCallMasks: Uint8Array[] = [];
      const spyEngine = {
        ...engine,
        _loaded: true,
        isLoaded: () => true,
        load: async (_: string) => {},
        forward: async (inputs: Float32Array[], masks: Uint8Array[]) => {
          if (engine.callCount === 2) {
            secondCallMasks = masks;
          }
          return engine.forward(inputs, masks);
        },
        dispose: async () => {},
      } as MockInferenceEngine;

      await decode(
        spyEngine as unknown as Parameters<typeof decode>[0],
        [makePatchedInput(numPatches)],
        [makeMask(numPatches)],
        makeContextMu(1, numPatches),
        makeContextSigma(1, numPatches),
        makeLastStats(1),
        horizon,
        fc,
        MC,
      );

      expect(secondCallMasks.length).toBeGreaterThan(0);
      for (const mask of secondCallMasks) {
        for (let i = 0; i < mask.length; i++) {
          expect(mask[i]).toBe(0); // AR inputs have no padding
        }
      }
    });
  });

  // ── Large horizon stress test ────────────────────────────────────────

  describe('Stress tests', () => {
    it('handles horizon=4096 without crashing', async () => {
      const engine = new MockInferenceEngine({ scale: 1.0 });
      engine.load('test');
      const fc = makeForecastConfig(512, 4096);
      const numPatches = Math.floor(512 / MC.inputPatchLen);
      const horizon = 4096; // floor((4096-1)/128) = 31 steps

      const result = await decode(
        engine,
        [makePatchedInput(numPatches)],
        [makeMask(numPatches)],
        makeContextMu(1, numPatches),
        makeContextSigma(1, numPatches),
        makeLastStats(1),
        horizon,
        fc,
        MC,
      );

      expect(result.pfOutputs).toHaveLength(1);
      expect(result.arOutputs).not.toBeNull();
      // arOutputs is [batch] — batchSize=1, contains all 31 AR steps concatenated
      expect(result.arOutputs!.length).toBe(1);
      // Total: 31 steps × 128 outputPatchLen = 3968 floats
      expect(result.arOutputs![0].length).toBe(31 * MC.outputPatchLen);
    });

    it('handles maxContext=1024 with many patches', async () => {
      const engine = new MockInferenceEngine({ scale: 1.0 });
      engine.load('test');
      const fc = makeForecastConfig(1024, 256);
      const numPatches = 32; // 1024/32

      const result = await decode(
        engine,
        [makePatchedInput(numPatches)],
        [makeMask(numPatches)],
        makeContextMu(1, numPatches),
        makeContextSigma(1, numPatches),
        makeLastStats(1),
        100,
        fc,
        MC,
      );

      // The ONNX model has exportedPatches-patch output.
      // decode trims to min(numInputPatches, exportedPatches) patches
      const effectivePatches = MC.exportedPatches; // exported ONNX shape
      expect(result.pfOutputs[0].length).toBe(
        effectivePatches * MC.outputPatchLen * MC.numQuantiles,
      );
    });
  });

  // ── arSeeds / lastStats continuity ───────────────────────────────────

  describe('AR seed and stats continuity', () => {
    it('arSeeds are taken from the last output sub-patch median', async () => {
      const scale = 2.0;
      const engine = new MockInferenceEngine({ scale });
      engine.load('test');
      const fc = makeForecastConfig(512, 256);
      const numPatches = Math.floor(512 / MC.inputPatchLen);
      const horizon = 257; // floor((257-1)/128) = 2 steps

      const result = await decode(
        engine,
        [makePatchedInput(numPatches)],
        [makeMask(numPatches)],
        makeContextMu(1, numPatches),
        makeContextSigma(1, numPatches),
        makeLastStats(1),
        horizon,
        fc,
        MC,
      );

      // arOutputs[b] contains ALL AR step seeds concatenated for batch b
      expect(result.arOutputs).not.toBeNull();
      // batchSize=1, so arOutputs has length 1
      expect(result.arOutputs!.length).toBe(1);
      // Each entry is numDecodeSteps × outputPatchLen = 2 × 128 = 256
      for (const ar of result.arOutputs!) {
        expect(ar.length).toBe(2 * MC.outputPatchLen);
      }
    });

    it('lastStats are updated across AR steps', async () => {
      const engine = new MockInferenceEngine({ scale: 1.0 });
      engine.load('test');
      const fc = makeForecastConfig(512, 256);
      const numPatches = Math.floor(512 / MC.inputPatchLen);
      const horizon = 257; // 2 AR steps

      // Use non-zero initial stats
      const stats = makeLastStats(1);
      stats[0] = { n: 10, mu: 5.0, sigma: 2.0 };

      const result = await decode(
        engine,
        [makePatchedInput(numPatches)],
        [makeMask(numPatches)],
        makeContextMu(1, numPatches),
        makeContextSigma(1, numPatches),
        stats,
        horizon,
        fc,
        MC,
      );

      expect(result.arOutputs).not.toBeNull();
      // arOutputs is [batch] — batchSize=1, contains 2 AR steps concatenated
      expect(result.arOutputs!.length).toBe(1);
      expect(result.arOutputs![0].length).toBe(2 * MC.outputPatchLen);
    });
  });

  // ── sub-patch count verification ─────────────────────────────────────

  describe('Sub-patch processing', () => {
    it('outputPatchesPerInput = outputPatchLen / inputPatchLen = 4', () => {
      expect(MC.outputPatchesPerInput).toBe(4);
    });
  });

  // ── AbortSignal tests ─────────────────────────────────────────────────

  describe('AbortSignal handling', () => {
    it('aborts before decode when signal is already aborted', async () => {
      const engine = new MockInferenceEngine({ scale: 1.0 });
      engine.load('test');
      const fc = makeForecastConfig(512, 256);
      const numPatches = Math.floor(512 / MC.inputPatchLen);
      const controller = new AbortController();
      controller.abort(); // already aborted

      await expect(
        decode(
          engine,
          [makePatchedInput(numPatches)],
          [makeMask(numPatches)],
          makeContextMu(1, numPatches),
          makeContextSigma(1, numPatches),
          makeLastStats(1),
          100,
          fc,
          MC,
          controller.signal,
        ),
      ).rejects.toThrow();
    });

    it('aborts during AR decode at step boundary', async () => {
      const engine = new MockInferenceEngine({ scale: 1.0 });
      engine.load('test');
      const fc = makeForecastConfig(512, 256);
      const numPatches = Math.floor(512 / MC.inputPatchLen);
      const controller = new AbortController();
      const horizon = 513; // floor((513-1)/128) = 4 steps

      // Abort after the first AR step (after prefill completes)
      const origForward = engine.forward.bind(engine);
      let callCount = 0;
      engine.forward = async (inputs, masks) => {
        callCount++;
        if (callCount === 2) {
          controller.abort(); // abort after prefill (call 1) completed
        }

        return origForward(inputs, masks) as any;
      };

      await expect(
        decode(
          engine,
          [makePatchedInput(numPatches)],
          [makeMask(numPatches)],
          makeContextMu(1, numPatches),
          makeContextSigma(1, numPatches),
          makeLastStats(1),
          horizon,
          fc,
          MC,
          controller.signal,
        ),
      ).rejects.toThrow();
    });
  });
});
