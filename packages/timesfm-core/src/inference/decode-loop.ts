/**
 * Autoregressive decode loop for TimesFM.
 *
 * This is the core inference algorithm:
 *
 *   Phase 1 — Prefill:
 *     Feed all context patches through the transformer.  The model produces
 *     output patches (each covering outputPatchLen future steps) and populates
 *     the KV cache.
 *
 *   Phase 2 — Autoregressive Decode:
 *     Take the last output patch's median value, split it into sub-patches,
 *     feed each back through the model to generate the next chunk of the
 *     forecast horizon.
 *
 * Mirrors the `decode()` method in timesfm_2p5_torch.py.
 *
 * @module decode-loop
 */

import {
  TIMESFM_25_CONFIG,
  type IInferenceEngine,
  type ModelConfig,
  type ForecastConfig,
} from '../types';
import { updateRunningStats, type RunningStats } from '../utils/stats';
import { revinBatch, revinBatch4D } from '../utils/revin';
import { concat, concatUint8 } from '../utils/tensor-utils';

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

export interface DecodeResult {
  /** Denormalised output: [batch][numPatches, outputPatchLen, numQuantiles] — prefill outputs. */
  pfOutputs: Float32Array[];
  /** Denormalised quantile spread: [batch][outputQuantileLen, numQuantiles] — from last patch. */
  quantileSpreads: Float32Array[];
  /**
   * Denormalised AR outputs: [batch] — each entry is the concatenated
   * flat quantile array from all autoregressive decode steps.
   * Indexing: arOutputs[b] contains all AR steps for batch element b,
   * each step produces outputPatchLen * numQuantiles floats in the same
   * interleaved (time × quantile) layout as the prefill output, allowing
   * uniform post-processing across all forecast steps.
   */
  arOutputs: Float32Array[] | null;
}

// ---------------------------------------------------------------------------
// Main decode function
// ---------------------------------------------------------------------------

/**
 * Run the full decode loop.
 *
 * **Concurrency safety**: This function is pure with respect to the engine
 * — it reads from the engine but never mutates shared state.  Multiple
 * `decode()` calls can run concurrently (e.g., main path + flip path in
 * `Promise.all`) without races, provided the engine supports concurrent
 * `forward()` calls.  ONNX Runtime sessions are safe for concurrent use.
 *
 * @param engine    Inference backend (ONNX Runtime).
 * @param normedInputs   Pre-normalised patched inputs [batch][numPatches * inputPatchLen].
 * @param patchedMasks   Patch masks [batch][numPatches * inputPatchLen].
 * @param contextMu      Per-patch means [batchSize * numPatches].
 * @param contextSigma   Per-patch stds [batchSize * numPatches].
 * @param lastStats      Running stats after last context patch (for AR decode start).
 * @param horizon        Forecast horizon.
 * @param fc             Forecast config.
 * @param mc             Model config.
 * @param signal         Optional AbortSignal for cancellation.
 */
export async function decode(
  engine: IInferenceEngine,
  normedInputs: Float32Array[],
  patchedMasks: Uint8Array[],
  contextMu: Float32Array[],
  contextSigma: Float32Array[],
  lastStats: RunningStats[],
  horizon: number,
  fc: ForecastConfig,
  mc: ModelConfig = TIMESFM_25_CONFIG,
  signal?: AbortSignal,
): Promise<DecodeResult> {
  const batchSize = normedInputs.length;
  const numInputPatches = Math.floor(fc.maxContext / mc.inputPatchLen);

  // Pad contextMu/contextSigma to match the exported model's patch count
  const paddedContextMu: Float32Array[] = [];
  const paddedContextSigma: Float32Array[] = [];

  for (let b = 0; b < batchSize; b++) {
    for (let p = 0; p < mc.exportedPatches; p++) {
      if (p < numInputPatches) {
        paddedContextMu.push(contextMu[b * numInputPatches + p]!);
        paddedContextSigma.push(contextSigma[b * numInputPatches + p]!);
      } else {
        paddedContextMu.push(new Float32Array([0]));
        paddedContextSigma.push(new Float32Array([1.0]));
      }
    }
  }
  const numDecodeSteps = Math.max(0, Math.floor((horizon - 1) / mc.outputPatchLen));

  // ---- Phase 1: Prefill ----

  // Abort check before long-running prefill
  signal?.throwIfAborted();

  // Run the model's forward pass
  const rawOutput = await engine.forward(normedInputs, patchedMasks);
  const { outputTimeSeries, outputQuantileSpread } = rawOutput;

  // Denormalise prefill outputs
  // outputTimeSeries is [batch][mc.exportedPatches * outputPatchLen * numQuantiles]
  const pfDenormed = revinBatch4D(
    outputTimeSeries,
    paddedContextMu,
    paddedContextSigma,
    true,
    mc.exportedPatches,
    mc.outputPatchLen,
    mc.numQuantiles,
  );

  // Trim padding back to actual patches: keep only the first numInputPatches outputs
  const pfTrimmed: Float32Array[] = [];
  for (let b = 0; b < batchSize; b++) {
    const perPatch = mc.outputPatchLen * mc.numQuantiles;
    const validLen = numInputPatches * perPatch;
    pfTrimmed.push(pfDenormed[b]!.slice(0, validLen));
  }

  // Extract quantile spread from the last patch
  const quantileSpreads: Float32Array[] = [];
  for (let b = 0; b < batchSize; b++) {
    const qsLen = mc.outputQuantileLen * mc.numQuantiles;
    const perPatch = outputQuantileSpread[b]!.length / mc.exportedPatches;
    const lastPatchQS = outputQuantileSpread[b]!.slice(outputQuantileSpread[b]!.length - perPatch);
    // Denormalise — process per-patch blocks rather than per-element
    const lastMu = paddedContextMu[(b + 1) * mc.exportedPatches - 1]![0]!;
    const lastSigma = paddedContextSigma[(b + 1) * mc.exportedPatches - 1]![0]!;
    const safeS = lastSigma! < 1e-6 ? 1.0 : lastSigma!;
    const denormed = new Float32Array(qsLen);
    for (let o = 0; o < mc.outputQuantileLen; o++) {
      const base = o * mc.numQuantiles;
      for (let q = 0; q < mc.numQuantiles; q++) {
        const idx = base + q;
        denormed[idx] = idx < lastPatchQS.length ? lastPatchQS[idx]! * safeS + lastMu! : 0;
      }
    }
    quantileSpreads.push(denormed);
  }

  // ---- Phase 2: Autoregressive Decode ----

  if (numDecodeSteps === 0) {
    return { pfOutputs: pfTrimmed, quantileSpreads, arOutputs: null };
  }

  // Extract last output patch's median as the seed for AR decode
  // arOutputsByBatch[b] accumulates all AR step outputs for batch element b
  const arOutputsByBatch: Float32Array[][] = Array.from({ length: batchSize }, () => []);

  let arSeeds: Float32Array[] = [];
  for (let b = 0; b < batchSize; b++) {
    const perPatch = mc.outputPatchLen * mc.numQuantiles;
    const lastPatchStart = pfTrimmed[b]!.length - perPatch;
    const seed = new Float32Array(mc.outputPatchLen);
    for (let o = 0; o < mc.outputPatchLen; o++) {
      const idx = lastPatchStart + o * mc.numQuantiles + mc.decodeIndex;
      seed[o] = pfTrimmed[b]![idx]!;
    }
    arSeeds.push(seed);
  }

  const arStats = lastStats.map((s) => ({ ...s }));

  for (let step = 0; step < numDecodeSteps; step++) {
    // Check for abort at each decode step boundary
    signal?.throwIfAborted();

    // Split the AR seed (length outputPatchLen=128) into m sub-patches of length inputPatchLen=32
    const m = mc.outputPatchesPerInput; // 4
    const arPatches: Float32Array[] = [];
    const arMasks: Uint8Array[] = [];
    const stepMus: Float32Array[] = [];
    const stepSigmas: Float32Array[] = [];

    for (let b = 0; b < batchSize; b++) {
      const patches: Float32Array[] = [];
      const masks: Uint8Array[] = [];
      let stats = { ...arStats[b]! };

      for (let sp = 0; sp < m; sp++) {
        const offset = sp * mc.inputPatchLen;
        const patch = arSeeds[b]!.slice(offset, offset + mc.inputPatchLen);
        const mask = new Uint8Array(mc.inputPatchLen); // all zeros

        const [updated] = updateRunningStats(stats, patch, mask);
        stats = updated!;

        patches.push(patch);
        masks.push(mask);
        stepMus.push(new Float32Array([stats.mu]));
        stepSigmas.push(new Float32Array([stats.sigma]));
      }

      arPatches.push(concat(patches));
      arMasks.push(concatUint8(masks));
      arStats[b] = stats;
    }

    // RevIN normalize
    const normedAr = revinBatch(
      arPatches,
      stepMus,
      stepSigmas,
      false, // forward
      m,
      mc.inputPatchLen,
    );

    // Run model forward
    const arRaw = await engine.forward(normedAr, arMasks);

    // Denormalise
    const arDenormed = revinBatch4D(
      arRaw.outputTimeSeries,
      stepMus,
      stepSigmas,
      true,
      m,
      mc.outputPatchLen,
      mc.numQuantiles,
    );

    // Take the LAST output sub-patch's full quantile output as the forecast
    // for this AR step, and its median as the seed for the next step.
    //
    // IMPORTANT: revinBatch4D allocates a full exportedPatches-sized result array,
    // zero-filling positions beyond the actual m sub-patches.  We MUST index by
    // (m - 1), NOT by array.length, to avoid reading from the zero-filled padding
    // region.  This contract is part of revinBatch4D's documented behavior — if
    // revinBatch4D's memory strategy changes, this code must be updated in lockstep.
    const nextSeeds: Float32Array[] = [];
    const perSubPatch = mc.outputPatchLen * mc.numQuantiles;
    const lastSubPatchStart = (m - 1) * perSubPatch;
    for (let b = 0; b < batchSize; b++) {
      // Extract median seed for the next AR decode step
      const seed = new Float32Array(mc.outputPatchLen);
      for (let o = 0; o < mc.outputPatchLen; o++) {
        const idx = lastSubPatchStart + o * mc.numQuantiles + mc.decodeIndex;
        seed[o] = arDenormed[b]![idx]!;
      }
      nextSeeds.push(seed);

      // Extract the full quantile forecast for this AR step
      // (outputPatchLen time steps × numQuantiles quantiles).
      // This preserves the same interleaved layout as the prefill output,
      // allowing the postprocessor to apply flip invariance, continuous
      // quantile head, and quantile crossing fix uniformly across all
      // forecast steps — both prefill and autoregressive.
      const arStepForecast = new Float32Array(mc.outputPatchLen * mc.numQuantiles);
      for (let o = 0; o < mc.outputPatchLen; o++) {
        const srcBase = lastSubPatchStart + o * mc.numQuantiles;
        const dstBase = o * mc.numQuantiles;
        for (let q = 0; q < mc.numQuantiles; q++) {
          arStepForecast[dstBase + q] = arDenormed[b]![srcBase + q]!;
        }
      }
      arOutputsByBatch[b]!.push(arStepForecast);
    }

    arSeeds = nextSeeds;
  }

  // Flatten per-batch AR outputs: arOutputs[b] = concat of all step outputs for batch b
  const arOutputs = arOutputsByBatch.map((arrs) => concat(arrs));

  return { pfOutputs: pfTrimmed, quantileSpreads, arOutputs };
}
