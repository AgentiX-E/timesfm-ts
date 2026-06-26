/**
 * Complete data preprocessing pipeline for TimesFM.
 *
 * Takes raw user-provided time series and produces the patched, padded,
 * normalized tensors required by the model's forward pass.
 *
 * Pipeline:
 *   1. Clean each series (trailing NaN → leading NaN → interpolate internal)
 *   2. Pad/truncate each series to `maxContext` length
 *   3. Split into patches of `inputPatchLen`
 *   4. Compute per-patch running statistics (RevIN μ, σ)
 *   5. Apply RevIN normalization
 *
 * Mirrors the logic in `TimesFM_2p5.forecast()` and `decode()`.
 */

import type { ForecastConfig, ModelConfig } from './types';
import { cleanSeries } from './utils/nan-handler';
import { createRunningStats, updateRunningStats, type RunningStats } from './utils/stats';
import { revinBatch } from './utils/revin';
import { leftPad, concat, concatUint8 } from './utils/tensor-utils';

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/**
 * Output of the full preprocessing pipeline — ready for model inference.
 */
export interface PreprocessedData {
  /** Patched input [batchSize][numPatches * inputPatchLen] — flat per batch entry. */
  patchedInputs: Float32Array[];
  /** Patch-level mask [batchSize][numPatches * inputPatchLen]. */
  patchedMasks: Uint8Array[];
  /** Per-patch means [batchSize * numPatches] — for RevIN reversal. */
  contextMu: Float32Array[];
  /** Per-patch std deviations [batchSize * numPatches] — for RevIN reversal. */
  contextSigma: Float32Array[];
  /** Per-batch-element running stats after the last patch. */
  lastStats: RunningStats[];
  /** Number of patches per series (= maxContext / inputPatchLen). */
  numPatches: number;
  /** Number of input series. */
  batchSize: number;
  /** The cleaned raw inputs (for post-processing reference). */
  cleanedInputs: Float32Array[];
  /** The truncated inputs (after truncation to maxContext, before padding). */
  truncatedInputs: Float32Array[];
}

// ---------------------------------------------------------------------------
// Main pipeline
// ---------------------------------------------------------------------------

/**
 * Run the full preprocessing pipeline on a batch of raw time series.
 *
 * @param inputs  Raw 1-D time series (any length, may contain NaN).
 * @param fc      Forecast configuration (controls maxContext).
 * @param mc      Model architecture config (controls patch sizes).
 */
export function preprocess(
  inputs: Float32Array[],
  fc: ForecastConfig,
  mc: ModelConfig,
): PreprocessedData {
  const batchSize = inputs.length;
  const { inputPatchLen } = mc;
  const numPatches = Math.floor(fc.maxContext / inputPatchLen);

  // ---- Step 1: Clean each series ----
  const cleanedInputs = inputs.map((s) => cleanSeries(s));

  // ---- Step 2: Pad/truncate to maxContext ----
  const padded: Float32Array[] = [];
  const fullMasks: Uint8Array[] = [];
  const truncatedInputs: Float32Array[] = [];

  for (const series of cleanedInputs) {
    const { padded: p, mask: m } = leftPad(series, fc.maxContext);
    padded.push(p);
    fullMasks.push(m);
    // Record the truncated version (last maxContext points)
    truncatedInputs.push(
      series.length > fc.maxContext ? series.slice(series.length - fc.maxContext) : series,
    );
  }

  // ---- Step 3: Split into patches ----
  const patchedInputs: Float32Array[] = [];
  const patchedMasks: Uint8Array[] = [];

  for (let b = 0; b < batchSize; b++) {
    const flatInput = padded[b];
    const flatMask = fullMasks[b];

    // Concatenate all patches into one flat array per batch element
    const patchValues: Float32Array[] = [];
    const patchMasks: Uint8Array[] = [];

    for (let p = 0; p < numPatches; p++) {
      const offset = p * inputPatchLen;
      patchValues.push(flatInput.slice(offset, offset + inputPatchLen));
      patchMasks.push(flatMask.slice(offset, offset + inputPatchLen));
    }

    patchedInputs.push(concat(patchValues));
    patchedMasks.push(concatUint8(patchMasks));
  }

  // ---- Step 4: Compute per-patch running statistics ----
  const contextMu: Float32Array[] = [];
  const contextSigma: Float32Array[] = [];
  const lastStats: RunningStats[] = [];

  for (let b = 0; b < batchSize; b++) {
    let stats = createRunningStats();
    const flatInput = padded[b];
    const flatMask = fullMasks[b];

    for (let p = 0; p < numPatches; p++) {
      const offset = p * inputPatchLen;
      const patchValues = flatInput.slice(offset, offset + inputPatchLen);
      const patchMask = flatMask.slice(offset, offset + inputPatchLen);

      const [updated] = updateRunningStats(stats, patchValues, patchMask);
      stats = updated;

      contextMu.push(new Float32Array([stats.mu]));
      contextSigma.push(new Float32Array([stats.sigma]));
    }

    lastStats.push({ ...stats });
  }

  // ---- Step 5: Apply RevIN normalization ----
  const normed = revinBatch(
    patchedInputs,
    contextMu,
    contextSigma,
    false, // forward normalization
    numPatches,
    inputPatchLen,
  );

  // Apply mask → zero out padded positions
  for (let b = 0; b < batchSize; b++) {
    const mask = patchedMasks[b];
    for (let i = 0; i < normed[b].length; i++) {
      if (mask[i] === 1) {
        normed[b][i] = 0;
      }
    }
  }

  return {
    patchedInputs: normed,
    patchedMasks,
    contextMu,
    contextSigma,
    lastStats,
    numPatches,
    batchSize,
    cleanedInputs,
    truncatedInputs,
  };
}
