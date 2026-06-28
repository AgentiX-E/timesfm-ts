/**
 * @experimental
 *
 * KV Cache for autoregressive transformer decoding.
 *
 * **Current status**: The ONNX-exported TimesFM model manages its own
 * internal KV cache.  This module provides an external KV cache
 * implementation prepared for a future pure-TypeScript Transformer
 * implementation.  It is **not used** by the current ONNX inference path.
 *
 * Each transformer layer maintains a cache of past Key and Value tensors
 * so that during autoregressive generation we only need to compute
 * attention for the new tokens, not the entire sequence.
 *
 * Types `KVCacheLayer` and `KVCache` are re-exported from `../types`
 * for convenience; `KVCacheLayer` is the canonical per-layer interface
 * and `KVCache` is the composed array-of-layers type.
 *
 * @see DecodeCache in torch/util.py (Python reference)
 */

import type { KVCacheLayer } from '../types';

// Re-export for convenience
export type { KVCacheLayer };

/**
 * Full KV cache across all transformer layers.
 */
export type KVCache = KVCacheLayer[];

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/**
 * Create a KV cache for all transformer layers.
 *
 * @param batchSize       Number of sequences in the batch.
 * @param maxCacheSize    Maximum number of patches that can be cached.
 * @param numLayers       Number of transformer layers (20 for TimesFM 2.5).
 * @param numHeads        Number of attention heads (16).
 * @param headDim         Dimension per head (80).
 */
export function createKVCache(
  batchSize: number,
  maxCacheSize: number,
  numLayers: number,
  numHeads: number,
  headDim: number,
): KVCache {
  const caches: KVCache = [];

  for (let l = 0; l < numLayers; l++) {
    const key = new Float32Array(batchSize * maxCacheSize * numHeads * headDim);
    const value = new Float32Array(batchSize * maxCacheSize * numHeads * headDim);

    caches.push({
      nextIndex: new Int32Array(batchSize), // all zeros
      numMasked: new Int32Array(batchSize), // all zeros
      key,
      value,
    });
  }

  return caches;
}

// ---------------------------------------------------------------------------
// Reset / clone
// ---------------------------------------------------------------------------

/**
 * Reset all KV caches to their initial (empty) state.
 */
export function resetKVCache(caches: KVCache): void {
  for (const cache of caches) {
    cache.nextIndex.fill(0);
    cache.numMasked.fill(0);
    cache.key.fill(0);
    cache.value.fill(0);
  }
}

/**
 * Deep-clone a KV cache (for flip-invariance dual-path decoding).
 */
export function cloneKVCache(caches: KVCache): KVCache {
  return caches.map((c) => ({
    nextIndex: new Int32Array(c.nextIndex),
    numMasked: new Int32Array(c.numMasked),
    key: new Float32Array(c.key),
    value: new Float32Array(c.value),
  }));
}

// ---------------------------------------------------------------------------
// Shape helpers for callers
// ---------------------------------------------------------------------------

/**
 * Compute the required KV cache size for a given context and horizon.
 *
 * cacheSize = numInputPatches + numDecodeSteps * outputPatchesPerInput
 */
export function computeCacheSize(
  numInputPatches: number,
  horizon: number,
  outputPatchLen: number,
  outputPatchesPerInput: number,
): number {
  const numDecodeSteps = Math.max(0, Math.floor((horizon - 1) / outputPatchLen));
  return numInputPatches + numDecodeSteps * outputPatchesPerInput;
}
