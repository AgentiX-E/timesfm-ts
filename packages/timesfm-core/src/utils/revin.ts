/**
 * Reversible Instance Normalization (RevIN).
 *
 * Mirrors the Python `revin()` function in torch/util.py and flax/util.py.
 *
 * RevIN normalizes time-series patches before feeding them into the
 * transformer, then reverses the normalization on the output so forecasts
 * are in the original data scale.
 *
 *   Forward:  y = (x - μ) / σ
 *   Reverse:  y = x * σ + μ
 *
 * When σ < ε (1e-6), the forward pass substitutes 1.0 to avoid division
 * by zero.  This happens for constant-valued series.
 */

const TOLERANCE = 1e-6;

// ---------------------------------------------------------------------------
// Scalar RevIN (one batch element, one patch)
// ---------------------------------------------------------------------------

/**
 * Apply RevIN to a single patch of values.
 *
 * @param values   The (possibly multi-dimensional) value array.
 * @param mu       Per-dimension mean(s).
 * @param sigma    Per-dimension std deviation(s).
 * @param reverse  If true, perform inverse normalisation (denormalize).
 */
export function revin(
  values: Float32Array,
  mu: Float32Array | Float32Array[],
  sigma: Float32Array | Float32Array[],
  reverse: boolean,
): Float32Array {
  const len = values.length;
  const result = new Float32Array(len);

  const muFlat = flattenParam(mu, len);
  const sigmaFlat = flattenParam(sigma, len);

  if (reverse) {
    // x * σ + μ
    for (let i = 0; i < len; i++) {
      result[i] = values[i]! * sigmaFlat[i]! + muFlat[i]!;
    }
  } else {
    // (x - μ) / max(σ, ε)
    for (let i = 0; i < len; i++) {
      const safeSigma = sigmaFlat[i]! < TOLERANCE ? 1.0 : sigmaFlat[i]!;
      result[i] = (values[i]! - muFlat[i]!) / safeSigma;
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Batch RevIN — for (batch, patches, patchLen) and (batch, patches, patchLen, q)
// ---------------------------------------------------------------------------

/**
 * Apply RevIN across a batch of patched time series.
 *
 * Shape: `values[b][p * patchLen + i]` where b = batch, p = patch, i = offset.
 *
 * @param values      Flat array per batch element (all patches concatenated).
 * @param mu          Mean per batch element, or per (batch, patch) pair.
 * @param sigma       Std per batch element, or per (batch, patch) pair.
 * @param reverse     If true, denormalize.
 * @param numPatches  Number of patches per batch element.
 * @param patchLen    Length of each patch (output: per-element).
 */
export function revinBatch(
  values: Float32Array[],
  mu: Float32Array[],
  sigma: Float32Array[],
  reverse: boolean,
  numPatches: number,
  patchLen: number,
): Float32Array[] {
  const batchSize = values.length;
  const result: Float32Array[] = [];

  // Determine broadcast pattern:
  // - mu/sigma length === batchSize     → per-batch (apply to all patches)
  // - mu/sigma length === batchSize * numPatches → per-patch
  const perPatch = mu.length === batchSize * numPatches;

  for (let b = 0; b < batchSize; b++) {
    const totalLen = values[b]!.length;
    const out = new Float32Array(totalLen);

    for (let p = 0; p < numPatches; p++) {
      const patchOffset = p * patchLen;
      const muIndex = perPatch ? b * numPatches + p : b;
      const m = mu[muIndex]![0]!; // scalar per patch/batch
      const s = sigma[muIndex]![0]!; // scalar per patch/batch

      for (let i = 0; i < patchLen && patchOffset + i < totalLen; i++) {
        const idx = patchOffset + i;
        if (reverse) {
          out[idx] = values[b]![idx]! * s + m;
        } else {
          const safeS = s < TOLERANCE ? 1.0 : s;
          out[idx] = (values[b]![idx]! - m) / safeS;
        }
      }
    }

    result.push(out);
  }

  return result;
}

// ---------------------------------------------------------------------------
// 4-D RevIN (batch, patches, patchLen, numQuantiles)
// ---------------------------------------------------------------------------

/**
 * Apply RevIN to quantile-shaped outputs.
 *
 * Shape: (batch, patches, patchLen, numQuantiles).
 *
 * mu/sigma are broadcast from (batch,) or (batch, patches) with
 * two trailing singleton dims added internally.
 *
 * **Contract**: The returned array always has exportedPatches-worth of
 * elements per batch entry.  When the caller only populated m < exportedPatches
 * sub-patches, the trailing (exportedPatches - m) sub-patches are zero-filled.
 * Callers MUST index by their known m, not by array.length.  Changing this
 * memory strategy requires updating all callers (notably decode-loop.ts).
 */
export function revinBatch4D(
  values: Float32Array[],
  mu: Float32Array[],
  sigma: Float32Array[],
  reverse: boolean,
  numPatches: number,
  patchLen: number,
  numQuantiles: number,
): Float32Array[] {
  const batchSize = values.length;
  const result: Float32Array[] = [];
  const perPatch = mu.length === batchSize * numPatches;

  for (let b = 0; b < batchSize; b++) {
    // Total elements per batch: numPatches * patchLen * numQuantiles
    const totalLen = values[b]!.length;
    const out = new Float32Array(totalLen);

    for (let p = 0; p < numPatches; p++) {
      for (let i = 0; i < patchLen; i++) {
        for (let q = 0; q < numQuantiles; q++) {
          const idx = (p * patchLen + i) * numQuantiles + q;
          const muIndex = perPatch ? b * numPatches + p : b;
          const m = mu[muIndex]![0]!;
          const s = sigma[muIndex]![0]!;

          if (reverse) {
            out[idx] = values[b]![idx]! * s + m;
          } else {
            const safeS = s < TOLERANCE ? 1.0 : s;
            out[idx] = (values[b]![idx]! - m) / safeS;
          }
        }
      }
    }

    result.push(out);
  }

  return result;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Flatten a RevIN parameter (mu or sigma) into a per-element array
 * matching the value length.
 *
 * Handles broadcasting:
 *   - length 1          → repeat to all elements
 *   - length === len    → direct (already broadcast)
 *   - length === len / patchLen → per-patch (repeats within each patch)
 */
function flattenParam(param: Float32Array | Float32Array[], len: number): Float32Array {
  if (param instanceof Float32Array) {
    return broadcast1D(param, len);
  }
  // Array of scalars — each element is a Float32Array([scalar])
  const flat = new Float32Array(len);
  if (param.length === len) {
    for (let i = 0; i < len; i++) flat[i] = param[i]![0]!;
  } else {
    // Assume it's per-patch sized — repeat values evenly.
    // Handle the remainder to avoid dropping trailing elements.
    const ratio = Math.floor(len / param.length);
    const remainder = len % param.length;
    let cursor = 0;
    for (let i = 0; i < param.length; i++) {
      const repeatCount = ratio + (i < remainder ? 1 : 0);
      for (let j = 0; j < repeatCount; j++) {
        flat[cursor + j] = param[i]![0]!;
      }
      cursor += repeatCount;
    }
  }
  return flat;
}

/**
 * Broadcast a scalar or 1-D array to length `len`.
 */
function broadcast1D(arr: Float32Array, len: number): Float32Array {
  if (arr.length === 1) {
    const result = new Float32Array(len);
    result.fill(arr[0]!);
    return result;
  }
  if (arr.length !== len) {
    throw new RangeError(`broadcast1D: array length ${arr.length} cannot be broadcast to ${len}`);
  }
  return arr;
}
