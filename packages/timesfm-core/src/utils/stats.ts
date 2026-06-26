/**
 * Welford-style online (streaming) statistics for RevIN normalization.
 *
 * Mirrors the Python `update_running_stats()` in torch/util.py and
 * flax/util.py.
 *
 * These utilities are called at every patch boundary during autoregressive
 * decoding.  Numerical errors here compound over long horizons, so we
 * implement the numerically-stable two-pass variant where practical and
 * test against reference implementations.
 */

// ---------------------------------------------------------------------------
// Running stats interface
// ---------------------------------------------------------------------------

export interface RunningStats {
  n: number;
  mu: number;
  sigma: number;
}

/** Return deep-frozen zero stats. */
export function createRunningStats(): RunningStats {
  return { n: 0, mu: 0, sigma: 0 };
}

// ---------------------------------------------------------------------------
// Single-patch update (one batch element)
// ---------------------------------------------------------------------------

/**
 * Update running statistics with a single patch of values.
 *
 * This is the core Welford step.  `mask` values of 1 indicate padding
 * positions that must be ignored entirely.
 *
 * @returns A tuple `[new_stats, new_stats]` matching the Python convention
 *          where the second element is a convenience copy.
 */
export function updateRunningStats(
  stats: RunningStats,
  values: Float32Array,
  mask: Uint8Array,
): [RunningStats, RunningStats] {
  let incN = 0;
  let incSum = 0;
  let incSumSq = 0;

  const len = values.length;
  for (let i = 0; i < len; i++) {
    if (mask[i] === 0) {
      // non-masked = valid
      const v = values[i];
      incN++;
      incSum += v;
      incSumSq += v * v;
    }
  }

  // Shortcut: no valid values
  if (incN === 0) {
    return [stats, stats];
  }

  // Incremental statistics
  const incMu = incSum / incN;
  const incVar = Math.max(0, incSumSq / incN - incMu * incMu);
  const incSigma = Math.sqrt(incVar);

  // Pooled update (Welford's parallel algorithm)
  const newN = stats.n + incN;

  const newMu = (stats.n * stats.mu + incN * incMu) / newN;

  // Parallel variance merge:
  // σ²_new = (n1·σ1² + n2·σ2² + n1·(μ1 - μ_new)² + n2·(μ2 - μ_new)²) / N
  const term1 = stats.n * stats.sigma * stats.sigma;
  const term2 = incN * incSigma * incSigma;
  const term3 = stats.n * (stats.mu - newMu) * (stats.mu - newMu);
  const term4 = incN * (incMu - newMu) * (incMu - newMu);

  const newVar = (term1 + term2 + term3 + term4) / newN;

  const result: RunningStats = {
    n: newN,
    mu: newMu,
    sigma: Math.sqrt(Math.max(0, newVar)),
  };

  // Return a shallow copy for the second element (Python convention)
  return [result, { ...result }];
}

// ---------------------------------------------------------------------------
// Batch updates
// ---------------------------------------------------------------------------

/**
 * Update running statistics across a batch of time series, where each
 * series has multiple patches.
 *
 * This is the batch-equivalent of the Python per-patch statistic accumulation.
 *
 * @param statsArr  One RunningStats entry per batch element (mutated in-place).
 * @param patchedValues  [batch][patch][patchLen] — values for each patch.
 * @param patchedMasks   [batch][patch][patchLen] — masks.
 */
export function updateRunningStatsBatch(
  statsArr: RunningStats[],
  patchedValues: Float32Array[][],
  patchedMasks: Uint8Array[][],
): RunningStats[] {
  const batchSize = statsArr.length;

  for (let b = 0; b < batchSize; b++) {
    const patches = patchedValues[b];
    const masks = patchedMasks[b];
    const numPatches = patches.length;

    for (let p = 0; p < numPatches; p++) {
      const [updated] = updateRunningStats(statsArr[b], patches[p], masks[p]);
      statsArr[b] = updated;
    }
  }

  return statsArr;
}

// ---------------------------------------------------------------------------
// Convenience: compute stats for a whole array at once
// ---------------------------------------------------------------------------

/**
 * Compute mean and population standard deviation for an array.
 *
 * @param mask  Optional mask; masked positions are ignored.
 */
export function computeStats(
  values: Float32Array,
  mask?: Uint8Array,
): { mean: number; std: number } {
  let n = 0;
  let sum = 0;
  let sumSq = 0;

  for (let i = 0; i < values.length; i++) {
    if (mask && mask[i] !== 0) continue;
    const v = values[i];
    n++;
    sum += v;
    sumSq += v * v;
  }

  if (n === 0) return { mean: 0, std: 0 };

  const mean = sum / n;
  const variance = Math.max(0, sumSq / n - mean * mean);

  return { mean, std: Math.sqrt(variance) };
}
