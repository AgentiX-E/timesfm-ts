/**
 * Welford-style online (streaming) statistics for RevIN normalization.
 *
 * Mirrors the Python `update_running_stats()` in torch/util.py and
 * flax/util.py.
 *
 * These utilities are called at every patch boundary during autoregressive
 * decoding.  Numerical errors here compound over long horizons, so we
 * implement the numerically-stable two-pass algorithm and skip NaN/Inf
 * values to prevent data corruption.
 */

// ---------------------------------------------------------------------------
// Running stats interface
// ---------------------------------------------------------------------------

export interface RunningStats {
  n: number;
  mu: number;
  sigma: number;
}

/** Return zero initialised running stats. */
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

  const len = values.length;
  for (let i = 0; i < len; i++) {
    if (mask[i] === 0) {
      // non-masked = valid
      const v = values[i]!;
      // Skip NaN and Infinity to prevent poisoning the running statistics.
      // A single NaN would make sum=NaN, mean=NaN, sigma=NaN, destroying
      // all downstream RevIN normalization.
      if (!Number.isFinite(v)) continue;
      incN++;
      incSum += v;
    }
  }

  // Shortcut: no valid values
  if (incN === 0) {
    return [stats, stats];
  }

  // Numerically-stable two-pass variance:
  // σ² = (Σ(v - μ)²) / N  rather than  Σv²/N - μ²
  // The one-pass E[X²] - E[X]² formula suffers from catastrophic cancellation
  // when values are large relative to their variance.
  const incMu = incSum / incN;

  // Two-pass: accumulate squared deviations from the computed mean
  let incVar = 0;
  for (let i = 0; i < len; i++) {
    if (mask[i] === 0 && Number.isFinite(values[i]!)) {
      const diff = values[i]! - incMu;
      incVar += diff * diff;
    }
  }
  incVar /= incN;
  const incSigma = Math.sqrt(Math.max(0, incVar));

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

  return [result, result];
}

// ---------------------------------------------------------------------------
// Convenience: compute stats for a whole array at once
// ---------------------------------------------------------------------------

/**
 * Compute mean and population standard deviation for an array.
 *
 * Uses the numerically-stable two-pass algorithm and skips NaN/Inf values
 * to prevent data corruption during z-score normalization.
 *
 * @param mask  Optional mask; masked positions are ignored.
 */
export function computeStats(
  values: Float32Array,
  mask?: Uint8Array,
): { mean: number; std: number } {
  // First pass: count valid (finite, unmasked) values and compute mean
  let n = 0;
  let sum = 0;

  for (let i = 0; i < values.length; i++) {
    if (mask && mask[i] !== 0) continue;
    const v = values[i]!;
    if (!Number.isFinite(v)) continue;
    n++;
    sum += v;
  }

  if (n === 0) return { mean: 0, std: 0 };

  const mean = sum / n;

  // Second pass: compute variance from the mean (numerically stable)
  let varSum = 0;
  for (let i = 0; i < values.length; i++) {
    if (mask && mask[i] !== 0) continue;
    const v = values[i]!;
    if (!Number.isFinite(v)) continue;
    const diff = v - mean;
    varSum += diff * diff;
  }

  const variance = Math.max(0, varSum / n);

  return { mean, std: Math.sqrt(variance) };
}
