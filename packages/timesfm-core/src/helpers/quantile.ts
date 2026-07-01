/**
 * Quantile helpers for working with TimesFM forecast output.
 *
 * Provides ergonomic access to per-series quantile forecasts
 * without manual Float32Array[][] indexing.
 */

import { QUANTILE_INDICES, type ForecastOutput } from '../types';

/**
 * Confidence level → quantile index mapping.
 *
 * TimesFM 2.5 provides 9 quantiles (q10 through q90).  The widest
 * prediction interval available is Q10–Q90, which provides **≥ 80%**
 * coverage.  Higher confidence levels (0.9, 0.95) map to the same
 * Q10–Q90 bounds because the model does not output finer-grained
 * percentiles.  Users needing tighter intervals should note that
 * 90% and 95% coverage is NOT guaranteed — these are at-least-80%
 * bounds labeled for convenience.
 */
const CI_MAP: Record<number, { lower: number; upper: number }> = {
  0.8: { lower: QUANTILE_INDICES.Q10, upper: QUANTILE_INDICES.Q90 },
  0.9: { lower: QUANTILE_INDICES.Q10, upper: QUANTILE_INDICES.Q90 },
  0.95: { lower: QUANTILE_INDICES.Q10, upper: QUANTILE_INDICES.Q90 },
};

/**
 * Extract a specific quantile forecast for a single series.
 *
 * Zero-copy — returns a direct reference to the underlying Float32Array.
 *
 * @param output        Forecast output from `model.forecast()`.
 * @param seriesIndex   Index of the series (0-based).
 * @param quantileIndex Quantile index (use QUANTILE_INDICES constants).
 *
 * @example
 * ```typescript
 * const q10 = getQuantile(output, 0, QUANTILE_INDICES.Q10);
 * ```
 */
export function getQuantile(
  output: ForecastOutput,
  seriesIndex: number,
  quantileIndex: number,
): Float32Array {
  if (seriesIndex < 0 || seriesIndex >= output.pointForecast.length) {
    throw new RangeError(
      `seriesIndex ${seriesIndex} out of range [0, ${output.pointForecast.length})`,
    );
  }
  if (quantileIndex < 0 || quantileIndex >= output.quantileForecast[seriesIndex]!.length) {
    const max = output.quantileForecast[seriesIndex]!.length;
    throw new RangeError(`quantileIndex ${quantileIndex} out of range [0, ${max})`);
  }
  return output.quantileForecast[seriesIndex]![quantileIndex]!;
}

/**
 * Get a prediction interval for a single series.
 *
 * Returns the lower and upper bounds of the prediction interval
 * at the given confidence level.  80% CI maps to Q10–Q90.
 *
 * @param output        Forecast output from `model.forecast()`.
 * @param seriesIndex   Index of the series (0-based).
 * @param confidence    Confidence level: 0.8, 0.9, or 0.95.
 *
 * @example
 * ```typescript
 * const { lower, upper } = getPredictionInterval(output, 0, 0.8);
 * ```
 */
export function getPredictionInterval(
  output: ForecastOutput,
  seriesIndex: number,
  confidence: 0.8 | 0.9 | 0.95,
): { lower: Float32Array; upper: Float32Array } {
  const mapping = CI_MAP[confidence];
  if (!mapping) {
    throw new RangeError(`Unsupported confidence: ${confidence}. Supported: 0.8, 0.9, 0.95`);
  }
  return {
    lower: getQuantile(output, seriesIndex, mapping.lower),
    upper: getQuantile(output, seriesIndex, mapping.upper),
  };
}
