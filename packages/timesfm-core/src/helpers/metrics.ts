/**
 * Time-series forecast evaluation metrics.
 *
 * Pure functions operating on Float32Array — zero dependencies,
 * zero allocations beyond the return value.
 *
 * Reference sources:
 *   - scikit-learn sklearn.metrics (BSD License)
 *   - Hyndman & Koehler (2006) "Another look at measures of forecast accuracy"
 *   - Google TimesFM evaluation harness
 */

/** Minimum absolute value threshold for division safety. */
const EPSILON = 1e-10;

/**
 * Mean Absolute Error.
 *
 * MAE = (1/n) * Σ|actual_i - predicted_i|
 */
export function mae(actual: Float32Array, predicted: Float32Array): number {
  if (actual.length !== predicted.length) {
    throw new RangeError(`Length mismatch: actual=${actual.length}, predicted=${predicted.length}`);
  }
  if (actual.length === 0) return 0;
  let sum = 0;
  let count = 0;
  for (let i = 0; i < actual.length; i++) {
    if (Number.isFinite(actual[i]!) && Number.isFinite(predicted[i]!)) {
      sum += Math.abs(actual[i]! - predicted[i]!);
      count++;
    }
  }
  return count > 0 ? sum / count : 0;
}

/**
 * Root Mean Square Error.
 *
 * RMSE = sqrt((1/n) * Σ(actual_i - predicted_i)²)
 */
export function rmse(actual: Float32Array, predicted: Float32Array): number {
  if (actual.length !== predicted.length) {
    throw new RangeError(`Length mismatch: actual=${actual.length}, predicted=${predicted.length}`);
  }
  if (actual.length === 0) return 0;
  let sumSq = 0;
  let count = 0;
  for (let i = 0; i < actual.length; i++) {
    if (Number.isFinite(actual[i]!) && Number.isFinite(predicted[i]!)) {
      const diff = actual[i]! - predicted[i]!;
      sumSq += diff * diff;
      count++;
    }
  }
  return count > 0 ? Math.sqrt(sumSq / count) : 0;
}

/**
 * Mean Absolute Percentage Error.
 *
 * MAPE = (100/n) * Σ|(actual_i - predicted_i) / actual_i|
 *
 * Points where |actual_i| < 1e-10 are skipped to avoid division by zero.
 */
export function mape(actual: Float32Array, predicted: Float32Array): number {
  if (actual.length !== predicted.length) {
    throw new RangeError(`Length mismatch: actual=${actual.length}, predicted=${predicted.length}`);
  }
  if (actual.length === 0) return 0;
  let sum = 0;
  let count = 0;
  for (let i = 0; i < actual.length; i++) {
    if (
      Number.isFinite(actual[i]) &&
      Number.isFinite(predicted[i]) &&
      Math.abs(actual[i]!) > EPSILON
    ) {
      sum += Math.abs((actual[i]! - predicted[i]!) / actual[i]!);
      count++;
    }
  }
  return count > 0 ? (sum / count) * 100 : 0;
}

/**
 * Symmetric Mean Absolute Percentage Error.
 *
 * SMAPE = (100/n) * Σ 2 * |actual_i - predicted_i| / (|actual_i| + |predicted_i|)
 *
 * Range: [0, 200].  Symmetric — swapping actual and predicted yields
 * the same result.
 */
export function smape(actual: Float32Array, predicted: Float32Array): number {
  if (actual.length !== predicted.length) {
    throw new RangeError(`Length mismatch: actual=${actual.length}, predicted=${predicted.length}`);
  }
  if (actual.length === 0) return 0;
  let sum = 0;
  let count = 0;
  for (let i = 0; i < actual.length; i++) {
    if (Number.isFinite(actual[i]!) && Number.isFinite(predicted[i]!)) {
      const denominator = Math.abs(actual[i]!) + Math.abs(predicted[i]!);
      if (denominator > EPSILON) {
        sum += (2 * Math.abs(actual[i]! - predicted[i]!)) / denominator;
        count++;
      }
    }
  }
  return count > 0 ? (sum / count) * 100 : 0;
}

/**
 * Mean Absolute Scaled Error.
 *
 * MASE = MAE(model) / MAE(naive)
 *
 * Values < 1 indicate the model outperforms the naive (no-change) forecast.
 */
export function mase(
  actual: Float32Array,
  predicted: Float32Array,
  naiveForecast: Float32Array,
): number {
  const modelMAE = mae(actual, predicted);
  const naiveMAE = mae(actual, naiveForecast);
  if (naiveMAE < EPSILON) return modelMAE < EPSILON ? 1 : Infinity;
  return modelMAE / naiveMAE;
}

/**
 * R² coefficient of determination.
 *
 * R² = 1 - SS_res / SS_tot
 *
 * Range: (-∞, 1].  1 = perfect fit, 0 = mean-predictor baseline,
 * negative = worse than the mean baseline.
 */
export function r2Score(actual: Float32Array, predicted: Float32Array): number {
  if (actual.length !== predicted.length) {
    throw new RangeError(`Length mismatch: actual=${actual.length}, predicted=${predicted.length}`);
  }
  if (actual.length === 0) return 0;

  let sumActual = 0;
  let n = 0;
  for (let i = 0; i < actual.length; i++) {
    if (Number.isFinite(actual[i]!)) {
      sumActual += actual[i]!;
      n++;
    }
  }
  if (n === 0) return 0;
  const meanActual = sumActual / n;

  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < actual.length; i++) {
    if (Number.isFinite(actual[i]!) && Number.isFinite(predicted[i]!)) {
      const diffRes = actual[i]! - predicted[i]!;
      const diffTot = actual[i]! - meanActual;
      ssRes += diffRes * diffRes;
      ssTot += diffTot * diffTot;
    }
  }

  if (ssTot < EPSILON) {
    // Constant target — R² is undefined. Return 1 if predictions are
    // perfect (zero residual), 0 otherwise (conservative default).
    return ssRes < EPSILON ? 1 : 0;
  }
  return 1 - ssRes / ssTot;
}

/**
 * Prediction Interval Coverage.
 *
 * The fraction of actual values that fall within [lower, upper].
 *
 * Range: [0, 1].  1 = all values covered, 0 = none covered.
 */
export function picCoverage(
  actual: Float32Array,
  lower: Float32Array,
  upper: Float32Array,
): number {
  if (actual.length !== lower.length || actual.length !== upper.length) {
    throw new RangeError('Length mismatch between actual, lower, and upper arrays');
  }
  if (actual.length === 0) return 0;
  let covered = 0;
  let total = 0;
  for (let i = 0; i < actual.length; i++) {
    if (Number.isFinite(actual[i]!) && Number.isFinite(lower[i]!) && Number.isFinite(upper[i]!)) {
      if (actual[i]! >= lower[i]! && actual[i]! <= upper[i]!) covered++;
      total++;
    }
  }
  return total > 0 ? covered / total : 0;
}

/**
 * Average Prediction Interval Width.
 *
 * The mean width of the prediction interval across all time steps.
 */
export function piWidth(lower: Float32Array, upper: Float32Array): number {
  if (lower.length !== upper.length) {
    throw new RangeError(`Length mismatch: lower=${lower.length}, upper=${upper.length}`);
  }
  if (lower.length === 0) return 0;
  let sum = 0;
  let count = 0;
  for (let i = 0; i < lower.length; i++) {
    if (Number.isFinite(lower[i]!) && Number.isFinite(upper[i]!)) {
      sum += upper[i]! - lower[i]!;
      count++;
    }
  }
  return count > 0 ? sum / count : 0;
}
