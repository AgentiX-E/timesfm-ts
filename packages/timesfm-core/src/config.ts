/**
 * ForecastConfig management and validation.
 *
 * Mirrors the Python `ForecastConfig` frozen-dataclass semantics plus the
 * auto-adjustment logic in `compile()` that rounds `maxContext` and
 * `maxHorizon` to patch-size multiples.
 */

import {
  DEFAULT_FORECAST_CONFIG,
  TIMESFM_25_CONFIG,
  type ForecastConfig,
  type ModelConfig,
} from './types';
import { ConfigValidationError } from './errors';

/**
 * Validate and normalise a ForecastConfig against a ModelConfig.
 *
 * Returns a *new* object; the input is never mutated.
 *
 * Adjustments applied (matching the Python `compile()` logic):
 *  - `maxContext` rounded up to the next multiple of `inputPatchLen`.
 *  - `maxHorizon` rounded up to the next multiple of `outputPatchLen`.
 *  - If `useContinuousQuantileHead` is true, `maxHorizon` must be ≤
 *    `outputQuantileLen`.
 *
 * @throws {RangeError} if context + horizon exceeds the model's limit.
 */
export function validateAndNormalizeConfig(
  fc: ForecastConfig,
  mc: ModelConfig = TIMESFM_25_CONFIG,
): ForecastConfig {
  // --- Copy to avoid mutating the caller's object. ---
  let { maxContext, maxHorizon } = fc;

  // --- Round context to patch boundary ---
  if (maxContext % mc.inputPatchLen !== 0) {
    maxContext = Math.ceil(maxContext / mc.inputPatchLen) * mc.inputPatchLen;
  }

  // --- Round horizon to output-patch boundary ---
  if (maxHorizon % mc.outputPatchLen !== 0) {
    maxHorizon = Math.ceil(maxHorizon / mc.outputPatchLen) * mc.outputPatchLen;
  }

  // --- Hard limits ---
  if (maxContext + maxHorizon > mc.contextLimit) {
    throw new ConfigValidationError(
      `Context + horizon (${maxContext} + ${maxHorizon} = ${
        maxContext + maxHorizon
      }) exceeds the model's context limit (${mc.contextLimit}).`,
    );
  }

  if (fc.useContinuousQuantileHead && maxHorizon > mc.outputQuantileLen) {
    throw new ConfigValidationError(
      `Continuous quantile head requires maxHorizon ≤ ${mc.outputQuantileLen}, got ${maxHorizon}.`,
    );
  }

  // --- Ensure sane defaults ---
  if (maxContext <= 0) {
    maxContext = mc.inputPatchLen;
  }
  if (maxHorizon <= 0) {
    maxHorizon = mc.outputPatchLen;
  }

  return {
    ...fc,
    maxContext,
    maxHorizon,
  };
}

/**
 * Create a ForecastConfig by merging user overrides into the defaults,
 * then normalising.
 */
export function createForecastConfig(
  overrides: Partial<ForecastConfig> = {},
  mc: ModelConfig = TIMESFM_25_CONFIG,
): ForecastConfig {
  const merged: ForecastConfig = { ...DEFAULT_FORECAST_CONFIG, ...overrides };
  return validateAndNormalizeConfig(merged, mc);
}

/**
 * Check whether two ForecastConfigs are equivalent (ignoring fields that
 * don't affect the compiled decode function).
 */
export function configsEqual(a: ForecastConfig, b: ForecastConfig): boolean {
  return (
    a.maxContext === b.maxContext &&
    a.maxHorizon === b.maxHorizon &&
    a.normalizeInputs === b.normalizeInputs &&
    a.useContinuousQuantileHead === b.useContinuousQuantileHead &&
    a.forceFlipInvariance === b.forceFlipInvariance &&
    a.inferIsPositive === b.inferIsPositive &&
    a.fixQuantileCrossing === b.fixQuantileCrossing &&
    a.returnBackcast === b.returnBackcast
  );
}
