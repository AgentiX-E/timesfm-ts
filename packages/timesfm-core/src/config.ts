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
import { createRequire } from 'node:module';

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
  const keys: Array<keyof ForecastConfig> = [
    'maxContext',
    'maxHorizon',
    'normalizeInputs',
    'useContinuousQuantileHead',
    'forceFlipInvariance',
    'inferIsPositive',
    'fixQuantileCrossing',
    'returnBackcast',
    'perCoreBatchSize',
  ];
  return keys.every((k) => a[k] === b[k]);
}

// ---------------------------------------------------------------------------
// Batch-size suggestion
// ---------------------------------------------------------------------------

/**
 * Suggest a `perCoreBatchSize` based on available system memory.
 *
 * TimesFM 2.5 200M loads ~1.5 GB into RAM (model weights + activations).
 * Each additional batch element adds ~200 MB for intermediate tensors.
 * This function computes a safe batch size that fits within a configurable
 * fraction of free memory (default: 50 %).
 *
 * The result is clamped to [1, 16] — the model processes elements
 * concurrently via `Promise.all`, so diminishing returns apply beyond 8.
 *
 * ```typescript
 * import { suggestBatchSize, createForecastConfig } from '@agentix-e/timesfm-core';
 *
 * const bs = suggestBatchSize();
 * const fc = createForecastConfig({ perCoreBatchSize: bs });
 * ```
 *
 * @param freeMemoryGB   Available RAM in GB (auto-detected via `os.freemem()`).
 * @param memoryFraction   Fraction of free RAM to use (0–1, default 0.5).
 * @returns Suggested batch size (1–16).
 */
export function suggestBatchSize(freeMemoryGB?: number, memoryFraction: number = 0.5): number {
  // ~1.5 GB for model + ONNX Runtime overhead
  const MODEL_OVERHEAD_GB = 1.5;
  // ~0.2 GB per additional batch element (intermediate tensors)
  const PER_BATCH_GB = 0.2;

  if (freeMemoryGB === undefined) {
    // Dynamic require via createRequire — Node.js always provides 'node:os'.
    // Non-Node runtimes must pass freeMemoryGB as a parameter.
    const _require = createRequire(import.meta.url);
    // eslint-disable-next-line @typescript-eslint/consistent-type-imports
    const os: typeof import('node:os') = _require('node:os');
    freeMemoryGB = os.freemem() / 1024 ** 3;
  }

  const usableGB = freeMemoryGB * memoryFraction - MODEL_OVERHEAD_GB;
  if (usableGB <= 0) return 1;
  return Math.max(1, Math.min(16, Math.floor(usableGB / PER_BATCH_GB)));
}
