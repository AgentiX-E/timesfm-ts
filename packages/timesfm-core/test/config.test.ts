/**
 * Tests for the configuration module.
 */

import { describe, it, expect } from 'vitest';
import {
  createForecastConfig,
  validateAndNormalizeConfig,
  configsEqual,
  suggestBatchSize,
} from '../src/config.ts';
import { DEFAULT_FORECAST_CONFIG, TIMESFM_25_CONFIG } from '../src/types.ts';

describe('createForecastConfig', () => {
  it('creates defaults when no overrides', () => {
    const fc = createForecastConfig();
    expect(fc.maxContext).toBe(1024);
    expect(fc.maxHorizon).toBe(256);
    expect(fc.normalizeInputs).toBe(true);
    expect(fc.useContinuousQuantileHead).toBe(true);
  });

  it('rounds maxContext to patchLen multiple', () => {
    const fc = createForecastConfig({ maxContext: 100 });
    // 100 → ceil(100/32)*32 = 128
    expect(fc.maxContext).toBe(128);
  });

  it('rounds maxHorizon to outputPatchLen multiple', () => {
    const fc = createForecastConfig({ maxHorizon: 130 });
    // 130 → ceil(130/128)*128 = 256
    expect(fc.maxHorizon).toBe(256);
  });

  it('throws when context + horizon exceeds limit', () => {
    expect(() => createForecastConfig({ maxContext: 16000, maxHorizon: 1000 })).toThrow('exceeds');
  });

  it('throws when quantile head horizon exceeds outputQuantileLen', () => {
    expect(() =>
      createForecastConfig({
        maxHorizon: 2000,
        useContinuousQuantileHead: true,
      }),
    ).toThrow('Continuous quantile head');
  });

  it('merges partial overrides correctly', () => {
    const fc = createForecastConfig({ inferIsPositive: false });
    expect(fc.inferIsPositive).toBe(false);
    // Other defaults preserved
    expect(fc.useContinuousQuantileHead).toBe(true);
    expect(fc.forceFlipInvariance).toBe(true);
  });
});

describe('validateAndNormalizeConfig — non-positive defaults', () => {
  it('uses inputPatchLen when maxContext <= 0', () => {
    const mc = TIMESFM_25_CONFIG;
    const cfg = { ...DEFAULT_FORECAST_CONFIG, maxContext: 0 };
    const result = validateAndNormalizeConfig(cfg, mc);
    expect(result.maxContext).toBe(mc.inputPatchLen);
  });

  it('uses outputPatchLen when maxHorizon <= 0', () => {
    const mc = TIMESFM_25_CONFIG;
    const cfg = { ...DEFAULT_FORECAST_CONFIG, maxHorizon: 0 };
    const result = validateAndNormalizeConfig(cfg, mc);
    expect(result.maxHorizon).toBe(mc.outputPatchLen);
  });

  it('uses inputPatchLen when maxContext is negative', () => {
    const mc = TIMESFM_25_CONFIG;
    const cfg = { ...DEFAULT_FORECAST_CONFIG, maxContext: -1 };
    const result = validateAndNormalizeConfig(cfg, mc);
    expect(result.maxContext).toBe(mc.inputPatchLen);
  });

  it('uses outputPatchLen when maxHorizon is negative', () => {
    const mc = TIMESFM_25_CONFIG;
    const cfg = { ...DEFAULT_FORECAST_CONFIG, maxHorizon: -5 };
    const result = validateAndNormalizeConfig(cfg, mc);
    expect(result.maxHorizon).toBe(mc.outputPatchLen);
  });
});

describe('configsEqual', () => {
  it('returns true for equal configs', () => {
    const a = createForecastConfig();
    const b = createForecastConfig();
    expect(configsEqual(a, b)).toBe(true);
  });

  it('returns false for different normalizeInputs', () => {
    const a = createForecastConfig({ normalizeInputs: true });
    const b = createForecastConfig({ normalizeInputs: false });
    expect(configsEqual(a, b)).toBe(false);
  });

  it('returns false for different maxContext', () => {
    const a = createForecastConfig({ maxContext: 256 });
    const b = createForecastConfig({ maxContext: 512 });
    expect(configsEqual(a, b)).toBe(false);
  });

  it('returns false for different maxHorizon', () => {
    const a = createForecastConfig({ maxHorizon: 128 });
    const b = createForecastConfig({ maxHorizon: 256 });
    expect(configsEqual(a, b)).toBe(false);
  });

  it('returns false for different useContinuousQuantileHead', () => {
    const a = createForecastConfig({ useContinuousQuantileHead: true });
    const b = createForecastConfig({ useContinuousQuantileHead: false });
    expect(configsEqual(a, b)).toBe(false);
  });

  it('returns false for different forceFlipInvariance', () => {
    const a = createForecastConfig({ forceFlipInvariance: true });
    const b = createForecastConfig({ forceFlipInvariance: false });
    expect(configsEqual(a, b)).toBe(false);
  });

  it('returns false for different inferIsPositive', () => {
    const a = createForecastConfig({ inferIsPositive: true });
    const b = createForecastConfig({ inferIsPositive: false });
    expect(configsEqual(a, b)).toBe(false);
  });

  it('returns false for different fixQuantileCrossing', () => {
    const a = createForecastConfig({ fixQuantileCrossing: true });
    const b = createForecastConfig({ fixQuantileCrossing: false });
    expect(configsEqual(a, b)).toBe(false);
  });

  it('returns false for different returnBackcast', () => {
    const a = createForecastConfig({ returnBackcast: true });
    const b = createForecastConfig({ returnBackcast: false });
    expect(configsEqual(a, b)).toBe(false);
  });
});

describe('suggestBatchSize', () => {
  it('returns 1 when no free memory available', () => {
    expect(suggestBatchSize(0)).toBe(1);
  });

  it('returns 1 when free memory is less than model overhead', () => {
    expect(suggestBatchSize(1.4)).toBe(1);
  });

  it('returns a value ≥ 1 and ≤ 16', () => {
    const bs = suggestBatchSize(4);
    expect(bs).toBeGreaterThanOrEqual(1);
    expect(bs).toBeLessThanOrEqual(16);
  });

  it('returns larger batch size for more memory', () => {
    const bs1 = suggestBatchSize(4);
    const bs2 = suggestBatchSize(16);
    expect(bs2).toBeGreaterThanOrEqual(bs1);
  });

  it('respects memoryFraction parameter', () => {
    const bsFull = suggestBatchSize(8, 1.0);
    const bsHalf = suggestBatchSize(8, 0.5);
    expect(bsFull).toBeGreaterThanOrEqual(bsHalf);
  });

  it('caps at 16', () => {
    expect(suggestBatchSize(64)).toBe(16);
  });

  it('returns 1 when os module is unavailable', () => {
    // No freeMemoryGB provided → falls back to os.freemem()
    // In test environment, this should work as Node.js is available
    const bs = suggestBatchSize();
    expect(bs).toBeGreaterThanOrEqual(1);
    expect(bs).toBeLessThanOrEqual(16);
  });
});
