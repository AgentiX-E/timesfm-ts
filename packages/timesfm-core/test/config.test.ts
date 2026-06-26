/**
 * Tests for the configuration module.
 */

import { describe, it, expect } from 'vitest';
import { createForecastConfig, validateAndNormalizeConfig, configsEqual } from '../src/config.ts';
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
