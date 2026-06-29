/**
 * Integration tests for hierarchical forecast reconciliation with real TimesFM model.
 *
 * Requires a real ONNX model (set TIMESFM_TEST_MODEL).
 * Excluded from unit-tier coverage via vitest.unit.config.ts.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { TimesFMModel, createForecastConfig } from '@agentix-e/timesfm-core';
import { reconcileForecast } from '../src/hierarchical';
import type { HierarchyDefinition } from '../src/types';
import { businessMetric } from '../../timesfm-core/test/test-fixtures';
import { getTestModelPath } from '../../timesfm-core/test/helpers';

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

const MODEL_PATH = getTestModelPath();
const skipTests = process.env.VITEST_SKIP_ONNX_TESTS === 'true' || !MODEL_PATH;
const describeIf = skipTests ? describe.skip : describe;

let model: TimesFMModel | null = null;

// 3-level hierarchy: Total → {West, East} → 4 stores
const HIERARCHY: HierarchyDefinition = {
  nodes: [
    { id: 'total', parentId: null },
    { id: 'west', parentId: 'total' },
    { id: 'east', parentId: 'total' },
    { id: 's1', parentId: 'west' },
    { id: 's2', parentId: 'west' },
    { id: 's3', parentId: 'east' },
    { id: 's4', parentId: 'east' },
  ],
};

const HORIZON = 12;
const SERIES_LEN = 128; // enough context for a short forecast

describeIf('reconcileForecast (integration)', () => {
  beforeAll(async () => {
    model = await TimesFMModel.fromPretrained({ modelPath: MODEL_PATH! });
    model.compile(
      createForecastConfig({
        maxContext: 128,
        maxHorizon: 64,
        normalizeInputs: true,
        forceFlipInvariance: true,
        inferIsPositive: false,
        perCoreBatchSize: 2,
      }),
    );
  }, 60000);

  afterAll(async () => {
    if (model) await model.dispose();
  });

  // ── Main forecast ──────────────────────────────────────────────────────

  it('reconciles a 3-level hierarchy with all 4 strategies', async () => {
    const strategies = ['bu', 'ols', 'wls', 'mint'] as const;
    const totalData = businessMetric(SERIES_LEN);
    const westData = businessMetric(SERIES_LEN);
    const eastData = businessMetric(SERIES_LEN);
    const s1Data = businessMetric(SERIES_LEN);
    const s2Data = businessMetric(SERIES_LEN);
    const s3Data = businessMetric(SERIES_LEN);
    const s4Data = businessMetric(SERIES_LEN);

    for (const strategy of strategies) {
      const result = await reconcileForecast(model!, {
        hierarchy: HIERARCHY,
        inputs: {
          total: totalData,
          west: westData,
          east: eastData,
          s1: s1Data,
          s2: s2Data,
          s3: s3Data,
          s4: s4Data,
        },
        horizon: HORIZON,
        reconcile: { strategy },
      });

      // All reconciled outputs exist and are horizon-length
      for (const id of ['total', 'west', 'east', 's1', 's2', 's3', 's4']) {
        const node = result.reconciled[id];
        expect(node).toBeDefined();
        expect(node.pointForecast.length).toBe(HORIZON);
        expect(node.quantileForecast.length).toBe(10);
        for (let q = 0; q < 10; q++) {
          expect(node.quantileForecast[q].length).toBe(HORIZON);
        }
      }
    }
  });

  // ── Coherence property ─────────────────────────────────────────────────

  it('ensures reconciled Total == West + East (within float tolerance)', async () => {
    const data = businessMetric(SERIES_LEN);

    const result = await reconcileForecast(model!, {
      hierarchy: HIERARCHY,
      inputs: {
        total: data,
        west: data,
        east: data,
        s1: data,
        s2: data,
        s3: data,
        s4: data,
      },
      horizon: HORIZON,
      reconcile: { strategy: 'mint' },
    });

    const r = result.reconciled;
    for (let h = 0; h < HORIZON; h++) {
      const sum12 = r.west.pointForecast[h] + r.east.pointForecast[h];
      expect(r.total.pointForecast[h]).toBeCloseTo(sum12, 2);
    }
  });

  it('ensures reconciled West == S1 + S2', async () => {
    const data = businessMetric(SERIES_LEN);

    const result = await reconcileForecast(model!, {
      hierarchy: HIERARCHY,
      inputs: { total: data, west: data, east: data, s1: data, s2: data, s3: data, s4: data },
      horizon: HORIZON,
      reconcile: { strategy: 'ols' },
    });

    for (let h = 0; h < HORIZON; h++) {
      const sumStore =
        result.reconciled.s1.pointForecast[h] + result.reconciled.s2.pointForecast[h];
      expect(result.reconciled.west.pointForecast[h]).toBeCloseTo(sumStore, 2);
    }
  });

  // ── All outputs finite ─────────────────────────────────────────────────

  it('produces finite outputs (no NaN/Infinity)', async () => {
    const data = businessMetric(SERIES_LEN);

    const result = await reconcileForecast(model!, {
      hierarchy: HIERARCHY,
      inputs: { total: data, west: data, east: data, s1: data, s2: data, s3: data, s4: data },
      horizon: HORIZON,
      reconcile: { strategy: 'bu' },
    });

    for (const id of ['total', 'west', 'east', 's1', 's2', 's3', 's4']) {
      for (let h = 0; h < HORIZON; h++) {
        expect(Number.isFinite(result.reconciled[id].pointForecast[h])).toBe(true);
      }
    }
  });

  // ── Error cases ────────────────────────────────────────────────────────

  it('throws when a node is missing from inputs', async () => {
    const data = businessMetric(SERIES_LEN);
    await expect(
      reconcileForecast(model!, {
        hierarchy: HIERARCHY,
        inputs: { total: data, west: data, east: data, s1: data }, // missing s2, s3, s4
        horizon: HORIZON,
      }),
    ).rejects.toThrow('missing');
  });
});
