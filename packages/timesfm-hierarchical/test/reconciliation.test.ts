/**
 * Unit tests for reconciliation strategies.
 *
 * Pure logic — uses deterministic seeded fixtures (businessMetric, hourlyTemp,
 * stockPrice) from test-fixtures.ts to generate realistic base forecasts.
 * No TimesFM model required.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { buildSummingMatrix } from '../src/summing-matrix';
import { computeProjectionMatrix, reconcileBaseForecasts } from '../src/reconciliation';
import type { HierarchyDefinition, BaseForecasts, ReconciliationStrategy } from '../src/types';
import { businessMetric, hourlyTemp, stockPrice } from '../../timesfm-core/test/test-fixtures';
import { Matrix } from 'ml-matrix';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/** 2-level hierarchy: Total → {A, B, C} — 4 nodes total, 3 bottom. */
const TWO_LEVEL: HierarchyDefinition = {
  nodes: [
    { id: 'total', parentId: null },
    { id: 'a', parentId: 'total' },
    { id: 'b', parentId: 'total' },
    { id: 'c', parentId: 'total' },
  ],
};

const HORIZON = 24;

/** Generate deterministic base forecasts for a 2-level hierarchy. */
function makeTwoLevelBaseForecasts(): BaseForecasts {
  const a = businessMetric(128 + HORIZON);
  const b = hourlyTemp(128 + HORIZON);
  const c = stockPrice(128 + HORIZON);

  // Total = A + B + C (for consistency — in real use these are independently forecast)
  const total = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) total[i] = a[i] + b[i] + c[i];

  return {
    total: total.slice(-HORIZON),
    a: a.slice(-HORIZON),
    b: b.slice(-HORIZON),
    c: c.slice(-HORIZON),
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('computeProjectionMatrix', () => {
  const summing = buildSummingMatrix(TWO_LEVEL);
  const S = new Matrix(summing.S.length, summing.S[0].length);
  for (let r = 0; r < summing.S.length; r++)
    for (let c = 0; c < summing.S[0].length; c++) S.set(r, c, summing.S[r][c]);

  it('BU projection P is n×m (3×4) with bottom-cols = I_3', () => {
    const P = computeProjectionMatrix(S, 'bu');
    expect(P.rows).toBe(3); // n
    expect(P.columns).toBe(4); // m
    // Bottom 3 columns (indices 1,2,3) should form identity — total at col 0 is zeroed
    for (let j = 0; j < 3; j++) {
      expect(P.get(j, 1 + j)).toBeCloseTo(1, 10);
    }
    // First column (total) should be all zeros
    for (let j = 0; j < 3; j++) {
      expect(P.get(j, 0)).toBeCloseTo(0, 10);
    }
  });

  it('OLS projection P is n×m (3×4)', () => {
    const P = computeProjectionMatrix(S, 'ols');
    expect(P.rows).toBe(3);
    expect(P.columns).toBe(4);
    // S*P should approximate the identity-like mapping for bottom-level
  });

  it('WLS with diagonal covariance produces P different from OLS', () => {
    const cov = new Matrix(4, 4);
    cov.set(0, 0, 100); // total has higher variance
    cov.set(1, 1, 1);
    cov.set(2, 2, 1);
    cov.set(3, 3, 1);

    const P_ols = computeProjectionMatrix(S, 'ols');
    const P_wls = computeProjectionMatrix(S, 'wls', { residualCovariance: cov });

    // They should differ when variances are unequal
    const diff = P_ols.get(0, 0) - P_wls.get(0, 0);
    expect(Math.abs(diff)).toBeGreaterThan(1e-9);
  });

  it('WLS falls back to OLS when no covariance provided', () => {
    const spy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const P_wls = computeProjectionMatrix(S, 'wls');
    const P_ols = computeProjectionMatrix(S, 'ols');

    // They should be identical (both use OLS under the hood)
    for (let r = 0; r < P_wls.rows; r++)
      for (let c = 0; c < P_wls.columns; c++)
        expect(P_wls.get(r, c)).toBeCloseTo(P_ols.get(r, c), 10);

    expect(spy).toHaveBeenCalledTimes(1);
    spy.mockRestore();
  });

  it('MinT falls back to OLS when no covariance provided', () => {
    const spy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const P_mint = computeProjectionMatrix(S, 'mint');
    const P_ols = computeProjectionMatrix(S, 'ols');

    for (let r = 0; r < P_mint.rows; r++)
      for (let c = 0; c < P_mint.columns; c++)
        expect(P_mint.get(r, c)).toBeCloseTo(P_ols.get(r, c), 10);

    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });

  it('MinT provides residualCovariance and computes correctly', () => {
    const cov = new Matrix(4, 4);
    cov.set(0, 0, 4);
    cov.set(1, 1, 2);
    cov.set(2, 2, 1);
    cov.set(3, 3, 0.5);
    // Off-diagonal: induced correlation between total and children
    cov.set(0, 1, 1);
    cov.set(1, 0, 1);

    const P = computeProjectionMatrix(S, 'mint', { residualCovariance: cov });
    expect(P.rows).toBe(3);
    expect(P.columns).toBe(4);
    for (let r = 0; r < P.rows; r++)
      for (let c = 0; c < P.columns; c++) expect(Number.isFinite(P.get(r, c))).toBe(true);
  });

  it('MinT with diagonal covariance reduces to WLS (property test)', () => {
    const cov = new Matrix(4, 4);
    cov.set(0, 0, 4);
    cov.set(1, 1, 1);
    cov.set(2, 2, 1);
    cov.set(3, 3, 1);

    const P_wls = computeProjectionMatrix(S, 'wls', { residualCovariance: cov });
    const P_mint = computeProjectionMatrix(S, 'mint', { residualCovariance: cov });

    // For diagonal W, MinT ≡ WLS
    for (let r = 0; r < P_mint.rows; r++)
      for (let c = 0; c < P_mint.columns; c++)
        expect(P_mint.get(r, c)).toBeCloseTo(P_wls.get(r, c), 10);
  });

  it('Ridge > 0 produces valid P (avoids singularity)', () => {
    // With large ridge, OLS P should still be computable
    const P = computeProjectionMatrix(S, 'ols', { ridge: 1.0 });
    expect(P.rows).toBe(3);
    expect(P.columns).toBe(4);
    // All values should be finite
    for (let r = 0; r < P.rows; r++)
      for (let c = 0; c < P.columns; c++) expect(Number.isFinite(P.get(r, c))).toBe(true);
  });

  it('throws on unknown strategy', () => {
    expect(() => computeProjectionMatrix(S, 'invalid' as ReconciliationStrategy)).toThrow(
      'Unknown',
    );
  });

  it('MinT with singular covariance falls back to OLS', () => {
    // A rank-1 covariance matrix (all rows identical) should be singular
    const cov = new Matrix(4, 4);
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) cov.set(r, c, 1.0);
    }
    const P_mint = computeProjectionMatrix(S, 'mint', { residualCovariance: cov });
    const P_ols = computeProjectionMatrix(S, 'ols');
    // Singular covariance → fallback to OLS, so they should match
    for (let r = 0; r < P_mint.rows; r++) {
      for (let c = 0; c < P_mint.columns; c++) {
        expect(P_mint.get(r, c)).toBeCloseTo(P_ols.get(r, c), 8);
      }
    }
  });
});

describe('reconcileBaseForecasts', () => {
  const summing = buildSummingMatrix(TWO_LEVEL);
  let baseForecasts: BaseForecasts;

  beforeEach(() => {
    baseForecasts = makeTwoLevelBaseForecasts();
  });

  it('BU reconciliation preserves bottom-level forecasts unchanged', () => {
    const result = reconcileBaseForecasts(summing, baseForecasts, { strategy: 'bu' });
    // Bottom-level should be identical to base
    for (let h = 0; h < HORIZON; h++) {
      expect(result.reconciled.a[h]).toBeCloseTo(baseForecasts.a[h], 6);
      expect(result.reconciled.b[h]).toBeCloseTo(baseForecasts.b[h], 6);
      expect(result.reconciled.c[h]).toBeCloseTo(baseForecasts.c[h], 6);
    }
  });

  it('BU reconciliation: total = sum of children', () => {
    const result = reconcileBaseForecasts(summing, baseForecasts, { strategy: 'bu' });
    for (let h = 0; h < HORIZON; h++) {
      const sum = result.reconciled.a[h] + result.reconciled.b[h] + result.reconciled.c[h];
      expect(result.reconciled.total[h]).toBeCloseTo(sum, 4);
    }
  });

  it('OLS reconciliation is coherent: S × bottom_reconciled == all_reconciled', () => {
    const result = reconcileBaseForecasts(summing, baseForecasts, { strategy: 'ols' });
    const { S, allNodeIds, bottomNodeIds } = summing;

    for (let h = 0; h < HORIZON; h++) {
      // Gather bottom reconciled
      const bottomVals = bottomNodeIds.map((id) => result.reconciled[id][h]);

      // For each node, compute S·bottom and compare to reconciled
      for (let i = 0; i < allNodeIds.length; i++) {
        let sum = 0;
        for (let j = 0; j < bottomNodeIds.length; j++) {
          sum += S[i][j] * bottomVals[j];
        }
        expect(result.reconciled[allNodeIds[i]][h]).toBeCloseTo(sum, 4);
      }
    }
  });

  it('WLS reconciliation is coherent with diagonal covariance', () => {
    const result = reconcileBaseForecasts(summing, baseForecasts, {
      strategy: 'wls',
      residualCovariance: [
        [4, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
      ],
    });
    const { S, allNodeIds, bottomNodeIds } = summing;

    for (let h = 0; h < HORIZON; h++) {
      const bottomVals = bottomNodeIds.map((id) => result.reconciled[id][h]);
      for (let i = 0; i < allNodeIds.length; i++) {
        let sum = 0;
        for (let j = 0; j < bottomNodeIds.length; j++) {
          sum += S[i][j] * bottomVals[j];
        }
        expect(result.reconciled[allNodeIds[i]][h]).toBeCloseTo(sum, 4);
      }
    }
  });

  it('MinT reconciliation is coherent', () => {
    const result = reconcileBaseForecasts(summing, baseForecasts, { strategy: 'mint' });
    const { S, allNodeIds, bottomNodeIds } = summing;

    for (let h = 0; h < HORIZON; h++) {
      const bottomVals = bottomNodeIds.map((id) => result.reconciled[id][h]);
      for (let i = 0; i < allNodeIds.length; i++) {
        let sum = 0;
        for (let j = 0; j < bottomNodeIds.length; j++) {
          sum += S[i][j] * bottomVals[j];
        }
        expect(result.reconciled[allNodeIds[i]][h]).toBeCloseTo(sum, 4);
      }
    }
  });

  it('MinT with explicit full covariance is coherent', () => {
    const cov: readonly (readonly number[])[] = [
      [4, 1, 1, 1],
      [1, 2, 0, 0],
      [1, 0, 1, 0],
      [1, 0, 0, 0.5],
    ];
    const result = reconcileBaseForecasts(summing, baseForecasts, {
      strategy: 'mint',
      residualCovariance: cov,
    });
    const { S, allNodeIds, bottomNodeIds } = summing;

    for (let h = 0; h < HORIZON; h++) {
      const bottomVals = bottomNodeIds.map((id) => result.reconciled[id][h]);
      for (let i = 0; i < allNodeIds.length; i++) {
        let sum = 0;
        for (let j = 0; j < bottomNodeIds.length; j++) {
          sum += S[i][j] * bottomVals[j];
        }
        expect(result.reconciled[allNodeIds[i]][h]).toBeCloseTo(sum, 4);
      }
    }
  });

  it('reconcile twice is idempotent (already coherent)', () => {
    const first = reconcileBaseForecasts(summing, baseForecasts, { strategy: 'ols' });
    const second = reconcileBaseForecasts(summing, first.reconciled, { strategy: 'ols' });

    // Second pass should be identical (already coherent)
    for (const id of summing.allNodeIds) {
      for (let h = 0; h < HORIZON; h++) {
        expect(second.reconciled[id][h]).toBeCloseTo(first.reconciled[id][h], 3);
      }
    }
  });

  it('returns correct strategy and summing matrix in result', () => {
    const result = reconcileBaseForecasts(summing, baseForecasts, { strategy: 'bu' });
    expect(result.strategy).toBe('bu');
    expect(result.summingMatrix).toEqual(summing.S);
    expect(result.projectionMatrix.length).toBe(summing.bottomNodeIds.length);
    expect(result.projectionMatrix[0].length).toBe(summing.allNodeIds.length);
  });

  it('throws when node is missing from baseForecasts', () => {
    const missing = { ...baseForecasts };
    delete missing.a;
    expect(() => reconcileBaseForecasts(summing, missing)).toThrow('missing');
  });
});
