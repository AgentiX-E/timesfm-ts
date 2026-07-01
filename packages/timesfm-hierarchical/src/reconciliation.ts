/**
 * Hierarchical forecast reconciliation strategies.
 *
 * Implements the four standard approaches from Hyndman et al. (2011):
 *
 *   BU  — Bottom-Up:       P = [0 | I_n]
 *   OLS — Ordinary LS:     P = (SᵀS)⁻¹ Sᵀ
 *   WLS — Weighted LS:     P = (SᵀW⁻¹S)⁻¹ SᵀW⁻¹ (W diagonal)
 *   MinT — Min-Trace:      P = (SᵀW⁻¹S)⁻¹ SᵀW⁻¹ (W full covariance)
 *
 * All strategies produce reconciled forecasts via:
 *   ŷ_h = S · P · ŷ_b
 * where ŷ_b is the stacked base forecast vector (length m).
 */

import { Matrix, solve } from 'ml-matrix';
import type { ReconciliationStrategy, ReconcileOptions, BaseForecasts } from './types';
import type { SummingMatrixResult } from './summing-matrix';
import { HierarchyValidationError, ConfigValidationError } from '@agentix-e/timesfm-core';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Convert a Matrix to number[][] for serialisation / diagnostics. */
function matrixTo2D(m: Matrix): number[][] {
  const rows: number[][] = [];
  for (let r = 0; r < m.rows; r++) {
    const row: number[] = [];
    for (let c = 0; c < m.columns; c++) row.push(m.get(r, c));
    rows.push(row);
  }
  return rows;
}

/** Build Matrix from number[][]. */
function matrixFrom2D(data: readonly (readonly number[])[]): Matrix {
  const rows = data.length;
  const cols = rows > 0 ? data[0]!.length : 0;
  const m = new Matrix(rows, cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) m.set(r, c, data[r]![c]!);
  }
  return m;
}

/** Build an m × m diagonal Matrix from a Float32Array of diagonal values. */
function diagFromArray(diag: Float32Array): Matrix {
  const m = new Matrix(diag.length, diag.length);
  for (let i = 0; i < diag.length; i++) m.set(i, i, diag[i]!);
  return m;
}

/** Ridge-regularised solve with automatic retry on failure. */
function solveWithRidge(A: Matrix, b: Matrix, ridge: number, maxRidge = 100): Matrix {
  const p = A.rows;
  const Aug = A.clone();
  // Add ridge to diagonal
  for (let i = 0; i < p; i++) {
    Aug.set(i, i, Aug.get(i, i) + ridge);
  }
  try {
    return solve(Aug, b);
  } catch {
    if (ridge >= maxRidge) {
      throw new HierarchyValidationError(
        `Reconciliation failed: matrix remains singular after increasing ridge to ${ridge}. ` +
          `Check for degenerate hierarchy or constant base forecasts.`,
      );
    }
    return solveWithRidge(A, b, ridge * 10, maxRidge);
  }
}

// ---------------------------------------------------------------------------
// Strategy implementations
// ---------------------------------------------------------------------------

/**
 * Bottom-Up (BU): P = [0_{n×(m-n)} | I_n]
 *
 * Simply picks the bottom-level base forecasts — no reconciliation math
 * beyond building the scaling matrix. BottomUp is the fastest strategy
 * and guarantees coherence by construction.
 */
function projectionBU(S: Matrix): Matrix {
  const m = S.rows;
  const n = S.columns;
  // P = I_n padded with zeros on the left → shape is [n × m], last n cols = I_n
  const P = new Matrix(n, m);
  for (let j = 0; j < n; j++) {
    P.set(j, m - n + j, 1);
  }
  return P;
}

/**
 * Ordinary Least Squares: P = (SᵀS)⁻¹ Sᵀ
 */
function projectionOLS(S: Matrix, ridge: number): Matrix {
  const StS = S.transpose().mmul(S); // n × n
  const St = S.transpose(); // n × m
  return solveWithRidge(StS, St, ridge);
}

/**
 * Weighted Least Squares: P = (SᵀW⁻¹S)⁻¹ SᵀW⁻¹
 *
 * W is diagonal with per-node residual variances.
 */
function projectionWLS(S: Matrix, W: Matrix, ridge: number): Matrix {
  // W⁻¹ = diag(1/w_i)
  const Winv = new Matrix(W.rows, W.columns);
  for (let i = 0; i < W.rows; i++) {
    const w = W.get(i, i);
    Winv.set(i, i, w > 1e-12 ? 1 / w : 1);
  }

  // StWinvS = Sᵀ W⁻¹ S (n × n)
  const StWinv = S.transpose().mmul(Winv);
  const StWinvS = StWinv.mmul(S);

  return solveWithRidge(StWinvS, StWinv, ridge);
}

/**
 * Minimum Trace: P = (SᵀW⁻¹S)⁻¹ SᵀW⁻¹
 *
 * Same formula as WLS but W is the full residual covariance (m × m).
 * Falls back to OLS when W is singular or not provided.
 */
function projectionMinT(S: Matrix, W: Matrix, ridge: number): Matrix {
  // Try to invert W; fall back to OLS on failure
  try {
    const Winv = solve(W, Matrix.eye(W.rows));
    const StWinv = S.transpose().mmul(Winv);
    const StWinvS = StWinv.mmul(S);
    return solveWithRidge(StWinvS, StWinv, ridge);
  } catch {
    return projectionOLS(S, ridge);
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Compute the projection matrix P (n × m) for the given strategy.
 *
 * @param S        The m × n summing matrix.
 * @param strategy The reconciliation strategy.
 * @param options  Strategy-specific parameters:
 *   - `residualCovariance` — m × m covariance Matrix for MinT / WLS.
 *   - `ridge` — ridge regularisation added to diagonal before inversion (default 1e-6).
 *
 * @returns The n × m projection matrix P.
 */
export function computeProjectionMatrix(
  S: Matrix,
  strategy: ReconciliationStrategy,
  options?: {
    readonly residualCovariance?: Matrix;
    readonly ridge?: number;
  },
): Matrix {
  const ridge = options?.ridge ?? 1e-6;

  switch (strategy) {
    case 'bu':
      return projectionBU(S);

    case 'ols':
      return projectionOLS(S, ridge);

    case 'wls': {
      let W: Matrix;
      if (options?.residualCovariance) {
        // Use diagonal of provided covariance
        const m = options.residualCovariance.rows;
        W = new Matrix(m, m);
        for (let i = 0; i < m; i++) {
          W.set(i, i, Math.max(options.residualCovariance.get(i, i), 1e-12));
        }
      } else {
        // No covariance → fall back to OLS with a warning

        console.warn(
          '[timesfm-hierarchical] WLS requires residualCovariance — falling back to OLS.',
        );
        return projectionOLS(S, ridge);
      }
      return projectionWLS(S, W, ridge);
    }

    case 'mint': {
      if (!options?.residualCovariance) {
        console.warn(
          '[timesfm-hierarchical] MinT requires residualCovariance — falling back to OLS.',
        );
        return projectionOLS(S, ridge);
      }
      return projectionMinT(S, options.residualCovariance, ridge);
    }

    default: {
      const exhaustive: never = strategy;
      throw new ConfigValidationError(`Unknown reconciliation strategy: ${exhaustive}`);
    }
  }
}

// ---------------------------------------------------------------------------
// Reconcile base forecasts (model-free)
// ---------------------------------------------------------------------------

/**
 * Reconcile base forecasts at all levels using the given strategy.
 *
 * This is a pure function that operates on pre-computed base forecasts.
 * It does not require a TimesFM model — use {@link reconcileBaseForecasts}
 * when you already have forecasts at all nodes (e.g., from a separate
 * forecasting step).
 *
 * @param summing       The summing matrix and node ordering (from buildSummingMatrix).
 * @param baseForecasts Point forecasts per node id — every node must have an entry.
 * @param options       Reconciliation strategy and parameters.
 *
 * @returns Reconciled forecasts per node + projection matrix for diagnostics.
 *
 * @throws {Error} if any node id in summing is missing from baseForecasts.
 */
export function reconcileBaseForecasts(
  summing: SummingMatrixResult,
  baseForecasts: BaseForecasts,
  options?: ReconcileOptions,
): ReconciledResult {
  const { S: sMat, allNodeIds, bottomNodeIds } = summing;
  const m = allNodeIds.length;
  const n = bottomNodeIds.length;

  // Validate every node has a base forecast
  const horizon = baseForecasts[allNodeIds[0]!]?.length;
  for (const id of allNodeIds) {
    const bf = baseForecasts[id];
    if (!bf || bf.length !== horizon) {
      throw new HierarchyValidationError(
        `Node "${id}" is missing from baseForecasts or has mismatched horizon length.`,
      );
    }
  }

  const strat: ReconciliationStrategy = options?.strategy ?? 'mint';

  // Build S Matrix
  const S = matrixFrom2D(sMat);

  // Build residual covariance if provided
  let residualCov: Matrix | undefined;
  if (options?.residualCovariance && options.residualCovariance.length > 0) {
    const cov = options.residualCovariance;
    residualCov = new Matrix(cov.length, cov.length > 0 ? cov[0]!.length : 0);
    for (let r = 0; r < cov.length; r++) {
      for (let c = 0; c < cov[r]!.length; c++) {
        residualCov.set(r, c, cov[r]![c]!);
      }
    }
  }

  // Auto-estimate WLS diagonal from base forecast variances when no covariance provided
  let effectiveCov: Matrix | undefined = residualCov;
  if ((strat === 'wls' || strat === 'mint') && !residualCov) {
    // Estimate per-node variance from base forecasts for WLS
    const diagVals = new Float32Array(m);
    for (let i = 0; i < m; i++) {
      const bf = baseForecasts[allNodeIds[i]!]!;
      let sum = 0,
        count = 0;
      for (let h = 0; h < bf.length; h++) {
        const v = bf[h]!;
        if (Number.isFinite(v)) {
          sum += v;
          count++;
        }
      }
      const mu = count > 0 ? sum / count : 0;
      // Two-pass variance (numerically stable, avoids catastrophic cancellation)
      let variance = 1e-12;
      if (count > 1) {
        let sqDiff = 0;
        for (let h = 0; h < bf.length; h++) {
          const v = bf[h]!;
          if (Number.isFinite(v)) {
            const d = v - mu;
            sqDiff += d * d;
          }
        }
        variance = Math.max(sqDiff / (count - 1), 1e-12);
      }
      diagVals[i] = variance;
    }
    effectiveCov = diagFromArray(diagVals);
  }

  // Fallback for MinT without covariance: use the auto-estimated diagonal
  const pOptions = {
    residualCovariance: effectiveCov,
    ridge: options?.ridge,
  };

  const P = computeProjectionMatrix(S, strat, pOptions);

  // ── Reconcile each horizon step ─────────────────────────────────────────
  const reconciled: Record<string, Float32Array> = {};

  for (const id of allNodeIds) {
    reconciled[id] = new Float32Array(horizon!);
  }

  for (let h = 0; h < horizon!; h++) {
    // Stack base forecasts into ŷ_b: length-m vector
    const yb: number[] = [];
    for (let i = 0; i < m; i++) {
      const bf = baseForecasts[allNodeIds[i]!]!;
      yb.push(bf[h]!);
    }

    // ŷ_h = S · P · ŷ_b
    // First: P · ŷ_b → bottom-level (length n)
    const bottom: number[] = [];
    for (let j = 0; j < n; j++) {
      let val = 0;
      for (let k = 0; k < m; k++) {
        val += P.get(j, k) * yb[k]!;
      }
      bottom.push(val);
    }

    // Then: S · bottom → all-level (length m)
    for (let i = 0; i < m; i++) {
      let val = 0;
      for (let j = 0; j < n; j++) {
        val += sMat[i]![j]! * bottom[j]!;
      }
      reconciled[allNodeIds[i]!]![h] = val;
    }
  }

  return {
    reconciled,
    projectionMatrix: matrixTo2D(P),
    strategy: strat,
    summingMatrix: sMat.map((row) => [...row]),
  };
}

/** Return type of reconcileBaseForecasts (avoids inline type in JSDoc). */
interface ReconciledResult {
  reconciled: Record<string, Float32Array>;
  projectionMatrix: number[][];
  strategy: ReconciliationStrategy;
  summingMatrix: number[][];
}
