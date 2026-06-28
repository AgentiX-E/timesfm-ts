/**
 * Hierarchical time-series reconciliation type definitions.
 *
 * Based on Hyndman, Ahmed, Athanasopoulos & Shang (2011)
 * "Optimal combination forecasts for hierarchical time series"
 * Computational Statistics & Data Analysis, 55(9), 2579–2589.
 */

import type { ForecastOutput } from '@agentix-e/timesfm-core';

// ---------------------------------------------------------------------------
// Hierarchy definition
// ---------------------------------------------------------------------------

/** A node in a time-series hierarchy (e.g. "Total > Region > Store"). */
export interface HierarchyNode {
  /** Unique node id (e.g. 'total', 'region:west', 'store:1'). */
  readonly id: string;
  /** Parent node id, or null for the root. */
  readonly parentId: string | null;
  /** Human-readable label. */
  readonly label?: string;
}

/**
 * Complete hierarchy definition.
 *
 * The hierarchy is a tree (or forest aggregated under a single root).
 * Leaf nodes (no children) are the "bottom level" — their forecasts are
 * summed up to produce every ancestor's forecast after reconciliation.
 */
export interface HierarchyDefinition {
  /** All nodes in the hierarchy. Must form a valid tree. */
  readonly nodes: readonly HierarchyNode[];
}

// ---------------------------------------------------------------------------
// Reconciliation strategies
// ---------------------------------------------------------------------------

/**
 * Reconciliation strategy, per Hyndman et al. (2011).
 *
 * - `'bu'`  Bottom-Up:      P = [0 | I_n] — only bottom-level forecasts used.
 * - `'ols'` Ordinary LS:    P = (SᵀS)⁻¹ Sᵀ — ignores forecast error structure.
 * - `'wls'` Weighted LS:    W = diag(σ²_1, …, σ²_m) per-level residual variance.
 * - `'mint'` Min-Trace:     W = full residual covariance (m × m).
 *                           Best theoretical properties, needs ≥ n observations.
 */
export type ReconciliationStrategy = 'bu' | 'ols' | 'wls' | 'mint';

/** Options for reconciliation. */
export interface ReconcileOptions {
  /** Which reconciliation strategy to apply. Default 'mint'. */
  readonly strategy?: ReconciliationStrategy;
  /**
   * Optional residual covariance for WLS / MinT.
   * Shape: [m][m] where m = total node count. When omitted for WLS, the
   * diagonal (per-level variance) is estimated from the base forecasts.
   * When omitted for MinT, falls back to OLS with a warning.
   */
  readonly residualCovariance?: readonly (readonly number[])[];
  /** Ridge regularization added to Wᵦ diagonal to guarantee invertibility. Default 1e-6. */
  readonly ridge?: number;
}

// ---------------------------------------------------------------------------
// Reconciliation result types
// ---------------------------------------------------------------------------

/** Result of reconciling a single forecast horizon. */
export interface ReconciliationResult {
  /** Reconciled point forecasts keyed by node id. Length m. */
  readonly reconciled: Record<string, Float32Array>;
  /** The projection matrix P used (for diagnostics / testing). */
  readonly projectionMatrix: number[][];
  /** Strategy that was applied. */
  readonly strategy: ReconciliationStrategy;
  /** Summing matrix S (m × n). */
  readonly summingMatrix: number[][];
}

/**
 * Input forecasts at every level of the hierarchy, keyed by node id.
 * Each Float32Array is the point forecast for that node over the horizon.
 * All arrays must have the same length (= horizon).
 */
export type BaseForecasts = Record<string, Float32Array>;

/** Input for the model-dependent reconcileForecast entry point. */
export interface HierarchicalForecastParams {
  /** The hierarchy topology. */
  readonly hierarchy: HierarchyDefinition;
  /**
   * Historical observations per node id. Each series is forecast
   * independently by TimesFM, then reconciled.
   */
  readonly inputs: Record<string, Float32Array>;
  /** Forecast horizon (passed to model.forecast). */
  readonly horizon: number;
  /** Reconciliation options. */
  readonly reconcile?: ReconcileOptions;
}

/** Extended forecast output with per-node reconciled forecasts. */
export interface HierarchicalForecastOutput extends ForecastOutput {
  /** Reconciled point + quantile forecasts per node id. */
  readonly reconciled: Record<
    string,
    {
      readonly pointForecast: Float32Array;
      readonly quantileForecast: Float32Array[];
    }
  >;
}
