/**
 * @agentix-e/timesfm-hierarchical
 *
 * Hierarchical time series reconciliation for TimesFM.
 *
 * Implements the Hyndman et al. (2011) framework for reconciling
 * base forecasts across a tree-structured hierarchy:
 *
 *   - **Bottom-Up (BU)** — P = [0 | I_n], simplest, guarantees coherence
 *   - **Ordinary Least Squares (OLS)** — P = (SᵀS)⁻¹Sᵀ, ignores error structure
 *   - **Weighted Least Squares (WLS)** — P = (SᵀW⁻¹S)⁻¹SᵀW⁻¹, diagonal W
 *   - **Minimum Trace (MinT)** — P = (SᵀW⁻¹S)⁻¹SᵀW⁻¹, full covariance W
 *
 * ## Quick Start
 *
 * ```typescript
 * import { TimesFMModel, createForecastConfig } from '@agentix-e/timesfm-core';
 * import { reconcileForecast } from '@agentix-e/timesfm-hierarchical';
 *
 * const model = await TimesFMModel.fromPretrained({ modelPath: './model.onnx' });
 * model.compile(createForecastConfig({ maxContext: 512, maxHorizon: 128 }));
 *
 * const result = await reconcileForecast(model, {
 *   hierarchy: {
 *     nodes: [
 *       { id: 'total', parentId: null },
 *       { id: 'regionA', parentId: 'total' },
 *       { id: 'regionB', parentId: 'total' },
 *     ],
 *   },
 *   inputs: { total: dataTotal, regionA: dataA, regionB: dataB },
 *   horizon: 24,
 *   reconcile: { strategy: 'mint' },
 * });
 * ```
 *
 * @module timesfm-hierarchical
 */

export { buildSummingMatrix } from './summing-matrix';
export type { SummingMatrixResult } from './summing-matrix';

export { computeProjectionMatrix, reconcileBaseForecasts } from './reconciliation';

export { reconcileForecast } from './hierarchical';

export type {
  HierarchyNode,
  HierarchyDefinition,
  ReconciliationStrategy,
  ReconcileOptions,
  ReconciliationResult,
  BaseForecasts,
  HierarchicalForecastParams,
  HierarchicalForecastOutput,
} from './types';
