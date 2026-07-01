/**
 * Hierarchical forecast reconciliation entry point.
 *
 * Forecasts every node independently with TimesFM, then reconciles
 * the stacked base forecasts across the hierarchy using the selected
 * strategy (MinT by default).
 *
 * Usage:
 * ```typescript
 * import { TimesFMModel, createForecastConfig } from '@agentix-e/timesfm-core';
 * import { reconcileForecast } from '@agentix-e/timesfm-hierarchical';
 *
 * const model = await TimesFMModel.fromPretrained({ modelPath });
 * model.compile(createForecastConfig({ maxContext: 512, maxHorizon: 128 }));
 *
 * const result = await reconcileForecast(model, {
 *   hierarchy: {
 *     nodes: [
 *       { id: 'total', parentId: null },
 *       { id: 'west',  parentId: 'total' },
 *       { id: 'east',  parentId: 'total' },
 *       { id: 's1',    parentId: 'west' },
 *       { id: 's2',    parentId: 'west' },
 *       { id: 's3',    parentId: 'east' },
 *       { id: 's4',    parentId: 'east' },
 *     ],
 *   },
 *   inputs: {
 *     total: totalSeries,
 *     west: westSeries,
 *     east: eastSeries,
 *     s1: store1Series,
 *     s2: store2Series,
 *     s3: store3Series,
 *     s4: store4Series,
 *   },
 *   horizon: 24,
 *   reconcile: { strategy: 'mint' },
 * });
 *
 * // result.reconciled.total.pointForecast — coherent aggregate
 * // result.reconciled.s1.pointForecast — reconciled bottom-level
 * ```
 *
 * @module hierarchical
 */

import {
  HierarchyValidationError,
  type TimesFMModel,
  type ForecastOutput,
} from '@agentix-e/timesfm-core';
import type { HierarchicalForecastOutput, HierarchicalForecastParams } from './types';
import { buildSummingMatrix } from './summing-matrix';
import { reconcileBaseForecasts } from './reconciliation';

/**
 * Forecast every node independently with TimesFM, then reconcile the
 * stacked base forecasts across the hierarchy.
 *
 * Steps:
 *   1. Validate the hierarchy structure.
 *   2. Build the summing matrix S.
 *   3. Call {@link TimesFMModel.forecast} once for all nodes (batched).
 *   4. Reconcile via the chosen strategy (MinT by default).
 *   5. Reconstruct per-node quantile forecasts from the reconciled
 *      point forecast (proportional adjustment from base quantiles).
 *
 * @param model  A compiled TimesFMModel instance.
 * @param params Forecast parameters including hierarchy definition,
 *               per-node input series, and horizon.
 *
 * @returns A {@link HierarchicalForecastOutput} with per-node reconciled
 *          forecasts and the top-level (unreconciled) raw model output
 *          for backward compatibility.
 *
 * @throws {Error} if the hierarchy is invalid or any node is missing
 *                 from the inputs map.
 */
export async function reconcileForecast(
  model: TimesFMModel,
  params: HierarchicalForecastParams,
): Promise<HierarchicalForecastOutput> {
  const { hierarchy, inputs, horizon, reconcile: recOptions } = params;

  // 1. Validate & build summing matrix
  const summing = buildSummingMatrix(hierarchy);
  const { allNodeIds } = summing;

  // 2. Validate all nodes have input series
  for (const id of allNodeIds) {
    if (!inputs[id]) {
      throw new HierarchyValidationError(
        `Node "${id}" is missing from the inputs map. ` +
          `All ${allNodeIds.length} nodes in the hierarchy must have input data.`,
      );
    }
  }

  // 3. Batch-forecast all nodes via TimesFM
  const inputList: Float32Array[] = [];
  const inputKeys: string[] = [];

  for (const id of allNodeIds) {
    inputList.push(inputs[id]!);
    inputKeys.push(id);
  }

  const rawOutput: ForecastOutput = await model.forecast(horizon, inputList);

  // 4. Extract per-node base forecasts from raw output
  const baseForecasts: Record<string, Float32Array> = {};
  for (let i = 0; i < inputKeys.length; i++) {
    baseForecasts[inputKeys[i]!] = rawOutput.pointForecast[i]!;
  }

  // 5. Reconcile
  const { reconciled: reconciledPoint } = reconcileBaseForecasts(
    summing,
    baseForecasts,
    recOptions,
  );

  // 6. Reconstruct per-node quantile forecasts using proportional adjustment
  const reconciled: HierarchicalForecastOutput['reconciled'] = {};

  for (let i = 0; i < allNodeIds.length; i++) {
    const id = allNodeIds[i]!;
    const basePoint = baseForecasts[id]!;
    const baseQuantile = rawOutput.quantileForecast[i]!;
    const reconcPoint = reconciledPoint[id]!;

    // Proportional adjustment: quantile_new[h][q] = baseQ[h][q] * (reconciled[h] / base[h])
    // with safe division and zero base → zero reconciled handling
    const adjustedQuantile: Float32Array[] = [];
    for (let q = 0; q < baseQuantile.length; q++) {
      const adjQ = new Float32Array(horizon);
      for (let h = 0; h < horizon; h++) {
        const baseVal = basePoint[h]!;
        const ratio = Math.abs(baseVal) > 1e-10 
          ? Math.max(0.1, Math.min(10, reconcPoint[h]! / baseVal))
          : 1;
        adjQ[h] = baseQuantile[q]![h]! * ratio;
      }
      adjustedQuantile.push(adjQ);
    }

    reconciled[id] = {
      pointForecast: reconcPoint,
      quantileForecast: adjustedQuantile,
    };
  }

  return {
    pointForecast: rawOutput.pointForecast,
    quantileForecast: rawOutput.quantileForecast,
    backcast: rawOutput.backcast,
    reconciled,
  } satisfies HierarchicalForecastOutput;
}
