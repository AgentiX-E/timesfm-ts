/**
 * @agentix-e/timesfm-web
 *
 * Browser-compatible TimesFM time-series forecasting using onnxruntime-web.
 *
 * ## Quick Start
 *
 * ```typescript
 * import { TimesFMModel, createForecastConfig } from '@agentix-e/timesfm-core';
 * import { TimesFMWebInferenceEngine, loadModelFromUrl } from '@agentix-e/timesfm-web';
 *
 * // 1. Download the model
 * const { buffer } = await loadModelFromUrl('/models/timesfm-2.5.onnx');
 *
 * // 2. Create web engine and load model
 * const engine = new TimesFMWebInferenceEngine(config);
 * await engine.load(buffer);
 *
 * // 3. Create model with injected engine
 * const model = await TimesFMModel.fromPretrained({
 *   modelPath: '/models/timesfm-2.5.onnx',
 *   engine,
 * });
 *
 * // 4. Forecast
 * model.compile(createForecastConfig({ maxContext: 512, maxHorizon: 128 }));
 * const result = await model.forecast(24, [inputData]);
 * ```
 *
 * @module timesfm-web
 */

export { TimesFMWebInferenceEngine } from './web-engine';
export type { WebEngineLogger } from './web-engine';
export { loadModelFromUrl, checkModelAvailability } from './model-loader';
export type { ModelLoaderOptions, ModelLoadResult } from './model-loader';
