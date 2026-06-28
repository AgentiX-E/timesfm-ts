/**
 * @agentix-e/timesfm-core
 *
 * Node.js/TypeScript reimplementation of Google Research's TimesFM —
 * a decoder-only foundation model for zero-shot time-series forecasting.
 */

// ---- Errors (public hierarchy) ----
export {
  TimesFMError,
  ModelNotCompiledError,
  ModelNotFoundError,
  ConfigValidationError,
  HorizonExceededError,
  DownloadError,
  ProxyAuthError,
  ChecksumMismatchError,
  InferenceError,
} from './errors';

// ---- Public API ----
export { TimesFMModel } from './model';

// ---- Model Downloader ----
export {
  downloadModel,
  defaultModelPath,
  isModelCached,
  getCachedModelPath,
} from './model-downloader';
export type { DownloadOptions, ProxyConfig } from './model-downloader';

// ---- Configuration ----
export {
  createForecastConfig,
  validateAndNormalizeConfig,
  configsEqual,
  suggestBatchSize,
} from './config';

// ---- Model Descriptor ----
export {
  type ModelDescriptor,
  type OnnxDescriptor,
  type ArchitectureDescriptor,
  type ProcessingDescriptor,
  type ModelIdentity,
  ENGINE_SUPPORTED_SCHEMA,
  PRECISION_DEFAULT,
  descriptorToModelConfig,
  loadModelDescriptor,
  resolveModelConfig,
} from './model-descriptor';
export type { ModelPrecision } from './model-descriptor';

// ---- Types ----
export type {
  ForecastConfig,
  ModelConfig,
  ModelLoadOptions,
  ForecastOutput,
  ForecastCallOptions,
  ProgressEvent,
  ProgressCallback,
  ITimesFMModel,
  CovariateForecastParams,
  CovariateForecastOutput,
  Series,
  BatchSeries,
  Mask,
  BatchMask,
} from './types';

export {
  DEFAULT_FORECAST_CONFIG,
  TIMESFM_25_CONFIG,
  QUANTILE_INDICES,
  QUANTILE_NAMES,
  createTimesFM25Config,
} from './types';

// ---- Inference Backend ----
export { TimesFMInferenceEngine } from './inference/onnx-engine';

export type { IInferenceEngine, RawModelOutput } from './types';

// ---- Utilities (for advanced users) ----
export {
  stripLeadingNaNs,
  linearInterpolateNaNs,
  cleanSeries,
  hasNaN,
  countNaN,
  stripTrailingNaNs,
  replaceInfWithNaN,
} from './utils/nan-handler';

export { createRunningStats, updateRunningStats, computeStats } from './utils/stats';
export type { RunningStats } from './utils/stats';

export { revin, revinBatch, revinBatch4D } from './utils/revin';

export {
  reshape2D,
  reshape3D,
  leftPad,
  concat,
  concatUint8,
  stack,
  sliceEach,
  takeLast,
  clipMin,
  clipMax,
  elementwiseMean,
  elementwiseDiff,
  negate,
  mean,
  std,
  allNonNegative,
  hasInvalid,
} from './utils/tensor-utils';

// ---- Pre/Post processing ----
export { preprocess } from './preprocessor';
export type { PreprocessedData } from './preprocessor';

export {
  flipQuantileArray,
  fixQuantileCrossing,
  applyContinuousQuantileHead,
  reverseInputNormalization,
  postProcess,
} from './postprocessor';

// ---- Decode loop ----
export { decode } from './inference/decode-loop';
export type { DecodeResult } from './inference/decode-loop';

// ---- KV Cache (@experimental — unstable API, may change) ----
export { createKVCache, resetKVCache, cloneKVCache, computeCacheSize } from './inference/kv-cache';
export type { KVCacheLayer, KVCache } from './inference/kv-cache';

// ---- Evaluation Metrics ----
export { mae, rmse, mape, smape, mase, r2Score, picCoverage, piWidth } from './helpers/metrics';

// ---- Quantile Helpers ----
export { getQuantile, getPredictionInterval } from './helpers/quantile';
