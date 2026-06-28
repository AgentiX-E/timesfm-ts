/**
 * Core type definitions for TimesFM TypeScript implementation.
 *
 * Mirrors the TimesFM Python source:
 *   - src/timesfm/configs.py (ForecastConfig, TransformerConfig, etc.)
 *   - src/timesfm/timesfm_2p5/timesfm_2p5_base.py (TimesFM_2p5_200M_Definition)
 */

import type { ModelPrecision } from './model-descriptor';
// ---------------------------------------------------------------------------
// Numeric aliases
// ---------------------------------------------------------------------------

/** A 1-D typed array representing a univariate time series. */
export type Series = Float32Array;

/** A batch of time series — one per element. */
export type BatchSeries = Series[];

/** Mask array: 1 = masked (padding/ignore), 0 = valid. */
export type Mask = Uint8Array;

/** Batch of masks. */
export type BatchMask = Mask[];

// ---------------------------------------------------------------------------
// Forecast configuration
// ---------------------------------------------------------------------------

/**
 * Configuration controlling all forecast-time behavior.
 *
 * This is the direct TypeScript equivalent of `ForecastConfig` in configs.py.
 * `compile()` may auto-adjust `maxContext` and `maxHorizon` to multiples of
 * the patch sizes.
 */
export interface ForecastConfig {
  /**
   * Maximum context length (number of historical time points).
   * Must be a multiple of `inputPatchLen` (32).  Auto-adjusted by `compile()`.
   */
  maxContext: number;

  /**
   * Maximum forecast horizon (number of future time points).
   * Must be a multiple of `outputPatchLen` (128).  Auto-adjusted by `compile()`.
   */
  maxHorizon: number;

  /** Whether to z-normalize each series before feeding to the model. */
  normalizeInputs: boolean;

  /** Number of series processed per device per batch. */
  perCoreBatchSize: number;

  /**
   * Use the 30M-parameter continuous quantile head for better
   * prediction-interval calibration at longer horizons.
   */
  useContinuousQuantileHead: boolean;

  /**
   * Ensure f(-x) = -f(x) by averaging the forecast of x and the negated
   * forecast of -x.
   */
  forceFlipInvariance: boolean;

  /**
   * If all input values are non-negative, clamp forecasts ≥ 0.
   * Set false for series that can take negative values (temperature, returns).
   */
  inferIsPositive: boolean;

  /**
   * Post-process quantiles to guarantee monotonicity:
   * q10 ≤ q20 ≤ … ≤ q90.
   */
  fixQuantileCrossing: boolean;

  /**
   * Return the model's reconstruction of the input (backcast) in addition
   * to the forecast.  Required for covariate (XReg) workflows.
   */
  returnBackcast: boolean;
}

/**
 * Sensible production defaults matching the Python reference's recommendations.
 * Frozen to match Python's frozen-dataclass semantics.
 */
export const DEFAULT_FORECAST_CONFIG: Readonly<ForecastConfig> = Object.freeze({
  maxContext: 1024,
  maxHorizon: 256,
  normalizeInputs: true,
  perCoreBatchSize: 1,
  useContinuousQuantileHead: true,
  forceFlipInvariance: true,
  inferIsPositive: true,
  fixQuantileCrossing: true,
  returnBackcast: false,
});

// ---------------------------------------------------------------------------
// Model architecture configuration (TimesFM 2.5 200M)
// ---------------------------------------------------------------------------

/**
 * Immutable definition of the TimesFM 2.5 200M model architecture.
 *
 * These values match `TimesFM_2p5_200M_Definition` in timesfm_2p5_base.py
 * exactly.  Changing any of them requires retraining the model.
 */
export interface ModelConfig {
  /** Maximum context + horizon (16,384 for v2.5). */
  readonly contextLimit: number;

  /** Number of patches in the exported ONNX model's fixed batch dimension. */
  readonly exportedPatches: number;

  /** Input patch length — number of time steps per input patch. */
  readonly inputPatchLen: number;

  /** Output patch length — number of time steps per output patch. */
  readonly outputPatchLen: number;

  /** Output length for the quantile spread head. */
  readonly outputQuantileLen: number;

  /** Number of output patches per input patch (= outputPatchLen / inputPatchLen). */
  readonly outputPatchesPerInput: number;

  /** Quantile levels (excluding the mean at index 0). */
  readonly quantiles: readonly number[];

  /** Index of the median quantile used for autoregressive decoding. */
  readonly decodeIndex: number;

  /** Number of transformer layers. */
  readonly numLayers: number;

  /** Number of attention heads. */
  readonly numHeads: number;

  /** Model dimension. */
  readonly modelDims: number;

  /** Head dimension (= modelDims / numHeads). */
  readonly headDim: number;

  /** Number of quantiles including the mean (10). */
  readonly numQuantiles: number;

  // Tokenizer config
  readonly tokenizerInputDims: number;
  readonly tokenizerHiddenDims: number;
  readonly tokenizerOutputDims: number;

  // Output projection configs
  readonly outputPointDims: number;
  readonly outputQuantileDims: number;
}

/**
 * Factory for the canonical TimesFM 2.5 200M architecture definition.
 */
export function createTimesFM25Config(): ModelConfig {
  const inputPatchLen = 32;
  const outputPatchLen = 128;
  const outputQuantileLen = 1024;
  const numLayers = 20;
  const numHeads = 16;
  const modelDims = 1280;
  const quantiles: readonly number[] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

  return Object.freeze({
    contextLimit: 16384,
    exportedPatches: 16,
    inputPatchLen,
    outputPatchLen,
    outputQuantileLen,
    outputPatchesPerInput: outputPatchLen / inputPatchLen, // 4
    quantiles: Object.freeze([...quantiles]),
    decodeIndex: 5, // index of the median in the output
    numLayers,
    numHeads,
    modelDims,
    headDim: modelDims / numHeads, // 80
    numQuantiles: quantiles.length + 1, // 10 (mean + 9 quantiles)
    tokenizerInputDims: inputPatchLen + inputPatchLen, // 64 (values + mask)
    tokenizerHiddenDims: 1280,
    tokenizerOutputDims: 1280,
    outputPointDims: 1280,
    outputQuantileDims: outputQuantileLen * (quantiles.length + 1), // 10240
  });
}

/**
 * The canonical TimesFM 2.5 configuration singleton.
 */
export const TIMESFM_25_CONFIG: ModelConfig = createTimesFM25Config();

// ---------------------------------------------------------------------------
// Model loading options
// ---------------------------------------------------------------------------

export interface ModelLoadOptions {
  /** Path to ONNX model file. Required. */
  modelPath: string;
  /** Execution provider: 'cpu' | 'cuda' | 'dml' (default 'cpu'). */
  executionProvider?: 'cpu' | 'cuda' | 'dml';
  /**
   * Hint for which precision variant to load.
   * The engine itself does not branch on precision — ONNX Runtime
   * transparently executes QDQ (INT8) graphs.  This field is
   * informational metadata forwarded from the model descriptor
   * for logging and diagnostics.
   */
  precision?: ModelPrecision;
  /**
   * Optional pre-built inference engine for dependency injection.
   *
   * When provided, the engine is used as-is (already loaded).
   * When omitted, a default TimesFMInferenceEngine (onnxruntime-node) is created.
   *
   * **Web use case**: Pass a TimesFMWebInferenceEngine from @agentix-e/timesfm-web
   * to run TimesFM in the browser with onnxruntime-web.
   *
   * **Testing use case**: Inject a mock engine for isolated unit testing.
   *
   * @example
   * ```typescript
   * import { TimesFMWebInferenceEngine } from '@agentix-e/timesfm-web';
   *
   * const webEngine = new TimesFMWebInferenceEngine(modelConfig);
   * await webEngine.load('https://example.com/timesfm-2.5.onnx');
   *
   * const model = await TimesFMModel.fromPretrained({
   *   modelPath: 'https://example.com/timesfm-2.5.onnx',
   *   engine: webEngine,
   * });
   * ```
   */
  engine?: IInferenceEngine;
  /** Custom cache directory for downloaded models. */
  cacheDir?: string;
  /**
   * Proxy configuration for model download in restricted network environments.
   *
   * **IMPORTANT**: This field is forwarded to {@link downloadModel} **only when
   * called explicitly**.  `fromPretrained()` itself does **not** download
   * models — it requires an existing `modelPath`.  If you use the separate
   * `downloadModel()` function to obtain the model before calling
   * `fromPretrained()`, pass your proxy config directly to `downloadModel()`.
   *
   * ```typescript
   * import { downloadModel, TimesFMModel } from '@agentix-e/timesfm-core';
   *
   * const modelPath = await downloadModel({
   *   proxy: { url: 'http://proxy:8080', username: 'user', password: 'pass' },
   * });
   * const model = await TimesFMModel.fromPretrained({ modelPath });
   * ```
   *
   * Priority: this option → TIMESFM_PROXY_URL env var → HTTPS_PROXY env var.
   *
   * For security, prefer passing the password via the TIMESFM_PROXY_PASSWORD
   * environment variable instead of embedding it in this config object.
   * The `password` field is accepted as a convenience for programmatic use
   * but will NOT be logged or serialized.
   */
  proxy?: {
    url: string;
    username?: string;
    password?: string;
  };
  /**
   * When `true`, the warmup inference (triggered during `load()`) is skipped.
   *
   * This is intended for **benchmarking** where the caller wants to measure
   * the true cold-start (first-inference) latency separately. Production
   * callers should leave this at the default (`false`) so that the first
   * user-facing `forecast()` call benefits from JIT-compiled execution plans.
   */
  skipWarmup?: boolean;
}

// ---------------------------------------------------------------------------
// Forecast call options
// ---------------------------------------------------------------------------

/** Progress event emitted during `forecast()`. */
export interface ProgressEvent {
  /** Current processing phase. */
  phase: 'preprocess' | 'prefill' | 'decode' | 'postprocess' | 'flip';
  /** Current step within the phase. */
  step: number;
  /** Total steps in the current phase. */
  total: number;
  /** Current batch index (0-based), if applicable. */
  batchIndex?: number;
  /** Total number of batches. */
  totalBatches?: number;
}

/** Callback for progress events during `forecast()`. */
export type ProgressCallback = (event: ProgressEvent) => void;

/** Optional parameters for `forecast()`. */
export interface ForecastCallOptions {
  /** AbortSignal to cancel a long-running forecast. */
  signal?: AbortSignal;
  /** Progress callback (called at phase boundaries and step increments). */
  onProgress?: ProgressCallback;
  /**
   * Per-call overrides for ForecastConfig fields.
   *
   * These apply only to this single `forecast()` invocation — the stored
   * config on the model instance is never mutated.  Useful for
   * covariate workflows that need `returnBackcast: true` without
   * a global recompile (avoiding race conditions).
   */
  configOverrides?: Partial<ForecastConfig>;
}

// ---------------------------------------------------------------------------
// Forecast output
// ---------------------------------------------------------------------------

/**
 * Result of a `forecast()` call.
 */
export interface ForecastOutput {
  /**
   * Point (median) forecast.
   * Shape: `[numSeries, horizon]` as an array of Float32Arrays.
   */
  pointForecast: Float32Array[];

  /**
   * Quantile forecasts.
   * Shape: `[numSeries, horizon, 10]` as nested arrays.
   * Quantile indices: 0=mean, 1=q10, 2=q20, …, 5=q50, …, 9=q90.
   */
  quantileForecast: Float32Array[][];

  /**
   * Model reconstruction of historical context (backcast).
   * Only populated when `returnBackcast` is true.
   * Shape: `[numSeries, contextLen]` as an array of Float32Arrays.
   */
  backcast?: Float32Array[];
}

// ---------------------------------------------------------------------------
// Public model interface (for testing and dependency injection)
// ---------------------------------------------------------------------------

/**
 * Public interface for TimesFMModel.
 *
 * Enables dependency injection and testing
 * while maintaining a stable contract.
 */
export interface ITimesFMModel {
  compile(fc: ForecastConfig): this;
  forecast(
    horizon: number,
    inputs: Float32Array[],
    options?: ForecastCallOptions,
  ): Promise<ForecastOutput>;
  dispose(): Promise<void>;
  readonly isCompiled: boolean;
  readonly forecastConfig: ForecastConfig | null;
  readonly modelConfig: ModelConfig;
}

// ---------------------------------------------------------------------------
// Inference engine interface (pluggable backend)
// ---------------------------------------------------------------------------

/**
 * Abstract interface for the model inference backend.
 *
 * Implementations:
 *   - `TimesFMInferenceEngine` — ONNX Runtime backend (onnxruntime-node)
 */
export interface IInferenceEngine {
  /** Load the model weights. When `options.skipWarmup` is true, the dummy warmup inference is skipped. */
  load(modelPath: string, options?: { skipWarmup?: boolean }): Promise<void>;

  /**
   * Run a single forward pass through the model.
   *
   * @param inputs  Patched & normed input series [batch, numPatches, inputPatchLen]
   * @param masks   Patch-level mask [batch, numPatches, inputPatchLen]
   * @returns Raw model outputs before any post-processing.
   */
  forward(inputs: Float32Array[], masks: Uint8Array[]): Promise<RawModelOutput>;

  /** Release all resources. */
  dispose(): Promise<void>;

  /** Whether the engine is loaded and ready. */
  isLoaded(): boolean;
}

/**
 * Raw output tensors from a single model forward pass.
 *
 * These correspond to the 4-return from the Python `forward()` method.
 */
export interface RawModelOutput {
  /** Input embeddings [batch, numPatches, modelDims]. */
  inputEmbeddings: Float32Array[];
  /** Output embeddings [batch, numPatches, modelDims]. */
  outputEmbeddings: Float32Array[];
  /** Point output [batch, numPatches, outputPatchLen * numQuantiles]. */
  outputTimeSeries: Float32Array[];
  /** Quantile spread output [batch, numPatches, outputQuantileLen * numQuantiles]. */
  outputQuantileSpread: Float32Array[];
}

// ---------------------------------------------------------------------------
// KV Cache
// ---------------------------------------------------------------------------

/**
 * KV Cache state for one transformer layer.
 *
 * Mirrors the Python `DecodeCache` dataclass in torch/util.py.
 */
export interface KVCache {
  /** Next position index to write into (per batch element). */
  nextIndex: Int32Array;
  /** Cumulative count of masked (padding) patches. */
  numMasked: Int32Array;
  /** Cached key tensors [batch, cacheSize, numHeads, headDim]. */
  key: Float32Array;
  /** Cached value tensors [batch, cacheSize, numHeads, headDim]. */
  value: Float32Array;
}

// ---------------------------------------------------------------------------
// Running statistics
// ---------------------------------------------------------------------------

/**
 * Running (online) statistics for RevIN normalization.
 */
export interface RunningStats {
  /** Count of valid (non-masked) values seen so far. */
  n: number;
  /** Running mean. */
  mu: number;
  /** Running standard deviation. */
  sigma: number;
}

// ---------------------------------------------------------------------------
// XReg (covariate regression)
// ---------------------------------------------------------------------------

/** Category type for categorical covariates. */
export type Category = number | string;

/** The two supported covariate forecast modes. */
export type XRegMode = 'xreg + timesfm' | 'timesfm + xreg';

export interface CovariateForecastParams {
  inputs: Float32Array[];
  dynamicNumericalCovariates?: Record<string, Float32Array[]>;
  dynamicCategoricalCovariates?: Record<string, Category[][]>;
  staticNumericalCovariates?: Record<string, number[]>;
  staticCategoricalCovariates?: Record<string, Category[]>;
  xregMode?: XRegMode;
  normalizeXregTargetPerInput?: boolean;
  ridge?: number;
  maxRowsPerCol?: number;
}

export interface CovariateForecastOutput extends ForecastOutput {
  xregOutputs: Float32Array[];
}

// ---------------------------------------------------------------------------
// Quantile indices (for external consumers)
// ---------------------------------------------------------------------------

export const QUANTILE_INDICES = {
  MEAN: 0,
  Q10: 1,
  Q20: 2,
  Q30: 3,
  Q40: 4,
  Q50: 5,
  Q60: 6,
  Q70: 7,
  Q80: 8,
  Q90: 9,
} as const;

/** The named quantile levels (for display / reference). */
export const QUANTILE_NAMES: Readonly<Record<number, string>> = {
  0: 'mean',
  1: 'q10',
  2: 'q20',
  3: 'q30',
  4: 'q40',
  5: 'q50',
  6: 'q60',
  7: 'q70',
  8: 'q80',
  9: 'q90',
};
