/**
 * TimesFM Model — the main public API.
 *
 * Usage:
 *
 * ```typescript
 * import { TimesFMModel } from '@agentix-e/timesfm-core';
 * import { createForecastConfig } from '@agentix-e/timesfm-core';
 *
 * // Always requires a valid ONNX model path
 * const model = await TimesFMModel.fromPretrained({
 *   modelPath: './models/timesfm-2.5.onnx',
 * });
 *
 * model.compile(createForecastConfig({
 *   maxContext: 1024,
 *   maxHorizon: 256,
 * }));
 *
 * const { pointForecast, quantileForecast } = await model.forecast(
 *   24,
 *   [new Float32Array([1, 2, 3, ...])],
 * );
 *
 * await model.dispose();
 * ```
 */

import {
  TIMESFM_25_CONFIG,
  type IInferenceEngine,
  type ForecastConfig,
  type ModelConfig,
  type ModelLoadOptions,
  type ForecastOutput,
  type ForecastCallOptions,
  type ITimesFMModel,
  type CovariateForecastParams,
  type CovariateForecastOutput,
} from './types';
import { ModelNotFoundError, ModelNotCompiledError, HorizonExceededError } from './errors';
import { validateAndNormalizeConfig } from './config';
import { TimesFMInferenceEngine } from './inference/onnx-engine';
import { preprocess } from './preprocessor';
import { decode } from './inference/decode-loop';
import { postProcess } from './postprocessor';
import { resolveModelConfig } from './model-descriptor';
import { computeStats } from './utils/stats';

// Cached dynamic import for optional @agentix-e/timesfm-xreg peer dependency.
//
// Uses a Promise-based singleton pattern to handle concurrent calls safely:
//   • First call initializes the promise (only one import() in flight)
//   • Subsequent calls wait on the same promise
//   • On failure, the promise is reset so the next call retries
//   • After success, _xregModule is set so subsequent calls resolve instantly
//
// eslint-disable-next-line @typescript-eslint/consistent-type-imports
type XRegModule = typeof import('@agentix-e/timesfm-xreg');
let _xregModule: XRegModule | null = null;
let _xregModulePromise: Promise<XRegModule> | null = null;
let _xregImportFn: (spec: string) => Promise<unknown> = (spec: string) => import(spec);

/**
 * Override the import function used to load @agentix-e/timesfm-xreg.
 *
 * **Test-only seam** — NOT part of the public API. Exported only so tests
 * can exercise the dynamic-import failure path without vitest mocking.
 *
 * Reset with `null` to restore default behaviour.
 *
 * @internal
 */
export function __test_setXregImport(fn: ((spec: string) => Promise<unknown>) | null): void {
  _xregImportFn = fn ?? ((spec: string) => import(spec));
  _xregModule = null; // Invalidate cache so next call uses the new fn
  _xregModulePromise = null; // Invalidate pending promise
}
import { allNonNegative } from './utils/tensor-utils';

// ---------------------------------------------------------------------------
// TimesFMModel
// ---------------------------------------------------------------------------

export class TimesFMModel implements ITimesFMModel {
  private _engine: IInferenceEngine;
  private _config: ModelConfig;
  private _forecastConfig: ForecastConfig | null = null;
  private _globalBatchSize: number = 0;
  private _compiled: boolean = false;

  private constructor(engine: IInferenceEngine, config: ModelConfig) {
    this._engine = engine;
    this._config = config;
  }

  // -----------------------------------------------------------------------
  // Factory
  // -----------------------------------------------------------------------

  /**
   * Create a TimesFM model from a pretrained ONNX checkpoint.
   *
   * The model architecture is resolved from the `model-descriptor.json`
   * co-located with the ONNX file.  When no descriptor is found, the
   * engine falls back to the canonical TimesFM 2.5 200M config.
   *
   * @param options.modelPath  Path to the TimesFM ONNX model file. Required.
   * @param options.executionProvider  'cpu' (default), 'cuda', or 'dml'.
   *
   * @throws {Error} if modelPath is not provided or the file doesn't exist.
   */
  static async fromPretrained(options: ModelLoadOptions): Promise<TimesFMModel> {
    if (!options.modelPath) {
      throw new ModelNotFoundError(
        'modelPath is required. Provide the path to a TimesFM ONNX model file.\n' +
          'To obtain a model:\n' +
          '  1. Run: python scripts/export-onnx.py (if you have the TimesFM PyTorch model)\n' +
          '  2. Or download a pre-converted ONNX model\n' +
          'See docs/GETTING-STARTED.md for details.',
      );
    }

    // Resolve architecture from model-descriptor.json (single source of truth).
    // Falls back to TIMESFM_25_CONFIG when no descriptor is present, maintaining
    // backward compatibility with bare .onnx files.
    const { config: mc, descriptor } = await resolveModelConfig(
      options.modelPath,
      TIMESFM_25_CONFIG,
    );

    if (descriptor) {
      // Log model identity for traceability
      // eslint-disable-next-line no-console
      console.log(
        `[TimesFM] Loaded descriptor: v${descriptor.model.version}-${descriptor.model.variant}` +
          ` (schema ${descriptor.schema}, hf_rev ${descriptor.model.hf_revision}), ` +
          `${descriptor.onnx.size_bytes > 0 ? (descriptor.onnx.size_bytes / 1024 ** 2).toFixed(0) + ' MB' : ''}`,
      );
    }

    const engine =
      options.engine ??
      new TimesFMInferenceEngine(mc, {
        executionProvider: options.executionProvider,
        intraOpNumThreads: options.intraOpNumThreads,
      });

    // Only load if the engine is not already loaded (external engine injection)
    if (!engine.isLoaded()) {
      await engine.load(options.modelPath, { skipWarmup: options.skipWarmup });
    }

    return new TimesFMModel(engine, mc);
  }

  // -----------------------------------------------------------------------
  // Compile
  // -----------------------------------------------------------------------

  /**
   * Compile the model with the given forecast configuration.
   *
   * This must be called before `forecast()`.  It validates and normalises
   * the config and sets up batch-size calculations.
   *
   * @returns `this` for method chaining.
   * @throws {RangeError} if context + horizon exceeds the model limit.
   */
  compile(fc: ForecastConfig): this {
    const normalized = validateAndNormalizeConfig(fc, this._config);
    this._forecastConfig = normalized;
    this._globalBatchSize = normalized.perCoreBatchSize;
    this._compiled = true;
    return this;
  }

  get isCompiled(): boolean {
    return this._compiled;
  }

  get forecastConfig(): ForecastConfig | null {
    return this._forecastConfig;
  }

  // -----------------------------------------------------------------------
  // Forecast
  // -----------------------------------------------------------------------

  /**
   * Forecast a batch of univariate time series.
   *
   * @param horizon  Number of future time points to forecast.
   * @param inputs   List of 1-D time series.  Each can have different length.
   *                 May contain NaN values (leading NaNs are stripped,
   *                 internal NaNs are linearly interpolated).
   * @param options  Optional AbortSignal and progress callback.
   *
   * @returns Point forecasts + quantile forecasts.
   *
   * @throws {Error} if the model has not been compiled.
   * @throws {DOMException} AbortError if the operation was cancelled.
   */
  async forecast(
    horizon: number,
    inputs: Float32Array[],
    options?: ForecastCallOptions,
  ): Promise<ForecastOutput> {
    if (!this._compiled || !this._forecastConfig) {
      throw new ModelNotCompiledError('Model not compiled. Call compile() before forecast().');
    }

    // Per-call config overrides (used internally by XReg covariate workflows
    // for returnBackcast without mutating global state).  The stored config
    // on the model instance is never modified.
    const fc: ForecastConfig = options?.configOverrides
      ? { ...this._forecastConfig, ...options.configOverrides }
      : this._forecastConfig;
    const signal = options?.signal;
    const onProgress = options?.onProgress;

    // Check for early abort
    signal?.throwIfAborted();

    if (horizon > fc.maxHorizon) {
      throw new HorizonExceededError(`Horizon ${horizon} exceeds maxHorizon ${fc.maxHorizon}.`);
    }

    // Pad batch to globalBatchSize — synthetic zero-padding series
    // are appended to align the input count to perCoreBatchSize.
    // These are tracked separately so we can skip them in the decode
    // pipeline and trim the output to numInputs, avoiding wasted ONNX
    // inference on dummy data (~12% compute saving at typical batch sizes).
    const paddedInputs = [...inputs];
    const numInputs = inputs.length;
    const remainder = numInputs % this._globalBatchSize;
    let padCount = 0;
    if (remainder !== 0) {
      padCount = this._globalBatchSize - remainder;
      for (let i = 0; i < padCount; i++) {
        paddedInputs.push(new Float32Array([0, 0, 0]));
      }
    }

    // Input z-score normalization
    let normalizedInputs = paddedInputs;
    const inputStats: { mu: number; sigma: number }[] = [];
    /** Per-series flag: true if raw (pre-normalized) input is all ≥ 0. */
    const isPositiveFlags: boolean[] = [];

    if (fc.normalizeInputs) {
      normalizedInputs = paddedInputs.map((arr) => {
        const { mean: mu, std: sigma } = computeStats(arr);
        const safeSigma = sigma < 1e-6 ? 1 : sigma;

        const isNonNeg = allNonNegative(arr);
        inputStats.push({ mu, sigma: safeSigma });
        isPositiveFlags.push(isNonNeg);

        const result = new Float32Array(arr.length);
        for (let i = 0; i < arr.length; i++) {
          result[i] = Number.isFinite(arr[i]) ? (arr[i] - mu) / safeSigma : 0;
        }
        return result;
      });
    } else {
      // When not normalizing, still compute positivity flags
      for (const arr of paddedInputs) {
        inputStats.push({ mu: 0, sigma: 1 });
        isPositiveFlags.push(allNonNegative(arr));
      }
    }

    const allPointForecasts: Float32Array[] = [];
    const allQuantileForecasts: Float32Array[][] = [];
    const allBackcasts: Float32Array[] = [];

    const batchSize = this._globalBatchSize;
    const numBatches = Math.ceil(normalizedInputs.length / batchSize);

    for (let bi = 0; bi < numBatches; bi++) {
      // Abort check at batch boundary
      signal?.throwIfAborted();

      const batchStart = bi * batchSize;
      const batchInputs = normalizedInputs.slice(batchStart, batchStart + batchSize);

      onProgress?.({
        phase: 'preprocess',
        step: bi + 1,
        total: numBatches,
        batchIndex: bi,
        totalBatches: numBatches,
      });

      const preprocessed = preprocess(batchInputs, fc, this._config);

      onProgress?.({
        phase: 'prefill',
        step: bi + 1,
        total: numBatches,
        batchIndex: bi,
        totalBatches: numBatches,
      });

      // ── Decode (main path, and optionally flip path in parallel) ──────

      let decodeResult: Awaited<ReturnType<typeof decode>>;
      let flipResult: Awaited<ReturnType<typeof decode>> | null = null;

      if (fc.forceFlipInvariance) {
        // Pre-build the negated flip inputs while the main decode runs.
        // Then launch both decode() calls concurrently via Promise.all
        // so the flip path doesn't add serial latency.  ONNX Runtime
        // sessions are safe for concurrent calls in the Node.js event loop.
        const negInputs = batchInputs.map((arr) => {
          const neg = new Float32Array(arr.length);
          for (let i = 0; i < arr.length; i++) neg[i] = -arr[i];
          return neg;
        });
        const flipPre = preprocess(negInputs, fc, this._config);

        signal?.throwIfAborted();

        onProgress?.({
          phase: 'flip',
          step: bi + 1,
          total: numBatches,
          batchIndex: bi,
          totalBatches: numBatches,
        });

        // Parallel: main + flip decode run concurrently
        [decodeResult, flipResult] = await Promise.all([
          decode(
            this._engine,
            preprocessed.patchedInputs,
            preprocessed.patchedMasks,
            preprocessed.contextMu,
            preprocessed.contextSigma,
            preprocessed.lastStats,
            fc.maxHorizon,
            fc,
            this._config,
            signal,
          ),
          decode(
            this._engine,
            flipPre.patchedInputs,
            flipPre.patchedMasks,
            flipPre.contextMu,
            flipPre.contextSigma,
            flipPre.lastStats,
            fc.maxHorizon,
            fc,
            this._config,
            signal,
          ),
        ]);
      } else {
        decodeResult = await decode(
          this._engine,
          preprocessed.patchedInputs,
          preprocessed.patchedMasks,
          preprocessed.contextMu,
          preprocessed.contextSigma,
          preprocessed.lastStats,
          fc.maxHorizon,
          fc,
          this._config,
          signal,
        );
      }

      onProgress?.({
        phase: 'postprocess',
        step: bi + 1,
        total: numBatches,
        batchIndex: bi,
        totalBatches: numBatches,
      });

      const batchInputStats = fc.normalizeInputs
        ? inputStats.slice(batchStart, batchStart + batchSize)
        : null;
      const batchIsPositive = isPositiveFlags.slice(batchStart, batchStart + batchSize);

      const output = postProcess(
        decodeResult,
        horizon,
        fc,
        this._config,
        batchInputStats,
        flipResult,
        batchIsPositive,
      );

      allPointForecasts.push(...output.pointForecast);
      allQuantileForecasts.push(...output.quantileForecast);
      if (output.backcast) {
        allBackcasts.push(...output.backcast);
      }
    }

    return {
      pointForecast: allPointForecasts.slice(0, numInputs),
      quantileForecast: allQuantileForecasts.slice(0, numInputs),
      backcast: allBackcasts.length > 0 ? allBackcasts.slice(0, numInputs) : undefined,
    };
  }

  /**
   * Forecast with exogenous covariates via {@link @agentix-e/timesfm-xreg}.
   *
   * Internally caches the dynamic import of the optional peer dependency
   * so that repeated calls avoid re-resolution overhead.
   *
   * Concurrent safety: uses a Promise-based singleton pattern so that
   * multiple concurrent first-time calls share a single import() and
   * import failures are retried on subsequent calls.
   */
  async forecastWithCovariates(params: CovariateForecastParams): Promise<CovariateForecastOutput> {
    // Fast path: module is already loaded (after first successful import).
    if (_xregModule) {
      return await _xregModule.forecastWithCovariates(this, params);
    }

    // Init path: start (or join) a single inflight import promise.
    if (!_xregModulePromise) {
      _xregModulePromise = _xregImportFn('@agentix-e/timesfm-xreg')
        .then((mod) => {
          _xregModule = mod as XRegModule;
          return _xregModule;
        })
        .catch((err) => {
          // Reset promise so the next call retries the import.
          _xregModulePromise = null;
          throw err;
        });
    }

    try {
      const mod = await _xregModulePromise;
      return await mod.forecastWithCovariates(this, params);
    } catch (err) {
      throw new Error(
        'forecastWithCovariates requires @agentix-e/timesfm-xreg.\n' +
          'Install it: npm install @agentix-e/timesfm-xreg\n\n' +
          `Original error: ${(err as Error).message}`,
      );
    }
  }

  async dispose(): Promise<void> {
    await this._engine.dispose();
    this._compiled = false;
    this._forecastConfig = null;
  }

  get engine(): IInferenceEngine {
    return this._engine;
  }
  get modelConfig(): ModelConfig {
    return this._config;
  }
  get globalBatchSize(): number {
    return this._globalBatchSize;
  }
}
