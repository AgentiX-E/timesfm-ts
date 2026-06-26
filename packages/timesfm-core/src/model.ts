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
import { validateAndNormalizeConfig } from './config';
import { TimesFMInferenceEngine } from './inference/onnx-engine';
import { preprocess } from './preprocessor';
import { decode } from './inference/decode-loop';
import { postProcess } from './postprocessor';
import { computeStats } from './utils/stats';
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
   * @param options.modelPath  Path to the TimesFM ONNX model file. Required.
   * @param options.executionProvider  'cpu' (default), 'cuda', or 'dml'.
   *
   * @throws {Error} if modelPath is not provided or the file doesn't exist.
   */
  static async fromPretrained(options: ModelLoadOptions): Promise<TimesFMModel> {
    if (!options.modelPath) {
      throw new Error(
        'modelPath is required. Provide the path to a TimesFM ONNX model file.\n' +
          'To obtain a model:\n' +
          '  1. Run: python scripts/export-onnx.py (if you have the TimesFM PyTorch model)\n' +
          '  2. Or download a pre-converted ONNX model\n' +
          'See docs/GETTING-STARTED.md for details.',
      );
    }

    const mc = TIMESFM_25_CONFIG;
    const engine = new TimesFMInferenceEngine(mc, {
      executionProvider: options.executionProvider,
    });
    await engine.load(options.modelPath);

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
      throw new Error('Model not compiled. Call compile() before forecast().');
    }

    const fc = this._forecastConfig;
    const signal = options?.signal;
    const onProgress = options?.onProgress;

    // Check for early abort
    signal?.throwIfAborted();

    if (horizon > fc.maxHorizon) {
      throw new Error(`Horizon ${horizon} exceeds maxHorizon ${fc.maxHorizon}.`);
    }

    // Pad batch to globalBatchSize
    const paddedInputs = [...inputs];
    const numInputs = inputs.length;
    const remainder = numInputs % this._globalBatchSize;
    if (remainder !== 0) {
      const padCount = this._globalBatchSize - remainder;
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

      const decodeResult = await decode(
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

      let flipResult = null;
      if (fc.forceFlipInvariance) {
        // Abort check before flip path
        signal?.throwIfAborted();

        onProgress?.({
          phase: 'flip',
          step: bi + 1,
          total: numBatches,
          batchIndex: bi,
          totalBatches: numBatches,
        });
        const negInputs = batchInputs.map((arr) => {
          const neg = new Float32Array(arr.length);
          for (let i = 0; i < arr.length; i++) neg[i] = -arr[i];
          return neg;
        });
        const flipPre = preprocess(negInputs, fc, this._config);
        flipResult = await decode(
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

  async forecastWithCovariates(params: CovariateForecastParams): Promise<CovariateForecastOutput> {
    // Dynamic import: timesfm-xreg is an optional dependency
    try {
      // Dynamic import: @agentix-e/timesfm-xreg is an optional dependency.
      // @ts-ignore — optional peer dependency, type-checked at install time
      const mod = await import('@agentix-e/timesfm-xreg');
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
