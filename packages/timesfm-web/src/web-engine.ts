/**
 * TimesFM Web Inference Engine — browser-compatible ONNX inference via
 * onnxruntime-web (WASM / WebGPU / WebGL).
 *
 * Implements the `IInferenceEngine` interface from @agentix-e/timesfm-core
 * so it can be injected into `TimesFMModel.fromPretrained()`.
 *
 * ## Execution Providers
 *
 * | Provider  | Speed   | Availability        |
 * |-----------|---------|---------------------|
 * | `webgpu`  | Fastest | Chrome 113+, Edge   |
 * | `wasm`    | Good    | All modern browsers |
 * | `webgl`   | Legacy  | Older browsers      |
 *
 * The engine tries providers in order: webgpu → wasm → webgl,
 * falling back to the next available one on failure.
 *
 * ## Model Loading
 *
 * `load()` accepts:
 * - A URL string (fetched via `fetch()`)
 * - An `ArrayBuffer` (pre-loaded model data)
 *
 * ## Usage
 *
 * ```typescript
 * import { TimesFMModel, createForecastConfig } from '@agentix-e/timesfm-core';
 * import { TimesFMWebInferenceEngine } from '@agentix-e/timesfm-web';
 *
 * const engine = new TimesFMWebInferenceEngine(config);
 * await engine.load('/models/timesfm-2.5.onnx');
 *
 * const model = await TimesFMModel.fromPretrained({
 *   modelPath: '/models/timesfm-2.5.onnx',
 *   engine,
 * });
 * model.compile(createForecastConfig({ maxContext: 512, maxHorizon: 128 }));
 * const result = await model.forecast(24, [inputData]);
 * ```
 *
 * @module web-engine
 */

import type { ModelConfig, IInferenceEngine, RawModelOutput } from '@agentix-e/timesfm-core';

// ---------------------------------------------------------------------------
// Logger interface
// ---------------------------------------------------------------------------

/**
 * Structured logger for Web engine diagnostics.
 *
 * Inject a custom implementation to capture provider selection events
 * for observability pipelines (OpenTelemetry, Datadog, etc.).
 */
export interface WebEngineLogger {
  /** Provider load attempt started. */
  info(msg: string, ctx?: Record<string, unknown>): void;
  /** Provider load attempt failed — will try next. */
  warn(msg: string, ctx?: Record<string, unknown>): void;
  /** All providers failed. */
  error(msg: string, ctx?: Record<string, unknown>): void;
}

const defaultLogger: WebEngineLogger = {
  info(msg) {
    // Default logger writes diagnostic messages to console.log
    // (ESLint allows console.error/warn; we use log for informational output).
    // eslint-disable-next-line no-console
    console.log(msg);
  },
  warn(msg) {
    console.warn(msg);
  },
  error(msg) {
    console.error(msg);
  },
};

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/**
 * Browser-compatible inference engine backed by onnxruntime-web.
 *
 * Supports WASM, WebGPU, and WebGL backends with automatic fallback.
 * Accepts model URLs (fetched) or pre-loaded ArrayBuffers.
 */
export class TimesFMWebInferenceEngine implements IInferenceEngine {
  private _config: ModelConfig;
  // eslint-disable-next-line @typescript-eslint/consistent-type-imports
  private _ortModule: typeof import('onnxruntime-web') | null = null;
  // eslint-disable-next-line @typescript-eslint/consistent-type-imports
  private _session: import('onnxruntime-web').InferenceSession | null = null;
  private _loaded = false;

  /** Preferred execution providers in fallback order. */
  private readonly _providers: Array<'webgpu' | 'wasm' | 'webgl'>;

  /** Custom WASM path for onnxruntime-web (used in Node.js testing). */
  private _wasmPath: string | null = null;

  /** Custom CDN version for the browser WASM fallback. */
  private _cdnVersion: string;

  /** Optional structured logger for diagnostics. When omitted, falls back to `console`. */
  private _logger: WebEngineLogger;

  /**
   * @param config          Model architecture configuration.
   * @param executionProviders  Execution providers to try, in order.
   *                            Default: `['webgpu', 'wasm']`
   * @param cdnVersion      onnxruntime-web CDN version for browser WASM fallback.
   *                        Default: matches the package's peerDependency version.
   * @param logger          Optional structured logger for diagnostics.
   *                        When omitted, provider selection messages are silenced
   *                        (only the final loaded provider is logged).
   */
  constructor(
    config: ModelConfig,
    executionProviders: Array<'webgpu' | 'wasm' | 'webgl'> = ['webgpu', 'wasm'],
    cdnVersion: string = '1.22.0',
    logger?: WebEngineLogger,
  ) {
    this._config = config;
    this._providers = executionProviders;
    this._cdnVersion = cdnVersion;
    this._logger = logger ?? defaultLogger;
  }

  /**
   * Set a custom WASM binary path for onnxruntime-web.
   *
   * Required when running in Node.js (not a browser) since the default
   * CDN URL won't work. Point this to the `dist/` directory of the
   * onnxruntime-web package.
   *
   * @example
   * ```typescript
   * // In Node.js testing:
   * import { createRequire } from 'node:module';
   * const require = createRequire(import.meta.url);
   * const wasmDir = require.resolve('onnxruntime-web').replace('/lib/index.js', '/dist/');
   * engine.setWasmPath(wasmDir);
   * ```
   */
  setWasmPath(wasmPath: string): void {
    this._wasmPath = wasmPath;
  }

  // -----------------------------------------------------------------------
  // IInferenceEngine — load
  // -----------------------------------------------------------------------

  /**
   * Load the ONNX model.
   *
   * @param modelPath  URL to the ONNX model file, **or** an ArrayBuffer.
   */
  async load(modelPath: string | ArrayBuffer, _options?: { skipWarmup?: boolean }): Promise<void> {
    const ort = await import('onnxruntime-web');
    this._ortModule = ort;

    // Configure WASM path — onnxruntime-web needs to locate the WASM binary.
    // Priority: 1) custom path set via setWasmPath()
    //           2) auto-detect from onnxruntime-web package (Node.js)
    //           3) jsdelivr CDN (browser default)
    if (this._wasmPath) {
      // Ensure trailing slash — onnxruntime-web concatenates filenames directly
      ort.env.wasm.wasmPaths = this._wasmPath.endsWith('/') ? this._wasmPath : this._wasmPath + '/';
    } else if (!ort.env.wasm.wasmPaths) {
      // Try Node.js detection: resolve onnxruntime-web from node_modules
      try {
        const { createRequire } = await import('node:module');
        const req = createRequire(import.meta.url);
        const pkgDir = req.resolve('onnxruntime-web');
        // onnxruntime-web's main entry is lib/index.js or dist/ort.node.min.js
        // WASM files are in dist/. Ensure trailing slash.
        let distDir = pkgDir.replace(/\/lib\/.+$/, '/dist/');
        if (!distDir.endsWith('/')) distDir += '/';
        ort.env.wasm.wasmPaths = distDir;
      } catch {
        // Browser fallback: use jsdelivr CDN with the configured version
        ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${this._cdnVersion}/dist/`;
      }
    }

    // Disable multi-threading in browser (not supported in all contexts)
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;

    // Try each provider in order until one succeeds
    let lastError: Error | null = null;

    for (const provider of this._providers) {
      try {
        // eslint-disable-next-line @typescript-eslint/consistent-type-imports
        const sessionOptions: import('onnxruntime-web').InferenceSession.SessionOptions = {
          executionProviders: [provider],
          graphOptimizationLevel: 'all',
          enableCpuMemArena: true,
          enableMemPattern: true,
        };

        // Accept both URL string and ArrayBuffer
        if (typeof modelPath === 'string') {
          this._session = await ort.InferenceSession.create(modelPath, sessionOptions);
        } else {
          this._session = await ort.InferenceSession.create(modelPath, sessionOptions);
        }

        this._logger.info(`[TimesFM Web] Loaded model with ${provider} provider`, { provider });
        this._loaded = true;
        return;
      } catch (err) {
        this._logger.warn(`[TimesFM Web] ${provider} provider failed — trying next`, {
          provider,
          error: (err as Error).message,
        });
        lastError = err as Error;
        // Continue to next provider
      }
    }

    const errorMsg = `[TimesFM Web] All execution providers failed. Last error: ${lastError?.message}`;
    this._logger.error(errorMsg, {
      providers: this._providers,
      lastError: lastError?.message,
    });
    throw new Error(errorMsg);
  }

  // -----------------------------------------------------------------------
  // IInferenceEngine — forward
  // -----------------------------------------------------------------------

  /**
   * Run inference for a batch of patched time series.
   *
   * Aligned with {@link TimesFMInferenceEngine.forward}: each batch
   * element is processed sequentially via the ONNX session and results
   * are returned as per-element arrays in the {@link RawModelOutput}.
   *
   * Builds the tokenizer input identically to the Node.js engine:
   * each 64-dim patch is [patch_values(32), patch_mask(32)] where
   * mask=0 means "visible" and mask=1 means "padding/ignored".
   */
  async forward(inputs: Float32Array[], masks: Uint8Array[]): Promise<RawModelOutput> {
    if (!this._session || !this._ortModule) {
      throw new Error('[TimesFM Web] Engine not loaded. Call load() first.');
    }

    const ort = this._ortModule;
    const session = this._session;
    const batchSize = inputs.length;
    const inputPatchLen = this._config.inputPatchLen;
    const tokenizerLen = this._config.tokenizerInputDims; // 64 = 32 values + 32 mask

    // Build output name mapping: preferred canonical name → actual name.
    // This matches the ONNX engine's dynamic output name resolution,
    // supporting models exported with non-standard naming conventions.
    const resolveOutputName = (preferred: string): string => {
      if (session.outputNames.includes(preferred)) return preferred;
      const canonicalOrder = ['input_emb', 'output_emb', 'output_ts', 'output_qs'];
      const idx = canonicalOrder.indexOf(preferred);
      if (idx >= 0 && idx < session.outputNames.length) return session.outputNames[idx];
      return preferred;
    };

    const inputName = session.inputNames[0] || 'inputs';

    // Run each batch element sequentially through the ONNX session.
    // onnxruntime-web's WASM backend is single-threaded, so Promise.all
    // wouldn't provide parallelism — sequential is both correct and optimal.
    const inputEmbs: Float32Array[] = [];
    const outputEmbs: Float32Array[] = [];
    const outputTSs: Float32Array[] = [];
    const outputQSs: Float32Array[] = [];

    for (let b = 0; b < batchSize; b++) {
      const input = inputs[b];
      const mask = masks[b];
      const numInputPatches = Math.floor(input.length / inputPatchLen);

      // Build padded input to match exported model shape, interleaving
      // values and masks per patch (identical to Node.js ONNX engine).
      const totalLen = this._config.exportedPatches * tokenizerLen;
      const flatInputs = new Float32Array(totalLen);
      const copyPatches = Math.min(numInputPatches, this._config.exportedPatches);

      for (let p = 0; p < this._config.exportedPatches; p++) {
        const basePatch = p * tokenizerLen;
        if (p < copyPatches) {
          for (let i = 0; i < inputPatchLen; i++) {
            flatInputs[basePatch + i] = input[p * inputPatchLen + i];
            flatInputs[basePatch + inputPatchLen + i] = mask[p * inputPatchLen + i];
          }
        } else {
          // Padding patch: values=0, mask=1 (ignored)
          for (let i = 0; i < inputPatchLen; i++) {
            flatInputs[basePatch + i] = 0;
            flatInputs[basePatch + inputPatchLen + i] = 1;
          }
        }
      }

      // eslint-disable-next-line @typescript-eslint/consistent-type-imports
      const feeds: Record<string, import('onnxruntime-web').Tensor> = {
        [inputName]: new ort.Tensor('float32', flatInputs, [
          1,
          this._config.exportedPatches,
          tokenizerLen,
        ]),
      };

      const results = await session.run(feeds);

      // eslint-disable-next-line @typescript-eslint/consistent-type-imports
      const extract = (t: import('onnxruntime-web').Tensor | undefined): Float32Array => {
        if (!t || !t.data) return new Float32Array(0);
        return new Float32Array(t.data as Float32Array);
      };

      inputEmbs.push(extract(results[resolveOutputName('input_emb')]));
      outputEmbs.push(extract(results[resolveOutputName('output_emb')]));
      outputTSs.push(extract(results[resolveOutputName('output_ts')]));
      outputQSs.push(extract(results[resolveOutputName('output_qs')]));
    }

    return {
      inputEmbeddings: inputEmbs,
      outputEmbeddings: outputEmbs,
      outputTimeSeries: outputTSs,
      outputQuantileSpread: outputQSs,
    };
  }

  // -----------------------------------------------------------------------
  // IInferenceEngine — dispose
  // -----------------------------------------------------------------------

  async dispose(): Promise<void> {
    if (this._session) {
      try {
        // onnxruntime-web's release() is available but may throw in some contexts
        if (typeof this._session.release === 'function') {
          this._session.release();
        }
      } catch {
        // GC will handle WASM cleanup eventually
      }
      this._session = null;
    }
    this._ortModule = null;
    this._loaded = false;
  }

  // -----------------------------------------------------------------------
  // IInferenceEngine — status
  // -----------------------------------------------------------------------

  isLoaded(): boolean {
    return this._loaded;
  }
}
