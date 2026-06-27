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

  /**
   * @param config          Model architecture configuration.
   * @param executionProviders  Execution providers to try, in order.
   *                            Default: `['webgpu', 'wasm']`
   */
  constructor(
    config: ModelConfig,
    executionProviders: Array<'webgpu' | 'wasm' | 'webgl'> = ['webgpu', 'wasm'],
  ) {
    this._config = config;
    this._providers = executionProviders;
  }

  // -----------------------------------------------------------------------
  // IInferenceEngine — load
  // -----------------------------------------------------------------------

  /**
   * Load the ONNX model.
   *
   * @param modelPath  URL to the ONNX model file, **or** an ArrayBuffer.
   */
  async load(modelPath: string | ArrayBuffer): Promise<void> {
    const ort = await import('onnxruntime-web');
    this._ortModule = ort;

    // Configure WASM path — onnxruntime-web needs to locate the WASM binary.
    // By default it looks for ort-wasm*.wasm relative to the current page.
    // Users can override via `ort.env.wasm.wasmPaths` before calling load().
    // We set a sensible default if not already configured.
    if (!ort.env.wasm.wasmPaths) {
      // Set the base path for WASM files — onnxruntime-web will append filenames
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
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

        // eslint-disable-next-line no-console
        console.log(`[TimesFM Web] Loaded model with ${provider} provider`);
        this._loaded = true;
        return;
      } catch (err) {
        console.warn(
          `[TimesFM Web] ${provider} provider failed: ${(err as Error).message}. Trying next...`,
        );
        lastError = err as Error;
        // Continue to next provider
      }
    }

    throw new Error(
      `[TimesFM Web] All execution providers failed. Last error: ${lastError?.message}`,
    );
  }

  // -----------------------------------------------------------------------
  // IInferenceEngine — forward
  // -----------------------------------------------------------------------

  async forward(inputs: Float32Array[], masks: Uint8Array[]): Promise<RawModelOutput> {
    if (!this._session || !this._ortModule) {
      throw new Error('[TimesFM Web] Engine not loaded. Call load() first.');
    }

    const ort = this._ortModule;
    const batchSize = inputs.length;
    const patchesPerSeries = inputs[0].length / this._config.tokenizerInputDims;

    // Build the combined input tensor: [batchSize, patches, tokenizerInputDims]
    const combined = new Float32Array(
      batchSize * patchesPerSeries * this._config.tokenizerInputDims,
    );
    const maskTensor = new Float32Array(batchSize * patchesPerSeries);
    for (let b = 0; b < batchSize; b++) {
      const offset = b * patchesPerSeries * this._config.tokenizerInputDims;
      combined.set(inputs[b], offset);
      // Convert Uint8Array mask to Float32Array (0.0 = visible, 1.0 = masked)
      for (let p = 0; p < patchesPerSeries; p++) {
        maskTensor[b * patchesPerSeries + p] = masks[b]?.[p] ?? 0;
      }
    }

    // eslint-disable-next-line @typescript-eslint/consistent-type-imports
    const feeds: Record<string, import('onnxruntime-web').Tensor> = {
      inputs: new ort.Tensor('float32', combined, [
        batchSize,
        patchesPerSeries,
        this._config.tokenizerInputDims,
      ]),
      patched_mask: new ort.Tensor('float32', maskTensor, [batchSize, patchesPerSeries]),
    };

    const results = await this._session.run(feeds);

    // Extract raw model outputs (field names match ONNX export)
    const ie = results['input_embedding']?.data as Float32Array;
    const oe = results['output_embedding']?.data as Float32Array;
    const ts = results['output_time_series']?.data as Float32Array;
    const qs = results['output_quantile_spread']?.data as Float32Array;

    return {
      inputEmbeddings: [ie ?? new Float32Array(0)],
      outputEmbeddings: [oe ?? new Float32Array(0)],
      outputTimeSeries: [ts ?? new Float32Array(0)],
      outputQuantileSpread: [qs ?? new Float32Array(0)],
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
