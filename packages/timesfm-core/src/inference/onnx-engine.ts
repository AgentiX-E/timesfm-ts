/**
 * ONNX Runtime inference engine for TimesFM.
 *
 * Production-grade inference engine backed by `onnxruntime-node`.
 * Handles fixed-shape exported models by padding variable-length
 * inputs and running sequential batch elements.
 */

import {
  TIMESFM_25_CONFIG,
  type IInferenceEngine,
  type RawModelOutput,
  type ModelConfig,
  type ModelLoadOptions,
} from '../types';

const PROVIDER_MAP: Record<string, string> = {
  cpu: 'CPUExecutionProvider',
  cuda: 'CUDAExecutionProvider',
  dml: 'DmlExecutionProvider',
};

// ---------------------------------------------------------------------------
// TimesFMInferenceEngine
// ---------------------------------------------------------------------------

export class TimesFMInferenceEngine implements IInferenceEngine {
  // eslint-disable-next-line @typescript-eslint/consistent-type-imports
  private _session: import('onnxruntime-node').InferenceSession | null = null;
  // eslint-disable-next-line @typescript-eslint/consistent-type-imports
  private _ortModule: typeof import('onnxruntime-node') | null = null;
  private _loaded = false;
  private readonly _config: ModelConfig;
  private _executionProvider: string;

  constructor(
    config: ModelConfig = TIMESFM_25_CONFIG,
    options: Pick<ModelLoadOptions, 'executionProvider'> = {},
  ) {
    this._config = config;
    this._executionProvider =
      PROVIDER_MAP[options.executionProvider ?? 'cpu'] ?? 'CPUExecutionProvider';
  }

  async load(modelPath: string): Promise<void> {
    this._ortModule = await import('onnxruntime-node');
    try {
      this._session = await this._ortModule.InferenceSession.create(modelPath, {
        executionProviders: [this._executionProvider],
      });
    } catch (err) {
      console.error(
        `[TimesFM] ${this._executionProvider} failed: ${(err as Error).message}. ` +
          `Falling back to default execution provider.`,
      );
      this._session = await this._ortModule.InferenceSession.create(modelPath);
    }

    // Warmup: run a single dummy inference to trigger JIT compilation.
    // First inference on ONNX Runtime is 2-5× slower due to lazy JIT
    // compilation.  Running a warmup call here eliminates "cold start"
    // variance from the first user-facing forecast() call.
    try {
      await this._warmup();
    } catch (err) {
      // Warmup failure is non-fatal, but log a warning since it may indicate
      // a model compatibility issue that will surface on first forecast().
      console.warn(
        `[TimesFM] Warmup inference failed: ${(err as Error).message}. ` +
          `First forecast() may be slower or fail.`,
      );
    }
    this._loaded = true;
  }

  /**
   * Run a minimal warmup inference to trigger ONNX Runtime's JIT compilation.
   *
   * ONNX Runtime lazily compiles execution graphs on first use.  The first
   * forward pass can be 2-5× slower than subsequent calls.  This warmup
   * absorbs that cost during model loading so user forecasts are consistent.
   *
   * Input and output names are read dynamically from the session metadata
   * to support models with non-standard naming conventions.
   */
  private async _warmup(): Promise<void> {
    if (!this._session || !this._ortModule) return;
    try {
      const ort = this._ortModule;
      const session = this._session;
      const tokenizerLen = this._config.tokenizerInputDims;

      // Read input name dynamically from the session (rather than hardcoding 'inputs')
      const inputName = session.inputNames[0];
      if (!inputName) return;

      // Build a minimal dummy input: 1 batch, exportedPatches patches, all zeros
      const dummyInput = new Float32Array(1 * this._config.exportedPatches * tokenizerLen);
      // eslint-disable-next-line @typescript-eslint/consistent-type-imports
      const feeds: Record<string, import('onnxruntime-node').Tensor> = {
        [inputName]: new ort.Tensor('float32', dummyInput, [
          1,
          this._config.exportedPatches,
          tokenizerLen,
        ]),
      };
      await session.run(feeds);
    } catch {
      // Warmup failure is non-fatal — first real inference will handle it
    }
  }

  isLoaded(): boolean {
    return this._loaded;
  }

  /**
   * Run inference for a batch of variable-length patched time series.
   *
   * The exported ONNX model has a fixed patch count defined by
   * `this._config.exportedPatches` and `this._config.tokenizerInputDims`.
   * We process each batch element sequentially, padding/truncating to exactly the
   * model's expected number of patches.  Padding patches are zero-filled with
   * mask=1 so the transformer's causal mask ignores them.
   *
   * Batch elements are processed concurrently via Promise.all for
   * maximum throughput when perCoreBatchSize > 1.
   */
  async forward(inputs: Float32Array[], masks: Uint8Array[]): Promise<RawModelOutput> {
    if (!this._session || !this._ortModule) {
      throw new Error('ONNX engine not loaded. Call load() first.');
    }

    const ort = this._ortModule;
    const session = this._session;
    const batchSize = inputs.length;
    const inputPatchLen = this._config.inputPatchLen;
    const tokenizerLen = this._config.tokenizerInputDims;

    // Read input/output names dynamically from the session metadata.
    // This supports models exported with non-standard naming conventions
    // without requiring hardcoded string constants.
    const inputName = session.inputNames[0];
    const outputNames = session.outputNames;
    if (!inputName) {
      throw new Error('Model session has no input names defined.');
    }

    // Build output name mapping: preferred canonical name → actual name
    const resolveOutputName = (preferred: string): string => {
      // Try exact match first
      if (outputNames.includes(preferred)) return preferred;
      // Fallback: match by positional index matching canonical order
      const canonicalOrder = ['input_emb', 'output_emb', 'output_ts', 'output_qs'];
      const idx = canonicalOrder.indexOf(preferred);
      if (idx >= 0 && idx < outputNames.length) return outputNames[idx];
      // Last resort: use the name as-is (let ONNX Runtime error if wrong)
      return preferred;
    };

    // Run all batch elements concurrently
    const results = await Promise.all(
      Array.from({ length: batchSize }, async (_, b) => {
        const input = inputs[b];
        const mask = masks[b];
        const numInputPatches = Math.floor(input.length / inputPatchLen);

        // Build padded input to match exported model shape
        const flatInputs = new Float32Array(1 * this._config.exportedPatches * tokenizerLen);
        const copyPatches = Math.min(numInputPatches, this._config.exportedPatches);

        for (let p = 0; p < this._config.exportedPatches; p++) {
          const basePatch = p * tokenizerLen;
          if (p < copyPatches) {
            for (let i = 0; i < inputPatchLen; i++) {
              flatInputs[basePatch + i] = input[p * inputPatchLen + i];
              flatInputs[basePatch + inputPatchLen + i] = mask[p * inputPatchLen + i];
            }
          } else {
            for (let i = 0; i < inputPatchLen; i++) {
              flatInputs[basePatch + i] = 0;
              flatInputs[basePatch + inputPatchLen + i] = 1;
            }
          }
        }

        // eslint-disable-next-line @typescript-eslint/consistent-type-imports
        const feeds: Record<string, import('onnxruntime-node').Tensor> = {
          [inputName]: new ort.Tensor('float32', flatInputs, [
            1,
            this._config.exportedPatches,
            tokenizerLen,
          ]),
        };

        const sessionResults = await session.run(feeds);

        // eslint-disable-next-line @typescript-eslint/consistent-type-imports
        const extract = (t: import('onnxruntime-node').Tensor) =>
          new Float32Array(t.data as Float32Array);

        return {
          inputEmb: extract(sessionResults[resolveOutputName('input_emb')]),
          outputEmb: extract(sessionResults[resolveOutputName('output_emb')]),
          outputTS: extract(sessionResults[resolveOutputName('output_ts')]),
          outputQS: extract(sessionResults[resolveOutputName('output_qs')]),
        };
      }),
    );

    // Reassemble in batch order
    const inputEmbs = results.map((r) => r.inputEmb);
    const outputEmbs = results.map((r) => r.outputEmb);
    const outputTSs = results.map((r) => r.outputTS);
    const outputQSs = results.map((r) => r.outputQS);

    return {
      inputEmbeddings: inputEmbs,
      outputEmbeddings: outputEmbs,
      outputTimeSeries: outputTSs,
      outputQuantileSpread: outputQSs,
    };
  }

  get executionProvider(): string {
    return this._executionProvider;
  }

  async dispose(): Promise<void> {
    if (this._session) {
      try {
        // Release native ONNX Runtime resources.
        // InferenceSession.release() is available in onnxruntime-node ≥ 1.17.
        // Using a runtime typeof guard instead of a brittle type assertion
        // keeps this safe across ONNX Runtime version upgrades.
        const s = this._session as Record<string, unknown>;
        if (typeof s.release === 'function') {
          await (s.release as () => Promise<void>)();
        }
      } catch {
        // Best-effort cleanup
      }
    }
    this._session = null;
    this._ortModule = null;
    this._loaded = false;
  }
}
