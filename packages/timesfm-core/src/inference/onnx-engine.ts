/**
 * ONNX Runtime inference engine for TimesFM.
 *
 * Production-grade inference engine backed by `onnxruntime-node`.
 * Handles fixed-shape exported models (batch=1, patches=16) by
 * padding variable-length inputs and running sequential batch elements.
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

/** The fixed shape of the exported TimesFM 2.5 ONNX model. */
const EXPORTED_PATCHES = 16;

// ---------------------------------------------------------------------------
// TimesFMInferenceEngine
// ---------------------------------------------------------------------------

export class TimesFMInferenceEngine implements IInferenceEngine {
  // eslint-disable-next-line @typescript-eslint/consistent-type-imports
  private _session: import('onnxruntime-node').InferenceSession | null = null;
  // eslint-disable-next-line @typescript-eslint/consistent-type-imports
  private _ortModule: typeof import('onnxruntime-node') | null = null;
  private _loaded = false;
  private _config: ModelConfig;
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
    this._loaded = true;
  }

  isLoaded(): boolean {
    return this._loaded;
  }

  /**
   * Run inference for a batch of variable-length patched time series.
   *
   * The exported ONNX model has fixed shape [1, 16, 64].  We process
   * each batch element sequentially, padding/truncating to exactly 16
   * patches.  Padding patches are zero-filled with mask=1 so the
   * transformer's causal mask ignores them.
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
    const tokenizerLen = inputPatchLen * 2; // 64

    // Run all batch elements concurrently
    const results = await Promise.all(
      Array.from({ length: batchSize }, async (_, b) => {
        const input = inputs[b];
        const mask = masks[b];
        const numInputPatches = Math.floor(input.length / inputPatchLen);

        // Build padded input: always [1, 16, 64]
        const flatInputs = new Float32Array(1 * EXPORTED_PATCHES * tokenizerLen);
        const copyPatches = Math.min(numInputPatches, EXPORTED_PATCHES);

        for (let p = 0; p < EXPORTED_PATCHES; p++) {
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
          inputs: new ort.Tensor('float32', flatInputs, [1, EXPORTED_PATCHES, tokenizerLen]),
        };

        const sessionResults = await session.run(feeds);

        // eslint-disable-next-line @typescript-eslint/consistent-type-imports
        const extract = (t: import('onnxruntime-node').Tensor) =>
          new Float32Array(t.data as Float32Array);

        return {
          inputEmb: extract(sessionResults['input_emb']),
          outputEmb: extract(sessionResults['output_emb']),
          outputTS: extract(sessionResults['output_ts']),
          outputQS: extract(sessionResults['output_qs']),
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
        // Release native ONNX Runtime resources
        await (this._session as { release?: () => Promise<void> }).release?.();
      } catch {
        // Best-effort cleanup
      }
    }
    this._session = null;
    this._ortModule = null;
    this._loaded = false;
  }
}
