/**
 * Mock Inference Engine for unit testing decode-loop without the 885 MB ONNX model.
 *
 * Implements IInferenceEngine with deterministic, configurable outputs.
 * Used by decode-loop.test.ts and other tests that need a fake engine.
 */

import { TIMESFM_25_CONFIG } from '@agentix-e/timesfm-core';
import type { IInferenceEngine, RawModelOutput, ModelConfig } from '@agentix-e/timesfm-core';

export interface MockEngineOptions {
  /** Output scale factor (default: 1.0). Higher values = larger outputs. */
  scale?: number;
  /** Number of forward calls to track. */
  callCount?: { value: number };
  /** Force a specific outputTimeSeries shape per batch element. */
  outputShape?: { patches: number; perPatch: number };
}

/**
 * A deterministic, configurable mock of IInferenceEngine.
 *
 * Each forward() call returns outputs filled with `scale * (b + 1)` for
 * batch element b, ensuring deterministic values that vary per batch element.
 */
export class MockInferenceEngine implements IInferenceEngine {
  private _loaded = false;
  private _scale: number;
  private _callCount: { value: number };
  private _outputShape: { patches: number; perPatch: number };
  private readonly _mc: ModelConfig;

  constructor(options: MockEngineOptions = {}, config: ModelConfig = TIMESFM_25_CONFIG) {
    const mc = config;
    this._mc = mc;
    this._scale = options.scale ?? 1.0;
    this._callCount = options.callCount ?? { value: 0 };
    const patches = options.outputShape?.patches ?? mc.exportedPatches;
    const perPatch = options.outputShape?.perPatch ?? mc.outputPatchLen * mc.numQuantiles;
    this._outputShape = { patches, perPatch };
  }

  async load(_modelPath: string): Promise<void> {
    this._loaded = true;
  }

  isLoaded(): boolean {
    return this._loaded;
  }

  get callCount(): number {
    return this._callCount.value;
  }

  /**
   * Returns deterministic outputs.
   *
   * outputTimeSeries: shape [batchSize, patches * perPatch]
   *   Filled with scale * (b + 1) for batch element b.
   * outputQuantileSpread: shape [batchSize, this._mc.outputQuantileLen * this._mc.numQuantiles]
   *   Filled with scale * (b + 1) * 0.5.
   */
  async forward(inputs: Float32Array[], _masks: Uint8Array[]): Promise<RawModelOutput> {
    this._callCount.value++;

    const batchSize = inputs.length;
    const { patches, perPatch } = this._outputShape;
    const tsl = patches * perPatch;
    const qsLen = this._mc.outputQuantileLen * this._mc.numQuantiles;

    const outputTimeSeries: Float32Array[] = [];
    const outputQuantileSpread: Float32Array[] = [];
    const inputEmbeddings: Float32Array[] = [];
    const outputEmbeddings: Float32Array[] = [];

    for (let b = 0; b < batchSize; b++) {
      const val = this._scale * (b + 1);

      const ts = new Float32Array(tsl);
      ts.fill(val);
      outputTimeSeries.push(ts);

      const qs = new Float32Array(qsLen);
      qs.fill(val * 0.5);
      outputQuantileSpread.push(qs);

      const emb = new Float32Array(patches * this._mc.modelDims);
      emb.fill(val);
      inputEmbeddings.push(emb);
      outputEmbeddings.push(emb);
    }

    return {
      inputEmbeddings,
      outputEmbeddings,
      outputTimeSeries,
      outputQuantileSpread,
    };
  }

  async dispose(): Promise<void> {
    this._loaded = false;
  }
}
