/**
 * Unit tests for ModelDescriptor — parsing, validation, and
 * descriptorToModelConfig conversion.
 *
 * No ONNX model required. Pure data-driven tests.
 */
import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import {
  descriptorToModelConfig,
  ENGINE_SUPPORTED_SCHEMA,
  loadModelDescriptor,
  resolveModelConfig,
  type ModelDescriptor,
} from '../../src/model-descriptor';
import { createTimesFM25Config, TIMESFM_25_CONFIG, type ModelConfig } from '../../src/types';

// ---------------------------------------------------------------------------
// Canonical TimesFM 2.5 descriptor (realistic)
// ---------------------------------------------------------------------------

const CANONICAL_DESCRIPTOR: ModelDescriptor = {
  schema: 1,
  model: {
    version: '2.5',
    variant: '200m',
    hf_revision: 'abc123def456',
    exported_at: '2026-06-26T12:00:00Z',
  },
  onnx: {
    input_name: 'inputs',
    input_shape: [1, 16, 64],
    outputs: {
      input_emb: [1, 16, 1280],
      output_emb: [1, 16, 1280],
      output_ts: [1, 16, 1280],
      output_qs: [1, 16, 10240],
    },
    opset: 18,
    sha256: 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    size_bytes: 928514048,
  },
  architecture: {
    input_patch_len: 32,
    output_patch_len: 128,
    output_quantile_len: 1024,
    num_layers: 20,
    num_heads: 16,
    model_dims: 1280,
    quantiles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    context_limit: 16384,
  },
  processing: {
    preprocessing: 'revin',
    postprocessing: ['flip_invariance', 'quantile_crossing_fix'],
  },
};

// ---------------------------------------------------------------------------
// descriptorToModelConfig
// ---------------------------------------------------------------------------

describe('descriptorToModelConfig', () => {
  it('produces a ModelConfig equivalent to createTimesFM25Config()', () => {
    const fromDescriptor = descriptorToModelConfig(CANONICAL_DESCRIPTOR);
    const canonical = createTimesFM25Config();

    // Every field must match
    expect(fromDescriptor.contextLimit).toBe(canonical.contextLimit);
    expect(fromDescriptor.exportedPatches).toBe(canonical.exportedPatches);
    expect(fromDescriptor.inputPatchLen).toBe(canonical.inputPatchLen);
    expect(fromDescriptor.outputPatchLen).toBe(canonical.outputPatchLen);
    expect(fromDescriptor.outputQuantileLen).toBe(canonical.outputQuantileLen);
    expect(fromDescriptor.outputPatchesPerInput).toBe(canonical.outputPatchesPerInput);
    expect(fromDescriptor.quantiles).toEqual(canonical.quantiles);
    expect(fromDescriptor.decodeIndex).toBe(canonical.decodeIndex);
    expect(fromDescriptor.numLayers).toBe(canonical.numLayers);
    expect(fromDescriptor.numHeads).toBe(canonical.numHeads);
    expect(fromDescriptor.modelDims).toBe(canonical.modelDims);
    expect(fromDescriptor.headDim).toBe(canonical.headDim);
    expect(fromDescriptor.numQuantiles).toBe(canonical.numQuantiles);
    expect(fromDescriptor.tokenizerInputDims).toBe(canonical.tokenizerInputDims);
    expect(fromDescriptor.tokenizerHiddenDims).toBe(canonical.tokenizerHiddenDims);
    expect(fromDescriptor.tokenizerOutputDims).toBe(canonical.tokenizerOutputDims);
    expect(fromDescriptor.outputPointDims).toBe(canonical.outputPointDims);
    expect(fromDescriptor.outputQuantileDims).toBe(canonical.outputQuantileDims);
  });

  it('extracts exportedPatches from onnx input_shape[1]', () => {
    const desc = { ...CANONICAL_DESCRIPTOR };
    desc.onnx = { ...desc.onnx, input_shape: [1, 32, 128] };
    const config = descriptorToModelConfig(desc);
    expect(config.exportedPatches).toBe(32);
    expect(config.tokenizerInputDims).toBe(64); // 32 + 32 (unchanged)
  });

  it('computes headDim as model_dims / num_heads', () => {
    const desc = { ...CANONICAL_DESCRIPTOR };
    desc.architecture = { ...desc.architecture, model_dims: 2560, num_heads: 32 };
    const config = descriptorToModelConfig(desc);
    expect(config.modelDims).toBe(2560);
    expect(config.numHeads).toBe(32);
    expect(config.headDim).toBe(80);
  });

  it('computes numQuantiles as quantiles.length + 1', () => {
    const desc = { ...CANONICAL_DESCRIPTOR };
    desc.architecture = {
      ...desc.architecture,
      quantiles: [0.25, 0.5, 0.75],
    };
    const config = descriptorToModelConfig(desc);
    expect(config.numQuantiles).toBe(4); // 1 mean + 3 quantiles (q25, q50, q75)
    expect(config.quantiles).toEqual([0.25, 0.5, 0.75]);
  });

  it('computes outputQuantileDims as output_quantile_len * numQuantiles', () => {
    const desc = { ...CANONICAL_DESCRIPTOR };
    desc.architecture = {
      ...desc.architecture,
      output_quantile_len: 512,
      quantiles: [0.25, 0.5, 0.75],
    };
    const config = descriptorToModelConfig(desc);
    // numQuantiles = 4 (mean + 3), outputQuantileLen = 512 → 2048
    expect(config.outputQuantileDims).toBe(2048);
  });

  it('returns a frozen object', () => {
    const config = descriptorToModelConfig(CANONICAL_DESCRIPTOR);
    expect(Object.isFrozen(config)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Schema compatibility (implicit via validateDescriptor logic)
// ---------------------------------------------------------------------------

describe('schema compatibility', () => {
  it('ENGINE_SUPPORTED_SCHEMA is 1', () => {
    expect(ENGINE_SUPPORTED_SCHEMA).toBe(1);
  });

  it('accepts schema equal to ENGINE_SUPPORTED_SCHEMA', () => {
    // descriptorToModelConfig doesn't validate schema directly,
    // but we verify the descriptor itself has schema === ENGINE_SUPPORTED_SCHEMA
    expect(CANONICAL_DESCRIPTOR.schema).toBe(ENGINE_SUPPORTED_SCHEMA);
  });
});

// ---------------------------------------------------------------------------
// Edge cases: non-TimesFM-2.5 architectures
// ---------------------------------------------------------------------------

describe('future model variants', () => {
  it('handles 500M model with doubled dims', () => {
    const desc: ModelDescriptor = {
      ...CANONICAL_DESCRIPTOR,
      architecture: {
        ...CANONICAL_DESCRIPTOR.architecture,
        model_dims: 2560,
        num_heads: 32,
        num_layers: 32,
        output_quantile_len: 2048,
      },
      onnx: {
        ...CANONICAL_DESCRIPTOR.onnx,
        input_shape: [1, 24, 64],
        outputs: {
          input_emb: [1, 24, 2560],
          output_emb: [1, 24, 2560],
          output_ts: [1, 24, 2560],
          output_qs: [1, 24, 20480],
        },
      },
    };
    const config = descriptorToModelConfig(desc);
    expect(config.exportedPatches).toBe(24);
    expect(config.modelDims).toBe(2560);
    expect(config.numHeads).toBe(32);
    expect(config.headDim).toBe(80);
    expect(config.numLayers).toBe(32);
    expect(config.outputQuantileDims).toBe(20480); // 2048 * 10
  });

  it('handles model with fewer quantiles', () => {
    const desc: ModelDescriptor = {
      ...CANONICAL_DESCRIPTOR,
      architecture: {
        ...CANONICAL_DESCRIPTOR.architecture,
        quantiles: [0.1, 0.5, 0.9],
      },
    };
    const config = descriptorToModelConfig(desc);
    expect(config.numQuantiles).toBe(4); // mean + 3
    expect(config.quantiles).toEqual([0.1, 0.5, 0.9]);
  });
});

// ---------------------------------------------------------------------------
// Helpers for file-system tests
// ---------------------------------------------------------------------------

/** Create a temp directory that gets cleaned up after the test. */
function withTempDir(fn: (dir: string) => Promise<void>): () => Promise<void> {
  return async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'timesfm-desc-test-'));
    try {
      await fn(dir);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  };
}

/** Write a descriptor JSON file into the given directory. */
function writeDescriptor(dir: string, filename: string, desc: unknown): string {
  const filePath = path.join(dir, filename);
  fs.writeFileSync(filePath, JSON.stringify(desc), 'utf-8');
  return filePath;
}

// ---------------------------------------------------------------------------
// loadModelDescriptor — file-system tests
// ---------------------------------------------------------------------------

describe('loadModelDescriptor', () => {
  // 1. Successfully loads a valid descriptor file
  it(
    'successfully loads a valid model-descriptor.json',
    withTempDir(async (dir) => {
      writeDescriptor(dir, 'model-descriptor.json', CANONICAL_DESCRIPTOR);
      const desc = await loadModelDescriptor(dir);
      expect(desc).not.toBeNull();
      expect(desc!.schema).toBe(1);
      expect(desc!.architecture.input_patch_len).toBe(32);
      expect(desc!.onnx.sha256).toBe(CANONICAL_DESCRIPTOR.onnx.sha256);
    }),
  );

  // 2. Returns null when no descriptor file exists
  it(
    'returns null when no descriptor file exists',
    withTempDir(async (dir) => {
      const desc = await loadModelDescriptor(dir);
      expect(desc).toBeNull();
    }),
  );

  // 3. -descriptor.json fallback (path ending with .onnx)
  it(
    'falls back to -descriptor.json when path ends with .onnx',
    withTempDir(async (dir) => {
      // Create a dummy .onnx file so the path looks like a model file
      const onnxPath = path.join(dir, 'model.onnx');
      fs.writeFileSync(onnxPath, 'dummy', 'utf-8');
      // Write the -descriptor.json file (the fallback candidate)
      const descPath = path.join(dir, 'model-descriptor.json');
      fs.writeFileSync(descPath, JSON.stringify(CANONICAL_DESCRIPTOR), 'utf-8');

      const desc = await loadModelDescriptor(onnxPath);
      expect(desc).not.toBeNull();
      expect(desc!.schema).toBe(1);
    }),
  );

  // 4. .onnx.meta.json fallback
  it(
    'falls back to .onnx.meta.json when path ends with .onnx',
    withTempDir(async (dir) => {
      const onnxPath = path.join(dir, 'model.onnx');
      fs.writeFileSync(onnxPath, 'dummy', 'utf-8');
      // Write the .onnx.meta.json file (third candidate)
      const metaPath = path.join(dir, 'model.onnx.meta.json');
      fs.writeFileSync(metaPath, JSON.stringify(CANONICAL_DESCRIPTOR), 'utf-8');

      const desc = await loadModelDescriptor(onnxPath);
      expect(desc).not.toBeNull();
      expect(desc!.schema).toBe(1);
    }),
  );

  // 5. Returns null for invalid JSON
  it(
    'returns null for malformed JSON',
    withTempDir(async (dir) => {
      const filePath = path.join(dir, 'model-descriptor.json');
      fs.writeFileSync(filePath, '{ invalid json !!! }', 'utf-8');
      const desc = await loadModelDescriptor(dir);
      expect(desc).toBeNull();
    }),
  );

  // 6. Returns null for descriptor with newer schema
  it(
    'returns null when schema > ENGINE_SUPPORTED_SCHEMA',
    withTempDir(async (dir) => {
      const newerDesc = { ...CANONICAL_DESCRIPTOR, schema: ENGINE_SUPPORTED_SCHEMA + 1 };
      writeDescriptor(dir, 'model-descriptor.json', newerDesc);
      const desc = await loadModelDescriptor(dir);
      expect(desc).toBeNull();
    }),
  );

  // 7. Returns null for descriptor missing architecture
  it(
    'returns null when architecture field is missing',
    withTempDir(async (dir) => {
      const { architecture: _, ...noArch } = CANONICAL_DESCRIPTOR as ModelDescriptor & {
        architecture?: unknown;
      };
      writeDescriptor(dir, 'model-descriptor.json', noArch);
      const desc = await loadModelDescriptor(dir);
      expect(desc).toBeNull();
    }),
  );

  // 8. Returns null for descriptor missing onnx
  it(
    'returns null when onnx field is missing',
    withTempDir(async (dir) => {
      const { onnx: _, ...noOnnx } = CANONICAL_DESCRIPTOR as ModelDescriptor & { onnx?: unknown };
      writeDescriptor(dir, 'model-descriptor.json', noOnnx);
      const desc = await loadModelDescriptor(dir);
      expect(desc).toBeNull();
    }),
  );

  // 9. Returns null for descriptor with invalid input_shape (< 3 dims)
  it(
    'returns null when onnx.input_shape has fewer than 3 dimensions',
    withTempDir(async (dir) => {
      const badShape = {
        ...CANONICAL_DESCRIPTOR,
        onnx: { ...CANONICAL_DESCRIPTOR.onnx, input_shape: [1, 16] },
      };
      writeDescriptor(dir, 'model-descriptor.json', badShape);
      const desc = await loadModelDescriptor(dir);
      expect(desc).toBeNull();
    }),
  );

  // 10. Returns null for non-object JSON (e.g. string)
  it(
    'returns null when JSON parses to a non-object',
    withTempDir(async (dir) => {
      const filePath = path.join(dir, 'model-descriptor.json');
      fs.writeFileSync(filePath, '"just a string"', 'utf-8');
      const desc = await loadModelDescriptor(dir);
      expect(desc).toBeNull();
    }),
  );

  it(
    'returns null when schema is not a number',
    withTempDir(async (dir) => {
      const badSchema = { ...CANONICAL_DESCRIPTOR, schema: 'not-a-number' };
      writeDescriptor(dir, 'model-descriptor.json', badSchema);
      const desc = await loadModelDescriptor(dir);
      expect(desc).toBeNull();
    }),
  );
});

// ---------------------------------------------------------------------------
// resolveModelConfig
// ---------------------------------------------------------------------------

describe('resolveModelConfig', () => {
  // 11. Falls back to default when no descriptor found
  it(
    'falls back to default config when no descriptor is found',
    withTempDir(async (dir) => {
      const modelPath = path.join(dir, 'nonexistent-model.onnx');
      const result = await resolveModelConfig(modelPath);
      expect(result.descriptor).toBeNull();
      expect(result.config).toBe(TIMESFM_25_CONFIG);
    }),
  );

  // 12. Uses descriptor when found
  it(
    'uses descriptor when model-descriptor.json is found alongside the model',
    withTempDir(async (dir) => {
      const modelPath = path.join(dir, 'model.onnx');
      fs.writeFileSync(modelPath, 'dummy-onnx', 'utf-8');
      writeDescriptor(dir, 'model-descriptor.json', CANONICAL_DESCRIPTOR);

      const result = await resolveModelConfig(modelPath);
      expect(result.descriptor).not.toBeNull();
      expect(result.descriptor!.schema).toBe(1);
      expect(result.config.inputPatchLen).toBe(CANONICAL_DESCRIPTOR.architecture.input_patch_len);
      expect(result.config.contextLimit).toBe(CANONICAL_DESCRIPTOR.architecture.context_limit);
    }),
  );

  // 13. Custom fallback
  it(
    'uses custom fallback config when provided',
    withTempDir(async (dir) => {
      const modelPath = path.join(dir, 'nonexistent-model.onnx');
      const customFallback: ModelConfig = {
        ...TIMESFM_25_CONFIG,
        contextLimit: 999,
        inputPatchLen: 99,
      };
      const result = await resolveModelConfig(modelPath, customFallback);
      expect(result.descriptor).toBeNull();
      expect(result.config.contextLimit).toBe(999);
      expect(result.config.inputPatchLen).toBe(99);
    }),
  );

  it(
    'finds model-descriptor.json when model has a non-standard filename',
    withTempDir(async (dir) => {
      const modelPath = path.join(dir, 'my-custom-model.onnx');
      fs.writeFileSync(modelPath, 'dummy-onnx', 'utf-8');
      // resolveModelConfig looks for model-descriptor.json in the directory
      writeDescriptor(dir, 'model-descriptor.json', CANONICAL_DESCRIPTOR);

      const result = await resolveModelConfig(modelPath);
      expect(result.descriptor).not.toBeNull();
      expect(result.descriptor!.schema).toBe(1);
    }),
  );
});
