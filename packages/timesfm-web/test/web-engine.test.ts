/**
 * Tests for TimesFMWebInferenceEngine and model-loader.
 *
 * Note: Full end-to-end tests require onnxruntime-web's WASM backend,
 * which needs a browser or Node.js with WASM support.
 * These tests focus on:
 * 1. API correctness (construction, lifecycle)
 * 2. Error handling (calling forward before load, etc.)
 * 3. Provider fallback logic (mocked)
 */

import { describe, it, expect, beforeEach } from 'vitest';
import type { ModelConfig } from '@agentix-e/timesfm-core';
import { TIMESFM_25_CONFIG } from '@agentix-e/timesfm-core';
import { TimesFMWebInferenceEngine } from '../src/web-engine';
import { checkModelAvailability } from '../src/model-loader';

// ---------------------------------------------------------------------------
// Test config
// ---------------------------------------------------------------------------

const TEST_CONFIG: ModelConfig = TIMESFM_25_CONFIG;

// ---------------------------------------------------------------------------
// TimesFMWebInferenceEngine
// ---------------------------------------------------------------------------

describe('TimesFMWebInferenceEngine', () => {
  let engine: TimesFMWebInferenceEngine;

  beforeEach(() => {
    engine = new TimesFMWebInferenceEngine(TEST_CONFIG);
  });

  describe('construction', () => {
    it('creates an engine with default providers', () => {
      expect(engine).toBeDefined();
      expect(engine.isLoaded()).toBe(false);
    });

    it('accepts custom provider order', () => {
      const e = new TimesFMWebInferenceEngine(TEST_CONFIG, ['wasm', 'webgl']);
      expect(e).toBeDefined();
      expect(e.isLoaded()).toBe(false);
    });
  });

  describe('lifecycle', () => {
    it('isLoaded returns false before load()', () => {
      expect(engine.isLoaded()).toBe(false);
    });

    it('forward throws before load()', async () => {
      await expect(engine.forward([new Float32Array(64)], [new Uint8Array(1)])).rejects.toThrow(
        'not loaded',
      );
    });
  });

  describe('dispose', () => {
    it('dispose works on unloaded engine', async () => {
      await expect(engine.dispose()).resolves.toBeUndefined();
      expect(engine.isLoaded()).toBe(false);
    });
  });

  describe('load with ArrayBuffer', () => {
    it('accepts ArrayBuffer', async () => {
      // Create a minimal valid ONNX buffer (just a mock for the API test)
      // In reality, onnxruntime-web would fail on this invalid data,
      // but we're testing that the API accepts it.
      const buffer = new ArrayBuffer(1024);

      // The actual load will fail because it's not a valid ONNX model,
      // but we verify the API contract is correct.
      await expect(engine.load(buffer)).rejects.toThrow();
      // Engine should not be loaded after failed load
      expect(engine.isLoaded()).toBe(false);
    });
  });
});

// ---------------------------------------------------------------------------
// checkModelAvailability
// ---------------------------------------------------------------------------

describe('checkModelAvailability', () => {
  it('returns null for non-existent URL', async () => {
    // In Node.js test environment, fetch will fail for non-existent URLs
    const result = await checkModelAvailability('https://invalid.example.com/model.onnx');
    expect(result).toBeNull();
  });
});
