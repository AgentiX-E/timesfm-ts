/**
 * Unit tests for model-downloader.ts — cache helpers, path resolution.
 */

import { describe, it, expect } from 'vitest';
import * as path from 'node:path';
import * as os from 'node:os';

// We test the cache helpers directly since they are exported.
import { isModelCached, getCachedModelPath, defaultModelPath } from '../src/model-downloader';

describe('model-downloader', () => {
  describe('defaultModelPath', () => {
    it('returns a path in the cache directory', () => {
      const p = defaultModelPath();
      const cacheBase = process.env.XDG_CACHE_HOME || path.join(os.homedir(), '.cache');
      expect(p).toContain(cacheBase);
      expect(p).toContain('agentix-timesfm-ts');
      expect(p).toContain('timesfm-2.5.onnx');
    });

    it('respects XDG_CACHE_HOME', () => {
      const original = process.env.XDG_CACHE_HOME;
      process.env.XDG_CACHE_HOME = '/tmp/custom-cache';
      const p = defaultModelPath();
      expect(p).toContain('/tmp/custom-cache');
      process.env.XDG_CACHE_HOME = original;
    });
  });

  describe('isModelCached', () => {
    it('returns false when no model exists', () => {
      expect(isModelCached()).toBe(false);
    });
  });

  describe('getCachedModelPath', () => {
    it('returns null when no model is cached', () => {
      expect(getCachedModelPath()).toBeNull();
    });
  });
});
