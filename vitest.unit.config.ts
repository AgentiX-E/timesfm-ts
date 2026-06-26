import { defineConfig } from 'vitest/config';
import { resolve } from 'path';

/**
 * Unit test configuration — runs tests that do NOT require the 885 MB ONNX model.
 *
 * These tests are fast, lightweight, and suitable for CI pre-merge checks.
 * They cover all pure-logic modules: NaN handling, tensor ops, config validation,
 * statistics, RevIN, and OneHotEncoder.
 */
export default defineConfig({
  resolve: {
    alias: {
      '@agentix/timesfm-core': resolve(__dirname, 'packages/timesfm-core/src/index.ts'),
      '@agentix/timesfm-xreg': resolve(__dirname, 'packages/timesfm-xreg/src/index.ts'),
    },
  },
  test: {
    globals: true,
    environment: 'node',
    include: [
      'packages/*/test/**/config.test.ts',
      'packages/*/test/**/nan-handler.test.ts',
      'packages/*/test/**/tensor-utils.test.ts',
      'packages/*/test/**/stats-revin.test.ts',
      'packages/*/test/**/one-hot-encoder.test.ts',
      'packages/*/test/**/decode-loop.test.ts',
      'packages/*/test/**/metrics.test.ts',
      'packages/*/test/**/quantile.test.ts',
    ],
    testTimeout: 10000,
    hookTimeout: 10000,
  },
});
