import { defineConfig } from 'vitest/config';
import { resolve } from 'path';

/**
 * Unit test configuration — runs tests that do NOT require the 885 MB ONNX model.
 *
 * These tests are fast, lightweight, and suitable for CI pre-merge checks.
 * They cover all pure-logic modules: NaN handling, tensor ops, config validation,
 * statistics, RevIN, OneHotEncoder, decode-loop (via MockInferenceEngine),
 * postprocessor, preprocessor, metrics, quantile helpers, model descriptor,
 * model-downloader (cache helpers only), and csv-forecast (mocked model).
 */
export default defineConfig({
  resolve: {
    alias: {
      '@agentix-e/timesfm-core': resolve(__dirname, 'packages/timesfm-core/src/index.ts'),
      '@agentix-e/timesfm-xreg': resolve(__dirname, 'packages/timesfm-xreg/src/index.ts'),
      '@agentix-e/timesfm-web': resolve(__dirname, 'packages/timesfm-web/src/index.ts'),
    },
  },
  test: {
    globals: true,
    environment: 'node',
    include: [
      'packages/*/test/**/*.test.ts',
    ],
    exclude: [
      // These tests require the 885 MB ONNX model (use pnpm test or pnpm test:coverage)
      '**/model.test.ts',
      '**/engine.test.ts',
      '**/web-integration.test.ts',
      '**/xreg-engine.test.ts',
    ],
    testTimeout: 10000,
    hookTimeout: 10000,
  },
});
