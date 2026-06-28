import { defineConfig } from 'vitest/config';
import { resolve } from 'path';

/**
 * Unit test configuration — runs tests that do NOT require the 885 MB ONNX model.
 *
 * These tests are fast, lightweight, and suitable for CI pre-merge checks.
 * They cover all pure-logic modules: NaN handling, tensor ops, config validation,
 * statistics, RevIN, OneHotEncoder, decode-loop (via MockInferenceEngine),
 * postprocessor, preprocessor, metrics, quantile helpers, model descriptor,
 * model-downloader (cache helpers only), csv-forecast (mocked model),
 * web-engine (mocked), and xreg-engine (mocked).
 *
 * **Local/CI Parity**: This config mirrors the CI unit-test job exactly.
 * Run `pnpm test:unit:coverage` locally to get the same results as CI.
 */
export default defineConfig({
  resolve: {
    alias: {
      '@agentix-e/timesfm-core': resolve(__dirname, 'packages/timesfm-core/src/index.ts'),
      '@agentix-e/timesfm-xreg': resolve(__dirname, 'packages/timesfm-xreg/src/index.ts'),
      '@agentix-e/timesfm-web': resolve(__dirname, 'packages/timesfm-web/src/index.ts'),
      '@agentix-e/timesfm-hierarchical': resolve(
        __dirname,
        'packages/timesfm-hierarchical/src/index.ts',
      ),
    },
  },
  test: {
    globals: true,
    environment: 'node',
    include: ['packages/*/test/**/*.test.ts'],
    exclude: [
      // These tests require the 885 MB ONNX model (use pnpm test or pnpm test:coverage)
      '**/model.test.ts',
      '**/engine.test.ts',
      '**/web-integration.test.ts',
      '**/xreg-engine.test.ts',
      '**/concurrency.test.ts',
      '**/hierarchical-engine.test.ts',
    ],
    testTimeout: 15000,
    hookTimeout: 15000,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'text-summary', 'json-summary', 'lcov', 'html'],
      include: [
        'packages/timesfm-core/src/**/*.ts',
        'packages/timesfm-xreg/src/**/*.ts',
        'packages/timesfm-cli/src/**/*.ts',
        'packages/timesfm-hierarchical/src/**/*.ts',
      ],
      exclude: [
        'packages/*/src/index.ts', // barrel re-exports only
        'packages/timesfm-cli/src/cli.ts', // Commander entry point (IO-only)
        'packages/timesfm-core/src/model-downloader.ts', // network IO (tested via cache helpers)
        'packages/timesfm-core/src/model.ts', // requires real ONNX model (covered by integration tests)
        'packages/timesfm-core/src/inference/onnx-engine.ts', // requires real ONNX model
        'packages/timesfm-core/src/inference/kv-cache.ts', // @experimental, not used by current ONNX path
        'packages/timesfm-core/src/types/', // pure type definitions
        'packages/timesfm-xreg/src/xreg-engine.ts', // requires real TimesFM model
        'packages/timesfm-web/src/**', // requires browser/WASM environment
        'packages/timesfm-hierarchical/src/hierarchical.ts', // requires real TimesFM model
        'packages/timesfm-hierarchical/src/types.ts', // pure type definitions
      ],
      thresholds: {
        lines: 95,
        functions: 95,
        branches: 95,
        statements: 95,
      },
      // Generate lcov for CI artifact upload
      reportsDirectory: './coverage',
    },
  },
});
