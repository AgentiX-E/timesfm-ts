import { defineConfig, type Plugin } from 'vitest/config';
import { resolve } from 'path';

/**
 * Vite plugin that rewrites `.js` import specifiers to `.ts` so that Vite
 * can resolve ESM-style source imports to the actual TypeScript files.
 */
function tsRewritePlugin(): Plugin {
  return {
    name: 'ts-rewrite',
    enforce: 'pre',
    async resolveId(source, importer, options) {
      if (!importer) return null;
      if (source.endsWith('.js')) {
        const tsPath = source.slice(0, -3) + '.ts';
        const resolved = await this.resolve(tsPath, importer, { ...options, skipSelf: true });
        if (resolved) return resolved;
      }
      if (!source.includes('.') && !source.startsWith('@') && !source.includes('/')) {
        const tsPath = source + '.ts';
        const resolved = await this.resolve(tsPath, importer, { ...options, skipSelf: true });
        if (resolved) return resolved;
      }
      return null;
    },
  };
}

export default defineConfig({
  plugins: [tsRewritePlugin()],
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
    globalSetup: ['./vitest.globalSetup.ts'],
    include: ['packages/*/test/**/*.test.ts'],
    // Single worker for real 885 MB model — avoids OOM from parallel loading
    pool: 'forks',
    poolOptions: {
      forks: {
        singleFork: true,
      },
    },
    // Longer timeout: real model takes ~2s to load + inference time
    testTimeout: 120000,
    hookTimeout: 120000,
    // ─── Coverage ───────────────────────────────────────────────────────────
    //
    // Target: ≥95% across all metrics (lines, branches, functions, statements).
    //
    // Integration tests use the real TimesFM ONNX model (no mocks, no synthetic
    // data). Core engine files (onnx-engine, xreg-engine, hierarchical) are now
    // INCLUDED in coverage counting — they are exercised by real-model tests.
    //
    // Exclusion rationale (limited to unavoidable cases):
    //   • index.ts (barrels)        — re-exports only, no logic to cover
    //   • cli.ts                    — Commander/stdio; tested via CLI smoke tests
    //   • model-downloader.ts       — network IO; cache helpers tested, download paths require GH Releases
    //   • kv-cache.ts               — @experimental, not used by current ONNX inference path
    //   • types/ & types.ts         — pure type definitions with no runtime code
    //
    // Local ↔ CI parity: both environments use the identical vitest.config.ts
    // and VITEST_SKIP_ONNX_TESTS env var to gate model-dependent test suites.
    coverage: {
      provider: 'v8',
      include: [
        'packages/timesfm-core/src/**/*.ts',
        'packages/timesfm-xreg/src/**/*.ts',
        'packages/timesfm-cli/src/**/*.ts',
        'packages/timesfm-web/src/**/*.ts',
        'packages/timesfm-hierarchical/src/**/*.ts',
      ],
      exclude: [
        'packages/*/src/index.ts', // barrel re-exports only — no runtime logic
        'packages/timesfm-cli/src/cli.ts', // Commander entry point (stdio); tested via CLI smoke tests
        'packages/timesfm-core/src/model-downloader.ts', // network IO; cache helpers tested, download paths require GH Releases
        'packages/timesfm-core/src/inference/kv-cache.ts', // @experimental — reserved for future native-KV ONNX export
        'packages/timesfm-core/src/types/', // pure type definitions — no runtime code
        'packages/timesfm-hierarchical/src/types.ts', // pure type definitions — no runtime code
      ],
      reporter: ['text', 'json', 'json-summary', 'html', 'lcov'],
      thresholds: {
        lines: 95,
        functions: 95,
        branches: 95,
        statements: 95,
      },
      reportsDirectory: './coverage',
    },
  },
});
