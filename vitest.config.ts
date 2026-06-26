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
      '@agentix/timesfm-core': resolve(__dirname, 'packages/timesfm-core/src/index.ts'),
      '@agentix/timesfm-xreg': resolve(__dirname, 'packages/timesfm-xreg/src/index.ts'),
    },
  },
  test: {
    globals: true,
    environment: 'node',
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
    // Coverage: target 100% across all metrics on logical code
    coverage: {
      provider: 'v8',
      include: [
        'packages/timesfm-core/src/**/*.ts',
        'packages/timesfm-xreg/src/**/*.ts',
        'packages/timesfm-cli/src/**/*.ts',
      ],
      exclude: [
        'packages/*/src/index.ts', // barrel re-exports only
        'packages/timesfm-cli/src/cli.ts', // Commander entry point (IO-only)
        'packages/timesfm-core/src/model-downloader.ts', // network IO (tested via cache helpers)
      ],
      reporter: ['text', 'json', 'html', 'lcov'],
      thresholds: {
        lines: 97,
        functions: 95,
        branches: 88,
        statements: 97,
      },
    },
  },
});
