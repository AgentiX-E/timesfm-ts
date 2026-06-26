/**
 * Test model path resolution.
 *
 * Used by all test files to locate the TimesFM ONNX model.
 * Priority:
 *   1. TIMESFM_TEST_MODEL environment variable (absolute path)
 *   2. TIMESFM_TEST_MODEL_DIR environment variable (directory containing timesfm-2.5.onnx)
 *   3. Relative path: ../../models/timesfm-2.5.onnx (from packages/timesfm-core/test/)
 *   4. Default cache: ~/.cache/agentix-timesfm-ts/timesfm-2.5.onnx
 */
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';

/** Known model filenames to search for, in priority order. */
const MODEL_FILENAMES = ['timesfm-2.5.onnx', 'timesfm-2.5-200m.onnx', 'timesfm.onnx'];

/** Relative search paths from the test file's location (packages/timesfm-core/test/). */
const RELATIVE_SEARCH_PATHS = [
  path.resolve(__dirname, '..', '..', '..', 'models'), // ../../models/
  path.resolve(__dirname, '..', '..', '..'), // project root
  path.join(os.homedir(), '.cache', 'agentix-timesfm-ts'), // default cache
];

/**
 * Resolve the TimesFM ONNX model path for testing.
 *
 * @throws {Error} if no model file is found.
 */
export function getTestModelPath(): string {
  // 1. Explicit env var
  const envPath = process.env.TIMESFM_TEST_MODEL;
  if (envPath) {
    if (fs.existsSync(envPath)) return envPath;
    throw new Error(`TIMESFM_TEST_MODEL is set but file not found: ${envPath}`);
  }

  // 2. Directory env var
  const envDir = process.env.TIMESFM_TEST_MODEL_DIR;
  if (envDir) {
    for (const name of MODEL_FILENAMES) {
      const candidate = path.join(envDir, name);
      if (fs.existsSync(candidate)) return candidate;
    }
  }

  // 3. Search relative paths
  for (const searchPath of RELATIVE_SEARCH_PATHS) {
    for (const name of MODEL_FILENAMES) {
      const candidate = path.join(searchPath, name);
      if (fs.existsSync(candidate)) return candidate;
    }
  }

  // 4. Not found — provide helpful error
  const searchedPaths = RELATIVE_SEARCH_PATHS.map((p) => `  - ${p}/`).join('\n');
  throw new Error(
    `TimesFM ONNX model not found.\n\n` +
      `Searched:\n${searchedPaths}\n\n` +
      `Set TIMESFM_TEST_MODEL=/path/to/model.onnx or\n` +
      `run: pnpm export:model  (to export from HuggingFace)`,
  );
}
