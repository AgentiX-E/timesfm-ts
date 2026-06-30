/**
 * Test model path resolution.
 *
 * Used by all test files to locate the TimesFM ONNX model.
 * Priority:
 *   1. TIMESFM_TEST_MODEL environment variable (absolute path)
 *   2. TIMESFM_TEST_MODEL_DIR environment variable (directory containing timesfm-2.5.onnx)
 *   3. Relative path: ../../models/timesfm-2.5.onnx (from packages/timesfm-core/test/)
 *   4. Default cache: ~/.cache/timesfm-ts/timesfm-2.5.onnx
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
  path.join(os.homedir(), '.cache', 'timesfm-ts'), // default cache
];

/**
 * Resolve the TimesFM ONNX model path for testing.
 *
 * Returns `null` if no model file is found, so callers can skip ONNX-dependent
 * tests gracefully instead of crashing at import time. The globalSetup already
 * warns when the model is missing.
 *
 * @returns The absolute path to the ONNX model, or `null` if not found.
 */
export function getTestModelPath(): string | null {
  // 1. Explicit env var
  const envPath = process.env.TIMESFM_TEST_MODEL;
  if (envPath) {
    if (fs.existsSync(envPath)) return envPath;
    console.warn(`[test helpers] TIMESFM_TEST_MODEL is set but file not found: ${envPath}`);
    return null;
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

  // 4. Not found — return null for graceful skip
  return null;
}
