/**
 * Vitest global setup — checks for TimesFM ONNX model availability
 * before running integration tests.
 *
 * If the model is not found, sets VITEST_SKIP_ONNX_TESTS=true so
 * individual test files can conditionally skip ONNX-dependent tests.
 * This prevents the test suite from crashing in environments without
 * the 885 MB model (e.g., fresh clones, CI without model cache).
 */
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export function setup(): void {
  const envPath = process.env.TIMESFM_TEST_MODEL;
  if (envPath && fs.existsSync(envPath)) {
    return; // Explicit path found
  }

  const searchPaths = [
    path.resolve(__dirname, 'models'),
    path.resolve(__dirname),
    path.join(os.homedir(), '.cache', 'timesfm-ts'),
  ];

  const filenames = ['timesfm-2.5.onnx', 'timesfm-2.5-200m.onnx', 'timesfm.onnx'];

  for (const dir of searchPaths) {
    for (const name of filenames) {
      if (fs.existsSync(path.join(dir, name))) {
        return; // Model found
      }
    }
  }

  // Model not found — signal tests to skip
  process.env.VITEST_SKIP_ONNX_TESTS = 'true';
  console.warn(
    '\n⚠️  TimesFM ONNX model not found. ONNX-dependent tests will be skipped.\n' +
      '    To run the full test suite:\n' +
      '      1. Export the model:      pnpm export:model\n' +
      '      2. Or one-click pipeline:  pnpm run pipeline\n' +
      '      3. Or set:                 TIMESFM_TEST_MODEL=/path/to/timesfm-2.5.onnx\n' +
      '      4. Or download via CLI:    npx @agentix-e/timesfm-cli setup -o ./models/timesfm-2.5.onnx\n' +
      '    Only pure-logic unit tests will run without the model.\n' +
      '    CI integration tests include the ONNX model and enforce ≥95% coverage thresholds.\n',
  );
}
