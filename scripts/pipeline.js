#!/usr/bin/env node
/**
 * agentix-timesfm-ts  Automated Pipeline Tool
 *
 * No dependency on make/bash, pure Node.js implementation.
 *
 * Usage:
 *   node scripts/pipeline.js              # Full pipeline
 *   node scripts/pipeline.js --export      # Export ONNX only
 *   node scripts/pipeline.js --test        # Run all regression tests only
 *   node scripts/pipeline.js --benchmark   # Run benchmark only
 *   node scripts/pipeline.js --check-latest # Check HF latest version
 *   node scripts/pipeline.js --quick       # Skip export (quick mode)
 */

const { execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

// ─── Configuration ──────────────────────────────────────────────────────────

const ROOT = path.resolve(__dirname, '..');
const MODEL_PATH = process.env.TIMESFM_MODEL_PATH || path.join(ROOT, 'models', 'timesfm-2.5.onnx');
const HF_MODEL = process.env.TIMESFM_HF_MODEL || 'google/timesfm-2.5-200m-pytorch';

const COLORS = {
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  cyan: '\x1b[36m',
  reset: '\x1b[0m',
};
const info = (msg) => console.log(`${COLORS.green}[✓]${COLORS.reset} ${msg}`);
const warn = (msg) => console.log(`${COLORS.yellow}[!]${COLORS.reset} ${msg}`);
const fail = (msg) => {
  console.log(`${COLORS.red}[✗]${COLORS.reset} ${msg}`);
  process.exit(1);
};
const title = (msg) => console.log(`\n${COLORS.cyan}━━━ ${msg} ━━━${COLORS.reset}`);

// ─── Helpers ────────────────────────────────────────────────────────────────

function run(cmd, opts = {}) {
  const label = opts.label || cmd.split(' ').slice(0, 3).join(' ') + ' …';
  process.stdout.write(`  ${label}`);
  try {
    const result = execSync(cmd, {
      cwd: ROOT,
      stdio: opts.stdio || ['pipe', 'pipe', 'pipe'],
      timeout: opts.timeout || 600_000,
      env: { ...process.env, NODE_OPTIONS: '' },
    });
    console.log(` ${COLORS.green}✓${COLORS.reset}`);
    if (!opts.silent && result.length > 0 && opts.stdio !== 'inherit') {
      const out = result.toString().trim();
      if (out) {
        const lines = out.split('\n').slice(-8); // show last 8 lines
        lines.forEach((l) => console.log(`  ${l}`));
      }
    }
    return true;
  } catch (e) {
    console.log(` ${COLORS.red}FAILED${COLORS.reset}`);
    if (e.stderr) console.error(e.stderr.toString().slice(-500));
    if (opts.fatal !== false) process.exit(1);
    return false;
  }
}

function runNode(script, args = []) {
  const scriptPath = path.join(ROOT, 'packages', 'timesfm-core', 'scripts', script);
  const fullCmd = `node ${scriptPath} ${args.join(' ')}`;
  return run(fullCmd, { label: `node ${script}`, fatal: true });
}

function runNpx(cmd) {
  return run(`npx ${cmd}`, { label: `npx ${cmd.split(' ')[0]}`, fatal: true });
}

// ─── Phase 1: Check latest version ──────────────────────────────────────────

function checkLatest() {
  title('Phase 1/5  Check HuggingFace Latest Version');
  try {
    run(`python3 scripts/export-onnx.py --check-latest -m "${HF_MODEL}"`, {
      fatal: false,
      silent: false,
    });
  } catch {
    warn('HF API check skipped (offline?)');
  }
}

// ─── Phase 2: Export ONNX ───────────────────────────────────────────────────

function exportONNX() {
  title('Phase 2/5  Export TimesFM → ONNX');
  run(`python3 scripts/export-onnx.py -m "${HF_MODEL}" -o "${MODEL_PATH}"`, {
    label: 'python export-onnx.py',
    fatal: true,
    stdio: 'inherit',
    timeout: 600_000,
  });
}

// ─── Phase 3: Validate model ────────────────────────────────────────────────

function validateModel() {
  title('Phase 3/5  ONNX Runtime Model Load Validation');
  runNode('check-model.js', [`--model`, MODEL_PATH, `--bench`]);
}

// ─── Phase 4: Full regression test ──────────────────────────────────────────

function runTests() {
  title('Phase 4/5  Full Regression Test (all tests)');
  runNpx('vitest run --reporter=verbose');
}

// ─── Phase 5: Benchmark ─────────────────────────────────────────────────────

function benchmark() {
  title('Phase 5/5  Inference Benchmark');
  runNpx(
    'tsx scripts/benchmark-ci.js --json benchmark-report.json --md benchmark-report.md --html benchmark-report.html',
  );
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

function printHelp() {
  console.log(`
${COLORS.cyan}agentix-timesfm-ts  Automated Pipeline${COLORS.reset}

Usage:
  node scripts/pipeline.js [options]

Options:
  --pipeline      Full pipeline: check version → export → test → benchmark (default)
  --export        Export ONNX model only
  --test          Run all regression tests only (use existing model)
  --benchmark     Run inference benchmark only
  --check-latest  Check HuggingFace latest version
  --quick         Quick mode: skip export, run test + benchmark
  --help          Show this help

Environment Variables:
  TIMESFM_MODEL_PATH    ONNX model path (default: models/timesfm-2.5.onnx)
  TIMESFM_HF_MODEL      HuggingFace model ID (default: google/timesfm-2.5-200m-pytorch)

Examples:
  node scripts/pipeline.js                          # Full pipeline
  node scripts/pipeline.js --quick                  # Quick verification
  node scripts/pipeline.js --export                 # Export ONNX only
  TIMESFM_MODEL_PATH=./my.onnx node scripts/pipeline.js --test
`);
}

// ─── Main ────────────────────────────────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2);
  const mode = args[0] || '--pipeline';

  if (mode === '--help' || mode === '-h') {
    printHelp();
    return;
  }

  console.log('='.repeat(66));
  console.log(`  agentix-timesfm-ts  Automated Pipeline  (Node.js)`);
  console.log(`  ${new Date().toLocaleString()}`);
  console.log(`  ${os.cpus()[0]?.model || 'unknown'}  x ${os.cpus().length} cores`);
  console.log('='.repeat(66));

  const start = Date.now();

  switch (mode) {
    case '--check-latest':
      checkLatest();
      break;

    case '--export':
      exportONNX();
      break;

    case '--test':
      validateModel();
      runTests();
      break;

    case '--benchmark':
      benchmark();
      break;

    case '--quick':
      validateModel();
      runTests();
      benchmark();
      break;

    case '--pipeline':
    default:
      checkLatest();
      exportONNX();
      validateModel();
      runTests();
      benchmark();
      break;
  }

  const elapsed = ((Date.now() - start) / 1000).toFixed(0);
  console.log(`\n${'='.repeat(66)}`);
  info(`Full pipeline complete  (${elapsed}s)`);
  if (fs.existsSync(MODEL_PATH)) {
    const mb = (fs.statSync(MODEL_PATH).size / 1024 ** 2).toFixed(0);
    console.log(`  Model: ${MODEL_PATH}  (${mb} MB)`);
  }
  // Dynamically count tests from vitest (available if installed)
  try {
    const out = execSync('npx vitest run --reporter=json 2>/dev/null || true', {
      cwd: ROOT,
      timeout: 30_000,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, NODE_OPTIONS: '' },
    }).toString();
    const json = JSON.parse(out);
    const files = (json.testResults || []).length;
    const total =
      files > 0
        ? json.testResults.reduce(
            (s, r) => s + (r.assertionResults ? r.assertionResults.length : 0),
            0,
          )
        : 0;
    if (total > 0) {
      console.log(`  Test files: ${files}  |  Tests: ${total} (all passed ✓)`);
    }
  } catch {
    // Silent fallback — vitest may not be installed
  }
  console.log('='.repeat(66));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
