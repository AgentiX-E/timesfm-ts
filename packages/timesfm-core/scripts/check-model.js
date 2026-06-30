#!/usr/bin/env node
/**
 * TimesFM ONNX model manager.
 *
 * Checks model status, reports path suggestions, and validates
 * that the ONNX model file is loadable by onnxruntime-node.
 *
 * Usage:
 *   node scripts/check-model.js --model ./models/timesfm-2.5.onnx
 *   node scripts/check-model.js --model ./models/timesfm-2.5.onnx --bench
 *
 * This does NOT download the model — use the Python export script
 * (scripts/export-onnx.py) to export TimesFM from HuggingFace.
 */

const fs = require('fs');
const path = require('path');
const os = require('os');
const { performance } = require('perf_hooks');

async function main() {
  const args = process.argv.slice(2);
  const modelIdx = args.indexOf('--model');
  const bench = args.includes('--bench');

  if (modelIdx === -1) {
    console.error('Usage: node scripts/check-model.js --model <path> [--bench]');
    console.error();
    console.error('  --model <path>   Path to TimesFM ONNX model file');
    console.error('  --bench          Run a quick inference benchmark');
    console.error();
    console.error('To obtain a model:');
    console.error('  1. Install TimesFM Python package:');
    console.error('     pip install timesfm[torch] onnx onnxruntime');
    console.error('  2. Run the export script:');
    console.error('     python scripts/export-onnx.py --output models/timesfm-2.5.onnx');
    process.exit(1);
  }

  const modelPath = args[modelIdx + 1];
  if (!modelPath) {
    console.error('Error: --model requires a path argument');
    process.exit(1);
  }

  console.log('='.repeat(60));
  console.log('  timesfm-ts — ONNX Model Checker');
  console.log('='.repeat(60));
  console.log();

  // ─── System info ───
  console.log('System:');
  console.log(`  OS:      ${os.type()} ${os.release()} (${os.arch()})`);
  console.log(`  Node.js: ${process.version}`);
  console.log(`  CPUs:    ${os.cpus().length}`);
  console.log(`  RAM:     ${(os.totalmem() / 1024 ** 3).toFixed(1)} GB`);
  console.log(`  Free:    ${(os.freemem() / 1024 ** 3).toFixed(1)} GB`);
  console.log();

  // ─── File check ───
  console.log('Model file:');
  console.log(`  Path: ${modelPath}`);
  const absPath = path.resolve(modelPath);

  if (!fs.existsSync(absPath)) {
    console.log(`  Status: ❌ NOT FOUND`);
    console.log();
    console.log('  To obtain a valid ONNX model:');
    console.log();
    console.log('  Option A — Export from HuggingFace (recommended):');
    console.log('    pip install "timesfm[torch]" onnx onnxruntime torch');
    console.log('    python scripts/export-onnx.py \\');
    console.log('      --model google/timesfm-2.5-200m-pytorch \\');
    console.log(`      --output ${modelPath}`);
    console.log();
    console.log('  Option B — Use a pre-converted model:');
    console.log('    Place the .onnx file at the path above.');
    console.log('    Expected size: ~800 MB for TimesFM 2.5 200M');
    process.exit(1);
  }

  const stats = fs.statSync(absPath);
  const sizeMB = stats.size / 1024 ** 2;
  console.log(`  Size:   ${sizeMB.toFixed(1)} MB`);

  if (sizeMB < 10) {
    console.log(`  Status: ⚠️  File too small to be a real TimesFM model (< 10 MB)`);
  } else if (sizeMB < 500) {
    console.log(`  Status: ⚠️  Smaller than expected (TimesFM 2.5 ≈ 800 MB). May be a test model.`);
  } else {
    console.log(`  Status: ✅ Size matches TimesFM 2.5 200M (~800 MB)`);
  }
  console.log();

  // ─── ONNX Runtime load test ───
  console.log('ONNX Runtime:');
  try {
    // Try to load onnxruntime-node — may be in a workspace package's node_modules
    let ort;
    try {
      ort = require('onnxruntime-node');
    } catch {
      // Try from the core package's node_modules
      const altPath = path.join(
        __dirname,
        '..',
        'packages',
        'timesfm-core',
        'node_modules',
        'onnxruntime-node',
      );
      ort = require(altPath);
    }
    console.log(`  Package: onnxruntime-node loaded`);
    console.log(`  Loading model...`);

    const t0 = performance.now();
    const session = await ort.InferenceSession.create(absPath);
    const loadTime = performance.now() - t0;
    console.log(`  Load time: ${loadTime.toFixed(0)} ms`);
    console.log(`  Status:    ✅ Loaded successfully`);

    // Print input/output info
    console.log();
    console.log('Model I/O:');
    console.log(`  Inputs:  ${session.inputNames.join(', ')}`);
    for (const name of session.inputNames) {
      const meta = session.inputMetadata?.[name];
      if (meta) console.log(`    ${name}: ${JSON.stringify(meta.dims)} (${meta.type})`);
    }
    console.log(`  Outputs: ${session.outputNames.join(', ')}`);
    for (const name of session.outputNames) {
      const meta = session.outputMetadata?.[name];
      if (meta) console.log(`    ${name}: ${JSON.stringify(meta.dims)} (${meta.type})`);
    }

    // ─── Benchmark ───
    if (bench) {
      console.log();
      console.log('Benchmark (CPU):');

      // Get input shape from metadata, or default to [1,16,64]
      const inputMeta = session.inputMetadata?.[session.inputNames[0]];
      let testShape;
      if (inputMeta && inputMeta.dims) {
        testShape = inputMeta.dims.map((d) => (typeof d === 'number' && d > 0 ? d : 1));
      } else {
        testShape = [1, 16, 64]; // default TimesFM 2.5 export shape
      }
      const totalElems = testShape.reduce((a, b) => a * b, 1);
      const testInput = new Float32Array(totalElems);
      for (let i = 0; i < totalElems; i++) testInput[i] = Math.random();

      const feeds = {
        [session.inputNames[0]]: new ort.Tensor('float32', testInput, testShape),
      };

      // Warmup
      for (let i = 0; i < 3; i++) await session.run(feeds);

      const times = [];
      for (let i = 0; i < 10; i++) {
        const start = performance.now();
        await session.run(feeds);
        times.push(performance.now() - start);
      }

      const avg = times.reduce((a, b) => a + b, 0) / times.length;
      times.sort((a, b) => a - b);
      const p50 = times[5];
      console.log(`  Shape:        ${JSON.stringify(testShape)}`);
      console.log(`  Avg latency:  ${avg.toFixed(1)} ms`);
      console.log(`  P50 latency:  ${p50.toFixed(1)} ms`);

      // Memory
      const mem = process.memoryUsage();
      console.log(`  RSS memory:   ${(mem.rss / 1024 ** 2).toFixed(1)} MB`);
    }

    session.release?.();
  } catch (err) {
    console.log(`  Status: ❌ Failed to load`);
    console.log(`  Error:  ${err.message}`);
    console.log();
    console.log('  Troubleshooting:');
    console.log('  • Is onnxruntime-node installed? npm install onnxruntime-node');
    console.log(
      '  • Is the file a valid ONNX model? Check with: python -c "import onnx; onnx.checker.check_model(\'' +
        absPath +
        '\')"',
    );
    console.log('  • Try re-exporting: python scripts/export-onnx.py --output ' + modelPath);
    process.exit(1);
  }

  console.log();
  console.log('='.repeat(60));
  console.log('  ✅ Model ready for use with timesfm-ts');
  console.log('='.repeat(60));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
