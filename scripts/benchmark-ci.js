#!/usr/bin/env node
/**
 * agentix-timesfm-ts  Benchmark Suite with JSON/Markdown Report
 *
 * Usage:
 *   node scripts/benchmark-ci.js                    # console output
 *   node scripts/benchmark-ci.js --json report.json  # JSON report
 *   node scripts/benchmark-ci.js --md report.md     # Markdown report
 *   node scripts/benchmark-ci.js --all              # Both JSON + Markdown
 *
 * Outputs structured benchmark data suitable for CI automation,
 * historical trend tracking, and GitHub Pages publication.
 *
 * Environment:
 *   TIMESFM_MODEL_PATH  — path to ONNX model
 *   BENCH_SKIP_ACCURACY — skip accuracy tests (faster)
 *   BENCH_ITERATIONS     — number of inference iterations (default 5)
 */

const { performance } = require('perf_hooks');
const os = require('os');
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

// ─── CLI ────────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const outJson = args.includes('--json')
  ? args[args.indexOf('--json') + 1] || 'benchmark-report.json'
  : null;
const outMd = args.includes('--md')
  ? args[args.indexOf('--md') + 1] || 'benchmark-report.md'
  : null;
const outAll = args.includes('--all');
const skipAccuracy = !!process.env.BENCH_SKIP_ACCURACY;

// ─── Model Path Resolution ──────────────────────────────────────────────────

function resolveModelPath() {
  if (process.env.TIMESFM_MODEL_PATH && fs.existsSync(process.env.TIMESFM_MODEL_PATH)) {
    return process.env.TIMESFM_MODEL_PATH;
  }
  const searchPaths = [
    path.join(__dirname, '..', 'models', 'timesfm-2.5.onnx'),
    path.join(__dirname, '..', '..', 'models', 'timesfm-2.5.onnx'),
    path.join(os.homedir(), '.cache', 'agentix-timesfm-ts', 'timesfm-2.5.onnx'),
  ];
  for (const p of searchPaths) {
    if (fs.existsSync(p)) return p;
  }
  return null;
}

// ─── System Info ─────────────────────────────────────────────────────────────

function getSystemInfo() {
  const cpus = os.cpus();
  const gitSha = (() => {
    try {
      return spawnSync('git', ['rev-parse', 'HEAD'], { cwd: __dirname + '/..', encoding: 'utf-8' })
        .stdout.trim()
        .slice(0, 8);
    } catch {
      return 'unknown';
    }
  })();

  const gitBranch = (() => {
    try {
      return spawnSync('git', ['rev-parse', '--abbrev-ref', 'HEAD'], {
        cwd: __dirname + '/..',
        encoding: 'utf-8',
      }).stdout.trim();
    } catch {
      return 'unknown';
    }
  })();

  return {
    timestamp: new Date().toISOString(),
    git_sha: gitSha,
    git_branch: gitBranch,
    platform: os.platform(),
    arch: os.arch(),
    cpu_model: cpus[0]?.model || 'unknown',
    cpu_cores: cpus.length,
    total_ram_gb: +(os.totalmem() / 1024 ** 3).toFixed(1),
    node_version: process.version,
    onnx_runtime_version: (() => {
      try {
        return require('onnxruntime-node/package.json').version;
      } catch {
        return 'unknown';
      }
    })(),
  };
}

// ─── Main ────────────────────────────────────────────────────────────────────

async function main() {
  const sysInfo = getSystemInfo();
  const report = {
    ...sysInfo,
    model: {},
    latency: [],
    accuracy: null,
    memory: {},
  };

  // ── Resolve model ──────────────────────────────────────────────────────────
  const modelPath = resolveModelPath();
  if (!modelPath) {
    console.error('No ONNX model found. Set TIMESFM_MODEL_PATH.');
    process.exit(1);
  }

  const modelStats = fs.statSync(modelPath);
  report.model = {
    path: modelPath,
    size_mb: +(modelStats.size / 1024 ** 2).toFixed(0),
  };

  console.log('TIMESFM BENCHMARK SUITE');
  console.log('='.repeat(70));
  console.log(`  Model:   ${modelPath} (${report.model.size_mb} MB)`);
  console.log(`  CPU:     ${sysInfo.cpu_model} x ${sysInfo.cpu_cores}`);
  console.log(`  RAM:     ${sysInfo.total_ram_gb} GB`);
  console.log(`  Node:    ${sysInfo.node_version}`);
  console.log(`  Git:     ${sysInfo.git_sha} (${sysInfo.git_branch})`);
  console.log('='.repeat(70));

  // ── Load ONNX Runtime ─────────────────────────────────────────────────────
  const ort = require('onnxruntime-node');
  const loadStart = performance.now();
  const session = await ort.InferenceSession.create(modelPath);
  report.model.load_time_sec = +((performance.now() - loadStart) / 1000).toFixed(2);
  console.log(`\n  Load time: ${report.model.load_time_sec}s`);

  // ── Latency Benchmark ─────────────────────────────────────────────────────
  const MODEL_PATCHES = 16;
  const inputPatchLen = 32;
  const dim = 64;
  const iterations = parseInt(process.env.BENCH_ITERATIONS || '5', 10);

  console.log('\n  ── Latency ──');

  const configs = [
    { patches: 4, label: 'ctx=128  (4 patches)', context: 128 },
    { patches: 8, label: 'ctx=256  (8 patches)', context: 256 },
    { patches: 16, label: 'ctx=512  (16 patches)', context: 512 },
  ];

  for (const cfg of configs) {
    // Build padded input matching exported model shape
    const input = new Float32Array(1 * MODEL_PATCHES * dim);
    for (let p = 0; p < cfg.patches; p++) {
      const bp = p * dim;
      for (let i = 0; i < inputPatchLen; i++) {
        input[bp + i] = Math.random();
        input[bp + inputPatchLen + i] = 0;
      }
    }
    for (let p = cfg.patches; p < MODEL_PATCHES; p++) {
      const bp = p * dim;
      for (let i = 0; i < inputPatchLen; i++) {
        input[bp + inputPatchLen + i] = 1;
      }
    }

    const feeds = { inputs: new ort.Tensor('float32', input, [1, MODEL_PATCHES, dim]) };

    // Warmup
    for (let i = 0; i < 2; i++) await session.run(feeds);

    // Measure
    const times = [];
    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      await session.run(feeds);
      times.push(performance.now() - start);
    }

    const avg = times.reduce((a, b) => a + b, 0) / times.length;
    const p50 = [...times].sort((a, b) => a - b)[Math.floor(times.length / 2)];
    const p99 = [...times].sort((a, b) => a - b)[Math.floor(times.length * 0.99)];

    report.latency.push({
      context: cfg.context,
      patches: cfg.patches,
      avg_ms: +avg.toFixed(1),
      p50_ms: +p50.toFixed(1),
      p99_ms: +p99.toFixed(1),
      throughput_seq_s: +(1000 / avg).toFixed(1),
    });

    console.log(
      `  ${cfg.label.padEnd(26)}  avg=${avg.toFixed(0).padStart(5)}ms  p50=${p50.toFixed(0).padStart(5)}ms  p99=${p99.toFixed(0).padStart(5)}ms  thr=${(1000 / avg).toFixed(1)} seq/s`,
    );
  }

  // ── Accuracy Benchmark ─────────────────────────────────────────────────────
  if (!skipAccuracy) {
    console.log('\n  ── Accuracy ──');

    const nSeries = 5;
    const horizon = 24;
    const seriesLen = 200;
    const naiveMAEs = [];
    const modelMAEs = [];
    const modelRMSEs = [];

    let seed = 42;
    function rand() {
      seed = (seed * 16807) % 2147483647;
      return (seed - 1) / 2147483646;
    }

    for (let s = 0; s < nSeries; s++) {
      const data = new Float32Array(seriesLen);
      const trend = rand() * 0.3 - 0.1;
      const seasonAmp = rand() * 20 + 5;
      const noiseAmp = rand() * 5 + 1;
      const base = rand() * 200;
      for (let i = 0; i < seriesLen; i++) {
        data[i] =
          base +
          trend * i +
          seasonAmp * Math.sin((2 * Math.PI * i) / 12) +
          (rand() - 0.5) * noiseAmp * 2;
      }

      const context = data.slice(0, seriesLen - horizon);
      const actual = data.slice(seriesLen - horizon);

      // Naive baseline
      const lastVal = context[context.length - 1];
      let naiveMae = 0;
      for (let h = 0; h < horizon; h++) naiveMae += Math.abs(actual[h] - lastVal);
      naiveMae /= horizon;
      naiveMAEs.push(naiveMae);

      // TimesFM
      const nPatches = Math.ceil(context.length / inputPatchLen);
      const actualPatches = Math.min(nPatches, MODEL_PATCHES);
      const padded = new Float32Array(MODEL_PATCHES * inputPatchLen);
      const mask = new Uint8Array(MODEL_PATCHES * inputPatchLen);

      const ctxStart = Math.max(0, MODEL_PATCHES * inputPatchLen - context.length);
      const contextToUse = context.slice(
        Math.max(0, context.length - actualPatches * inputPatchLen),
      );
      for (let i = 0; i < ctxStart; i++) mask[i] = 1;
      padded.set(contextToUse, ctxStart);

      const flatInput = new Float32Array(1 * MODEL_PATCHES * dim);
      for (let p = 0; p < MODEL_PATCHES; p++) {
        const bp = p * dim;
        for (let i = 0; i < inputPatchLen; i++) {
          flatInput[bp + i] = padded[p * inputPatchLen + i];
          flatInput[bp + inputPatchLen + i] = mask[p * inputPatchLen + i];
        }
      }

      const result = await session.run({
        inputs: new ort.Tensor('float32', flatInput, [1, MODEL_PATCHES, dim]),
      });

      const outputTS = result['output_ts'].data;
      const perPatchOut = 128 * 10;
      const lastValidPatch = actualPatches - 1;
      const lastPatchStart = lastValidPatch * perPatchOut;
      let maeVal = 0,
        rmseVal = 0;
      for (let h = 0; h < Math.min(horizon, 128); h++) {
        const pred = outputTS[lastPatchStart + h * 10 + 5] || 0;
        maeVal += Math.abs(actual[h] - pred);
        rmseVal += (actual[h] - pred) ** 2;
      }
      maeVal /= Math.min(horizon, 128);
      rmseVal = Math.sqrt(rmseVal / Math.min(horizon, 128));
      modelMAEs.push(maeVal);
      modelRMSEs.push(rmseVal);
    }

    const avgNaiveMAE = naiveMAEs.reduce((a, b) => a + b, 0) / naiveMAEs.length;
    const avgModelMAE = modelMAEs.reduce((a, b) => a + b, 0) / modelMAEs.length;
    const avgModelRMSE = modelRMSEs.reduce((a, b) => a + b, 0) / modelRMSEs.length;
    const scaledMAE = avgModelMAE / avgNaiveMAE;

    report.accuracy = {
      naive_mae: +avgNaiveMAE.toFixed(4),
      model_mae: +avgModelMAE.toFixed(4),
      model_rmse: +avgModelRMSE.toFixed(4),
      scaled_mae: +scaledMAE.toFixed(4),
      better_than_naive: scaledMAE < 1,
    };

    console.log(`  Naive MAE:        ${avgNaiveMAE.toFixed(4)}`);
    console.log(`  TimesFM MAE:      ${avgModelMAE.toFixed(4)}`);
    console.log(`  TimesFM RMSE:     ${avgModelRMSE.toFixed(4)}`);
    console.log(
      `  Scaled MAE:       ${scaledMAE.toFixed(4)}  (${scaledMAE < 1 ? '✅ Better than Naive' : '⚠️'})`,
    );
  }

  // ── Memory ─────────────────────────────────────────────────────────────────
  if (global.gc) global.gc();
  const mem = process.memoryUsage();
  report.memory = {
    rss_mb: +(mem.rss / 1024 / 1024).toFixed(0),
    heap_used_mb: +(mem.heapUsed / 1024 / 1024).toFixed(1),
    heap_total_mb: +(mem.heapTotal / 1024 / 1024).toFixed(1),
  };

  console.log('\n  ── Memory ──');
  console.log(`  RSS:     ${report.memory.rss_mb} MB`);
  console.log(`  Heap:    ${report.memory.heap_used_mb} MB`);

  // ── Cleanup ────────────────────────────────────────────────────────────────
  session.release?.();

  // ── Output Reports ─────────────────────────────────────────────────────────
  const doJson = outJson || outAll;
  const doMd = outMd || outAll;
  const jsonPath = outJson && typeof outJson === 'string' ? outJson : 'benchmark-report.json';
  const mdPath = outMd && typeof outMd === 'string' ? outMd : 'benchmark-report.md';

  if (doJson && jsonPath) {
    fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2));
    console.log(`\n  ✅ JSON report written: ${jsonPath}`);
  }

  if (doMd && mdPath) {
    const md = generateMarkdownReport(report);
    fs.writeFileSync(mdPath, md);
    console.log(`  ✅ Markdown report written: ${mdPath}`);
  }
}

// ─── Markdown Report Generator ───────────────────────────────────────────────

function generateMarkdownReport(report) {
  return `# TimesFM 2.5 200M Benchmark Report

> Generated: ${report.timestamp} · Git: \`${report.git_sha}\` · Node: ${report.node_version}

## System

| Property | Value |
|----------|-------|
| CPU | ${report.cpu_model} × ${report.cpu_cores} |
| RAM | ${report.total_ram_gb} GB |
| Platform | ${report.platform} / ${report.arch} |
| Node.js | ${report.node_version} |
| ONNX Runtime | ${report.onnx_runtime_version} |

## Model

| Property | Value |
|----------|-------|
| Size | ${report.model.size_mb} MB |
| Load time | ${report.model.load_time_sec}s |

## Inference Latency

| Context | Patches | Avg (ms) | P50 (ms) | P99 (ms) | Throughput (seq/s) |
|---------|---------|----------|----------|----------|---------------------|
${report.latency.map((l) => `| ${l.context} | ${l.patches} | ${l.avg_ms} | ${l.p50_ms} | ${l.p99_ms} | ${l.throughput_seq_s} |`).join('\n')}

## Memory

| Metric | Value |
|--------|-------|
| RSS | ${report.memory.rss_mb} MB |
| Heap Used | ${report.memory.heap_used_mb} MB |
| Heap Total | ${report.memory.heap_total_mb} MB |

${
  report.accuracy
    ? `
## Prediction Accuracy

| Metric | Value |
|--------|-------|
| Naive MAE | ${report.accuracy.naive_mae} |
| TimesFM MAE | ${report.accuracy.model_mae} |
| TimesFM RMSE | ${report.accuracy.model_rmse} |
| **Scaled MAE** | **${report.accuracy.scaled_mae}** ${report.accuracy.better_than_naive ? '✅ Better than Naive' : '⚠️'} |
`
    : ''
}
---
*Automated benchmark by agentix-timesfm-ts CI*
`;
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
