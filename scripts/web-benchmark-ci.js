#!/usr/bin/env node
/**
 * timesfm-ts  Web (WASM) Benchmark Suite
 *
 * Benchmarks onnxruntime-web (WASM backend) in a Node.js environment.
 * Mirrors benchmark-ci.js but uses onnxruntime-web instead of onnxruntime-node
 * for a cross-runtime comparison (Node.js native vs WASM).
 *
 * Usage:
 *   node scripts/web-benchmark-ci.js                              # console output
 *   node scripts/web-benchmark-ci.js --json report.json           # JSON report
 *   node scripts/web-benchmark-ci.js --md report.md               # Markdown report
 *   node scripts/web-benchmark-ci.js --html report.html           # HTML report
 *   node scripts/web-benchmark-ci.js --all                        # JSON + Markdown + HTML
 *
 * Environment:
 *   TIMESFM_MODEL_PATH   — path to ONNX model (required)
 *   WEB_BENCH_ITERATIONS — number of warm inference iterations (default 5)
 *   WEB_BENCH_BATCH_SIZES — comma-separated batch sizes (default "1,2,4")
 *   WEB_BENCH_SKIP_ACCURACY — skip accuracy tests (faster)
 */

const { performance } = require('perf_hooks');
const os = require('os');
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

// ─── CLI ────────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);

function getArgValue(flag) {
  const idx = args.indexOf(flag);
  return idx >= 0 && idx + 1 < args.length ? args[idx + 1] : null;
}

function hasArg(flag) {
  return args.includes(flag);
}

const outJson = getArgValue('--json') || (hasArg('--json') ? 'web-benchmark-report.json' : null);
const outMd = getArgValue('--md') || (hasArg('--md') ? 'web-benchmark-report.md' : null);
const outHtml = getArgValue('--html') || (hasArg('--html') ? 'web-benchmark-report.html' : null);
const outAll = hasArg('--all');
const skipAccuracy = !!process.env.WEB_BENCH_SKIP_ACCURACY;

// Batch sizes (comma-separated or default — web is slower, so smaller range)
const batchSizes = (process.env.WEB_BENCH_BATCH_SIZES || '1,2,4')
  .split(',')
  .map(Number)
  .filter((n) => n > 0);

// ─── Model Path Resolution ──────────────────────────────────────────────────

function resolveModelPath() {
  if (process.env.TIMESFM_MODEL_PATH && fs.existsSync(process.env.TIMESFM_MODEL_PATH)) {
    return process.env.TIMESFM_MODEL_PATH;
  }
  const searchPaths = [
    path.join(__dirname, '..', 'models', 'timesfm-2.5.onnx'),
    path.join(__dirname, '..', '..', 'models', 'timesfm-2.5.onnx'),
    path.join(os.homedir(), '.cache', 'timesfm-ts', 'timesfm-2.5.onnx'),
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
    runtime: 'onnxruntime-web (WASM)',
  };
}

// ─── HTML Report Generator ──────────────────────────────────────────────────

function generateHtmlReport(report) {
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  const jsonEmbed = JSON.stringify(report);

  const latencyRows = report.latency
    .map(
      (l) => `
    <tr>
      <td>${l.context}</td>
      <td>${l.batch_size || 1}</td>
      <td>${l.patches}</td>
      <td><strong>${l.avg_ms}</strong></td>
      <td>${l.p50_ms}</td>
      <td>${l.p99_ms}</td>
      <td>${l.throughput_seq_s}</td>
      <td>${l.cold_start_ms != null ? l.cold_start_ms : '—'}</td>
    </tr>`,
    )
    .join('');

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TimesFM Web (WASM) Benchmark Report</title>
  <style>
    :root {
      --bg: #ffffff;
      --fg: #1a1a2e;
      --muted: #6b7280;
      --border: #e5e7eb;
      --th-bg: #f3f4f6;
      --hover: #f8fafc;
      --accent: #2563eb;
      --accent2: #1e40af;
      --pass: #16a34a;
      --fail: #dc2626;
      --code-bg: #1e293b;
      --code-fg: #e2e8f0;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #0f172a;
        --fg: #e2e8f0;
        --muted: #94a3b8;
        --border: #334155;
        --th-bg: #1e293b;
        --hover: #1e293b;
        --accent: #60a5fa;
        --accent2: #93c5fd;
        --pass: #4ade80;
        --fail: #f87171;
        --code-bg: #0f172a;
        --code-fg: #e2e8f0;
      }
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      max-width: 960px;
      margin: 2rem auto;
      padding: 0 1rem;
      line-height: 1.6;
      color: var(--fg);
      background: var(--bg);
    }
    h1 { border-bottom: 3px solid var(--accent); padding-bottom: 0.5rem; margin-bottom: 0.5rem; }
    h2 { margin-top: 2.5rem; color: var(--accent2); }
    table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }
    th, td { border: 1px solid var(--border); padding: 8px 12px; text-align: left; }
    th { background: var(--th-bg); font-weight: 600; }
    tr:hover { background: var(--hover); }
    .pass { color: var(--pass); font-weight: 600; }
    .fail { color: var(--fail); font-weight: 600; }
    .meta { color: var(--muted); font-size: 0.9rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin: 1rem 0; }
    .card { border: 1px solid var(--border); border-radius: 8px; padding: 1rem; }
    .card h3 { font-size: 0.85rem; text-transform: uppercase; color: var(--muted); margin-bottom: 0.5rem; }
    .card .value { font-size: 1.5rem; font-weight: 700; }
    footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); color: var(--muted); font-size: 0.85rem; text-align: center; }
    @media (max-width: 600px) {
      body { margin: 1rem auto; }
      table { font-size: 0.8rem; }
      th, td { padding: 6px 8px; }
    }
  </style>
</head>
<body>
  <h1>🌐 TimesFM 2.5 200M Web (WASM) Benchmark Report</h1>
  <p class="meta">
    Generated: ${esc(report.timestamp)} · Git: <code>${esc(report.git_sha)}</code> · ${esc(report.cpu_model)} × ${report.cpu_cores}
  </p>

  <h2>💻 System</h2>
  <div class="grid">
    <div class="card"><h3>CPU</h3><div class="value" style="font-size:1rem;">${esc(report.cpu_model)} × ${report.cpu_cores}</div></div>
    <div class="card"><h3>RAM</h3><div class="value">${report.total_ram_gb} GB</div></div>
    <div class="card"><h3>Platform</h3><div class="value" style="font-size:1rem;">${esc(report.platform)} / ${esc(report.arch)}</div></div>
    <div class="card"><h3>Node.js</h3><div class="value" style="font-size:1rem;">${esc(report.node_version)}</div></div>
  </div>

  <h2>🧠 Model</h2>
  <div class="grid">
    <div class="card"><h3>Runtime</h3><div class="value" style="font-size:1rem;">${esc(report.runtime)}</div></div>
    <div class="card"><h3>Size</h3><div class="value">${report.model.size_mb} MB</div></div>
    <div class="card"><h3>Load Time</h3><div class="value">${report.model.load_time_sec}s</div></div>
    <div class="card"><h3>Cold/Warm Ratio</h3><div class="value">${report.cold_warm_ratio != null ? report.cold_warm_ratio.toFixed(2) + '×' : '—'}</div></div>
  </div>

  <h2>⚡ Inference Latency (WASM)</h2>
  <table>
    <thead>
      <tr><th>Context</th><th>Batch</th><th>Patches</th><th>Avg (ms)</th><th>P50 (ms)</th><th>P99 (ms)</th><th>Throughput (seq/s)</th><th>Cold Start (ms)</th></tr>
    </thead>
    <tbody>${latencyRows}</tbody>
  </table>

  <h2>💾 Memory Footprint</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>
      <tr><td>RSS</td><td>${report.memory.rss_mb} MB</td></tr>
      <tr><td>Heap Used</td><td>${report.memory.heap_used_mb} MB</td></tr>
      <tr><td>Heap Total</td><td>${report.memory.heap_total_mb} MB</td></tr>
    </tbody>
  </table>

  <p>
    <a href="web-benchmark-report.json">📄 Raw JSON data</a> ·
    <a href="web-benchmark-report.md">📝 Markdown version</a>
  </p>

  <footer>
    Automated web benchmark by <strong>timesfm-ts</strong> CI · ${esc(report.timestamp)}
  </footer>

  <script type="application/json" id="benchmark-data">
${jsonEmbed}
  </script>
</body>
</html>`;
}

// ─── Markdown Report Generator ───────────────────────────────────────────────

function generateMarkdownReport(report) {
  const latencyTable = report.latency
    .map(
      (l) =>
        `| ${l.context} | ${l.batch_size || 1} | ${l.patches} | ${l.avg_ms} | ${l.p50_ms} | ${l.p99_ms} | ${l.throughput_seq_s} | ${l.cold_start_ms != null ? l.cold_start_ms : '—'} |`,
    )
    .join('\n');

  return `# TimesFM 2.5 200M Web (WASM) Benchmark Report

> Generated: ${report.timestamp} · Git: \`${report.git_sha}\` · Node: ${report.node_version}

## System

| Property | Value |
|----------|-------|
| CPU | ${report.cpu_model} × ${report.cpu_cores} |
| RAM | ${report.total_ram_gb} GB |
| Platform | ${report.platform} / ${report.arch} |
| Node.js | ${report.node_version} |
| Runtime | ${report.runtime} |

## Model

| Property | Value |
|----------|-------|
| Size | ${report.model.size_mb} MB |
| Load time | ${report.model.load_time_sec}s |
| Cold/Warm Ratio | ${report.cold_warm_ratio != null ? report.cold_warm_ratio.toFixed(2) + '×' : 'N/A'} |

## Inference Latency (WASM)

| Context | Batch | Patches | Avg (ms) | P50 (ms) | P99 (ms) | Throughput (seq/s) | Cold Start (ms) |
|---------|-------|---------|----------|----------|----------|---------------------|------------------|
${latencyTable}

## Memory

| Metric | Value |
|--------|-------|
| RSS | ${report.memory.rss_mb} MB |
| Heap Used | ${report.memory.heap_used_mb} MB |
| Heap Total | ${report.memory.heap_total_mb} MB |

---
*Automated web benchmark by timesfm-ts CI*
`;
}

// ─── WASM Path Setup ─────────────────────────────────────────────────────────

/**
 * Resolve the onnxruntime-web WASM binary path for Node.js.
 *
 * onnxruntime-web's WASM backend *requires* the .wasm files bundled
 * in its dist/ directory.  This function uses three strategies:
 *
 *   1. `createRequire` from the *workspace root* package.json
 *      to resolve the exact dist/ location (handles pnpm symlinks).
 *   2. Glob-search the pnpm store directory for wasm files
 *      e.g. \`node_modules/.pnpm/onnxruntime-web@1.x/.../dist/\`
 *      for the raw .wasm files (robust pnpm fallback).
 *   3. Direct path check at `node_modules/onnxruntime-web/dist/`
 *      for non-pnpm environments (npm, yarn, etc.).
 *
 * NOTE: onnxruntime-web ≥ 1.20 uses `ort-wasm-simd-threaded.wasm`.
 * Earlier versions use `ort-wasm-simd.wasm`. We check for both.
 */
function resolveWasmPath() {
  const { createRequire } = require('node:module');
  const rootDir = path.resolve(__dirname, '..');

  // Canonical markers — any of these .wasm files confirm a valid dist/
  const WASM_MARKERS = ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd.wasm'];

  function hasWasm(dir) {
    for (const marker of WASM_MARKERS) {
      if (fs.existsSync(path.join(dir, marker))) return true;
    }
    return false;
  }

  // Strategy 1: Use createRequire from the workspace root package.json.
  // This forces resolution from `@agentix-e/timesfm-web`'s perspective,
  // correct even when tsx runs the script from a different CWD/context.
  try {
    const rootPkg = path.join(rootDir, 'package.json');
    if (fs.existsSync(rootPkg)) {
      const req = createRequire(rootPkg);
      const resolved = req.resolve('onnxruntime-web');
      const baseDir = path.dirname(resolved);

      // The resolved file is typically `dist/ort.node.min.mjs` or `dist/ort.node.min.js`.
      // The WASM files live alongside it in the same dist/ directory.
      if (hasWasm(baseDir)) {
        return baseDir + '/';
      }

      // If resolution returned something else (e.g. `lib/index.js` in older
      // versions), walk up to find the real dist/.
      let cur = baseDir;
      for (let i = 0; i < 3; i++) {
        cur = path.dirname(cur);
        const candidate = path.join(cur, 'dist');
        if (hasWasm(candidate)) {
          return candidate + '/';
        }
        // Stop if we've reached filesystem root or the dist directory itself
        if (cur === '/' || path.basename(cur) === 'dist') break;
      }
    }
  } catch {
    // createRequire failed — continue to fallback
  }

  // Strategy 2: pnpm store layout — search `node_modules/.pnpm/onnxruntime-web@*`
  const pnpmStore = path.join(rootDir, 'node_modules', '.pnpm');
  if (fs.existsSync(pnpmStore)) {
    try {
      const entries = fs.readdirSync(pnpmStore);
      for (const entry of entries) {
        if (entry.startsWith('onnxruntime-web@')) {
          const distDir = path.join(pnpmStore, entry, 'node_modules', 'onnxruntime-web', 'dist');
          if (hasWasm(distDir)) {
            return distDir + '/';
          }
        }
      }
    } catch {
      // continue
    }
  }

  // Strategy 3: Plain node_modules (npm / yarn / non-pnpm)
  const plainDist = path.join(rootDir, 'node_modules', 'onnxruntime-web', 'dist');
  if (hasWasm(plainDist)) {
    return plainDist + '/';
  }

  return null;
}

// ─── Main ────────────────────────────────────────────────────────────────────

async function main() {
  const sysInfo = getSystemInfo();
  const report = {
    ...sysInfo,
    model: {},
    latency: [],
    memory: {},
    cold_warm_ratio: null,
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

  console.log('TIMESFM WEB (WASM) BENCHMARK SUITE');
  console.log('='.repeat(70));
  console.log(`  Model:      ${modelPath} (${report.model.size_mb} MB)`);
  console.log(`  CPU:        ${sysInfo.cpu_model} x ${sysInfo.cpu_cores}`);
  console.log(`  RAM:        ${sysInfo.total_ram_gb} GB`);
  console.log(`  Node:       ${sysInfo.node_version}`);
  console.log(`  Runtime:    onnxruntime-web (WASM)`);
  console.log(`  Git:        ${sysInfo.git_sha} (${sysInfo.git_branch})`);
  console.log(`  Batch sizes: [${batchSizes.join(', ')}]`);
  console.log(`  Iterations: ${parseInt(process.env.WEB_BENCH_ITERATIONS || '5', 10)}`);
  console.log('='.repeat(70));

  // ── Configure WASM path ────────────────────────────────────────────────────
  const wasmPath = resolveWasmPath();
  if (!wasmPath) {
    console.error(
      'Cannot find onnxruntime-web WASM binaries. Install onnxruntime-web:\n' +
        '  pnpm add -D onnxruntime-web',
    );
    process.exit(1);
  }
  console.log(`\n  WASM path: ${wasmPath}`);

  // ── Load onnxruntime-web ───────────────────────────────────────────────────
  //
  // onnxruntime-web is ESM-only and lives inside the pnpm store as a
  // devDependency of @agentix-e/timesfm-web.  Bare `import('onnxruntime-web')`
  // fails from the root workspace — we resolve the full path to the ESM entry
  // point from the dist/ directory we already located.
  const ortEntry = path.join(wasmPath.replace(/\/$/, ''), 'ort.node.min.mjs');
  if (!fs.existsSync(ortEntry)) {
    console.error(`onnxruntime-web ESM entry not found: ${ortEntry}`);
    process.exit(1);
  }

  const ort = await import(require('url').pathToFileURL(ortEntry).href);

  // Configure WASM environment before any session creation
  ort.env.wasm.wasmPaths = wasmPath;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;

  // ── Load ONNX session ──────────────────────────────────────────────────────
  const loadStart = performance.now();
  const modelBuffer = fs.readFileSync(modelPath).buffer;
  const sessionOptions = {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
    enableCpuMemArena: true,
    enableMemPattern: true,
  };
  const session = await ort.InferenceSession.create(modelBuffer, sessionOptions);
  let sessionReleased = false;
  report.model.load_time_sec = +((performance.now() - loadStart) / 1000).toFixed(2);
  console.log(`\n  Load time: ${report.model.load_time_sec}s`);

  // ── Latency Benchmark ──────────────────────────────────────────────────────
  const MODEL_PATCHES = 16;
  const dim = 64;
  const iterations = parseInt(process.env.WEB_BENCH_ITERATIONS || '5', 10);

  console.log('\n  ── Latency ──');

  const contextConfigs = [
    { patches: 4, context: 128 },
    { patches: 8, context: 256 },
    { patches: 16, context: 512 },
  ];

  let firstColdStartMs = null;

  for (const cfg of contextConfigs) {
    for (const bs of batchSizes) {
      const label = `ctx=${String(cfg.context).padStart(3)}  batch=${bs}`;

      // Build input tensor: [1, MODEL_PATCHES, dim]
      const input = new Float32Array(1 * MODEL_PATCHES * dim);

      // Fill active patches with random data
      for (let p = 0; p < cfg.patches; p++) {
        const bp = p * dim;
        for (let i = 0; i < 32; i++) {
          input[bp + i] = Math.random();
          input[bp + 32 + i] = 0; // padding flag for active patches
        }
      }
      // Fill padding patches with mask=1 (fully masked)
      for (let p = cfg.patches; p < MODEL_PATCHES; p++) {
        const bp = p * dim;
        for (let i = 0; i < 32; i++) {
          input[bp + 32 + i] = 1; // mask flag for padding
        }
      }

      const inputName = session.inputNames[0] || 'inputs';
      const feeds = { [inputName]: new ort.Tensor('float32', input, [1, MODEL_PATCHES, dim]) };

      // Cold start
      const coldStart = performance.now();
      await session.run(feeds);
      const coldStartMs = +(performance.now() - coldStart).toFixed(1);

      if (firstColdStartMs === null) {
        firstColdStartMs = coldStartMs;
      }

      // Warmup: 2 passes
      for (let i = 0; i < 2; i++) await session.run(feeds);

      // Measure warm inference
      if (global.gc) global.gc();
      const heapBefore = process.memoryUsage().heapUsed;

      const times = [];
      for (let i = 0; i < iterations; i++) {
        const start = performance.now();
        for (let b = 0; b < bs; b++) {
          await session.run(feeds);
        }
        times.push(performance.now() - start);
      }

      const heapAfter = process.memoryUsage().heapUsed;
      const heapDeltaMb = +((heapAfter - heapBefore) / 1024 / 1024).toFixed(2);

      const avg = times.reduce((a, b) => a + b, 0) / times.length;
      const p50 = [...times].sort((a, b) => a - b)[Math.floor(times.length / 2)];
      const p99 = [...times].sort((a, b) => a - b)[Math.floor(times.length * 0.99)];

      report.latency.push({
        context: cfg.context,
        batch_size: bs,
        patches: cfg.patches,
        avg_ms: +avg.toFixed(1),
        p50_ms: +p50.toFixed(1),
        p99_ms: +p99.toFixed(1),
        throughput_seq_s: +(1000 / avg).toFixed(1),
        cold_start_ms: coldStartMs,
        heap_used_mb: heapDeltaMb > 0 ? heapDeltaMb : 0,
      });

      console.log(
        `  ${label.padEnd(20)}  avg=${avg.toFixed(0).padStart(5)}ms  cold=${coldStartMs.toFixed(0).padStart(5)}ms  p50=${p50.toFixed(0).padStart(5)}ms  p99=${p99.toFixed(0).padStart(5)}ms  thr=${(1000 / avg).toFixed(1)} seq/s`,
      );
    }
  }

  // Compute cold/warm ratio
  const warmAvgs = report.latency.filter((l) => l.batch_size === 1).map((l) => l.avg_ms);
  if (firstColdStartMs !== null && warmAvgs.length > 0) {
    const avgWarm = warmAvgs.reduce((a, b) => a + b, 0) / warmAvgs.length;
    report.cold_warm_ratio = +(firstColdStartMs / avgWarm).toFixed(2);
    console.log(
      `\n  Cold/Warm ratio: ${report.cold_warm_ratio}× (cold=${firstColdStartMs.toFixed(0)}ms, avg warm=${avgWarm.toFixed(0)}ms)`,
    );
  }

  // ── Stability (Memory Leak Detection) ─────────────────────────────────────
  {
    console.log('\n  ── Stability (memory leak check) ──');

    const STABILITY_ITERS = parseInt(process.env.WEB_BENCH_STABILITY_ITERS || '50', 10);
    const modelPatches = 16;
    const dim2 = 64;
    const input = new Float32Array(1 * modelPatches * dim2);
    for (let p = 0; p < modelPatches; p++) {
      const bp = p * dim2;
      for (let i = 0; i < 32; i++) {
        input[bp + i] = Math.random();
        input[bp + 32 + i] = p >= 8 ? 1 : 0;
      }
    }
    const inputName2 = session.inputNames[0] || 'inputs';
    const feeds = {
      [inputName2]: new ort.Tensor('float32', input, [1, modelPatches, dim2]),
    };

    // 3 warmup iterations
    for (let i = 0; i < 3; i++) await session.run(feeds);

    if (global.gc) global.gc();
    const heapBaseline = process.memoryUsage().heapUsed;

    const snapshots = [];
    for (let i = 0; i < STABILITY_ITERS; i++) {
      await session.run(feeds);
      if (i % 10 === 9 || i === 0) {
        if (global.gc) global.gc();
        const heap = process.memoryUsage().heapUsed;
        const deltaMb = +((heap - heapBaseline) / 1024 / 1024).toFixed(2);
        const pct = +((deltaMb / (heapBaseline / 1024 / 1024)) * 100).toFixed(1);
        snapshots.push({
          iter: i + 1,
          heap_mb: +(heap / 1024 / 1024).toFixed(1),
          delta_mb: deltaMb,
          delta_pct: pct,
        });
      }
    }

    const lastDelta = snapshots[snapshots.length - 1].delta_pct;
    const stable = Math.abs(lastDelta) <= 5;
    report.stability = {
      iterations: STABILITY_ITERS,
      baseline_heap_mb: +(heapBaseline / 1024 / 1024).toFixed(1),
      final_delta_pct: lastDelta,
      stable,
      snapshots,
    };

    console.log(
      `  Baseline heap: ${report.stability.baseline_heap_mb} MB · ${STABILITY_ITERS} iters · ` +
        `Final Δ: ${lastDelta > 0 ? '+' : ''}${lastDelta}% ${stable ? '✅ stable' : '⚠️ growth'}`,
    );
  }

  // ── Accuracy Benchmark (full TimesFM pipeline via web engine) ─────────────
  if (!skipAccuracy) {
    console.log('\n  ── Accuracy (full pipeline, web engine) ──');

    // Release raw session before loading full model to save memory
    try {
      session.release();
    } catch {
      // Already disposed
    }
    sessionReleased = true;

    // Import from TypeScript source via tsx
    const coreMod = await import(
      path.join(__dirname, '..', 'packages', 'timesfm-core', 'src', 'index.ts')
    );
    const core = coreMod.default || coreMod;
    const { TimesFMModel, createForecastConfig, TIMESFM_25_CONFIG } = core;

    // Import web engine
    const webMod = await import(
      path.join(__dirname, '..', 'packages', 'timesfm-web', 'src', 'index.ts')
    );
    const webExports = webMod.default || webMod;
    const { TimesFMWebInferenceEngine } = webExports;

    // Import test fixtures
    const fixturesMod = await import(
      path.join(__dirname, '..', 'packages', 'timesfm-core', 'test', 'test-fixtures.ts')
    );
    const { businessMetric, stockPrice, hourlyTemp, eCommerce, regimeShift } =
      typeof fixturesMod.default === 'object' && fixturesMod.default !== null
        ? fixturesMod.default
        : fixturesMod;

    const horizon = 12;
    const seriesLen = 200;
    const seriesFixtures = [
      businessMetric(seriesLen),
      stockPrice(seriesLen),
      hourlyTemp(seriesLen),
      eCommerce(seriesLen),
      regimeShift(seriesLen),
    ];

    const naiveMAEs = [];
    const modelMAEs = [];
    const modelRMSEs = [];
    const constMAEs = [];

    // Create web engine with WASM backend
    const webEngine = new TimesFMWebInferenceEngine(TIMESFM_25_CONFIG, ['wasm']);
    webEngine.setWasmPath(wasmPath);

    const accModel = await TimesFMModel.fromPretrained({
      modelPath,
      engine: webEngine,
    });

    accModel.compile(
      createForecastConfig({
        maxContext: 512,
        maxHorizon: 128,
        normalizeInputs: true,
        useContinuousQuantileHead: true,
        forceFlipInvariance: true,
        inferIsPositive: false,
        fixQuantileCrossing: true,
      }),
    );

    for (const data of seriesFixtures) {
      const context = data.slice(0, seriesLen - horizon);
      const actual = data.slice(seriesLen - horizon);

      // Naive (no-change) baseline
      const lastVal = context[context.length - 1];
      let naiveMae = 0;
      for (let h = 0; h < horizon; h++) naiveMae += Math.abs(actual[h] - lastVal);
      naiveMae /= horizon;
      naiveMAEs.push(naiveMae);

      // Full TimesFM pipeline forecast via web engine
      const { pointForecast } = await accModel.forecast(horizon, [context]);
      const pred = pointForecast[0];

      let maeVal = 0,
        rmseVal = 0;
      for (let h = 0; h < Math.min(horizon, pred.length); h++) {
        const diff = actual[h] - pred[h];
        maeVal += Math.abs(diff);
        rmseVal += diff * diff;
      }
      maeVal /= Math.min(horizon, pred.length);
      rmseVal = Math.sqrt(rmseVal / Math.min(horizon, pred.length));
      modelMAEs.push(maeVal);
      modelRMSEs.push(rmseVal);

      // Constant mean baseline
      const constMean = context.reduce((a, b) => a + b, 0) / context.length;
      let constMae = 0;
      for (let h = 0; h < horizon; h++) constMae += Math.abs(actual[h] - constMean);
      constMae /= horizon;
      constMAEs.push(constMae);
    }

    await accModel.dispose();

    const avgNaiveMAE = naiveMAEs.reduce((a, b) => a + b, 0) / naiveMAEs.length;
    const avgModelMAE = modelMAEs.reduce((a, b) => a + b, 0) / modelMAEs.length;
    const avgModelRMSE = modelRMSEs.reduce((a, b) => a + b, 0) / modelRMSEs.length;
    const avgConstMAE = constMAEs.reduce((a, b) => a + b, 0) / constMAEs.length;
    const scaledMAE = avgModelMAE / avgNaiveMAE;
    const improMae = avgNaiveMAE > 0 ? ((avgNaiveMAE - avgModelMAE) / avgNaiveMAE) * 100 : 0;
    const improConst = avgConstMAE > 0 ? ((avgConstMAE - avgModelMAE) / avgConstMAE) * 100 : 0;

    report.accuracy = {
      naive_mae: +avgNaiveMAE.toFixed(4),
      model_mae: +avgModelMAE.toFixed(4),
      model_rmse: +avgModelRMSE.toFixed(4),
      const_mae: +avgConstMAE.toFixed(4),
      scaled_mae: +scaledMAE.toFixed(4),
      better_than_naive: scaledMAE < 1,
      improvement_vs_naive_pct: +improMae.toFixed(1),
      improvement_vs_const_pct: +improConst.toFixed(1),
    };

    console.log(`  Fixtures: businessMetric, stockPrice, hourlyTemp, eCommerce, regimeShift`);
    console.log(`  Naive MAE (no-change):    ${avgNaiveMAE.toFixed(4)}`);
    console.log(`  Const Mean MAE:            ${avgConstMAE.toFixed(4)}`);
    console.log(`  TimesFM MAE (web engine):  ${avgModelMAE.toFixed(4)}`);
    console.log(`  TimesFM RMSE:              ${avgModelRMSE.toFixed(4)}`);
    console.log(
      `  Scaled MAE vs Naive:       ${scaledMAE.toFixed(4)}  (${scaledMAE < 1 ? '✅' : '⚠️'})`,
    );
    console.log(`  Improvement vs Naive:      ${improMae.toFixed(1)}%`);
    console.log(`  Improvement vs Const Mean: ${improConst.toFixed(1)}%`);
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
  if (!sessionReleased && session && typeof session.release === 'function') {
    try {
      session.release();
    } catch {
      // Already disposed
    }
  }

  // ── Output Reports ─────────────────────────────────────────────────────────
  const doJson = outJson || outAll;
  const doMd = outMd || outAll;
  const doHtml = outHtml || outAll;
  const jsonPath = outJson && typeof outJson === 'string' ? outJson : 'web-benchmark-report.json';
  const mdPath = outMd && typeof outMd === 'string' ? outMd : 'web-benchmark-report.md';
  const htmlPath = outHtml && typeof outHtml === 'string' ? outHtml : 'web-benchmark-report.html';

  if (doJson && jsonPath) {
    fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2));
    console.log(`\n  ✅ JSON report written: ${jsonPath}`);
  }

  if (doMd && mdPath) {
    const md = generateMarkdownReport(report);
    fs.writeFileSync(mdPath, md);
    console.log(`  ✅ Markdown report written: ${mdPath}`);
  }

  if (doHtml && htmlPath) {
    const html = generateHtmlReport(report);
    fs.writeFileSync(htmlPath, html);
    console.log(`  ✅ HTML report written: ${htmlPath}`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
