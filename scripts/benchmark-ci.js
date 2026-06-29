#!/usr/bin/env node
/**
 * agentix-timesfm-ts  Benchmark Suite with JSON / Markdown / HTML Reports
 *
 * Usage:
 *   node scripts/benchmark-ci.js                              # console output
 *   node scripts/benchmark-ci.js --json report.json            # JSON report
 *   node scripts/benchmark-ci.js --md report.md               # Markdown report
 *   node scripts/benchmark-ci.js --html report.html           # HTML report
 *   node scripts/benchmark-ci.js --all                        # JSON + Markdown + HTML
 *   node scripts/benchmark-ci.js --check-regression            # Regression check
 *
 * Environment:
 *   TIMESFM_MODEL_PATH   — path to ONNX model (required)
 *   BENCH_SKIP_ACCURACY  — skip accuracy tests (faster)
 *   BENCH_ITERATIONS     — number of warm inference iterations (default 5)
 *   BENCH_BATCH_SIZES    — comma-separated batch sizes (default "1,2,4,8")
 *
 * Regression detection:
 *   --baseline <path>         — baseline JSON to compare against
 *   --regression-threshold N  — % slowdown threshold for warning (default 10)
 *   --check-regression        — run only regression check (requires --current + --baseline)
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

const outJson = getArgValue('--json') || (hasArg('--json') ? 'benchmark-report.json' : null);
const outMd = getArgValue('--md') || (hasArg('--md') ? 'benchmark-report.md' : null);
const outHtml = getArgValue('--html') || (hasArg('--html') ? 'benchmark-report.html' : null);
const outAll = hasArg('--all');
const skipAccuracy = !!process.env.BENCH_SKIP_ACCURACY;

// Regression mode
const checkRegressionOnly = hasArg('--check-regression');
const baselinePath = getArgValue('--baseline');
const currentPath = getArgValue('--current');
const regressionThreshold = parseInt(getArgValue('--regression-threshold') || '10', 10);

// Batch sizes (comma-separated or default)
const batchSizes = (process.env.BENCH_BATCH_SIZES || '1,2,4,8')
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

// ─── Regression Check ───────────────────────────────────────────────────────

/**
 * Compare current benchmark against baseline and detect regressions.
 *
 * Returns an object with:
 *   - has_regression: boolean
 *   - has_critical:   boolean (any > threshold*1.5)
 *   - comparisons:     array of {context, batch_size, current_ms, baseline_ms, delta_pct, level}
 *   - summary:         human-readable summary string
 */
function checkRegression(current, baseline, threshold) {
  if (!baseline || !baseline.latency || baseline.latency.length === 0) {
    return {
      has_regression: false,
      has_critical: false,
      comparisons: [],
      summary: 'No baseline data available — skipping regression check.',
    };
  }

  const baselineMap = new Map();
  for (const b of baseline.latency) {
    const key = `${b.context}_${b.batch_size || 1}`;
    baselineMap.set(key, b.avg_ms);
  }

  const comparisons = [];
  let hasRegression = false;
  let hasCritical = false;

  for (const c of current.latency) {
    const key = `${c.context}_${c.batch_size || 1}`;
    const baselineAvg = baselineMap.get(key);
    if (baselineAvg === undefined || baselineAvg === 0) continue;

    const deltaPct = +(((c.avg_ms - baselineAvg) / baselineAvg) * 100).toFixed(1);
    let level = 'ok';
    if (deltaPct > threshold * 1.5) {
      level = 'critical';
      hasCritical = true;
      hasRegression = true;
    } else if (deltaPct > threshold) {
      level = 'warning';
      hasRegression = true;
    } else if (deltaPct > threshold / 2) {
      level = 'notice';
    }

    comparisons.push({
      context: c.context,
      batch_size: c.batch_size || 1,
      current_ms: c.avg_ms,
      baseline_ms: baselineAvg,
      delta_pct: deltaPct,
      level,
    });
  }

  const criticals = comparisons.filter((c) => c.level === 'critical');
  const warnings = comparisons.filter((c) => c.level === 'warning');
  let summary = '';
  if (criticals.length > 0) {
    summary += `❌ CRITICAL: ${criticals.length} data point(s) exceed ${threshold * 1.5}% slowdown. `;
  }
  if (warnings.length > 0) {
    summary += `⚠️ WARNING: ${warnings.length} data point(s) exceed ${threshold}% slowdown. `;
  }
  if (!hasRegression) {
    summary = '✅ No performance regression detected.';
  }

  return { has_regression: hasRegression, has_critical: hasCritical, comparisons, summary };
}

// ─── HTML Report Generator ──────────────────────────────────────────────────

/**
 * Generate a self-contained HTML benchmark report.
 *
 * Uses template literals (no external dependencies) with:
 * - Dark/light theme support via prefers-color-scheme
 * - Responsive layout for mobile
 * - Embedded JSON data for programmatic consumption
 * - Regression comparison section (if available)
 */
function generateHtmlReport(report, regression) {
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  const jsonEmbed = JSON.stringify(report);

  // Build latency table rows (grouped by batch_size)
  const batchSizesInReport = [...new Set(report.latency.map((l) => l.batch_size || 1))].sort(
    (a, b) => a - b,
  );
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

  // Memory section
  const perConfigMemory = report.latency.some((l) => l.heap_used_mb != null);

  // Regression comparison section
  let regressionSection = '';
  if (regression && regression.comparisons && regression.comparisons.length > 0) {
    const regRows = regression.comparisons
      .map((c) => {
        const cls =
          c.level === 'critical'
            ? 'reg-critical'
            : c.level === 'warning'
              ? 'reg-warning'
              : c.level === 'notice'
                ? 'reg-notice'
                : '';
        const icon =
          c.level === 'critical'
            ? '❌'
            : c.level === 'warning'
              ? '⚠️'
              : c.level === 'notice'
                ? 'ℹ️'
                : '✅';
        return `
    <tr class="${cls}">
      <td>${c.context}</td>
      <td>${c.batch_size}</td>
      <td>${c.current_ms}</td>
      <td>${c.baseline_ms}</td>
      <td class="${cls}">${icon} ${c.delta_pct > 0 ? '+' : ''}${c.delta_pct}%</td>
    </tr>`;
      })
      .join('');
    regressionSection = `
    <h2>📈 Regression vs Baseline</h2>
    ${regression.baseline_info ? `<p class="meta">Baseline: ${esc(regression.baseline_info)}</p>` : ''}
    <p class="meta">${esc(regression.summary)} · Threshold: ${regressionThreshold}%</p>
    <table>
      <thead><tr><th>Context</th><th>Batch</th><th>Current (ms)</th><th>Baseline (ms)</th><th>Delta</th></tr></thead>
      <tbody>${regRows}</tbody>
    </table>`;
  }

  // Accuracy section
  let accuracySection = '';
  if (report.accuracy) {
    accuracySection = `
    <h2>🎯 Prediction Accuracy</h2>
    <p class="meta">Full TimesFM pipeline with RevIN normalization, flip invariance, continuous quantile head</p>
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td>Naive MAE (no-change)</td><td>${report.accuracy.naive_mae}</td></tr>
        <tr><td>Constant Mean MAE</td><td>${report.accuracy.const_mae}</td></tr>
        <tr><td><strong>TimesFM MAE</strong></td><td><strong>${report.accuracy.model_mae}</strong></td></tr>
        <tr><td>TimesFM RMSE</td><td>${report.accuracy.model_rmse}</td></tr>
        <tr><td>Scaled MAE vs Naive</td><td>${report.accuracy.scaled_mae} ${report.accuracy.better_than_naive ? '<span class="pass">✅ Better</span>' : '<span class="fail">⚠️</span>'}</td></tr>
        <tr><td>Improvement vs Naive</td><td><strong>${report.accuracy.improvement_vs_naive_pct}%</strong></td></tr>
        <tr><td>Improvement vs Const Mean</td><td><strong>${report.accuracy.improvement_vs_const_pct}%</strong></td></tr>
      </tbody>
    </table>`;
  }

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TimesFM Benchmark Report</title>
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
      --warning: #d97706;
      --code-bg: #1e293b;
      --code-fg: #e2e8f0;
      --crit-bg: #fef2f2;
      --warn-bg: #fffbeb;
      --notice-bg: #eff6ff;
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
        --warning: #fbbf24;
        --code-bg: #0f172a;
        --code-fg: #e2e8f0;
        --crit-bg: #450a0a;
        --warn-bg: #451a03;
        --notice-bg: #172554;
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
    pre { background: var(--code-bg); color: var(--code-fg); padding: 1rem; border-radius: 6px; overflow-x: auto; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
    .badge-pass { background: var(--pass); color: #fff; }
    .badge-fail { background: var(--fail); color: #fff; }
    .reg-critical { background: var(--crit-bg); }
    .reg-warning { background: var(--warn-bg); }
    .reg-notice { background: var(--notice-bg); }
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
  <h1>📊 TimesFM 2.5 200M Benchmark Report</h1>
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
    <div class="card"><h3>Size</h3><div class="value">${report.model.size_mb} MB</div></div>
    <div class="card"><h3>Load Time</h3><div class="value">${report.model.load_time_sec}s</div></div>
    <div class="card"><h3>ONNX Runtime</h3><div class="value" style="font-size:1rem;">${esc(report.onnx_runtime_version)}</div></div>
    <div class="card"><h3>Cold/Warm Ratio</h3><div class="value">${report.cold_warm_ratio != null ? report.cold_warm_ratio.toFixed(2) + '×' : '—'}</div></div>
  </div>

  <h2>⚡ Inference Latency</h2>
  <table>
    <thead>
      <tr><th>Context</th><th>Batch</th><th>Patches</th><th>Avg (ms)</th><th>P50 (ms)</th><th>P99 (ms)</th><th>Throughput (seq/s)</th><th>Cold Start (ms)</th></tr>
    </thead>
    <tbody>${latencyRows}</tbody>
  </table>

  ${regressionSection}
  ${accuracySection}

  <h2>💾 Memory Footprint</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>
      <tr><td>RSS</td><td>${report.memory.rss_mb} MB</td></tr>
      <tr><td>Heap Used</td><td>${report.memory.heap_used_mb} MB</td></tr>
      <tr><td>Heap Total</td><td>${report.memory.heap_total_mb} MB</td></tr>
    </tbody>
  </table>

  ${
    report.stability
      ? `
  <h2>🔬 Memory Stability (${report.stability.iterations} iters)</h2>
  <p class="meta">
    Baseline heap: ${report.stability.baseline_heap_mb} MB ·
    Final &Delta;: ${report.stability.final_delta_pct > 0 ? '+' : ''}${report.stability.final_delta_pct}% ·
    ${
      report.stability.stable
        ? '<span class="pass">✅ Stable</span>'
        : '<span class="fail">⚠️ Growth detected</span>'
    }
  </p>
  <table>
    <thead><tr><th>Iteration</th><th>Heap (MB)</th><th>Delta (MB)</th><th>Delta %</th></tr></thead>
    <tbody>${report.stability.snapshots
      .map(
        (s) => `
      <tr>
        <td>${s.iter}</td>
        <td>${s.heap_mb}</td>
        <td>${s.delta_mb > 0 ? '+' : ''}${s.delta_mb}</td>
        <td>${s.delta_pct > 0 ? '<span class="warn">+' : ''}${s.delta_pct}%${s.delta_pct > 0 ? '</span>' : ''}</td>
      </tr>`,
      )
      .join('')}
    </tbody>
  </table>`
      : ''
  }

  <p>
    <a href="benchmark-report.json">📄 Raw JSON data</a> ·
    <a href="benchmark-report.md">📝 Markdown version</a>
  </p>

  <footer>
    Automated benchmark by <strong>agentix-timesfm-ts</strong> CI · ${esc(report.timestamp)}
  </footer>

  <!-- Embedded JSON for programmatic access -->
  <script type="application/json" id="benchmark-data">
${jsonEmbed}
  </script>
</body>
</html>`;
}

// ─── Markdown Report Generator ───────────────────────────────────────────────

function generateMarkdownReport(report, regression) {
  // Latency table
  const latencyTable = report.latency
    .map(
      (l) =>
        `| ${l.context} | ${l.batch_size || 1} | ${l.patches} | ${l.avg_ms} | ${l.p50_ms} | ${l.p99_ms} | ${l.throughput_seq_s} | ${l.cold_start_ms != null ? l.cold_start_ms : '—'} |`,
    )
    .join('\n');

  // Regression section
  let regSection = '';
  if (regression && regression.comparisons && regression.comparisons.length > 0) {
    const regRows = regression.comparisons
      .map((c) => {
        const icon =
          c.level === 'critical'
            ? '❌'
            : c.level === 'warning'
              ? '⚠️'
              : c.level === 'notice'
                ? 'ℹ️'
                : '✅';
        return `| ${c.context} | ${c.batch_size} | ${c.current_ms} | ${c.baseline_ms} | ${icon} ${c.delta_pct > 0 ? '+' : ''}${c.delta_pct}% |`;
      })
      .join('\n');
    regSection = `
## Regression vs Baseline

> ${regression.summary} · Threshold: ${regressionThreshold}%

| Context | Batch | Current (ms) | Baseline (ms) | Delta |
|---------|-------|-------------|-------------|-------|
${regRows}
`;
  }

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
| Cold/Warm Ratio | ${report.cold_warm_ratio != null ? report.cold_warm_ratio.toFixed(2) + '×' : 'N/A'} |

## Inference Latency

| Context | Batch | Patches | Avg (ms) | P50 (ms) | P99 (ms) | Throughput (seq/s) | Cold Start (ms) |
|---------|-------|---------|----------|----------|----------|---------------------|------------------|
${latencyTable}

## Memory

| Metric | Value |
|--------|-------|
| RSS | ${report.memory.rss_mb} MB |
| Heap Used | ${report.memory.heap_used_mb} MB |
| Heap Total | ${report.memory.heap_total_mb} MB |

${regSection}${
    report.accuracy
      ? `
## Prediction Accuracy (full TimesFM pipeline)

> Uses the complete TimesFM pipeline with RevIN normalization, flip invariance,
> and continuous quantile head — the same path as production model.forecast().

| Metric | Value |
|--------|-------|
| Naive MAE (no-change) | ${report.accuracy.naive_mae} |
| Constant Mean MAE | ${report.accuracy.const_mae} |
| **TimesFM MAE** | **${report.accuracy.model_mae}** |
| TimesFM RMSE | ${report.accuracy.model_rmse} |
| Scaled MAE vs Naive | ${report.accuracy.scaled_mae} ${report.accuracy.better_than_naive ? '✅' : '⚠️'} |
| Improvement vs Naive | ${report.accuracy.improvement_vs_naive_pct}% |
| Improvement vs Const Mean | ${report.accuracy.improvement_vs_const_pct}% |
`
      : ''
  }---
*Automated benchmark by agentix-timesfm-ts CI*
`;
}

// ─── Main ────────────────────────────────────────────────────────────────────

async function main() {
  // ── Regression-only mode ───────────────────────────────────────────────────
  if (checkRegressionOnly) {
    if (!currentPath) {
      console.error('--check-regression requires --current <path>');
      process.exit(1);
    }
    const baseline =
      baselinePath && fs.existsSync(baselinePath)
        ? JSON.parse(fs.readFileSync(baselinePath, 'utf-8'))
        : null;
    const current = JSON.parse(fs.readFileSync(currentPath, 'utf-8'));

    if (!baseline || Object.keys(baseline).length === 0) {
      console.log('No baseline data — skipping regression check.');
      process.exit(0);
    }

    const regression = checkRegression(current, baseline, regressionThreshold);
    console.log(regression.summary);
    if (regression.comparisons.length > 0) {
      console.log('');
      for (const c of regression.comparisons) {
        const icon =
          c.level === 'critical'
            ? '❌'
            : c.level === 'warning'
              ? '⚠️'
              : c.level === 'notice'
                ? 'ℹ️'
                : '✅';
        console.log(
          `  ${icon} ctx=${c.context} batch=${c.batch_size}: ${c.current_ms}ms vs ${c.baseline_ms}ms (${c.delta_pct > 0 ? '+' : ''}${c.delta_pct}%)`,
        );
      }
    }
    process.exit(regression.has_critical ? 1 : 0);
  }

  // ── Full benchmark run ─────────────────────────────────────────────────────
  const sysInfo = getSystemInfo();
  const report = {
    ...sysInfo,
    model: {},
    latency: [],
    accuracy: null,
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

  console.log('TIMESFM BENCHMARK SUITE');
  console.log('='.repeat(70));
  console.log(`  Model:      ${modelPath} (${report.model.size_mb} MB)`);
  console.log(`  CPU:        ${sysInfo.cpu_model} x ${sysInfo.cpu_cores}`);
  console.log(`  RAM:        ${sysInfo.total_ram_gb} GB`);
  console.log(`  Node:       ${sysInfo.node_version}`);
  console.log(`  Git:        ${sysInfo.git_sha} (${sysInfo.git_branch})`);
  console.log(`  Batch sizes: [${batchSizes.join(', ')}]`);
  console.log(`  Iterations: ${parseInt(process.env.BENCH_ITERATIONS || '5', 10)}`);
  console.log('='.repeat(70));

  // ── Load ONNX Runtime (resolve from pnpm workspace if needed) ─────────────
  const childProcess = require('child_process');
  let ort;
  try {
    ort = require('onnxruntime-node');
  } catch {
    const rootDir = path.resolve(__dirname, '..');
    try {
      const modPath = childProcess
        .execSync(
          'find node_modules -name "onnxruntime-node" -type d -path "*/node_modules/onnxruntime-node" 2>/dev/null | head -1',
          { cwd: rootDir, encoding: 'utf-8', shell: true },
        )
        .trim();
      if (modPath) {
        ort = require(path.join(rootDir, modPath));
      }
    } catch {
      // fall through
    }
  }
  if (!ort) {
    console.error('Cannot find onnxruntime-node. Install it first: pnpm install');
    process.exit(1);
  }

  const loadStart = performance.now();
  const session = await ort.InferenceSession.create(modelPath);
  let sessionReleased = false;
  report.model.load_time_sec = +((performance.now() - loadStart) / 1000).toFixed(2);
  console.log(`\n  Load time: ${report.model.load_time_sec}s`);

  // ── Latency Benchmark (context × batch_size matrix) ────────────────────────
  const MODEL_PATCHES = 16;
  const inputPatchLen = 32;
  const dim = 64;
  const iterations = parseInt(process.env.BENCH_ITERATIONS || '5', 10);

  console.log('\n  ── Latency ──');

  const contextConfigs = [
    { patches: 4, context: 128 },
    { patches: 8, context: 256 },
    { patches: 16, context: 512 },
  ];

  // Track the very first cold start for cold/warm ratio calculation
  let firstColdStartMs = null;

  for (const cfg of contextConfigs) {
    for (const bs of batchSizes) {
      const label = `ctx=${String(cfg.context).padStart(3)}  batch=${bs}`;

      // Build a single-series input: [1, MODEL_PATCHES, dim]
      // The ONNX model has a fixed first dimension of 1, so we always feed [1, 16, 64].
      // For batch_size > 1, we run sequential inferences and sum the time.
      const input = new Float32Array(1 * MODEL_PATCHES * dim);

      // Fill active patches with random data
      for (let p = 0; p < cfg.patches; p++) {
        const bp = p * dim;
        for (let i = 0; i < inputPatchLen; i++) {
          input[bp + i] = Math.random();
          input[bp + inputPatchLen + i] = 0; // padding flag
        }
      }
      // Fill padding patches with mask=1
      for (let p = cfg.patches; p < MODEL_PATCHES; p++) {
        const bp = p * dim;
        for (let i = 0; i < inputPatchLen; i++) {
          input[bp + inputPatchLen + i] = 1; // mask flag
        }
      }

      const feeds = { inputs: new ort.Tensor('float32', input, [1, MODEL_PATCHES, dim]) };

      // Measure one cold-start inference (first call, no pre-warming)
      const coldStart = performance.now();
      await session.run(feeds);
      const coldStartMs = +(performance.now() - coldStart).toFixed(1);

      // Track first cold start for cold/warm ratio
      if (firstColdStartMs === null) {
        firstColdStartMs = coldStartMs;
      }

      // Warmup: 2 passes to trigger JIT compilation
      for (let i = 0; i < 2; i++) await session.run(feeds);

      // Clear GC before measurement for consistent heap readings
      if (global.gc) global.gc();
      const heapBefore = process.memoryUsage().heapUsed;

      // Measure warm inference — for batch_size > 1, run sequential inferences
      // and sum the time. This reflects real-world throughput when the model
      // has a fixed batch dimension of 1.
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

  // Compute cold/warm ratio per context (paired comparison — same context, same batch_size=1)
  // Each context's cold start is compared against its own warm average to produce a
  // meaningful per-context ratio.  The reported ratio is the geometric mean across
  // all context configurations (avoids the skew of averaging across contexts with
  // very different absolute latency ranges).
  const coldWarmPairs = [];
  for (const entry of report.latency) {
    if (entry.batch_size === 1 && entry.cold_start_ms != null && entry.avg_ms > 0) {
      coldWarmPairs.push({
        ctx: entry.context,
        ratio: entry.cold_start_ms / entry.avg_ms,
      });
    }
  }
  if (coldWarmPairs.length > 0) {
    const geoMeanLog =
      coldWarmPairs.reduce((sum, p) => sum + Math.log(p.ratio), 0) / coldWarmPairs.length;
    report.cold_warm_ratio = +Math.exp(geoMeanLog).toFixed(2);
    if (coldWarmPairs.length >= 3) {
      console.log(
        `\n  Cold/Warm ratio: ${report.cold_warm_ratio}× (per-context: ${coldWarmPairs.map((p) => `ctx=${p.ctx}:${p.ratio.toFixed(2)}×`).join(', ')})`,
      );
    } else {
      console.log(`\n  Cold/Warm ratio: ${report.cold_warm_ratio}×`);
    }
  }

  // ── Stability (Memory Leak Detection) ─────────────────────────────────────
  // Run repeated inference on the raw ONNX session and monitor heap growth.
  // A healthy ONNX Runtime session should stabilize within ±5% after warmup.
  // Must run BEFORE the accuracy section (which releases the session).
  {
    console.log('\n  ── Stability (memory leak check) ──');

    const STABILITY_ITERS = parseInt(process.env.BENCH_STABILITY_ITERS || '100', 10);
    const modelPatches = 16;
    const dim2 = 64;
    const input = new Float32Array(1 * modelPatches * dim2);
    for (let p = 0; p < modelPatches; p++) {
      const bp = p * dim2;
      for (let i = 0; i < 32; i++) {
        input[bp + i] = Math.random();
        input[bp + 32 + i] = p >= 8 ? 1 : 0; // mask padding patches beyond 8
      }
    }
    const feeds = {
      inputs: new ort.Tensor('float32', input, [1, modelPatches, dim2]),
    };

    // 3 warmup iterations
    for (let i = 0; i < 3; i++) await session.run(feeds);

    if (global.gc) global.gc();
    const heapBaseline = process.memoryUsage().heapUsed;

    const snapshots = [];
    for (let i = 0; i < STABILITY_ITERS; i++) {
      await session.run(feeds);
      if (i % 25 === 24 || i === 0) {
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

  // ── Accuracy Benchmark (full TimesFM pipeline) ────────────────────────────
  if (!skipAccuracy) {
    console.log('\n  ── Accuracy (full pipeline) ──');
    console.log(
      '  (Using real-world test fixtures — business metric, stock price, seasonal temp, etc.)',
    );

    // Close raw ONNX session before loading the full model to save memory
    try {
      session.release();
    } catch {
      // Already disposed
    }
    sessionReleased = true;

    // Import from TypeScript source via tsx.
    const coreMod = await import(
      path.join(__dirname, '..', 'packages', 'timesfm-core', 'src', 'index.ts')
    );
    const core = coreMod.default || coreMod;
    const { TimesFMModel, createForecastConfig } = core;

    // ── Real-world test fixture generators (imported from test-fixtures.ts) ═══
    // Single source of truth — the same generators used by the full test suite.
    const fixturesMod = await import(
      path.join(__dirname, '..', 'packages', 'timesfm-core', 'test', 'test-fixtures.ts')
    );
    const { businessMetric, stockPrice, hourlyTemp, eCommerce, regimeShift } =
      typeof fixturesMod.default === 'object' && fixturesMod.default !== null
        ? fixturesMod.default
        : fixturesMod;

    // Use 5 diverse real-world fixture types (deterministic via seed=42).
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

    // Load full TimesFM model with proper preprocessing pipeline
    const accModel = await TimesFMModel.fromPretrained({ modelPath });
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

      // Full TimesFM pipeline forecast
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

      // Constant mean baseline for reference
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
    console.log(`  TimesFM MAE (full pipe):   ${avgModelMAE.toFixed(4)}`);
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

  // ── Regression check ───────────────────────────────────────────────────────
  let regression = null;
  if (baselinePath && fs.existsSync(baselinePath)) {
    try {
      const baseline = JSON.parse(fs.readFileSync(baselinePath, 'utf-8'));
      if (baseline && baseline.latency) {
        regression = checkRegression(report, baseline, regressionThreshold);
        // Store baseline info in regression object for HTML report
        regression.baseline_info = `${baseline.timestamp || 'unknown'} (${baseline.git_sha || 'unknown'})`;
        console.log(`\n  ── Regression ──`);
        console.log(`  ${regression.summary}`);
        for (const c of regression.comparisons) {
          const icon =
            c.level === 'critical'
              ? '❌'
              : c.level === 'warning'
                ? '⚠️'
                : c.level === 'notice'
                  ? 'ℹ️'
                  : '✅';
          console.log(
            `    ${icon} ctx=${c.context} batch=${c.batch_size}: ${c.current_ms}ms vs ${c.baseline_ms}ms (${c.delta_pct > 0 ? '+' : ''}${c.delta_pct}%)`,
          );
        }
      }
    } catch (err) {
      console.log(`\n  ⚠️  Could not load baseline: ${err.message}`);
    }
  }

  // ── Output Reports ─────────────────────────────────────────────────────────
  const doJson = outJson || outAll;
  const doMd = outMd || outAll;
  const doHtml = outHtml || outAll;
  const jsonPath = outJson && typeof outJson === 'string' ? outJson : 'benchmark-report.json';
  const mdPath = outMd && typeof outMd === 'string' ? outMd : 'benchmark-report.md';
  const htmlPath = outHtml && typeof outHtml === 'string' ? outHtml : 'benchmark-report.html';

  if (doJson && jsonPath) {
    fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2));
    console.log(`\n  ✅ JSON report written: ${jsonPath}`);
  }

  if (doMd && mdPath) {
    const md = generateMarkdownReport(report, regression);
    fs.writeFileSync(mdPath, md);
    console.log(`  ✅ Markdown report written: ${mdPath}`);
  }

  if (doHtml && htmlPath) {
    const html = generateHtmlReport(report, regression);
    fs.writeFileSync(htmlPath, html);
    console.log(`  ✅ HTML report written: ${htmlPath}`);
  }

  // ── Exit with regression status if applicable ──────────────────────────────
  if (regression && regression.has_critical) {
    console.log('\n  ❌ Critical performance regression detected!');
    process.exit(1);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
