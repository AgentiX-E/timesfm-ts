#!/usr/bin/env node
/**
 * Generate a unified benchmark report combining Node.js (native ONNX) and
 * Web (WASM) inference latency data into a single HTML page.
 *
 * Usage:
 *   node scripts/generate-combined-report.js \
 *     --node benchmark-report.json \
 *     --web web-benchmark-report.json \
 *     --out docs/benchmark/index.html
 */

const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
function getArg(flag) {
  const idx = args.indexOf(flag);
  return idx >= 0 && idx + 1 < args.length ? args[idx + 1] : null;
}

const nodePath = getArg('--node');
const webPath = getArg('--web');
const outPath = getArg('--out');

if (!nodePath || !webPath || !outPath) {
  console.error('Usage: generate-combined-report.js --node <json> --web <json> --out <html>');
  process.exit(1);
}

const nodeData = JSON.parse(fs.readFileSync(nodePath, 'utf-8'));
const webData = JSON.parse(fs.readFileSync(webPath, 'utf-8'));

const esc = (s) =>
  String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

// ── Build latency comparison table ────────────────────────────────────────────
// Match contexts between Node and WASM, produce a combined table
const nodeLatency = nodeData.latency || [];
const webLatency = webData.latency || [];

// Build lookup maps: key = "context_batch"
function latencyKey(l) {
  return `${l.context}_${l.batch_size || 1}`;
}
const nodeMap = new Map(nodeLatency.map((l) => [latencyKey(l), l]));
const webMap = new Map(webLatency.map((l) => [latencyKey(l), l]));

// Union of all keys, sorted by context then batch
const allKeys = new Set([...nodeMap.keys(), ...webMap.keys()]);
const sortedKeys = [...allKeys].sort((a, b) => {
  const [ca] = a.split('_').map(Number);
  const [cb] = b.split('_').map(Number);
  if (ca !== cb) return ca - cb;
  const [, ba] = a.split('_').map(Number);
  const [, bb] = b.split('_').map(Number);
  return ba - bb;
});

let comparisonRows = '';
for (const key of sortedKeys) {
  const n = nodeMap.get(key);
  const w = webMap.get(key);
  const ctx = n ? n.context : w ? w.context : '?';
  const bs = n ? n.batch_size || 1 : w ? w.batch_size || 1 : 1;

  const nodeAvg = n ? n.avg_ms.toFixed(1) : '—';
  const nodeP99 = n ? n.p99_ms.toFixed(1) : '—';
  const webAvg = w ? w.avg_ms.toFixed(1) : '—';
  const webP99 = w ? w.p99_ms.toFixed(1) : '—';

  // Compute slowdown ratio
  let ratio = '—';
  let ratioClass = '';
  if (n && w) {
    const r = w.avg_ms / n.avg_ms;
    ratio = r.toFixed(1) + '×';
    if (r > 5) ratioClass = 'class="fail"';
    else if (r > 3) ratioClass = 'class="warn"';
  }

  comparisonRows += `<tr>
    <td>${ctx}</td>
    <td>${bs}</td>
    <td>${nodeAvg}</td>
    <td>${nodeP99}</td>
    <td>${webAvg}</td>
    <td>${webP99}</td>
    <td ${ratioClass}>${ratio}</td>
  </tr>`;
}

// ── Node-only latency table ───────────────────────────────────────────────────
let nodeRows = '';
for (const l of nodeLatency) {
  nodeRows += `<tr>
    <td>${l.context}</td>
    <td>${l.batch_size || 1}</td>
    <td>${l.patches}</td>
    <td>${l.avg_ms}</td>
    <td>${l.p50_ms}</td>
    <td>${l.p99_ms}</td>
    <td>${l.throughput_seq_s}</td>
    <td>${l.cold_start_ms != null ? l.cold_start_ms : '—'}</td>
  </tr>`;
}

// ── WASM-only latency table ───────────────────────────────────────────────────
let webRows = '';
for (const l of webLatency) {
  webRows += `<tr>
    <td>${l.context}</td>
    <td>${l.batch_size || 1}</td>
    <td>${l.avg_ms}</td>
    <td>${l.p50_ms}</td>
    <td>${l.p99_ms}</td>
    <td>${l.throughput_seq_s}</td>
    <td>${l.cold_start_ms != null ? l.cold_start_ms : '—'}</td>
  </tr>`;
}

// ── System info comparison ────────────────────────────────────────────────────
const nodeRam = nodeData.total_ram_gb || '?';
const nodeCpu = nodeData.cpu_model || '?';
const webRam = webData.total_ram_gb || '?';
const webCpu = webData.cpu_model || '?';

// ── Accuracy section (Node only — WASM doesn't run full pipeline) ─────────────
let accuracySection = '';
if (nodeData.accuracy) {
  const a = nodeData.accuracy;
  accuracySection = `
    <h2>🎯 Prediction Accuracy</h2>
    <p class="meta">Full TimesFM pipeline with RevIN normalization, flip invariance, continuous quantile head</p>
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td>Naive MAE (no-change)</td><td>${a.naive_mae}</td></tr>
        <tr><td>Constant Mean MAE</td><td>${a.const_mae}</td></tr>
        <tr><td><strong>TimesFM MAE</strong></td><td><strong>${a.model_mae}</strong></td></tr>
        <tr><td>TimesFM RMSE</td><td>${a.model_rmse}</td></tr>
        <tr><td>Scaled MAE vs Naive</td><td>${a.scaled_mae} ${a.better_than_naive ? '<span class="pass">✅ Better</span>' : '<span class="fail">⚠️</span>'}</td></tr>
        <tr><td>Improvement vs Naive</td><td><strong>${a.improvement_vs_naive_pct}%</strong></td></tr>
        <tr><td>Improvement vs Const Mean</td><td><strong>${a.improvement_vs_const_pct}%</strong></td></tr>
      </tbody>
    </table>`;
}

const html = `<!DOCTYPE html>
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
      --warn: #d97706;
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
        --warn: #fbbf24;
        --code-bg: #0f172a;
        --code-fg: #e2e8f0;
      }
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      max-width: 1100px;
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
    th { background: var(--th-bg); font-weight: 600; white-space: nowrap; }
    tr:hover { background: var(--hover); }
    .pass { color: var(--pass); font-weight: 600; }
    .fail { color: var(--fail); font-weight: 600; }
    .warn { color: var(--warn); font-weight: 600; }
    .meta { color: var(--muted); font-size: 0.9rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; margin: 1rem 0; }
    .card { border: 1px solid var(--border); border-radius: 8px; padding: 1rem; }
    .card h3 { font-size: 0.85rem; text-transform: uppercase; color: var(--muted); margin-bottom: 0.5rem; }
    .card .value { font-size: 1.4rem; font-weight: 700; }
    .tabs { display: flex; gap: 0; margin: 1.5rem 0 0 0; border-bottom: 2px solid var(--border); }
    .tabs button {
      padding: 0.5rem 1.5rem;
      border: none;
      background: none;
      cursor: pointer;
      font-size: 0.95rem;
      font-weight: 500;
      color: var(--muted);
      border-bottom: 2px solid transparent;
      margin-bottom: -2px;
      transition: color 0.2s, border-color 0.2s;
    }
    .tabs button.active { color: var(--accent); border-bottom-color: var(--accent); }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); color: var(--muted); font-size: 0.85rem; text-align: center; }
    .nav-links { display: flex; gap: 1rem; margin: 0.5rem 0; }
    .nav-links a { color: var(--accent); text-decoration: none; font-size: 0.9rem; }
    .nav-links a:hover { text-decoration: underline; }
    @media (max-width: 700px) {
      body { margin: 1rem auto; }
      table { font-size: 0.75rem; }
      th, td { padding: 4px 6px; }
    }
  </style>
</head>
<body>
  <h1>📊 TimesFM 2.5 200M Benchmark Report</h1>
  <p class="meta">
    Generated: ${esc(nodeData.timestamp)} · Git: <code>${esc(nodeData.git_sha)}</code> · Node: ${esc(nodeData.node_version)}
    ${nodeData.onnx_runtime_version ? ` · ONNX Runtime ${esc(nodeData.onnx_runtime_version)}` : ''}
  </p>

  <div class="nav-links">
    <a href="benchmark-report.html">📊 Node.js Full Report</a>
    <a href="../web-benchmark/web-benchmark-report.html">🌐 WASM Full Report</a>
    <a href="benchmark-report.json">📄 Node JSON</a>
    <a href="../web-benchmark/web-benchmark-report.json">📄 WASM JSON</a>
  </div>

  <h2>💻 System</h2>
  <div class="grid">
    <div class="card"><h3>CPU</h3><div class="value" style="font-size:1rem;">${esc(nodeCpu)} × ${nodeData.cpu_cores || '?'}</div></div>
    <div class="card"><h3>RAM</h3><div class="value">${nodeRam} GB</div></div>
    <div class="card"><h3>Platform</h3><div class="value" style="font-size:1rem;">${esc(nodeData.platform || '?')} / ${esc(nodeData.arch || '?')}</div></div>
    <div class="card"><h3>Node.js</h3><div class="value" style="font-size:1rem;">${esc(nodeData.node_version || '?')}</div></div>
  </div>

  <h2>🧠 Model</h2>
  <div class="grid">
    <div class="card"><h3>Size</h3><div class="value">${nodeData.model ? nodeData.model.size_mb : '?'} MB</div></div>
    <div class="card"><h3>Node Load Time</h3><div class="value">${nodeData.model ? nodeData.model.load_time_sec : '?'}s</div></div>
    <div class="card"><h3>ONNX Runtime</h3><div class="value" style="font-size:1rem;">${esc(nodeData.onnx_runtime_version || '?')}</div></div>
    <div class="card"><h3>Cold/Warm Ratio</h3><div class="value">${nodeData.cold_warm_ratio != null ? nodeData.cold_warm_ratio.toFixed(2) + '×' : '—'}</div></div>
  </div>

  <!-- Comparison: Node vs WASM -->
  <h2>⚡ Inference Latency — Node.js vs WASM</h2>
  <table>
    <thead><tr>
      <th>Context</th><th>Batch</th>
      <th>Node Avg (ms)</th><th>Node P99 (ms)</th>
      <th>WASM Avg (ms)</th><th>WASM P99 (ms)</th>
      <th>WASM/Node</th>
    </tr></thead>
    <tbody>${comparisonRows}</tbody>
  </table>

  <!-- Tabs for detailed per-backend tables -->
  <div class="tabs">
    <button class="active" onclick="switchTab('node')">🖥️ Node.js (native ONNX)</button>
    <button onclick="switchTab('wasm')">🌐 Web (WASM)</button>
  </div>

  <div id="tab-node" class="tab-content active">
    <h2>🖥️ Node.js Inference Latency</h2>
    <table>
      <thead><tr>
        <th>Context</th><th>Batch</th><th>Patches</th>
        <th>Avg (ms)</th><th>P50 (ms)</th><th>P99 (ms)</th>
        <th>Throughput (seq/s)</th><th>Cold Start (ms)</th>
      </tr></thead>
      <tbody>${nodeRows}</tbody>
    </table>
  </div>

  <div id="tab-wasm" class="tab-content">
    <h2>🌐 WASM Inference Latency</h2>
    <table>
      <thead><tr>
        <th>Context</th><th>Batch</th>
        <th>Avg (ms)</th><th>P50 (ms)</th><th>P99 (ms)</th>
        <th>Throughput (seq/s)</th><th>Cold Start (ms)</th>
      </tr></thead>
      <tbody>${webRows}</tbody>
    </table>
  </div>

  ${accuracySection}

  <h2>💾 Memory Footprint (Node.js)</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>
      <tr><td>RSS</td><td>${nodeData.memory ? nodeData.memory.rss_mb : '?'} MB</td></tr>
      <tr><td>Heap Used</td><td>${nodeData.memory ? nodeData.memory.heap_used_mb : '?'} MB</td></tr>
      <tr><td>Heap Total</td><td>${nodeData.memory ? nodeData.memory.heap_total_mb : '?'} MB</td></tr>
    </tbody>
  </table>

  <footer>
    Automated benchmark by <strong>agentix-timesfm-ts</strong> CI · ${esc(nodeData.timestamp)}
  </footer>

  <script>
    function switchTab(name) {
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tabs button').forEach(el => el.classList.remove('active'));
      document.getElementById('tab-' + name).classList.add('active');
      event.target.classList.add('active');
    }
    // Embed JSON data
    window.__benchmarkData = { node: ${JSON.stringify(nodeData)}, web: ${JSON.stringify(webData)} };
  </script>
</body>
</html>`;

fs.writeFileSync(outPath, html);
console.log(`✅ Combined benchmark report written: ${outPath}`);
