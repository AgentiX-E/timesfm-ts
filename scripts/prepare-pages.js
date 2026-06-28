/**
 * prepare-pages.js — Generate GitHub Pages static HTML files
 *
 * Creates:
 *   1. docs/index.html              — Root landing page with navigation cards
 *   2. docs/coverage/index.html     — Coverage dashboard (lines/branches/functions/statements)
 *   3. docs/web-benchmark/          — Ensures directory for WASM benchmark data
 *
 * Used by the CI deploy-pages job as an independent script to avoid shell escaping
 * issues that arise with inline `node -e '...'` containing JS template literals.
 */

const fs = require('fs');
const path = require('path');

// ── Helpers ───────────────────────────────────────────────────────────────

function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

// ── Web benchmark data directory ─────────────────────────────────────────

function ensureWebBenchmarkDir() {
  ensureDir('docs/web-benchmark');
  console.log('[prepare-pages] Ensured docs/web-benchmark/ directory');
}

// ── Coverage index ───────────────────────────────────────────────────────

function writeCoverageIndex() {
  ensureDir('docs/coverage');

  let html;
  try {
    const summaryPath = path.join('docs', 'coverage', 'coverage-summary.json');
    const summary = require(path.resolve(summaryPath)).total;
    const pct = (k) => summary[k].pct.toFixed(1);
    const hasLcov = fs.existsSync(path.join('docs', 'coverage', 'lcov-report', 'index.html'));

    const lines = [
      '<!DOCTYPE html>',
      '<html lang="en">',
      '<head>',
      '<meta charset="UTF-8">',
      '<title>Coverage Report &middot; agentix-timesfm-ts</title>',
      '<style>',
      'body{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;line-height:1.6;color:#1a1a2e}',
      'h1{border-bottom:3px solid #2563eb;padding-bottom:.5rem}',
      'h2{margin-top:2rem;color:#1e40af}',
      '.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin:1.5rem 0}',
      '.card{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:1.25rem;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.05)}',
      '.card .pct{font-size:2.5rem;font-weight:700;color:#16a34a}',
      '.card .label{color:#6b7280;font-size:.875rem;margin-top:.25rem}',
      '.btn{display:inline-block;padding:0.75rem 2rem;background:#2563eb;color:#fff;border-radius:6px;text-decoration:none;font-weight:600;margin-top:1rem}',
      '.btn:hover{background:#1d4ed8}',
      '</style>',
      '</head>',
      '<body>',
      '<h1>&#x1F4C8; Coverage Report</h1>',
      '<p>Generated ' +
        new Date().toISOString() +
        ' &middot; <a href="https://github.com/AgentiX-E/agentix-timesfm-ts">agentix-timesfm-ts</a></p>',
      '<div class=grid>',
      '  <div class=card><div class=pct>' +
        pct('lines') +
        '%</div><div class=label>Lines</div></div>',
      '  <div class=card><div class=pct>' +
        pct('branches') +
        '%</div><div class=label>Branches</div></div>',
      '  <div class=card><div class=pct>' +
        pct('functions') +
        '%</div><div class=label>Functions</div></div>',
      '  <div class=card><div class=pct>' +
        pct('statements') +
        '%</div><div class=label>Statements</div></div>',
      '</div>',
      '<p>Thresholds: &ge;95% lines &middot; functions &middot; branches &middot; statements</p>',
      hasLcov
        ? '<a class="btn" href="lcov-report/index.html">&#x1F4CA; View Detailed Report</a>'
        : '',
      '</body>',
      '</html>',
    ];
    html = lines.join('\n') + '\n';
    console.log(
      '[prepare-pages] Coverage report generated (lines: ' +
        pct('lines') +
        '%, branches: ' +
        pct('branches') +
        '%, functions: ' +
        pct('functions') +
        '%, statements: ' +
        pct('statements') +
        '%)',
    );
  } catch (e) {
    html =
      [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head><meta charset="UTF-8"><title>Coverage</title></head>',
        '<body>',
        '<h1>&#x1F4C8; Coverage</h1>',
        '<p>Report pending &mdash; check back after CI completes.</p>',
        '</body>',
        '</html>',
      ].join('\n') + '\n';
    console.log(
      '[prepare-pages] Coverage report pending (summary JSON not found: ' + e.message + ')',
    );
  }

  fs.writeFileSync('docs/coverage/index.html', html);
}

// ── Root landing page ─────────────────────────────────────────────────────

function writeRootLandingPage() {
  ensureDir('docs');
  const html =
    [
      '<!DOCTYPE html>',
      '<html lang="en">',
      '<head>',
      '<meta charset="UTF-8">',
      '<title>agentix-timesfm-ts &middot; Docs</title>',
      '<style>',
      'body{font-family:system-ui,sans-serif;max-width:800px;margin:3rem auto;padding:0 1.5rem;line-height:1.7;color:#1a1a2e;background:#fafbfc}',
      'h1{font-size:2rem;border-bottom:3px solid #2563eb;padding-bottom:0.5rem}',
      '.card{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:1.5rem;margin:1.5rem 0;box-shadow:0 1px 3px rgba(0,0,0,0.05)}',
      '.card h2{margin-top:0;font-size:1.3rem}.card a{color:#2563eb;text-decoration:none;font-weight:500}.card a:hover{text-decoration:underline}',
      '.card p{color:#6b7280;margin:0.5rem 0 0}',
      '</style>',
      '</head>',
      '<body>',
      '<h1>&#x1F680; agentix-timesfm-ts</h1>',
      '<p>Node.js/TypeScript reimplementation of Google Research&apos;s TimesFM 2.5 &mdash; zero-shot time series forecasting.</p>',
      '<div class=card><h2>&#x1F4DA; <a href="api/index.html">API Documentation</a></h2><p>Full TypeDoc reference for all 5 packages (timesfm-core, timesfm-xreg, timesfm-cli, timesfm-web, timesfm-hierarchical)</p></div>',
      '<div class=card><h2>&#x1F4CA; <a href="benchmark/">Benchmark Reports</a></h2><p>Inference latency, throughput, and prediction accuracy benchmarks (Node.js + WASM comparison) with historical trending</p></div>',
      '<div class=card><h2>&#x1F4C8; <a href="coverage/">Test Coverage</a></h2><p>Code coverage reports (lines, branches, functions, statements) &mdash; target &ge;95%</p></div>',
      '<div class=card><h2>&#x1F4D6; <a href="https://github.com/AgentiX-E/agentix-timesfm-ts/blob/master/docs/ARCHITECTURE.md">Architecture Guide</a></h2><p>Component design, data flow, and design principles</p></div>',
      '<div class=card><h2>&#x1F4A0; <a href="https://github.com/AgentiX-E/agentix-timesfm-ts/blob/master/docs/ACCURACY.md">Accuracy Validation</a></h2><p>Methodology and results comparing Python vs TypeScript TimesFM on real-world datasets</p></div>',
      '<div class=card><h2>&#x1F504; <a href="https://github.com/AgentiX-E/agentix-timesfm-ts/blob/master/docs/MIGRATION.md">Migration Guide</a></h2><p>Step-by-step guide for migrating from Google TimesFM (Python) to Agentix TimesFM (TypeScript)</p></div>',
      '<div class=card><h2>&#x1F527; <a href="https://github.com/AgentiX-E/agentix-timesfm-ts/blob/master/docs/TROUBLESHOOTING.md">Troubleshooting</a></h2><p>Common issues and solutions for model download, memory, WASM, and more</p></div>',
      '<div class=card><h2>&#x1F4BB; <a href="https://github.com/AgentiX-E/agentix-timesfm-ts">Source Code</a></h2><p>GitHub repository with full source, contributing guide, and CI workflows</p></div>',
      '</body>',
      '</html>',
    ].join('\n') + '\n';

  fs.writeFileSync('docs/index.html', html);
  console.log('[prepare-pages] Wrote docs/index.html');
}

// ── Main ─────────────────────────────────────────────────────────────────

ensureWebBenchmarkDir();
writeCoverageIndex();
writeRootLandingPage();
console.log('[prepare-pages] All pages generated successfully.');
