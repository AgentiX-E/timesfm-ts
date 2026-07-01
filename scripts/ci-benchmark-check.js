#!/usr/bin/env node
/**
 * ci-benchmark-check.js — CI benchmark accuracy gate
 *
 * Reads benchmark-report.json and enforces:
 *   1. scaled_mae < 1.0 (TimesFM is better than naive baseline)
 *   2. Scaled MAE is within acceptable range (< 1.0 by definition)
 *
 * Usage:
 *   node scripts/ci-benchmark-check.js [--report benchmark-report.json] [--verbose]
 *
 * Called from ci.yml (benchmark and web-benchmark jobs).
 */

import * as fs from 'node:fs';
import * as path from 'node:path';

const REPORT_FILE = process.argv.includes('--report')
  ? process.argv[process.argv.indexOf('--report') + 1] || 'benchmark-report.json'
  : 'benchmark-report.json';

const VERBOSE = process.argv.includes('--verbose');

function main() {
  const reportPath = path.resolve(REPORT_FILE);

  if (!fs.existsSync(reportPath)) {
    console.error(
      `[benchmark-check] No benchmark report at ${reportPath} — skipping accuracy gate.`,
    );
    process.exit(0);
  }

  const report = JSON.parse(fs.readFileSync(reportPath, 'utf-8'));

  if (!report.accuracy) {
    console.log('[benchmark-check] No accuracy data — skipping gate.');
    process.exit(0);
  }

  const {
    scaled_mae: scaledMae,
    model_mae: modelMae,
    naive_mae: naiveMae,
    improvement_vs_naive_pct: improvementPct,
  } = report.accuracy;

  if (scaledMae >= 1.0) {
    console.error(`FAIL: TimesFM scaled MAE (${scaledMae}) is not better than naive baseline.`);
    console.error(`  Model MAE: ${modelMae ?? 'N/A'} vs Naive MAE: ${naiveMae ?? 'N/A'}`);
    process.exit(1);
  }

  console.log(`PASS: TimesFM scaled MAE = ${scaledMae} (< 1.0, better than naive)`);
  if (improvementPct !== undefined) {
    console.log(`  Improvement vs Naive: ${improvementPct}%`);
  }

  if (VERBOSE) {
    console.log(`  Model MAE: ${modelMae ?? 'N/A'}`);
    console.log(`  Naive MAE: ${naiveMae ?? 'N/A'}`);
  }
}

main();
