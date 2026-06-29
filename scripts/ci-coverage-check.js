#!/usr/bin/env node
/**
 * ci-coverage-check.js — CI coverage threshold verification
 *
 * Reads coverage/coverage-summary.json and enforces ≥95% on all four metrics
 * (lines, branches, functions, statements).  Exits with code 1 if any metric
 * falls below the threshold.
 *
 * Usage:
 *   node scripts/ci-coverage-check.js [--tier unit|integration] [--verbose]
 *
 * Called from ci.yml (unit-test and integration-test jobs) instead of inline
 * node -e scripts to avoid shell escaping issues.
 */

const fs = require('fs');
const path = require('path');

const THRESHOLDS = { lines: 95, branches: 95, functions: 95, statements: 95 };
const TIER = process.argv.includes('--tier')
  ? process.argv[process.argv.indexOf('--tier') + 1] || 'unit'
  : 'unit';
const VERBOSE = process.argv.includes('--verbose');

function main() {
  const summaryPath = path.resolve('coverage', 'coverage-summary.json');

  if (!fs.existsSync(summaryPath)) {
    console.error(
      `[${TIER}] FAIL: coverage-summary.json not found. Tests may have failed silently.`,
    );
    process.exit(1);
  }

  const summary = require(summaryPath);

  if (!summary || !summary.total) {
    console.error(`[${TIER}] FAIL: No total coverage data. All metrics are 0%.`);
    process.exit(1);
  }

  const s = summary.total;

  // Catch zero-coverage edge case
  const allZero = ['lines', 'branches', 'functions', 'statements'].every(
    (k) => !s[k] || s[k].pct === 0,
  );
  if (allZero) {
    console.error(`[${TIER}] FAIL: No meaningful coverage data. All metrics are 0%.`);
    process.exit(1);
  }

  let failed = false;
  for (const [metric, threshold] of Object.entries(THRESHOLDS)) {
    const pct = s[metric]?.pct ?? 0;
    const status = pct >= threshold ? '\u2705' : '\u274C';
    console.log(`${status} ${TIER} ${metric}: ${pct.toFixed(1)}% (threshold: ${threshold}%)`);
    if (pct < threshold) failed = true;
  }

  if (failed) {
    console.error(
      `\nFAIL: ${TIER} coverage thresholds not met (\u226595% required on all metrics).`,
    );
    process.exit(1);
  }

  if (VERBOSE) {
    console.log(`\n\u2705 All ${TIER} coverage thresholds met (\u226595%).`);
  }
}

main();
