#!/usr/bin/env node
/**
 * agentix-timesfm-ts Official Benchmark Suite
 *
 * Based on TimesFM ICML 2024 paper evaluation protocol:
 *   - Datasets: benchmark datasets
 *   - Metrics: MAE, Scaled MAE (vs Naive), RMSE
 *   - Method: Zero-shot, no training
 *
 * Usage: node scripts/benchmark.js --model <path> [--datasets <dir>]
 */

// Load core from relative path when running from workspace
let TimesFMModel, createForecastConfig;
try {
  ({ TimesFMModel, createForecastConfig } = require('@agentix/timesfm-core'));
} catch {
  ({ TimesFMModel, createForecastConfig } = require('../packages/timesfm-core/src/model'));
  // Override createForecastConfig to use the correct import
  ({ createForecastConfig } = require('../packages/timesfm-core/src/config'));
}
const fs = require('fs');
const path = require('path');
let parse;
try {
  ({ parse } = require('csv-parse/sync'));
} catch {
  parse = function () {
    throw new Error('csv-parse not installed');
  };
}
const { performance } = require('perf_hooks');

// ─── Configuration ─────────────────────────────────────────────────────────

const BENCHMARK_CONFIG = {
  // Match TimesFM paper evaluation protocol
  horizon: 24,
  context: 512,
  maxSeriesPerDataset: 100,
  datasets: [
    { name: 'benchmark_monthly', freq: 'monthly', expectedNaiveMAE: 'variable' },
    { name: 'benchmark_daily', freq: 'daily', expectedNaiveMAE: 'variable' },
    { name: 'benchmark_hourly', freq: 'hourly', expectedNaiveMAE: 'variable' },
  ],
};

// ─── Naive Forecast Baseline ───────────────────────────────────────────────

function naiveForecast(values, horizon) {
  const lastValue = values[values.length - 1];
  return new Float32Array(horizon).fill(lastValue);
}

function computeMAE(actual, predicted) {
  let sum = 0;
  const len = Math.min(actual.length, predicted.length);
  for (let i = 0; i < len; i++) sum += Math.abs(actual[i] - predicted[i]);
  return sum / len;
}

function computeRMSE(actual, predicted) {
  let sum = 0;
  const len = Math.min(actual.length, predicted.length);
  for (let i = 0; i < len; i++) sum += (actual[i] - predicted[i]) ** 2;
  return Math.sqrt(sum / len);
}

// ─── CSV Loader ────────────────────────────────────────────────────────────

function loadDataset(csvPath) {
  const raw = fs.readFileSync(csvPath, 'utf-8');
  const records = parse(raw, { columns: true, skip_empty_lines: true });
  if (records.length === 0) return [];

  const cols = Object.keys(records[0]);
  const series = [];
  for (const col of cols) {
    const values = [];
    for (const row of records) {
      const v = parseFloat(row[col]);
      values.push(Number.isFinite(v) ? v : NaN);
    }
    // Remove trailing NaN
    while (values.length > 0 && Number.isNaN(values[values.length - 1])) values.pop();
    if (values.length > 50) {
      series.push(new Float32Array(values));
    }
  }
  return series.slice(0, BENCHMARK_CONFIG.maxSeriesPerDataset);
}

// ─── Main Benchmark ────────────────────────────────────────────────────────

async function runBenchmark(modelPath, dataDir) {
  console.log('='.repeat(72));
  console.log('  agentix-timesfm-ts — Benchmark Report');
  console.log('  Based on TimesFM ICML 2024 Paper Evaluation Protocol');
  console.log('='.repeat(72));
  console.log();

  // System info
  const os = require('os');
  console.log('[System Environment]');
  console.log(`  CPU:     ${os.cpus()[0]?.model || 'unknown'}`);
  console.log(`  Cores:   ${os.cpus().length}`);
  console.log(`  Memory:  ${(os.totalmem() / 1024 ** 3).toFixed(1)} GB`);
  console.log(`  GPU:     ❌ Unavailable (CPU only mode)`);
  console.log(`  Node.js: ${process.version}`);
  console.log();

  // Load model
  console.log('[Load Model]');
  const t0 = performance.now();
  const model = await TimesFMModel.fromPretrained({ modelPath });
  const t1 = performance.now();
  console.log(`  Model Path: ${modelPath}`);
  console.log(`  Load Time: ${((t1 - t0) / 1000).toFixed(2)}s`);
  console.log();

  // Compile
  const fc = createForecastConfig({
    maxContext: BENCHMARK_CONFIG.context,
    maxHorizon: BENCHMARK_CONFIG.horizon,
    normalizeInputs: true,
    useContinuousQuantileHead: true,
    forceFlipInvariance: true,
    inferIsPositive: true,
    fixQuantileCrossing: true,
    perCoreBatchSize: 1,
  });
  model.compile(fc);
  console.log(`  maxContext: ${fc.maxContext}, maxHorizon: ${fc.maxHorizon}`);
  console.log();

  // ─── Per-dataset evaluation ───
  console.log('[Dataset Evaluation]');
  console.log();

  const allResults = [];

  for (const ds of BENCHMARK_CONFIG.datasets) {
    const csvPath = path.join(dataDir, `${ds.name}.csv`);
    if (!fs.existsSync(csvPath)) {
      console.log(`  ⚠️ ${ds.name}: Data file not found (${csvPath})`);
      continue;
    }

    const series = loadDataset(csvPath);
    if (series.length === 0) {
      console.log(`  ⚠️ ${ds.name}: No valid series`);
      continue;
    }

    console.log(`  📊 ${ds.name} (${ds.freq}) — ${series.length} series`);

    const horizon = BENCHMARK_CONFIG.horizon;
    const naiveMAEs = [];
    const timesfmMAEs = [];
    const timesfmRMSEs = [];
    const times = [];
    let skipped = 0;

    for (let i = 0; i < series.length; i++) {
      const full = series[i];
      if (full.length < horizon + 32) {
        skipped++;
        continue;
      }

      const context = full.slice(0, full.length - horizon);
      const actual = full.slice(full.length - horizon);

      // Naive baseline
      const naivePred = naiveForecast(context, horizon);
      naiveMAEs.push(computeMAE(actual, naivePred));

      // TimesFM
      const tStart = performance.now();
      try {
        const result = await model.forecast(horizon, [context]);
        const pred = result.pointForecast[0];
        timesfmMAEs.push(computeMAE(actual, pred));
        timesfmRMSEs.push(computeRMSE(actual, pred));
      } catch (e) {
        skipped++;
      }
      times.push(performance.now() - tStart);

      // Progress
      if ((i + 1) % 20 === 0 || i === series.length - 1) {
        process.stderr.write(`\r    Progress: ${i + 1}/${series.length}`);
      }
    }
    process.stderr.write('\n');

    // Aggregate
    const avgNaiveMAE = naiveMAEs.reduce((a, b) => a + b, 0) / naiveMAEs.length;
    const avgTimesFMMAE = timesfmMAEs.reduce((a, b) => a + b, 0) / timesfmMAEs.length;
    const avgTimesFMRMSE = timesfmRMSEs.reduce((a, b) => a + b, 0) / timesfmRMSEs.length;
    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
    const scaledMAE = avgTimesFMMAE / avgNaiveMAE;

    console.log(`    Valid series: ${series.length - skipped}/${series.length}`);
    console.log(`    Naive MAE:       ${avgNaiveMAE.toFixed(4)}`);
    console.log(`    TimesFM MAE:     ${avgTimesFMMAE.toFixed(4)}`);
    console.log(`    TimesFM RMSE:    ${avgTimesFMRMSE.toFixed(4)}`);
    console.log(`    Scaled MAE:       ${scaledMAE.toFixed(4)}  (1.0 = Naive, <1 = Better)`);
    console.log(`    Avg Time:         ${avgTime.toFixed(0)} ms/series`);
    console.log();

    allResults.push({
      dataset: ds.name,
      frequency: ds.freq,
      numSeries: series.length - skipped,
      naiveMAE: avgNaiveMAE,
      timesfmMAE: avgTimesFMMAE,
      timesfmRMSE: avgTimesFMRMSE,
      scaledMAE,
      avgTimeMs: avgTime,
    });
  }

  // ─── Summary ───
  console.log('[Summary]');
  console.log();

  if (allResults.length > 0) {
    // Geometric mean of scaled MAEs (matches paper Table 4)
    const gmScaledMAE = Math.exp(
      allResults.reduce((sum, r) => sum + Math.log(r.scaledMAE), 0) / allResults.length,
    );
    const amScaledMAE = allResults.reduce((sum, r) => sum + r.scaledMAE, 0) / allResults.length;

    console.log(`  Scaled MAE (Arithmetic Mean, AM): ${amScaledMAE.toFixed(4)}`);
    console.log(`  Scaled MAE (Geometric Mean, GM): ${gmScaledMAE.toFixed(4)}`);
    console.log(`  Total series:            ${allResults.reduce((s, r) => s + r.numSeries, 0)}`);
    console.log(`  Total datasets:          ${allResults.length}`);
    console.log();

    // ─── Comparison with paper ───
    console.log('[Comparison with TimesFM Paper]');
    console.log();
    console.log('  Paper Monash GM Scaled MAE:  0.6846 (TimesFM 2.0 200M ZS)');
    console.log(`  This run Scaled MAE (GM):   ${gmScaledMAE.toFixed(4)}`);
    console.log();
    console.log('  ⚠️ Note:');
    console.log('  • Paper uses full Monash 18 dataset; this test uses benchmark data');
    console.log('  • Paper uses TimesFM 2.0 200M PyTorch model');
    console.log('  • This test uses 57MB lightweight ONNX test model (not real weights)');
    console.log('  • Scaled MAE < 1 means better than Naive baseline');
    console.log();
  }

  // ─── Performance ───
  console.log('[Performance Stats]');
  const allTimes = allResults.flatMap((r) => [r.avgTimeMs]);
  const avg = allTimes.reduce((a, b) => a + b, 0) / allTimes.length;
  console.log(`  Avg inference per series: ${avg.toFixed(0)} ms`);
  console.log(`  Throughput (est.):       ${(1000 / avg).toFixed(1)} series/s (CPU)`);
  console.log();

  // Cleanup
  await model.dispose();

  console.log('='.repeat(72));
  console.log('  Benchmark Complete');
  console.log('='.repeat(72));

  return allResults;
}

// ─── CLI ───────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const modelIdx = args.indexOf('--model');
const dataIdx = args.indexOf('--datasets');

const modelPath = modelIdx >= 0 ? args[modelIdx + 1] : null;
const dataDir = dataIdx >= 0 ? args[dataIdx + 1] : path.join(__dirname, '..', 'benchmarks', 'data');

if (!modelPath) {
  console.error('Usage: node scripts/benchmark.js --model <path> [--datasets <dir>]');
  console.error('Example: node scripts/benchmark.js --model ./models/timesfm-2.5.onnx');
  process.exit(1);
}

runBenchmark(modelPath, dataDir)
  .then(() => process.exit(0))
  .catch((e) => {
    console.error(e);
    process.exit(1);
  });
