#!/usr/bin/env node
/* eslint-disable no-console */
/**
 * TimesFM CLI — command-line time-series forecasting.
 *
 * Usage:
 *   # First time: download model
 *   timesfm setup
 *
 *   # With proxy (corporate network)
 *   timesfm setup --proxy-url http://proxy.company.com:8080
 *   timesfm setup --proxy-url http://proxy.company.com:8080 --proxy-username user
 *   timesfm setup --proxy-url http://proxy:8080 --proxy-username user --proxy-password pass
 *   # Password via environment variable (recommended for security)
 *   TIMESFM_PROXY_PASSWORD=pass timesfm setup --proxy-url http://proxy:8080 --proxy-username user
 *   # Or via file (Docker/K8s secrets):
 *   TIMESFM_PROXY_PASSWORD_FILE=/run/secrets/proxy-password timesfm setup --proxy-url http://proxy:8080 --proxy-username user
 *
 *   # Forecast
 *   timesfm forecast --horizon 24 input.csv
 *   timesfm forecast --model ./custom.onnx --horizon 52 input.csv
 *
 * @module timesfm-cli
 */

import { Command } from 'commander';
import { existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import { csvForecast, type CSVForecastOptions } from './csv-forecast';

const require = createRequire(import.meta.url);
const { version } = require('../package.json') as { version: string };

const program = new Command();

program
  .name('timesfm')
  .description('Zero-shot time series forecasting with TimesFM (Node.js)')
  .version(version);

// ─── info — model metadata ─────────────────────────────────────────────────

program
  .command('info')
  .description('Show model metadata and system information')
  .option('-m, --model <path>', 'Path to TimesFM ONNX model')
  .action(async (options: Record<string, unknown>) => {
    try {
      const core = await import('@agentix-e/timesfm-core');
      const modelPath =
        (options.model as string) || process.env.TIMESFM_MODEL_PATH || core.defaultModelPath();

      console.log(`TimesFM CLI  —  @agentix-e/timesfm-cli  v${version}`);
      console.log(`Model path:  ${modelPath}`);
      if (existsSync(modelPath)) {
        const { resolveModelConfig } = core;
        const { config, descriptor } = await resolveModelConfig(modelPath);
        console.log(
          `Architecture: TimesFM v${descriptor?.model.version ?? '?'}-${descriptor?.model.variant ?? '?'}`,
        );
        console.log(`HF revision: ${descriptor?.model.hf_revision ?? 'unknown'}`);
        console.log(
          `Parameters:  ${config.numLayers} layers × ${config.numHeads} heads × ${config.modelDims} dims`,
        );
        console.log(`Context limit: ${config.contextLimit}`);
        console.log(
          `Input patch:  ${config.inputPatchLen}  |  Output patch: ${config.outputPatchLen}`,
        );
        console.log(`Quantiles:    ${config.quantiles.join(', ')}`);
        if (descriptor?.onnx.size_bytes) {
          console.log(`ONNX size:    ${(descriptor.onnx.size_bytes / 1024 ** 2).toFixed(0)} MB`);
        }
        const prec = descriptor?.onnx.precision ?? 'fp32';
        console.log(`Precision:    ${prec}`);
        // System info
        const os = await import('node:os');
        console.log(`\nSystem:`);
        console.log(`  Platform: ${os.platform()} / ${os.arch()}`);
        console.log(`  Node.js:  ${process.version}`);
        console.log(`  CPU:      ${os.cpus()[0]?.model ?? 'unknown'} × ${os.cpus().length}`);
        console.log(`  RAM:      ${(os.totalmem() / 1024 ** 3).toFixed(1)} GB total`);
        console.log(`  Free RAM: ${(os.freemem() / 1024 ** 3).toFixed(1)} GB`);
        const bs = core.suggestBatchSize();
        console.log(`  Suggested batch size: ${bs}`);
      } else {
        console.log('Model not found. Run "timesfm setup" to download.');
      }
    } catch (err) {
      console.error('Error:', err instanceof Error ? err.message : String(err));
      process.exit(1);
    }
  });

// ─── setup — download model ────────────────────────────────────────────────
// Tracks the most recent download path for cross-command convenience.
// CLI invocations are single-process, so this is safe.
let _lastSetupPath: string | null = null;

program
  .command('setup')
  .description('Download the TimesFM 2.5 200M ONNX model (~885 MB)')
  .option('-f, --force', 'Force re-download even if already cached')
  .option(
    '-o, --output <path>',
    'Custom output path (default: ~/.cache/timesfm-ts/timesfm-2.5.onnx)',
  )
  .option('--proxy-url <url>', 'Proxy URL for downloading through corporate firewall')
  .option('--proxy-username <user>', 'Proxy authentication username')
  .option('--proxy-password <pass>', 'Proxy authentication password (prefer env variable)')
  .option('--precision <fp32|int8>', 'Model precision variant (default: fp32)')
  .action(async (options: Record<string, unknown>) => {
    try {
      const core = await import('@agentix-e/timesfm-core');

      // Build proxy config from CLI args + environment variables
      // Priority: CLI args → TIMESFM_PROXY_* env vars → standard *_proxy env vars
      const proxyUrl = (options.proxyUrl as string) || process.env.TIMESFM_PROXY_URL || undefined;
      const proxyUsername =
        (options.proxyUsername as string) || process.env.TIMESFM_PROXY_USERNAME || undefined;
      // Password: CLI arg takes precedence, then env var
      const proxyPassword =
        (options.proxyPassword as string) || process.env.TIMESFM_PROXY_PASSWORD || undefined;

      const proxyConfig = proxyUrl
        ? { url: proxyUrl, username: proxyUsername, password: proxyPassword }
        : undefined;

      const prec = (options.precision as string) || process.env.TIMESFM_PRECISION || 'fp32';
      if (prec !== 'fp32' && prec !== 'int8') {
        console.error(`Error: Unknown precision "${prec}". Supported: fp32, int8`);
        process.exit(1);
      }

      const dest = await core.downloadModel({
        force: options.force === true,
        dest: options.output as string | undefined,
        proxy: proxyConfig,
        precision: prec,
      });
      _lastSetupPath = dest;
      console.log(`\nModel ready: ${dest}`);
      if (prec !== 'fp32') {
        console.log(`Precision:   ${prec}`);
      }
      console.log(`   Run: timesfm forecast --horizon 24 your-data.csv`);
    } catch (err) {
      console.error('Error:', err instanceof Error ? err.message : String(err));
      process.exit(1);
    }
  });

// ─── forecast ──────────────────────────────────────────────────────────────

/**
 * Resolve the model path with this priority:
 *   1. Explicit `--model` flag
 *   2. `TIMESFM_MODEL_PATH` environment variable
 *   3. Path from `timesfm setup -o <path>` in the same process (programmatic use only)
 *   4. Default cache path (~/.cache/timesfm-ts/timesfm-2.5.onnx)
 *   5. Auto-download to default cache path
 */
async function resolveModelPath(explicitPath: string | undefined): Promise<string> {
  // 1. Explicit --model flag
  if (explicitPath) return explicitPath;

  const core = await import('@agentix-e/timesfm-core');

  // 2. Environment variable
  const envPath = process.env.TIMESFM_MODEL_PATH;
  if (envPath && existsSync(envPath)) return envPath;

  // 3. Last setup path (in-process memory)
  if (_lastSetupPath && existsSync(_lastSetupPath)) return _lastSetupPath;

  // 4. Default cache
  const cached = core.getCachedModelPath();
  if (cached) return cached;

  // 5. Auto-download (proxy is resolved automatically from env vars inside downloadModel)
  console.error('No model found. Downloading TimesFM 2.5 200M…');
  return core.downloadModel();
}

program
  .command('forecast')
  .description('Forecast time series from a CSV file')
  .argument('<input>', 'Path to input CSV file')
  .requiredOption('-H, --horizon <number>', 'Forecast horizon (number of steps)', (v: string) =>
    parseInt(v, 10),
  )
  .option('-m, --model <path>', 'Path to TimesFM ONNX model (auto-download if omitted)')
  .option('-d, --date-col <name>', 'Date column name (default: "date")', 'date')
  .option('-v, --value-cols <names>', 'Comma-separated value column names (default: all numeric)')
  .option('-o, --output <path>', 'Output file path (default: stdout)')
  .option('--output-format <format>', 'Output format: csv or json', 'csv')
  .option('--context <number>', 'Max context length', (v: string) => parseInt(v, 10), 1024)
  .option('--no-normalize', 'Disable input normalization')
  .option('--no-flip-invariance', 'Disable flip invariance')
  .option('--no-positive', 'Disable positive-value inference')
  .option('--no-fix-quantile-crossing', 'Disable quantile crossing fix')
  .option('--no-continuous-quantile-head', 'Disable continuous quantile head')
  .action(async (input: string, options: Record<string, unknown>) => {
    try {
      const resolvedPath = await resolveModelPath(options.model as string | undefined);
      if (!resolvedPath) {
        throw new Error('Failed to resolve model path.');
      }

      const forecastOptions: CSVForecastOptions = {
        inputPath: input,
        horizon: options.horizon as number,
        modelPath: resolvedPath,
        dateCol: options.dateCol as string,
        valueCols: options.valueCols
          ? (options.valueCols as string).split(',').map((s: string) => s.trim())
          : undefined,
        outputPath: options.output as string | undefined,
        outputFormat: ((options.outputFormat as string) || 'csv') as 'csv' | 'json',
        maxContext: (options.context as number) || 1024,
        normalizeInputs: options.normalize !== false,
        forceFlipInvariance: options.flipInvariance !== false,
        inferIsPositive: options.positive !== false,
        fixQuantileCrossing: options.fixQuantileCrossing !== false,
        useContinuousQuantileHead: options.continuousQuantileHead !== false,
      };

      await csvForecast(forecastOptions);
    } catch (err) {
      console.error('Error:', err instanceof Error ? err.message : String(err));
      process.exit(1);
    }
  });

program.parse();
