# @agentix-e/timesfm-cli

> Command-line interface for TimesFM — download models and forecast time series from CSV files.

[![npm](https://img.shields.io/npm/v/@agentix-e/timesfm-cli?color=blue)](https://www.npmjs.com/package/@agentix-e/timesfm-cli)
[![API Docs](https://img.shields.io/badge/docs-TypeDoc-blue)](https://agentix-e.github.io/agentix-timesfm-ts/api/modules/timesfm-cli.html)

## Overview

`@agentix-e/timesfm-cli` provides the `timesfm` command-line tool for zero-shot time series forecasting. It handles model downloads, CSV I/O, and command-line argument parsing via Commander.

## Installation

```bash
npm install -g @agentix-e/timesfm-cli
# or use npx
npx @agentix-e/timesfm-cli setup
```

## Quick Start

```bash
# Download model (first time only, ~885 MB)
timesfm setup

# With proxy (corporate network)
timesfm setup --proxy-url http://proxy.company.com:8080
timesfm setup --proxy-url http://proxy:8080 --proxy-username user --proxy-password pass

# Password via environment variable (recommended for security)
TIMESFM_PROXY_PASSWORD=pass timesfm setup --proxy-url http://proxy:8080 --proxy-username user

# Or via file (Docker/Kubernetes secrets)
TIMESFM_PROXY_PASSWORD_FILE=/run/secrets/proxy-password timesfm setup --proxy-url http://proxy:8080 --proxy-username user

# Forecast from CSV
timesfm forecast --horizon 24 data.csv

# Custom model path and options
timesfm forecast --model ./custom.onnx --horizon 52 --context 512 data.csv

# JSON output
timesfm forecast --horizon 24 --output-format json data.csv > forecast.json

# Show model info
timesfm info --model ./models/timesfm-2.5.onnx
```

## Model Path Resolution

The `forecast` command resolves the model in this order:

1. `--model` CLI flag
2. `TIMESFM_MODEL_PATH` environment variable
3. Path from `timesfm setup -o <path>` (same process)
4. Default cache (`~/.cache/agentix-timesfm-ts/`)
5. Auto-download

## API Documentation

📚 **Full API reference**: [agentix-e.github.io/agentix-timesfm-ts/api/modules/timesfm-cli.html](https://agentix-e.github.io/agentix-timesfm-ts/api/modules/timesfm-cli.html)

Key exports:

- CLI entry point (Commander-based) — `timesfm setup`, `timesfm forecast`, `timesfm info`
- `csvForecast` — Programmatic CSV forecasting with full config control
- `CSVForecastOptions` / `CSVForecastLogger` — TypeScript types for programmatic CSV forecasting

## License

Apache 2.0
