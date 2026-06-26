# agentix-timesfm-ts

> Node.js/TypeScript reimplementation of Google Research's TimesFM — a decoder-only foundation model for zero-shot time-series forecasting.

[![CI](https://github.com/AgentiX-E/agentix-timesfm-ts/actions/workflows/ci.yml/badge.svg)](https://github.com/AgentiX-E/agentix-timesfm-ts/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-blue)](https://www.typescriptlang.org/)
[![Node.js](https://img.shields.io/badge/Node.js-%3E%3D20-green)](https://nodejs.org/)

## Overview

**agentix-timesfm-ts** brings Google's TimesFM 2.5 (200M parameters, decoder-only transformer) to the Node.js ecosystem. It provides zero-shot time series forecasting — feed it any univariate time series and get point forecasts with calibrated prediction intervals, no training required.

### Architecture

```
Raw Time Series → [Preprocessor] → [ONNX Runtime] → [Postprocessor] → Forecasts
                   (NaN cleaning,      (Trained       (Flip invariance,
                    patch splitting,     TimesFM 2.5     quantile calibration,
                    RevIN normalize)     model)          crossing fix, etc.)
```

### Key Features

- **Zero-shot forecasting** — no training needed
- **Point forecasts + 10 quantile bands** (mean, q10–q90)
- **Variable-length inputs** — different series lengths in one batch
- **Automatic NaN handling** — leading NaN stripped, internal NaN interpolated
- **Covariate support** — dynamic/static numerical & categorical exogenous variables (XReg)
- **Production-grade** — built on ONNX Runtime's native C++ backend (CPU, CUDA, DirectML)

## Packages

| Package                   | npm                                                                                                                              | Description                                                               |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `@agentix-e/timesfm-core` | [![npm](https://img.shields.io/npm/v/@agentix-e/timesfm-core?color=blue)](https://www.npmjs.com/package/@agentix-e/timesfm-core) | Core inference engine + preprocessing + postprocessing + model downloader |
| `@agentix-e/timesfm-xreg` | [![npm](https://img.shields.io/npm/v/@agentix-e/timesfm-xreg?color=blue)](https://www.npmjs.com/package/@agentix-e/timesfm-xreg) | Covariate regression extension (Ridge + OneHot)                           |
| `@agentix-e/timesfm-cli`  | [![npm](https://img.shields.io/npm/v/@agentix-e/timesfm-cli?color=blue)](https://www.npmjs.com/package/@agentix-e/timesfm-cli)   | CLI tool (includes `timesfm setup` auto model download)                   |

> **Layered strategy**: npm packages contain only code (~150 KB), models (885 MB zip / ~928 MB ONNX) are downloaded on-demand via GitHub Releases.

## Quick Start

### Option 1 — npm install (recommended, code only)

```bash
npm install @agentix-e/timesfm-cli

# Auto download model ~885 MB (first time only)
npx timesfm setup

# Forecast
npx timesfm forecast --horizon 24 data.csv
```

### Option 2 — Programmatic usage

```typescript
import { TimesFMModel, downloadModel, createForecastConfig } from '@agentix-e/timesfm-core';

// Auto download model (first time only, cached thereafter)
const modelPath = await downloadModel();

const model = await TimesFMModel.fromPretrained({ modelPath });
model.compile(createForecastConfig({ maxContext: 1024, maxHorizon: 256 }));

const { pointForecast, quantileForecast } = await model.forecast(24, [
  new Float32Array([1, 2, 3 /* ... */]),
]);
```

### Option 3 — Build from source + HuggingFace export

```bash
git clone https://github.com/AgentiX-E/agentix-timesfm-ts.git
cd agentix-timesfm-ts && pnpm install && pnpm build

# One-click pipeline
pnpm run pipeline
```

> Detailed docs: [docs/GETTING-STARTED.md](docs/GETTING-STARTED.md) | [docs/MODEL-UPDATE.md](docs/MODEL-UPDATE.md)

## ForecastConfig Reference

| Parameter                   | Type    | Default | Description                                |
| --------------------------- | ------- | ------- | ------------------------------------------ |
| `maxContext`                | number  | 1024    | Maximum context window (rounded to 32x)    |
| `maxHorizon`                | number  | 256     | Maximum forecast horizon (rounded to 128x) |
| `normalizeInputs`           | boolean | true    | Z-score normalize inputs                   |
| `useContinuousQuantileHead` | boolean | true    | Better prediction intervals                |
| `forceFlipInvariance`       | boolean | true    | Ensure f(-x) = -f(x)                       |
| `inferIsPositive`           | boolean | true    | Clamp forecasts ≥ 0 for positive inputs    |
| `fixQuantileCrossing`       | boolean | true    | Ensure monotonic quantiles                 |
| `returnBackcast`            | boolean | false   | Return input reconstruction                |

## Output Shape Reference

| Output                | Shape        | Description                       |
| --------------------- | ------------ | --------------------------------- |
| `pointForecast`       | `(B, H)`     | Median forecast                   |
| `quantileForecast`    | `(B, H, 10)` | Full distribution                 |
| `quantileForecast[0]` | `(B, H)`     | Mean                              |
| `quantileForecast[1]` | `(B, H)`     | 10th percentile                   |
| `quantileForecast[5]` | `(B, H)`     | 50th percentile (= pointForecast) |
| `quantileForecast[9]` | `(B, H)`     | 90th percentile                   |

## Project Structure

```
agentix-timesfm-ts/
├── packages/
│   ├── timesfm-core/           # Core inference engine
│   │   ├── src/
│   │   │   ├── index.ts        # Public API
│   │   │   ├── model.ts        # TimesFMModel class
│   │   │   ├── config.ts       # Configuration management
│   │   │   ├── types.ts        # Type definitions
│   │   │   ├── preprocessor.ts # Data preprocessing
│   │   │   ├── postprocessor.ts# Output postprocessing
│   │   │   ├── inference/
│   │   │   │   ├── onnx-engine.ts  # ONNX Runtime inference engine
│   │   │   │   ├── kv-cache.ts     # KV Cache management
│   │   │   │   └── decode-loop.ts  # Autoregressive decode
│   │   │   └── utils/
│   │   │       ├── nan-handler.ts  # NaN stripping/interpolation
│   │   │       ├── stats.ts        # Welford running statistics
│   │   │       ├── revin.ts        # RevIN normalization
│   │   │       └── tensor-utils.ts # Low-level tensor ops
│   │   └── test/               # 237 tests (unit) + 26 integration tests
│   ├── timesfm-xreg/           # Covariate regression
│   │   ├── src/
│   │   │   ├── index.ts
│   │   │   ├── xreg-engine.ts     # Ridge regression engine
│   │   │   └── one-hot-encoder.ts # Scikit-learn compatible OHE
│   │   └── test/               # 18 tests
│   └── timesfm-cli/            # CLI tool
│       ├── src/
│       │   ├── cli.ts          # Commander-based CLI
│       │   └── csv-forecast.ts # CSV I/O forecasting
│       └── test/
├── scripts/
│   ├── pipeline.js              # Node.js fully automated pipeline
│   └── export-onnx.py           # PyTorch → ONNX exporter
├── .github/
│   └── workflows/               # CI/CD automation
│       ├── ci.yml               # PR checks + integration tests
│       ├── release.yml          # npm publish + model GitHub Release
│       └── nightly.yml          # Daily model version monitoring
├── models/                      # ONNX models (gitignored)
└── vitest.config.ts
```

## Development

```bash
# Install
pnpm install && pnpm build

# One-click full pipeline (model export + tests + benchmarks)
pnpm run pipeline

# Run tests only
pnpm test
pnpm run test:watch

# Lint + benchmarks
pnpm run lint
pnpm run benchmark

# Check HF latest version
pnpm run check:latest
```

## References

- **Paper**: [A Decoder-Only Foundation Model for Time-Series Forecasting](https://arxiv.org/abs/2310.10688) (ICML 2024)
- **Original Project**: [google-research/timesfm](https://github.com/google-research/timesfm)
- **ONNX Runtime**: [onnxruntime.ai](https://onnxruntime.ai/)
- **HuggingFace Models**: [google/timesfm-2.5-200m-pytorch](https://huggingface.co/google/timesfm-2.5-200m-pytorch)

## System Requirements

| Component             | Minimum                 | Recommended                        |
| --------------------- | ----------------------- | ---------------------------------- |
| **OS**                | Linux / macOS / Windows | Linux (production)                 |
| **Node.js**           | ≥ 20.x                  | ≥ 22.x                             |
| **RAM**               | 4 GB                    | 8 GB+                              |
| **Disk (code)**       | 10 MB                   | —                                  |
| **Disk (model)**      | 1 GB                    | SSD                                |
| **GPU** (optional)    | 2 GB VRAM               | 4 GB+ VRAM (CUDA)                  |
| **Python** (optional) | ≥ 3.10                  | Only needed for HuggingFace export |

### Pre-install dependencies

| Usage method                          | Requires pre-install                                                  |
| ------------------------------------- | --------------------------------------------------------------------- |
| **npm install + auto model download** | Node.js ≥ 20 only                                                     |
| **Export model from HuggingFace**     | Python ≥ 3.10 + `pip install "timesfm[torch]" onnx onnxruntime torch` |
| **Build from source**                 | Node.js ≥ 20 + pnpm                                                   |

> `onnxruntime-node` includes prebuilt C++ native modules, supports Linux x64 / arm64, macOS x64 / arm64 (Apple Silicon), Windows x64. **No additional system packages required**.

## CLI Quick Reference

```bash
# Download model
timesfm setup                              # Default: ~/.cache/agentix-timesfm-ts/
timesfm setup -o ./models/my-model.onnx    # Custom path
timesfm setup -f                           # Force re-download

# Download with proxy (corporate / restricted networks)
# Option A: Standard environment variables (auto-detected)
export HTTPS_PROXY=http://proxy.company.com:8080
timesfm setup

# Option B: Explicit proxy with authentication
timesfm setup --proxy-url http://proxy.company.com:8080
timesfm setup --proxy-url http://proxy.company.com:8080 --proxy-username user
# Password is always read from environment variable (never in CLI args):
TIMESFM_PROXY_PASSWORD=pass timesfm setup --proxy-url http://proxy:8080 --proxy-username user

# Option C: TIMESFM-specific environment variables
TIMESFM_PROXY_URL=http://proxy:8080 TIMESFM_PROXY_USERNAME=user TIMESFM_PROXY_PASSWORD=pass timesfm setup

# Forecast (model path priority)
timesfm forecast --horizon 24 data.csv                     # Auto: cache → download
timesfm forecast -m ./custom.onnx --horizon 24 data.csv    # Explicit path
TIMESFM_MODEL_PATH=./prod.onnx timesfm forecast --horizon 24 data.csv  # Environment variable

# Model path resolution priority: ① --model ② $TIMESFM_MODEL_PATH ③ setup session ④ default cache ⑤ auto download
```

## License

This project is open source under [Apache 2.0](LICENSE).

### Relationship with Google TimesFM

- Google TimesFM ([google-research/timesfm](https://github.com/google-research/timesfm)) is licensed under **Apache 2.0**
- The TypeScript/Node.js code in this project is an **original implementation**, also released under Apache 2.0
- TimesFM pretrained model weights (downloaded from HuggingFace) follow Google's model license terms
- This project's `scripts/export-onnx.py` is used to help users export models, does not directly distribute model weights
- ONNX files in GitHub Releases are derivative works exported by users from HuggingFace

### License compatibility

| Component               | License             | Description        |
| ----------------------- | ------------------- | ------------------ |
| agentix-timesfm-ts code | Apache 2.0          | Fully original     |
| TimesFM model weights   | Apache 2.0 (Google) | HuggingFace hosted |
| ONNX Runtime            | MIT (Microsoft)     | npm dependency     |
| ml-matrix               | MIT                 | npm dependency     |
