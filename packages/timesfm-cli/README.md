# @agentix-e/timesfm-cli

Command-line interface for TimesFM 2.5 — zero-shot time-series forecasting directly from CSV files. No code required.

[![npm](https://img.shields.io/npm/v/@agentix-e/timesfm-cli?color=blue)](https://www.npmjs.com/package/@agentix-e/timesfm-cli)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../../LICENSE)

📚 [API Documentation](https://agentix-e.github.io/agentix-timesfm-ts/api/modules/timesfm_cli.html) · 💻 [Source](https://github.com/AgentiX-E/agentix-timesfm-ts)

```bash
npm install -g @agentix-e/timesfm-cli
```

## Quick Start

```bash
# One-time: download the model (~885 MB, cached for future runs)
timesfm setup

# Forecast the next 24 steps
timesfm forecast --horizon 24 sales.csv
```

## Commands

### `timesfm info`

Show model metadata and system information.

```bash
timesfm info                    # Auto-detect model location
timesfm info -m ./model.onnx    # Explicit model path
```

### `timesfm setup`

Download the TimesFM 2.5 ONNX model. Subsequent runs use the cached copy.

```bash
timesfm setup                        # → ~/.cache/agentix-timesfm-ts/
timesfm setup -o ./my-model.onnx     # custom path
timesfm setup -f                     # force re-download
```

#### Proxy Support (corporate / restricted networks)

```bash
# Option A: Standard environment variables (auto-detected)
export HTTPS_PROXY=http://proxy.company.com:8080
timesfm setup

# Option B: Explicit proxy with authentication
timesfm setup --proxy-url http://proxy.company.com:8080
timesfm setup --proxy-url http://proxy.company.com:8080 --proxy-username user
# Password via CLI (convenience) or environment variable (recommended for security):
TIMESFM_PROXY_PASSWORD=pass timesfm setup --proxy-url http://proxy:8080 --proxy-username user
timesfm setup --proxy-url http://proxy:8080 --proxy-username user --proxy-password pass

# Option C: TIMESFM-specific environment variables
TIMESFM_PROXY_URL=http://proxy:8080 TIMESFM_PROXY_USERNAME=user TIMESFM_PROXY_PASSWORD=pass timesfm setup

# Option D: All proxy parameters via CLI (password in CLI for convenience, but prefer env var)
timesfm setup --proxy-url http://proxy:8080 --proxy-username user --proxy-password pass
```

Proxy resolution priority: `--proxy-url` → `TIMESFM_PROXY_URL` → `HTTPS_PROXY` → `https_proxy` → `HTTP_PROXY` → `http_proxy`. `NO_PROXY` / `no_proxy` are respected. Password priority: `--proxy-password` → `TIMESFM_PROXY_PASSWORD`. For security, prefer the environment variable to avoid exposing credentials in shell history.

### `timesfm forecast`

| Flag                            | Type            | Default      | Description                                          |
| ------------------------------- | --------------- | ------------ | ---------------------------------------------------- |
| `-H, --horizon`                 | number          | **required** | Forecast steps (e.g. `24` = next 24 data points)     |
| `-m, --model`                   | path            | auto         | TimesFM ONNX model path (auto-download if omitted)   |
| `-d, --date-col`                | string          | `date`       | Name of the date column                              |
| `-v, --value-cols`              | string          | all numeric  | Comma-separated column names to forecast             |
| `-o, --output`                  | path            | stdout       | Output file path                                     |
| `--output-format`               | `csv` \| `json` | `csv`        | Output format                                        |
| `--context`                     | number          | `1024`       | Max context length (longer = more history consumed)  |
| `--no-normalize`                | flag            | —            | Disable input normalisation                          |
| `--no-flip-invariance`          | flag            | —            | Disable flip invariance (2× faster, less calibrated) |
| `--no-positive`                 | flag            | —            | Allow negative forecasts                             |
| `--no-fix-quantile-crossing`    | flag            | —            | Skip monotonic quantile enforcement                  |
| `--no-continuous-quantile-head` | flag            | —            | Use simpler quantile head                            |

### Model Path Resolution

The `--model` flag is optional. The CLI resolves the model in this order:

1. `--model <path>` flag
2. `TIMESFM_MODEL_PATH` environment variable
3. Path from the last `timesfm setup -o <path>` in the current session
4. Default cache: `~/.cache/agentix-timesfm-ts/timesfm-2.5.onnx`
5. Auto-download to default cache (proxy settings auto-detected from environment variables)

## CSV Input

```csv
date,sales,revenue
2024-01-01,100,5000
2024-01-02,102,5100
2024-01-03,105,5250
2024-01-04,,            ← empty = NaN, auto-interpolated
2024-01-05,110,5500
```

- First column defaults to dates (configurable with `--date-col`)
- Empty cells are linearly interpolated
- Each numeric column is forecast independently

## CSV Output

```csv
series_id,horizon_step,point_forecast,q10,q50,q90
sales,1,112.3,108.1,112.3,116.5
sales,2,114.1,109.3,114.1,118.9
revenue,1,5600.0,5400.0,5600.0,5800.0
```

## JSON Output

```bash
timesfm forecast -H 24 -o forecast.json --output-format json sales.csv
```

```json
{
  "model": "timesfm-2.5",
  "horizon": 24,
  "series": {
    "sales": {
      "point_forecast": [112.3, 114.1, ...],
      "lower_80": [108.1, 109.3, ...],
      "upper_80": [116.5, 118.9, ...],
      "quantiles": {
        "q10": [...], "q20": [...], "q50": [...], "q90": [...]
      }
    }
  }
}
```

## Environment Variables

| Variable                 | Description                                         |
| ------------------------ | --------------------------------------------------- |
| `TIMESFM_MODEL_PATH`     | Path to the ONNX model file                         |
| `TIMESFM_PROXY_URL`      | Proxy URL for model download                        |
| `TIMESFM_PROXY_USERNAME` | Proxy authentication username                       |
| `TIMESFM_PROXY_PASSWORD` | Proxy authentication password (env var recommended) |
| `HTTPS_PROXY`            | Standard proxy (auto-detected as fallback)          |

## License

Apache 2.0
