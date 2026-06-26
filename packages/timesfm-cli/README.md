# @agentix-e/timesfm-cli

Command-line interface for TimesFM 2.5 — zero-shot time-series forecasting directly from CSV files. No code required.

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

### `timesfm setup`

Download the TimesFM 2.5 ONNX model. Subsequent runs use the cached copy.

```bash
timesfm setup                        # → ~/.cache/agentix-timesfm-ts/
timesfm setup -o ./my-model.onnx     # custom path
timesfm setup -f                     # force re-download
```

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
5. Auto-download to default cache

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

| Variable             | Description                 |
| -------------------- | --------------------------- |
| `TIMESFM_MODEL_PATH` | Path to the ONNX model file |

## License

Apache 2.0
