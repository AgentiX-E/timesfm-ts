# TimesFM Model Update & Full Regression Test Guide

> This document describes how to automatically fetch the latest TimesFM model, export it to ONNX, and run a complete regression test.
>
> **No make / bash required** — provides both Node.js and pnpm entry points.

---

## I. One-Click Full Pipeline

```bash
# Node.js (project native language)
node scripts/pipeline.js

# pnpm script (simplest)
pnpm run pipeline
```

**Pipeline executes 5 phases automatically:**

```
Phase 1 ─── Check HuggingFace latest version
    │         python scripts/export-onnx.py --check-latest
    │         → Output: model ID, SHA revision, last update time
    ▼
Phase 2 ─── Export TimesFM → ONNX (≈30s)
    │         python scripts/export-onnx.py
    │         → Output: models/timesfm-2.5.onnx (≈885 MB)
    │         → Validation: PyTorch vs ONNX max_diff < 1e-3
    ▼
Phase 3 ─── ONNX Runtime model load test
    │         node packages/timesfm-core/scripts/check-model.js --bench
    │         → Output: load time, input/output shapes, inference latency
    ▼
Phase 4 ─── Full regression test (111 tests)
    │         npx vitest run
    │         → Output: pass/fail stats, coverage report
    ▼
Phase 5 ─── Inference benchmark
              node packages/timesfm-core/scripts/real-benchmark.js
              → Output: latency (P50/P95/P99), Scaled MAE, memory usage
```

---

## II. Quick Commands

| Command                       | Purpose                                   |
| ----------------------------- | ----------------------------------------- |
| `pnpm run pipeline`           | **One-click full pipeline** (Recommended) |
| `node scripts/pipeline.js`    | Node.js full pipeline                     |
| `pnpm run pipeline:quick`     | Quick mode (skip export)                  |
| `pnpm run pipeline:export`    | Export ONNX only                          |
| `pnpm run pipeline:test`      | Run regression tests only                 |
| `pnpm run pipeline:benchmark` | Run inference benchmark only              |
| `pnpm run check:latest`       | Query HF latest version                   |
| `pnpm test`                   | Run vitest tests only                     |
| `pnpm run pipeline`           | **One-click full pipeline** (Recommended) |

---

## III. Python Export Script

`scripts/export-onnx.py` supports the following options:

```bash
# Check latest version
python scripts/export-onnx.py --check-latest

# Standard export (default google/timesfm-2.5-200m-pytorch)
python scripts/export-onnx.py

# Specify model and output path
python scripts/export-onnx.py \
  -m google/timesfm-2.5-200m-pytorch \
  -o models/timesfm-2.5.onnx

# Validate existing ONNX file only
python scripts/export-onnx.py --validate-only -o models/timesfm-2.5.onnx

# Skip ONNX validation (faster)
python scripts/export-onnx.py --skip-validation
```

**Export internal steps:**

```
1. Download TimesFM 2.5 200M PyTorch weights from HuggingFace
2. Wrap with TimesFMWrapper (unified [B,P,64] single input)
3. torch.onnx.export (dynamo=True, opset=18)
4. Merge external data files → single-file ONNX
5. Validate: ONNX Runtime vs PyTorch per-output max_diff < 1e-3
6. Write metadata JSON (version, timestamp, shape info)
```

---

## IV. Environment Variables

| Variable             | Default                           | Description          |
| -------------------- | --------------------------------- | -------------------- |
| `TIMESFM_MODEL_PATH` | `models/timesfm-2.5.onnx`         | ONNX model file path |
| `TIMESFM_HF_MODEL`   | `google/timesfm-2.5-200m-pytorch` | HuggingFace model ID |

```bash
# Custom path — all three are equivalent
TIMESFM_MODEL_PATH=./my-model.onnx pnpm run pipeline
TIMESFM_MODEL_PATH=./my-model.onnx node scripts/pipeline.js
TIMESFM_MODEL_PATH=./my-model.onnx python scripts/pipeline.py
```

---

## V. Check Model Version

```bash
# Three ways
npm run check-latest
node scripts/pipeline.js --check-latest
python scripts/pipeline.py --check-latest

# View metadata of exported model
cat models/timesfm-2.5.onnx.meta.json
```

Output example:

```json
{
  "source": "google/timesfm-2.5-200m-pytorch",
  "exported_at": "2026-06-25T14:02:00Z",
  "size_mb": 885.3,
  "input_shape": "[1, 16, 64]",
  "hf_revision": "abc123def456"
}
```

---

## VI. Update to Latest TimesFM Version

```bash
# 1. Check if there is a newer version
pnpm run check:latest

# 2. If there is an update, run full pipeline
pnpm run pipeline

# 3. After confirming tests pass, commit model metadata
git add models/timesfm-2.5.onnx.meta.json
git commit -m "chore: update TimesFM ONNX model (HF rev: $(cat models/timesfm-2.5.onnx.meta.json | python3 -c 'import json,sys; print(json.load(sys.stdin)["hf_revision"])'))"
```

---

## VII. CI/CD Integration Example

```yaml
# .github/workflows/model-test.yml
name: TimesFM Model Regression
on:
  schedule:
    - cron: '0 6 * * 1' # Every Monday 6 AM
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '22' }
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install "timesfm[torch]" onnx onnxruntime torch
      - run: pnpm install && pnpm build
      - run: pnpm run pipeline
```

---

## VIII. Troubleshooting

| Issue                     | Solution                                                              |
| ------------------------- | --------------------------------------------------------------------- |
| `HF API check skipped`    | Offline environment — normal, does not affect export                  |
| `ONNX export FAILED`      | Check PyTorch/timesfm version: `pip list \| grep timesfm`             |
| `OOM during export`       | Reduce MODEL_PATCHES (edit export-onnx.py) or increase RAM            |
| `Test suite FAILED`       | View failure details: `npx vitest run --reporter=verbose`             |
| `Model too large for git` | ONNX files should not be committed to Git — only commit metadata JSON |

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│          pnpm run pipeline / node scripts/pipeline.js     │
│                      │                                   │
│     ┌────────────────┼────────────────┐                 │
│     ▼                ▼                ▼                 │
│  Phase 1-2         Phase 3-4         Phase 5            │
│  (Python)          (Node.js)         (Node.js)          │
│                      │                                   │
│  export-onnx.py      check-model.js    real-benchmark.js│
│       │              vitest (111 tests)                  │
│       ▼              │                  │               │
│  timesfm-2.5.onnx    │                  │               │
│  (885 MB)            │                  │               │
│       │              ▼                  ▼               │
│       └──────────→ ONNX Runtime ←────── Result report   │
│                      │                                   │
│                      ▼                                   │
│               ✅ All passed / ❌ Failure details         │
└──────────────────────────────────────────────────────────┘
```
