# TimesFM Model Update Guide

> This document describes the automated and manual model update flows for agentix-timesfm-ts.

---

## Automated Model Updates (Default)

The project uses a **dual-channel release architecture**:

### Code Channel
```
git tag v* → release.yml
              ├─ quality (lint + test)
              ├─ publish-npm (OIDC + provenance)
              └─ github-release (TypeDoc + release notes)
```
Code releases publish npm packages only. They do **not** include ONNX model files.

### Model Channel
```
nightly.yml (cron: 2 AM UTC daily)
    │
    ├─ Compare HF revision of google/timesfm-2.5-200m-pytorch
    │  against committed models/model-descriptor.json
    │
    ├─ New revision detected? → Trigger model-release.yml
    │     ├─ detect (idempotency check via model-<sha> tag)
    │     ├─ export-model (PyTorch → ONNX with validation)
    │     ├─ validate (full test suite + benchmark)
    │     ├─ github-release (model-<sha> + model-latest tags)
    │     └─ update-manifest (commit descriptor → auto-close issue)
    │
    └─ No change → no-op
```

### Download Channel
```
npm install @agentix-e/timesfm-cli
npx timesfm setup
    │
    └─ Downloads from: github.com/.../releases/download/model-latest/timesfm-2.5.onnx
       + model-descriptor.json for SHA-256 verification
```

The `model-latest` tag is a rolling pointer that always points to the most recently validated model release. Users always get the latest model without upgrading their npm package.

---

## Manual Model Update

If you need to manually trigger a model export and release:

### 1. Trigger model release workflow

```bash
# Automatic (force re-release of current HF revision)
gh workflow run model-release.yml -f force=true

# Or via GitHub UI: Actions → Model Release → Run workflow
```

### 2. Local export for testing

```bash
# Full pipeline: check HF → export → validate → test → benchmark
pnpm run pipeline

# Export only (skip tests)
pnpm run pipeline:export

# Export with descriptor only (no ONNX regeneration)
python3 scripts/export-onnx.py --descriptor-only
```

### 3. Update the committed descriptor

```bash
# After local export with a new HF revision
git add models/model-descriptor.json
git commit -m "chore(model): update descriptor to HF rev <sha>"
```

---

## ModelDescriptor Contract

The `model-descriptor.json` file (committed to the repo and distributed with each ONNX release) defines the architecture contract between the model and the TypeScript engine:

| Field | Source | Purpose |
|-------|--------|---------|
| `schema` | Constant (1) | Forward compatibility version |
| `model.hf_revision` | HuggingFace API | Traceability to exact PyTorch checkpoint |
| `onnx.input_shape` | ONNX graph | Runtime shape validation |
| `onnx.sha256` | Computed from ONNX file | Download integrity verification |
| `architecture.*` | PyTorch model params | Configures the TypeScript engine |
| `processing.*` | Hardcoded per model | Pre/post-processing strategy flags |

The engine reads this descriptor at runtime via `loadModelDescriptor()` and converts it to a `ModelConfig` via `descriptorToModelConfig()`. All hardcoded architecture constants in the TypeScript code have been eliminated — the descriptor is the single source of truth.

---

## Version Compatibility

| Schema | Engine Requirement | Notes |
|--------|-------------------|-------|
| 1 | `@agentix-e/timesfm-core` ≥ 0.2.1 | Current |
| > 1 | Upgrade required | Engine logs warning and falls back to `TIMESFM_25_CONFIG` |

If a future TimesFM model version has a different architecture, updating `export-onnx.py` to generate the new descriptor is sufficient — no TypeScript code changes are needed as long as the schema version is compatible.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Nightly check not detecting changes | Verify `models/model-descriptor.json` is committed with the current HF revision |
| Model release fails | Check the workflow run logs; force re-run with `force: true` |
| `downloadModel()` fails | Verify `model-latest` release exists and contains both `.onnx` and `model-descriptor.json` |
| Engine rejects model | Descriptor schema > `ENGINE_SUPPORTED_SCHEMA` — upgrade `@agentix-e/timesfm-core` |
| Local tests need model | Export locally: `python3 scripts/export-onnx.py` or `pnpm run pipeline:export` |
