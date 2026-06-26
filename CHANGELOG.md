# Changelog

All notable changes to agentix-timesfm-ts will be documented in this file.

## [Unreleased]

### Added

- **Evaluation metrics module** (`@agentix/timesfm-core/helpers/metrics`): MAE, RMSE, MAPE, SMAPE, MASE, R², PIC Coverage, PI Width
- **Quantile helpers** (`@agentix/timesfm-core/helpers/quantile`): `getQuantile()`, `getPredictionInterval()`
- **AbortSignal support**: `forecast()` accepts optional `{ signal }` for cancellation
- **Progress callback**: `forecast()` accepts optional `{ onProgress }` for phase-level progress
- **ESLint + Prettier**: Flat config (`eslint.config.mjs`) with `@typescript-eslint` strict rules
- **20+ decode-loop tests** with `MockInferenceEngine` (no ONNX model required)
- **Dependabot** config for automated npm and GitHub Actions dependency updates
- **`.npmrc`** with `engine-strict=true`

### Fixed

- **`normalizeInputs=false` bug**: `inputStats` array now correctly contains one entry per series
- **Wrong GitHub URLs**: All `package.json` repository URLs corrected to `AgentiX-E/agentix-timesfm-ts`
- **CI `continue-on-error` removed**: Unit test job now properly fails on test failures
- **CI model caching**: ONNX model export is cached between CI jobs
- **KVCache module** marked as `@experimental` with usage documentation

### Changed

- **CLI** now uses ESM `import` for `fs` module instead of `require()`
- **Lint & format** apply to all project files, not just source
- **`pnpm` engine** constraint added (>=10.0.0)
- **Model downloader** checksum loaded dynamically from metadata JSON

## [0.1.0] — 2026-06-25

### Initial release

- TimesFM 2.5 200M ONNX inference engine
- Preprocessing pipeline (NaN handling, patch splitting, RevIN normalization)
- Postprocessing pipeline (flip invariance, quantile crossing fix, positive clamping)
- Autoregressive decode loop
- Covariate regression extension (XReg)
- CLI tool (CSV/JSON I/O)
- Model downloader with streaming and caching
- 133 tests, 99.46% line coverage
