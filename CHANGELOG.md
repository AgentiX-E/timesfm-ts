# Changelog

All notable changes to agentix-timesfm-ts will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **P0 fixes** by @Lambertyan — comprehensive audit-driven quality improvements
  - Accuracy Gate in CI benchmark workflow (fails if TimesFM MAE ≥ naive baseline)
  - Real-world test fixtures in accuracy benchmark (businessMetric, stockPrice, hourlyTemp, eCommerce, regimeShift)
  - Auto-discovering test globs in vitest.unit.config.ts (wildcard include + exclude)
  - Coverage landing page with link to detailed lcov-report on GitHub Pages
- CLI proxy support documentation in `packages/timesfm-cli/README.md`
- `AUDIT_REPORT.md` — comprehensive codebase audit against 10 quality dimensions

### Changed

- **xreg-engine.ts**: Normalization uses numerically-stable two-pass variance algorithm (was one-pass E[X²]−E[X]², susceptible to catastrophic cancellation)
- **ci.yml**: Removed redundant `pnpm test` step (coverage run already validates all tests, saving ~3 min CI wall time)
- **model-release.yml**: Replaced `gh release delete` + `git push -f` (breaks on protected branches) with unique HF-revision tags and `gh release upload --clobber` for `timesfm-latest`

### Fixed

- Prettier formatting consistency across all modified files (v3.8.4, matching lockfile)

## [0.3.1] — 2026-06-26

### Added

- Combined Node.js vs WASM benchmark report on GitHub Pages
- Web (WASM) inference engine with WebGPU / WASM / WebGL backends
- Model descriptor system for self-describing ONNX models
- autoregressive decode loop with KV cache support
- RevIN normalization (Reversible Instance Normalization)
- NaN handling utilities (strip, interpolate, clean)
- Continuous quantile head for calibrated prediction intervals
- Flip invariance enforcement (f(−x) = −f(x))
- Quantile crossing fix for monotonic quantile bands
- CLI tool with `timesfm setup` and `timesfm forecast` commands
- CSV input/output support with auto NaN interpolation
- Model downloader with proxy support (3-tier cascade: options → env vars → standard vars)
- SHA-256 checksum verification for downloaded models
- Welford-style running statistics for numerically stable inference
- 4-package monorepo: `@agentix-e/timesfm-core`, `@agentix-e/timesfm-xreg`, `@agentix-e/timesfm-cli`, `@agentix-e/timesfm-web`
- Full CI/CD pipeline: lint, unit test (Node 20 + 22), build check, integration test, benchmark, deploy to GitHub Pages
- Nightly model version monitoring workflow
- Model release workflow with automated validation
- TypeScript strict mode with 95%+ coverage thresholds

### Documentation

- Comprehensive README with architecture diagram, quick start, config reference
- API documentation via TypeDoc, auto-published to GitHub Pages
- Getting Started and Model Update guides
- Individual package READMEs with API doc links

## [0.1.0] — Initial

- Initial project scaffold
- Core TimesFM model architecture definition
- ONNX Runtime inference engine integration
- Basic forecast pipeline (preprocess → inference → postprocess)
