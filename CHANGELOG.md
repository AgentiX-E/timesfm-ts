# Changelog

All notable changes to agentix-timesfm-ts will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Integrated benchmark jobs in `ci.yml`** — Node.js and WASM benchmarks run as part of the main CI pipeline with GitHub Pages deployment, accuracy gate, and regression detection. All benchmark functionality is consolidated in `ci.yml` (no standalone benchmark workflow needed).
- **`--proxy-password` CLI flag** for `timesfm setup` — proxy authentication password can now be passed via CLI argument (in addition to the existing `TIMESFM_PROXY_PASSWORD` environment variable).
- Coverage threshold enforcement in CI unit-test and integration-test jobs — both workflows now explicitly check ≥95% on all four metrics (lines, branches, functions, statements) with gate failure.
- `vitest.unit.config.ts` now generates `lcov` coverage format for CI artifact compatibility.
- **`web-engine.ts`** and **`model-loader.ts`** added to TypeDoc entry points for comprehensive browser API documentation.
- `vitest.globalSetup.ts` migrated to ESM-compatible `import.meta.url` pattern (replaced `__dirname`).

### Changed

- **Removed standalone web-benchmark landing page** per project requirements — WASM benchmark data is now only presented in the combined Node.js vs WASM comparison report (`docs/benchmark/index.html`).
- **`prepare-pages.js`** — simplified to no longer generate `docs/web-benchmark/index.html` (only ensures directory exists for WASM data files).
- **`docs/index.html` root landing page** — merged Web/WASM benchmark link into the main Benchmark Reports card for consolidated navigation.
- **CI `deploy-pages` job** — generate-combined-report step now has file-existence fallback for missing benchmark data.
- **`pnpm ci` and `pnpm ci:local`** commands unified — both now run build + lint + format + unit tests with coverage (matching CI pre-merge checks exactly).
- **`pnpm ci:full`** — consolidated to build + lint + format + unit coverage + full coverage (removed redundant intermediate test run).
- **`vitest.config.ts` coverage excludes** — no longer excludes `model.ts`, `onnx-engine.ts`, and `timesfm-web/src/**` from coverage (these are covered by integration tests with real ONNX model).
- **`model-release.yml`** — fixed Node.js version from non-existent `24` to `22`; initial `timesfm-latest` release creation no longer pushes a git tag (avoids protected-branch conflicts).
- **`web-engine.ts`** — output name resolution now uses dynamic `resolveOutputName()` matching the ONNX engine, supporting models with non-standard naming conventions.
- **`benchmark-ci.js`** — fixture generators now imported from `test-fixtures.ts` instead of inline duplication (single source of truth, same seed=42 determinism).

### Fixed

- **P0-P2 comprehensive fixes** by @Lambertyan — comprehensive audit-driven quality improvements
  - Accuracy Gate in CI benchmark workflow (fails if TimesFM MAE ≥ naive baseline)
  - Real-world test fixtures in accuracy benchmark (businessMetric, stockPrice, hourlyTemp, eCommerce, regimeShift)
  - Auto-discovering test globs in vitest.unit.config.ts (wildcard include + exclude)
  - Coverage landing page with link to detailed lcov-report on GitHub Pages
- **P2 improvements** by @Lambertyan — concurrency safety and benchmark precision
  - `skipWarmup` option on `TimesFMModel.fromPretrained()` / `IInferenceEngine.load()` for precise benchmark cold-start measurement
  - Concurrency stress test: 4 test scenarios verifying ONNX Runtime session concurrent safety (concurrent forecasts, different inputs, flip invariance, high-concurrency batch)
  - CI deploy-pages shell-escaping fix: inline `node -e` replaced with `scripts/prepare-pages.js`
  - Unit test coverage config with 95% thresholds in `vitest.unit.config.ts`, `pnpm test:unit:coverage` command
- CLI proxy support documentation in `packages/timesfm-cli/README.md`
- `AUDIT_REPORT.md` — comprehensive codebase audit against 10 quality dimensions
- `COMPREHENSIVE_IMPROVEMENT_PLAN.md` — full actionable improvement plan with priority matrix

### Changed

- **xreg-engine.ts**: Normalization uses numerically-stable two-pass variance algorithm (was one-pass E[X²]−E[X]², susceptible to catastrophic cancellation)
- **onnx-engine.ts**: `load()` now accepts `{ skipWarmup?: boolean }` option for benchmark precision; `IInferenceEngine` interface updated
- **ci.yml**: Removed redundant `pnpm test` step (coverage run already validates all tests, saving ~3 min CI wall time)
- **model-release.yml**: Replaced `gh release delete` + `git push -f` (breaks on protected branches) with unique HF-revision tags and `gh release upload --clobber` for `timesfm-latest`

### Fixed

- CI `deploy-pages` job shell-escaping error (inline `node -e` with JS template literals interpreted as bash command substitution)
- Prettier formatting consistency across all modified files (v3.8.4, matching lockfile)
- Local/CI test consistency: `pnpm ci:local` now runs with coverage checks matching CI behavior

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
