# Contributing to agentix-timesfm-ts

## Development Environment

```bash
git clone https://github.com/AgentiX-E/agentix-timesfm-ts.git
cd agentix-timesfm-ts
pnpm install && pnpm build
```

**Prerequisites**: Node.js ≥ 20.x, pnpm ≥ 10.x

## Project Structure

```
agentix-timesfm-ts/
├── packages/
│   ├── timesfm-core/       # Core inference engine
│   ├── timesfm-xreg/       # Covariate regression
│   ├── timesfm-cli/        # CLI tool
│   └── timesfm-web/        # Browser inference engine (WASM/WebGPU)
├── docs/                   # Documentation
├── scripts/                # Pipeline/export scripts
├── .github/workflows/      # CI/CD automation
└── models/                 # ONNX models (gitignored)
```

## Development Workflow

```bash
# Install dependencies and build
pnpm install && pnpm build

# Before committing — auto lint + format via pre-commit hook
# (husky + lint-staged run automatically on git commit)

# Run all tests (requires ONNX model)
pnpm test

# Watch mode
pnpm test:watch

# Unit tests only (no model needed, fast — covers pure logic)
pnpm test:unit

# Coverage report
pnpm test:coverage

# Lint
pnpm lint
pnpm lint:fix

# Format
pnpm format
pnpm format:check

# Local CI simulation (mirrors CI fast checks: build + lint + format + unit tests with coverage)
pnpm ci:local

# Full CI simulation (includes integration tests — requires ONNX model)
pnpm ci

# Full local verification (adds integration tests + full coverage)
pnpm ci:full

# One-click full pipeline (model export + tests + benchmarks)
pnpm run pipeline

# Manual precommit check (same checks that git hook runs)
pnpm precommit
```

### Pre-commit Hook

On `git commit`, **husky + lint-staged** auto-runs:

- `eslint --fix` on staged `.ts/.js` files
- `prettier --write` on staged `.ts/.js/.json/.md/.yaml` files

This guarantees format/lint issues never reach CI. To bypass in emergencies:

```bash
git commit --no-verify
```

## Model Management

```bash
# Export ONNX model
pnpm export:model

# Check latest HuggingFace version
pnpm check:latest

# Validate existing models
pnpm check:model

# Run inference benchmarks
pnpm benchmark
```

## Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new feature
fix: fix bug in X
docs: update documentation
test: add tests for Y
chore: update dependencies
perf: improve performance of Z
refactor: restructure module W
```

## Code Style

- TypeScript strict mode
- Functions over classes (utils module)
- Float32Array/Uint8Array for all tensor operations
- No `any` types (unless required by ONNX Runtime interfaces)
- Clear export naming: `camelCase` function names, `PascalCase` class names
- Interfaces use `I` prefix (e.g. `ITimesFMModel`, `IInferenceEngine`)

## Adding New Features

1. Implement in the corresponding `packages/*/src/`
2. Write vitest tests (unit + integration)
3. Run `pnpm ci` to confirm CI passes
4. Update relevant documentation
5. Submit PR

## Model Updates

TimesFM model versions are published on HuggingFace. Check for updates:

```bash
pnpm check:latest
```

Export new model and run full regression:

```bash
pnpm run pipeline
```

## Versioning & Publishing

This project uses [Changesets](https://github.com/changesets/changesets) for version management:

```bash
# 1. Create a changeset (describes your changes)
pnpm changeset

# 2. Version packages (updates package.json versions + CHANGELOGs)
pnpm changeset version

# 3. Publish to npm (run by CI on release)
pnpm changeset publish
```

All four packages (`timesfm-core`, `timesfm-xreg`, `timesfm-cli`, `timesfm-web`) are **fixed together** — they always share the same version number. When creating a changeset, specify which packages are affected; the version bump will be applied uniformly.

**Important**: CI automatically publishes to npm when release tags are pushed. The release workflow (`release.yml`) handles OIDC-based npm provenance attestation.
