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
│   └── timesfm-cli/        # CLI tool
├── docs/                   # Documentation
├── scripts/                # Pipeline/export scripts
├── .github/workflows/      # CI/CD automation
└── models/                 # ONNX models (gitignored)
```

## Development Workflow

```bash
# Install dependencies and build
pnpm install && pnpm build

# Run all tests
pnpm test

# Watch mode
pnpm test:watch

# Unit tests only (no models needed, fast)
pnpm test:unit

# Coverage report
pnpm test:coverage

# Lint
pnpm lint
pnpm lint:fix

# Format
pnpm format
pnpm format:check

# One-click full pipeline (model export + tests + benchmarks)
pnpm run pipeline

# Quick mode (tests + benchmarks only, no model re-export)
pnpm run pipeline:quick

# CI local simulation
pnpm ci
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
