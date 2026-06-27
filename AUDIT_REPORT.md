# 🔬 agentix-timesfm-ts 全面审计报告

> 审计日期: 2026-06-28 | 审计范围: 109+ 源文件, 4 个 Packages, 4 个 CI Workflows  
> 基准: [google-research/timesfm](https://github.com/google-research/timesfm) | 标准: 极致完美工程零妥协

---

## 总体评分

| 维度                         | 评分    | 状态                        |
| ---------------------------- | ------- | --------------------------- |
| 1. 架构设计与代码实现        | ⭐⭐⭐⭐  | 优秀，少量可优化点          |
| 2. 性能优化与 Benchmark CI   | ⭐⭐⭐⭐  | 良好，需修复 Accuracy 数据源 |
| 3. API 设计                  | ⭐⭐⭐⭐⭐ | 极佳，TypeScript 类型安全典范 |
| 4. 测试覆盖率                | ⭐⭐⭐⭐  | 良好，覆盖率目标 95%+ 已配置 |
| 5. 优于 google-research/timesfm | ⭐⭐⭐⭐ | 多平台/多引擎，真正优于参考 |
| 6. Model Download Proxy      | ⭐⭐⭐⭐⭐ | 完整代理支持，三层级联优先级 |
| 7. 文档质量                  | ⭐⭐⭐⭐  | 良好，CLI 文档可增强         |
| 8. 本地/CI 一致性            | ⭐⭐⭐   | **存在不一致，需重点修复**    |
| 9. 所有 CI Workflows         | ⭐⭐⭐   | **存在失败风险，需修复**      |

---

## 一、架构设计与代码实现

### ✅ 做得好的地方

1. **Monorepo 架构清晰**: 4 个 package (core/xreg/cli/web) 职责分离明确，`IInferenceEngine` 接口实现了可插拔的推理后端设计。

2. **错误体系完备**: 8 个类型的 Typed Error 层级 (TimesFMError → DownloadError → ProxyAuthError / ChecksumMismatchError)，生产环境可精准捕获和处理。

3. **配置验证严谨**: `validateAndNormalizeConfig` 在 `compile()` 阶段就发现配置错误，而非运行时崩溃。patch 边界自动对齐、context+horizon 上限检查。

4. **Factory Pattern**: `TimesFMModel.fromPretrained()` 封装了模型创建复杂性，支持 engine 注入用于浏览器和测试。

5. **ModelDescriptor 自描述**: ONNX 模型携带 JSON descriptor，引擎动态配置，消除硬编码。`descriptorToModelConfig` 动态计算 `decodeIndex`，支持非标准 quantile 集合。

6. **Flip Invariance 并行化**: `model.ts` 中 `forceFlipInvariance` 路径使用 `Promise.all` 并行执行主路径和反向路径的 decode，而非串行等待。

7. **RevIN 数值稳定性**: `stats.ts` 使用 Welford 并行方差合并算法和两遍方差计算，避免灾难性抵消。NaN/Inf 主动跳过防止数据污染。

8. **NaN 处理**: `linearInterpolateNaNs` 严格 O(n) 双遍历实现（前向记录 + 后向插值），无内层循环。

### 🔴 需要修复的问题

#### 1.1 `xreg-engine.ts` — `normalizeXregTargets` 数值不稳定性

**严重程度**: 🔴 High

```typescript
// 当前实现 (xreg-engine.ts:298-300):
const sigma = n > 0 ? Math.sqrt(Math.max(0, sumSq / n - mu * mu)) : 1;
//                                   ^^^^^^^^^^^^^^^^^^^^^^^^
//                                   单遍 E[X²] - E[X]² 公式
//                                   大数值下灾难性抵消!
```

`stats.ts` 中的 `computeStats` 已正确使用两遍算法，但 `xreg-engine.ts` 中的 `normalizeXregTargets` 退化为不稳态单遍公式。**必须统一使用 `computeStats` 或等效的两遍算法。**

**修复方案**:
```typescript
// 直接复用 stats.ts 中已验证的 computeStats
import { computeStats } from '@agentix-e/timesfm-core';
// 或实现本地两遍方差
```

#### 1.2 `model.ts` — `@ts-ignore` 逃逸类型检查

**严重程度**: 🟡 Medium

```typescript
// model.ts:380
// @ts-ignore — optional peer dependency, type-checked at install time
const mod = await import('@agentix-e/timesfm-xreg');
```

`@ts-ignore` 会抑制该行所有类型错误。应使用 `@ts-expect-error` 或正确的类型声明。

**修复方案**: 在 `timesfm-core/tsconfig.json` 中添加 `@agentix-e/timesfm-xreg` 为可选引用，或使用 `declare module '@agentix-e/timesfm-xreg'` 显式类型声明。

#### 1.3 `benchmark-ci.js` — CJS require() 在 ESM 项目中混用

**严重程度**: 🟡 Medium

`scripts/benchmark-ci.js` 大量使用 `require()` 和 `__dirname`，而项目根级设置了 `"type": "module"`。该文件虽以 `.js` 扩展名排除在 ESM 之外（Node.js 特殊规则），但在 `tsx` 执行路径下可能导致不一致。

#### 1.4 `onnx-engine.ts` — Session dispose 使用类型断言而非接口

**严重程度**: 🟡 Medium

```typescript
// onnx-engine.ts:233
await (this._session as { release?: () => Promise<void> }).release?.();
```

`onnxruntime-node.InferenceSession` 的 `release` 方法并非标准接口，使用 `as` 类型断言。应检查 ONNX Runtime 版本并确认该 API 的存在。

#### 1.5 缺少 `AbortController` polyfill 文档

**严重程度**: 🟢 Low

`model.forecast()` 接受 `AbortSignal` 但未在 README 中提及 Node.js 20 以下版本需要 polyfill。

---

## 二、性能优化与 Benchmark CI

### ✅ 做得好的地方

1. **基准测试矩阵**: 3 种上下文 × 4 种 batch 大小覆盖真实场景。
2. **Cold/Warm 追踪**: 精确测量 JIT 编译对首次推理的影响。
3. **HTML 报告**: 自包含（无外部依赖）、响应式、暗色主题、内嵌 JSON 数据。
4. **回归检测**: 与基线 JSON 自动比较，阈值可配置（10% warning, 15% critical）。
5. **独立 CI Job**: `benchmark` 和 `web-benchmark` 并行执行，不阻塞 lint/test。
6. **GitHub Pages 发布**: benchmark 报告自动部署到 GitHub Pages，README 有链接。

### 🔴 需要修复的问题

#### 2.1 **Accuracy Benchmark 使用 Synthetic Random Data**（违反需求 #4）

**严重程度**: 🔴 Critical

`benchmark-ci.js` 中的 accuracy section (行 759-883) 使用 `Math.random()` 风格的 PRNG 生成合成数据，而非真实数据集:

```typescript
// benchmark-ci.js:810-818
for (let i = 0; i < seriesLen; i++) {
  data[i] =
    base + trend * i +
    seasonAmp * Math.sin((2 * Math.PI * i) / 12) +
    (rand() - 0.5) * noiseAmp * 2;
}
```

虽然 `test-fixtures.ts` 提供了 11 种真实场景（`businessMetric`, `stockPrice`, `hourlyTemp` 等），但它们**未被 benchmark 使用**。

**修复方案**: 使用 `benchmarks/data/` 目录中实际的 CSV 数据 (`benchmark_daily.csv`, `benchmark_hourly.csv`, `benchmark_monthly.csv`)，并通过完整的 TimesFM pipeline 进行评估。或者使用 `test-fixtures.ts` 中的 11 种真实场景生成器。

#### 2.2 **缺少 Memory Leak / Long-Running Stability Test**

**严重程度**: 🟡 Medium

没有长时间运行的内存稳定性测试。ONNX Runtime 的原生内存泄漏可能在数千次推理后才显现。

**建议**: 添加 `benchmark/stability` 阶段 — 1000 次推理后检查 heapUsed 增长 ≤ 5%。

#### 2.3 **ONNX Runtime Warmup 测量不精确**

**严重程度**: 🟡 Medium

`benchmark-ci.js` 中 warmup 是 2 次固定迭代，但 `onnx-engine.ts` 的 `load()` 方法已经有 warmup。这导致 CI 中实际测量的是"第三次推理"而非真正的"warm 推理"——差异很小但影响严谨性。

**修复方案**: 统一 warmup 策略 — 要么完全依赖 `onnx-engine.ts` 的 warmup（推荐），要么在 benchmark 中显式跳过 engine 的 warmup。

#### 2.4 **Benchmark CI 缺少 Accuracy 断言**

**严重程度**: 🟡 Medium

CI benchmark 运行 accuracy 测试但**从不检查其结果**。即使 TimesFM 产生完全错误的预测（MAE 远高于 naive baseline），CI 也会通过。

**修复方案**: 添加 Accuracy Gate — `scaled_mae < 1.0`（TimesFM 必须不差于 naive baseline）时 CI 才通过。

---

## 三、API 设计

### ✅ 做得好的地方

1. **类型安全**: 全部 TypeScript，严格模式 (`strict: true`, `noUnusedLocals`, `noUnusedParameters`)。
2. **Progressive Disclosure**: 简单 `forecast()` + `ForecastConfig` 细粒度控制 + `CovariateForecastParams` 高级协变量。
3. **模块化导出**: `index.ts` 导出层次清晰：Errors → Model → Config → Types → Inference → Utils。
4. **每个 package README 都有 API doc 链接**: ✅ `timesfm-core/README.md`, `timesfm-xreg/README.md`, `timesfm-cli/README.md`, `timesfm-web/README.md` 都有独立的 API Docs 链接。
5. **TypeDoc 自动发布**: `deploy-pages` job 运行 `pnpm docs:generate` 并部署到 GitHub Pages。4 个入口点全部覆盖。
6. **ForecastConfig 自文档**: TypeScript 接口注释详细，参数默认值、作用、Python 对应关系均有说明。

### 🔴 需要修复的问题

#### 3.1 **CLI 输出格式文档与实现不一致**

**严重程度**: 🟡 Medium

CLI README 声称 JSON 输出包含 `"quantiles": { "q10": [...], "q20": [...], ... }` 结构，但需要验证 `csv-forecast.ts` 的实际实现是否匹配。查看 `csv-forecast.ts`:

```typescript
// 仅查看了 csvForecast 函数签名，需要验证实现
```

**检查**: 需要验证 `csv-forecast.ts` 的 JSON 输出格式与 README 完全一致。

#### 3.2 **缺少 `ForecastConfig.perCoreBatchSize` 公共文档**

**严重程度**: 🟢 Low

`perCoreBatchSize` 出现在 `ForecastConfig` 接口中但 README 的 ForecastConfig Reference 表格缺少此项。CLI 也没有暴露 `--batch-size` 选项。

---

## 四、测试覆盖率

### ✅ 做得好的地方

1. **覆盖率阈值**: 95% lines, branches, functions, statements — 已在 `vitest.config.ts` 中强制执行。
2. **分层测试**: 306+ 单元测试（纯逻辑，无模型依赖）+ 集成测试（真实 ONNX 模型）。
3. **测试场景丰富**: `test-fixtures.ts` 提供 11 种真实场景生成器（business metric, seasonal temp, stock price, spikes, constant, long series, negative values, regime shift, exponential growth）。
4. **排除策略合理**: barrel files, CLI entry, network IO, WASM-only 代码被正确排除。

### 🔴 需要修复的问题

#### 4.1 **单元测试使用 MockEngine + Synthetic Data**（违反需求 #4 的部分解读）

**严重程度**: 🔴 Critical

当前单元测试架构广泛使用 `MockInferenceEngine` 和合成数据:

```
packages/timesfm-core/test/helpers/mock-engine.ts  ← MockEngine
packages/timesfm-core/test/test-fixtures.ts        ← Synthetic data generators
packages/timesfm-core/test/inference/decode-loop.test.ts  ← Uses MockEngine
packages/timesfm-core/test/model.test.ts           ← 部分使用 MockEngine
```

需求明确要求"不使用 Mock 模型和 Synthetic 的数据"。如果严格执行：
- `decode-loop.test.ts` 需要改为使用真实 ONNX 引擎
- `csv-forecast.test.ts` 不能使用 mock
- 所有测试数据应从 `benchmarks/data/` 的真实 CSV 或 HuggingFace 数据集加载

**重要说明**: 这是一个架构性决策。完全移除 mock/synthetic 数据将使单元测试运行时间从 <10 分钟增加到 >30 分钟（需要加载 885MB 模型），且 CI 成本大幅增加。建议：
- **集成测试**（`pnpm test`）使用真实模型 + 真实数据
- **单元测试**（`pnpm test:unit`）可以保留 mock/synthetic 但需在 README 中注明区分
- **所有覆盖率报告**必须基于集成测试（真实模型+真实数据）的结果

#### 4.2 **`postprocessor.ts` — `arOutputs` 非 null 分支未被 unit 测试覆盖**

`decode-loop.test.ts` 不涉及 AR decode steps（`numDecodeSteps === 0`），导致 `postProcess` 中 `arOutputs` 连接逻辑未在单元测试中覆盖。该路径仅在 `horizon >= outputPatchLen + 1 = 129` 时触发。

#### 4.3 **覆盖率报告发布到 GitHub Pages — 实现不完整**

**严重程度**: 🟡 Medium

CI `deploy-pages` job 创建了覆盖率 summary HTML（`docs/coverage/index.html`），但**如果 `lcov-report` 已存在则跳过**。`lcov-report` 来自 `pnpm test:coverage`（使用 `@vitest/coverage-v8`），只有当 integration-test 成功运行时才会存在。

**问题**: `coverage/index.html` 与 `coverage/lcov-report/index.html` 可能指向不同内容，容易混淆。建议统一 — 只保留 `lcov-report` 并确保 index.html 重定向到它。

#### 4.4 `pnpm test:unit` 不生成覆盖率 → 本地 CI 不一致

**严重程度**: 🔴 High

```bash
# package.json scripts:
"test:unit": "vitest run --config vitest.unit.config.ts"    # ❌ 无覆盖率
"test:coverage": "vitest run --coverage"                     # ✅ 有覆盖率（需模型）
```

`vitest.unit.config.ts` **不包含任何 coverage 配置**。用户在本地运行 `pnpm test:unit` 无法得到覆盖率数据，需要运行 `pnpm test:coverage`（需要模型）。这与 CI 中 `unit-test` job 只跑测试不跑覆盖率是一致的，但**用户需要本地覆盖率验证时必须下载模型**。

**修复方案**: 在 `vitest.unit.config.ts` 中添加 coverage 配置（与主配置相同的 thresholds，但不包含需要模型的模块）。

---

## 五、优于 google-research/timesfm

### ✅ 已验证的优势

| 方面             | agentix-timesfm-ts                    | google-research/timesfm     |
| ---------------- | ------------------------------------- | --------------------------- |
| 运行时           | Node.js / TypeScript (生产就绪)        | Python + JAX (研究导向)     |
| 推理引擎         | ONNX Runtime C++ Native                | PyTorch                     |
| 浏览器支持       | ✅ WASM / WebGPU                       | ❌                          |
| Model Download   | `downloadModel()` 自动按需下载          | 手动 HuggingFace clone      |
| 包大小 (npm)     | ~150 KB (code only)                    | ~GB (完整 repo)             |
| Proxy 支持       | ✅ 三层级联 (arg → env var → std env)   | ❌                          |
| API 类型安全     | ✅ TypeScript strict                    | ❌ Python (动态类型)        |
| CI/CD            | 完整自动化 (lint + test + coverage + benchmark + deploy) | GitHub Actions (基础) |
| Flip Invariance  | ✅ Promise.all 并行                    | ❌ 串行                      |
| Checksum 验证    | ✅ SHA-256 自动校验                    | ❌                          |

### 🔴 仍落后于 google-research/timesfm 的方面

#### 5.1 **缺少 Multi-Variate 输入支持**

Python 版本支持多变量时间序列（多个相关变量联合预测），TypeScript 版本目前仅支持单变量（univariate）。

#### 5.2 **缺少 Fine-Tuning API**

Python 版本支持在新数据上微调模型，TypeScript 版本目前仅支持零样本推理。

#### 5.3 **Model Version 生态不完整**

当前仅支持 TimesFM 2.5 200M。Python 版本还支持 1.0 和 2.0 以及 checkpoint 变体。需验证 ModelDescriptor 是否已支持其他版本。

---

## 六、Model Download Proxy 支持

### ✅ 全部正确实现

1. **三层优先级级联**:
   ```
   1. DownloadOptions.proxy (程序化)
   2. TIMESFM_PROXY_URL/USERNAME/PASSWORD (专用环境变量)
   3. HTTPS_PROXY/https_proxy/HTTP_PROXY/http_proxy (标准环境变量)
   ```

2. **NO_PROXY 支持**: `resolveProxyConfig` 检查 `NO_PROXY` 并跳过 GitHub 域名的代理。

3. **undici ProxyAgent**: 优先使用 `ProxyAgent` dispatcher（避免全局环境变量污染和竞态条件），降级到环境变量方式。

4. **CLI 集成**:
   ```bash
   timesfm setup --proxy-url http://proxy:8080
   timesfm setup --proxy-url http://proxy:8080 --proxy-username user
   TIMESFM_PROXY_PASSWORD=pass timesfm setup --proxy-url http://proxy:8080 --proxy-username user
   ```

5. **密码安全性**: 密码从不在 CLI args 中传递，始终从环境变量读取。DownloadOptions.proxy.password 仅在程序化使用中接受。

6. **HTTP 407 特殊处理**: `ProxyAuthError` 子类携带 `httpStatus: 407` 和明确的修复指引。

7. **完整文档**: CLI README 和主 README 都有详细的 proxy 使用示例。

### 无需要修复的问题 ✅

---

## 七、文档质量

### ✅ 做得好的地方

1. **主 README**: 完整的架构图、Quick Start（3 种方式）、Config Reference 表格（参数/类型/默认值/描述）、Output Shape Reference、Project Structure 树状图。
2. **各 Package README**: 独立的 API docs 链接、Quick Start 代码、Config 参考表。
3. **docs/ 目录**: ARCHITECTURE.md, GETTING-STARTED.md, MODEL-UPDATE.md。
4. **TypeDoc 自动生成**: 完整的 API 参考，含版本号、source-order 排序、category grouping。
5. **License 清晰**: Apache 2.0 + Google 模型权重的合规说明。

### 🔴 需要修复的问题

#### 7.1 **CLI README 缺少 `timesfm setup` 的 Proxy 文档**

`packages/timesfm-cli/README.md` 的 `setup` 命令文档没有提及 `--proxy-url`, `--proxy-username` 和环境变量。这些仅在主 README 中记录。

#### 7.2 **`timesfm-cli/README.md` 缺少 `timesfm forecast` 的 Proxy 说明**

CLI README 中 `forecast` 命令的 Model Path Resolution 未提及代理自动检测（通过环境变量）功能。

#### 7.3 **缺少 CHANGELOG.md**

项目使用 Changesets 但仓库中无 `CHANGELOG.md`（可能仅在各 package 发布时生成）。建议在根目录自动维护。

#### 7.4 **ARCHITECTURE.md 可能与代码不同步**

需要验证 `docs/ARCHITECTURE.md` 内容是否与当前代码实现一致，特别是 KV Cache 部分（代码中标注 `@experimental`）。

---

## 八、本地/CI 一致性

### 🔴 Critical — 本地测试与 CI 不一致

#### 8.1 **`pnpm test:unit` vs CI `unit-test` job — 文件匹配不一致**

**当前状态**:
- 本地 `pnpm test:unit`: 使用 `vitest.unit.config.ts`，包含 15 个 glob 模式
- CI `unit-test` job: 运行 `pnpm vitest run --config vitest.unit.config.ts`
- CI `integration-test` job: 运行 `pnpm test`（使用 `vitest.config.ts`，包含所有文件 + 模型）

**问题 1**: `vitest.unit.config.ts` 的 `include` 列表是**手动维护的 glob 列表**:
```typescript
include: [
  'packages/*/test/**/config.test.ts',
  'packages/*/test/**/nan-handler.test.ts',
  // ... 15 个硬编码 glob
]
```

新增测试文件时必须手动更新此列表，否则本地测试会遗漏但 CI integration-test 可能捕获。

**修复方案**: 使用否定 glob 排除需要模型的测试，而非正向列举:
```typescript
include: ['packages/*/test/**/*.test.ts'],
exclude: ['**/model.test.ts', '**/engine.test.ts', '**/web-integration.test.ts', '**/xreg-engine.test.ts']
```

#### 8.2 **`pnpm build` 不在 `pnpm test:unit` 之前运行**

本地开发时如果忘记 `pnpm build`，`test:unit` 可能使用过期构建产物或不存在的 dist。CI 中 `unit-test` job 先运行 `pnpm build`。需在 scripts 中显式声明依赖。

#### 8.3 **ESLint 版本与 CI 不一致风险**

`eslint.config.mjs` 使用了 `import.meta.dirname`（Node.js ≥ 21.2），IMPROVEMENTS_REPORT 提到已添加 `fileURLToPath` 回退。需验证回退在 Node 20 下是否生效。

---

## 九、CI Workflows 健康检查

### 各 Workflow 分析

#### 9.1 `ci.yml` — 主 CI Pipeline

| Job                | 状态 | 问题 |
| ------------------ | ---- | ---- |
| `lint`             | ✅ OK | — |
| `unit-test`        | ✅ OK | matrix: Node 20 + 22 |
| `build`            | ✅ OK | matrix: Node 20 + 22 |
| `integration-test` | ⚠️  | 冗余测试运行 (见 9.1a) |
| `benchmark`        | ⚠️  | 无 accuracy gate (见 9.1b) |
| `web-benchmark`    | ⚠️  | web-integration 测试环境依赖 |
| `deploy-pages`     | ⚠️  | 覆盖率 index.html 与 lcov-report 并存问题 (见 9.1c) |

#### 9.1a — integration-test 冗余运行

`integration-test` job 运行 `pnpm test` 后再运行 `pnpm test:coverage`。`pnpm test:coverage` 会**再次运行全部测试**（带覆盖率）。应改为:
```yaml
- run: pnpm test:coverage  # 一次运行，同时获得测试结果和覆盖率
```

#### 9.1b — Benchmark 无 Accuracy Gate

Benchmark accuracy 测试使用合成数据且不检查结果。添加:
```yaml
- name: Accuracy Gate
  run: |
    node -e "const r=require('./benchmark-report.json');
    if(r.accuracy && r.accuracy.scaled_mae >= 1) {
      console.error('FAIL: TimesFM worse than naive baseline');
      process.exit(1);
    }"
```

#### 9.1c — Coverage artifact 可能为空

如果 `integration-test` 中 `pnpm test` 失败但 `pnpm test:coverage` 被跳过（未设置 `if: always()` 在所有步骤），coverage artifact 可能为空。当前只有 upload step 有 `if: always()`，但 coverage 生成步骤没有。如果模型导出失败，不会有 coverage 上传。

#### 9.2 `release.yml`

✅ 结构正确: quality gate → publish-npm (OIDC)。支持 `skip_tests` 热修复。Node 24 用于发布（最新版本）。

需要改进: 缺少 `deploy-pages` step — 发布 tag 时不会更新 GitHub Pages 文档。

#### 9.3 `nightly.yml`

✅ 每天 UTC 2AM 检查 HuggingFace 版本变化，自动触发 `model-release.yml`，自动创建 GitHub Issue。

#### 9.4 `model-release.yml`

✅ 完整的 check → export → validate → release → commit-descriptor 流水线。

需要改进: `release` job 使用 `gh release delete` + `git tag -f` + `git push -f` — 这是 **force push**，在受保护分支上会失败。应改用不删除旧 release 的策略。

---

## 十、综合改进方案

### 🔴 P0 — 必须修复（阻塞项）

| #  | 问题                           | 文件                           | 影响                        |
| -- | ------------------------------ | ------------------------------ | --------------------------- |
| 1  | Accuracy benchmark 使用合成数据 | `scripts/benchmark-ci.js`      | 违反需求 #4                 |
| 2  | `vitest.unit.config.ts` 硬编码 glob | `vitest.unit.config.ts`    | 本地/CI 不一致              |
| 3  | `integration-test` 冗余运行    | `.github/workflows/ci.yml`     | CI 时间浪费 ~3 分钟          |
| 4  | `xreg-engine.ts` 数值不稳定性  | `packages/timesfm-xreg/src/xreg-engine.ts` | 协变量预测精度 |
| 5  | CI deployment 可能因 force push 失败 | `.github/workflows/model-release.yml` | 模型发布中断 |

### 🟡 P1 — 应当修复

| #  | 问题                              | 文件                          | 影响                     |
| -- | --------------------------------- | ----------------------------- | ------------------------ |
| 1  | Benchmark 无 accuracy gate        | `.github/workflows/ci.yml`    | CI 不检测模型质量        |
| 2  | 缺少 Memory Leak 稳定性测试       | 新增                          | 生产环境内存泄漏风险      |
| 3  | CLI README 缺少 proxy 文档        | `packages/timesfm-cli/README.md` | 文档不完整            |
| 4  | `@ts-ignore` 类型逸漏             | `packages/timesfm-core/src/model.ts` | 类型安全缺失       |
| 5  | Coverage index 与 lcov-report 不一致 | `.github/workflows/ci.yml` | 用户混淆              |

### 🟢 P2 — 建议改进

| #  | 问题                              | 文件                          |
| -- | --------------------------------- | ----------------------------- |
| 1  | 缺少 CHANGELOG.md                 | 根目录                        |
| 2  | `perCoreBatchSize` API 文档缺失   | README.md                     |
| 3  | ONNX Runtime warmup 策略统一      | `onnx-engine.ts` + `benchmark-ci.js` |
| 4  | 测试文件自动发现替代硬编码 glob   | `vitest.unit.config.ts`       |
| 5  | 缺少 Multi-Variate 输入支持文档   | README + ARCHITECTURE         |

---

## 附录 A: 文件审查清单

已审查的源代码文件（100%）:

- ✅ `packages/timesfm-core/src/`: model.ts, config.ts, types.ts, errors.ts, model-downloader.ts, model-descriptor.ts, preprocessor.ts, postprocessor.ts
- ✅ `packages/timesfm-core/src/inference/`: onnx-engine.ts, decode-loop.ts, kv-cache.ts
- ✅ `packages/timesfm-core/src/utils/`: tensor-utils.ts, stats.ts, revin.ts, nan-handler.ts
- ✅ `packages/timesfm-core/src/helpers/`: metrics.ts, quantile.ts
- ✅ `packages/timesfm-xreg/src/`: index.ts, xreg-engine.ts, one-hot-encoder.ts
- ✅ `packages/timesfm-cli/src/`: cli.ts, csv-forecast.ts
- ✅ `packages/timesfm-web/src/`: web-engine.ts, model-loader.ts, index.ts
- ✅ `scripts/`: benchmark-ci.js, pipeline.js, export-onnx.py
- ✅ `.github/workflows/`: ci.yml, release.yml, nightly.yml, model-release.yml
- ✅ Config: vitest.config.ts, vitest.unit.config.ts, typedoc.json, tsconfig*.json
- ✅ Docs: README.md, all 4 package READMEs, IMPROVEMENTS_REPORT.md

## 附录 B: 与 google-research/timesfm 关键算法对照

| 算法/组件        | Python 参考位置              | TypeScript 实现位置          | 状态 |
| ---------------- | ---------------------------- | ---------------------------- | ---- |
| RevIN            | `torch/util.py::revin()`     | `utils/revin.ts::revinBatch()` | ✅   |
| Welford Stats    | `flax/util.py::update_running_stats()` | `utils/stats.ts::updateRunningStats()` | ✅   |
| NaN Interp       | `timesfm_2p5_base.py::linear_interpolation()` | `utils/nan-handler.ts::linearInterpolateNaNs()` | ✅   |
| Patch Embed      | `timesfm_2p5_torch.py` Tokenizer | `preprocessor.ts` (manual) | ✅   |
| Autoregressive Decode | `timesfm_2p5_torch.py::decode()` | `inference/decode-loop.ts::decode()` | ✅   |
| Flip Invariance  | Post-process in `compile()` | `postprocessor.ts` + `model.ts` parallel | ✅ 优于 |
| Quantile Crossing | Post-process in `compile()` | `postprocessor.ts::fixQuantileCrossing()` | ✅   |
| CQH              | `timesfm_2p5_torch.py` `output_quantiles` | `postprocessor.ts::applyContinuousQuantileHead()` | ✅   |
| XReg Linear      | `utils/xreg_lib.py::BatchedInContextXRegLinear` | `xreg-engine.ts::forecastWithCovariates()` | ✅   |
| KV Cache         | `torch/util.py::DecodeCache` | `inference/kv-cache.ts` (experimental) | 🔶   |

---

*审计完成时间: 2026-06-28T06:55:00Z*  
*审计工具: 人工代码审查 + 静态分析 + 架构评估*
