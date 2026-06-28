# 故障排除指南

常见问题和解决方案。

## 模型下载

### 下载失败：连接超时

```
Error: Failed to download model: ETIMEDOUT
```

**解决方案**：

1. 检查网络连接
2. 配置代理（见下文）
3. 手动下载模型文件放到 `~/.cache/agentix-timesfm-ts/`

### 使用代理下载

```bash
# 通过 CLI 参数
timesfm setup --proxy-url http://proxy.company.com:8080

# 带认证的代理
timesfm setup --proxy-url http://proxy.company.com:8080 \
  --proxy-username myuser \
  --proxy-password mypass

# 通过环境变量
export TIMESFM_PROXY_URL=http://proxy.company.com:8080
export TIMESFM_PROXY_USERNAME=myuser
export TIMESFM_PROXY_PASSWORD=mypass
timesfm setup
```

### 代理返回 407 (需要认证)

```
ProxyAuthError: Proxy authentication required. Received HTTP 407
```

**解决方案**：

- 确认用户名/密码正确
- 检查代理 URL 格式：`http://host:port`（不支持 SOCKS）
- 如果使用 NTLM/Kerberos 认证，请先通过代理隧道转发

### 手动下载模型

```bash
# 从 GitHub Releases 下载
wget https://github.com/AgentiX-E/agentix-timesfm-ts/releases/download/timesfm-latest/timesfm-2.5.onnx \
  -O ~/.cache/agentix-timesfm-ts/timesfm-2.5.onnx
```

## 内存问题

### OOM (内存溢出)

```
FATAL ERROR: Reached heap limit Allocation failed — JavaScript heap out of memory
```

TimesFM ONNX 模型约 885 MB (fp32)。推理时还需要 ~1-2 GB 额外内存。

**解决方案**：

1. 增加 Node.js 内存限制：
   ```bash
   NODE_OPTIONS="--max-old-space-size=4096" npx timesfm forecast --data input.csv
   ```
2. 使用 int8 精度模型 (约 230 MB)：
   ```bash
   # 需要先从 HuggingFace 导出 int8 模型
   python3 scripts/export-onnx.py --precision int8 --output models/timesfm-2.5-int8.onnx
   ```
3. 使用较小的上下文窗口：
   ```typescript
   model.compile(createForecastConfig({ maxContext: 256, maxHorizon: 64 }));
   ```

## 推理问题

### 不知道模型是否编译

```
ModelNotCompiledError: Model has not been compiled. Call model.compile() before forecast().
```

**解决方案**：

```typescript
const model = await TimesFMModel.fromPretrained({ modelPath: 'auto' });
model.compile(createForecastConfig({ maxContext: 512, maxHorizon: 128 }));
const result = await model.forecast(24, inputs); // ✅
```

### 预测结果为 NaN

**可能原因**：

1. 输入包含 NaN 或 Infinity
2. 输入长度 < 1
3. 输入全为零（RevIN 无法归一化）

**解决方案**：

```typescript
// 预处理器会自动清理 NaN（线性插值），但全零序列可能出问题
// 确保输入至少有非零值
```

### ONNX 运行时不可用

```
Error: onnxruntime-node not found. Install with: npm install onnxruntime-node
```

**解决方案**：

```bash
npm install onnxruntime-node
```

注意：`onnxruntime-node` 需要原生二进制文件，确保系统满足要求。

## TypeScript 编译问题

### 动态导入失败

```
Error: Cannot find module '@agentix-e/timesfm-xreg'
```

**解决方案**：

```bash
npm install @agentix-e/timesfm-xreg
```

协变量和分层包是可选依赖，不会自动安装。

## 浏览器 (WASM) 问题

### WASM 文件加载失败

```
Error: WASM binary not found
```

**解决方案**：

1. 确认 WASM 文件在正确路径：
   ```typescript
   import { TimesFMWebInferenceEngine } from '@agentix-e/timesfm-web';
   const engine = new TimesFMWebInferenceEngine({
     wasmPath: '/path/to/onnxruntime-web/',
   });
   ```
2. 如果从 CDN 加载，确保允许跨域：
   ```typescript
   const engine = new TimesFMWebInferenceEngine({
     useCDN: true, // 从 jsdelivr CDN 加载
   });
   ```

### WebGPU 不可用 → 回退到 WASM

这是正常行为。WebGPU 在以下情况不可用：

- 大多数桌面浏览器（Chrome 113+, Edge 113+）
- 移动浏览器通常不支持

自动回退链: `webgpu → wasm → webgl`

## CI/CD 问题

### 覆盖率丢失

```
FAIL: coverage/coverage-summary.json not generated
```

**解决方案**：检查 vitest 配置是否正确，确保模型已缓存（集成测试需要 ONNX 模型）。

### 基准测试失败

```
FAIL: TimesFM scaled MAE (X.XX) is not better than naive baseline
```

这表明模型推理管线的精度回归。检查最近的更改是否影响了预处理/后处理逻辑。

## 获取帮助

- [GitHub Issues](https://github.com/AgentiX-E/agentix-timesfm-ts/issues)
- [API 文档](https://agentix-e.github.io/agentix-timesfm-ts/api/)
- [架构文档](./ARCHITECTURE.md)
- [迁移指南](./MIGRATION.md)
