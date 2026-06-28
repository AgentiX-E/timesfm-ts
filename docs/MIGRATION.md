# 从 Google TimesFM (Python) 迁移到 Agentix TimesFM (TypeScript)

本指南帮助现有 Google TimesFM Python 用户迁移到 TypeScript/Node.js 版本。

## 快速对比

| 操作 | Python API | TypeScript API |
|------|-----------|---------------|
| 安装 | `pip install timesfm` | `npm install @agentix-e/timesfm-core` |
| 导入 | `import timesfm` | `import { TimesFMModel } from '@agentix-e/timesfm-core'` |
| 加载模型 | `tf = timesfm.TimesFm(...)` | `const model = await TimesFMModel.fromPretrained({ modelPath })` |
| 编译 | 加载时自动 | `model.compile(config)` |
| 预测 | `tf.forecast(inputs, horizon)` | `await model.forecast(horizon, inputs)` |
| 分位数 | `forecast_output.quantiles` | `result.quantileForecast` |
| 协变量 | `timesfm.TimeSeriesFMWithCovariates` | `import { forecastWithCovariates } from '@agentix-e/timesfm-xreg'` |
| CLI | 无 | `npx timesfm forecast --data input.csv` |

## 逐步迁移

### 1. 安装

```bash
# Python
pip install "timesfm[torch]"

# TypeScript (Node.js)
npm install @agentix-e/timesfm-core
```

### 2. 模型获取

```python
# Python: 自动从 HuggingFace 下载
tf = timesfm.TimesFm(
    context_len=512,
    horizon_len=128,
    backend="gpu",
    model_path="google/timesfm-2.5-200m-pytorch"
)
```

```typescript
// TypeScript: 使用 timesfm CLI 设置
// $ npx timesfm setup
// 或通过代码自动下载：
import { TimesFMModel } from '@agentix-e/timesfm-core';

const model = await TimesFMModel.fromPretrained({
  modelPath: 'auto', // 自动下载到缓存
});
```

### 3. 单序列预测

```python
# Python
import numpy as np
data = np.array([1.0, 2.0, 3.0, ...])  # shape: (context_len,)
forecast = tf.forecast([data], horizon=24)
point_forecast = forecast[:, 0, 0].numpy()  # 提取点预测
```

```typescript
// TypeScript
const data = new Float32Array([1.0, 2.0, 3.0, ...]);
const model = await TimesFMModel.fromPretrained({ modelPath: 'auto' });
model.compile(createForecastConfig({ maxContext: 512, maxHorizon: 128 }));

const result = await model.forecast(24, [data]);
const pointForecast = result.pointForecast[0]; // Float32Array(24)
```

### 4. 批量预测

```python
# Python
batch = [series1, series2, series3]
forecast = tf.forecast(batch, horizon=24)
```

```typescript
// TypeScript
const batch = [series1, series2, series3]; // Float32Array[]
const result = await model.forecast(24, batch);
// result.pointForecast[0], result.pointForecast[1], result.pointForecast[2]
```

### 5. 获取预测区间

```python
# Python
forecast = tf.forecast([data], horizon=24)
upper = forecast[:, :, 9]  # 上界 (90%)
lower = forecast[:, :, 0]  # 下界 (10%)
median = forecast[:, :, 5] # 中位数
```

```typescript
// TypeScript
import { getPredictionInterval } from '@agentix-e/timesfm-core';

const result = await model.forecast(24, [data]);
const { lower, upper, median } = getPredictionInterval(result, 0.9);
```

### 6. 评估指标

```python
# Python (需要手动实现或使用 sklearn)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(actual, predicted)
```

```typescript
// TypeScript (内置)
import { mae, rmse, smape, mape } from '@agentix-e/timesfm-core';

const error = mae(predicted, actual);
```

## 关键差异

### 显式编译步骤

TypeScript 版本要求显式 `compile()` 调用：

```typescript
// ✅ 正确
const model = await TimesFMModel.fromPretrained({ modelPath });
model.compile(createForecastConfig({ maxContext: 512, maxHorizon: 128 }));
await model.forecast(horizon, inputs);

// ❌ 错误 — 在编译前预测会抛出 ModelNotCompiledError
const model = await TimesFMModel.fromPretrained({ modelPath });
await model.forecast(horizon, inputs); // 抛出!
```

### 输入顺序

Python API: `tf.forecast(inputs, horizon)`  
TypeScript API: `model.forecast(horizon, inputs)`

注意参数顺序是相反的！

### 数据类型

| Python | TypeScript |
|--------|-----------|
| `np.float32` 或 `np.ndarray` | `Float32Array` |
| `torch.Tensor` | `Float32Array` |
| `list[float]` | `Float32Array` |

### 资源管理

```typescript
// TypeScript 版本支持显式销毁
const model = await TimesFMModel.fromPretrained({ modelPath });
try {
  // 使用模型
} finally {
  await model.dispose(); // 释放 ONNX 运行时内存
}
```

## 协变量预测

```python
# Python — TimeSeriesFMWithCovariates
from timesfm import TimesFm, TimesFMWithCovariates
```

```typescript
// TypeScript — @agentix-e/timesfm-xreg
import { forecastWithCovariates } from '@agentix-e/timesfm-xreg';

const result = await forecastWithCovariates(model, {
  y: targetSeries,
  X: covariates,
  horizon: 24,
  mode: 'xreg+timesfm',
});
```

## 分层协调

TypeScript 版本独有的功能 — Python 版本无对应功能：

```typescript
import { reconcileForecast } from '@agentix-e/timesfm-hierarchical';

const reconciled = reconcileForecast({
  hierarchy: {
    total: ['region_a', 'region_b'],
    region_a: ['store_a1', 'store_a2'],
    region_b: ['store_b1', 'store_b2'],
  },
  baseForecasts: baseForecastsMap,
  strategy: 'mint',
});
```

## 下一步

- [API 文档](https://agentix-e.github.io/agentix-timesfm-ts/api/)
- [架构文档](./ARCHITECTURE.md)
- [精度验证](./ACCURACY.md)
- [故障排除](./TROUBLESHOOTING.md)
