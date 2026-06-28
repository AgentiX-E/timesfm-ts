# TimesFM 精度验证

本文档记录了 Google TimesFM (Python, PyTorch) 与 Agentix TimesFM (TypeScript, ONNX Runtime) 之间的精度对比方法和结果。

## 对比方法论

### 运行时环境

| 维度     | Google TimesFM (Python)           | Agentix TimesFM (TypeScript)                |
| -------- | --------------------------------- | ------------------------------------------- |
| 模型权重 | `google/timesfm-2.5-200m-pytorch` | 相同权重，通过 `export-onnx.py` 导出为 ONNX |
| 推理后端 | PyTorch (eager mode)              | ONNX Runtime (CPU)                          |
| 数值精度 | float32                           | float32                                     |
| 推理管线 | `timesfm.TimesFM.forecast()`      | `TimesFMModel.forecast()`                   |
| 预处理   | RevIN (PyTorch)                   | RevIN (纯 TypeScript)                       |
| 后处理   | Quantile head (PyTorch)           | Quantile head (纯 TypeScript)               |

### 数据集

以下公共领域数据集用于精度验证：

1. **Air Passengers** (144 个月度值): 经典 Box-Jenkins 基准
2. **Monthly Sunspots** (300 个月度值): SIDC 太阳黑子数据
3. **Electricity Production** (396 个月度值): 澳大利亚 IPEA
4. **Melbourne Daily Temperatures** (400 个日度值): 澳大利亚气象局
5. **US Gas Production** (240 个月度值): 美国能源信息署

### 评估指标

| 指标             | 公式                                     | 说明                                        |
| ---------------- | ---------------------------------------- | ------------------------------------------- |
| **MAE**          | Σ\|ŷᵢ - yᵢ\| / n                         | 平均绝对误差                                |
| **RMSE**         | √(Σ(ŷᵢ - yᵢ)² / n)                       | 均方根误差                                  |
| **SMAPE**        | 200/n × Σ\|ŷᵢ - yᵢ\| / (\|ŷᵢ\| + \|yᵢ\|) | 对称平均绝对百分比误差 (0-200 范围)         |
| **Scaled MAE**   | MAE_model / MAE_naive                    | 相对于朴素基线的缩放 MAE (< 1.0 = 优于朴素) |
| **PIC Coverage** | CI中包含真实值的比例                     | 预测区间覆盖 (< 目标置信度 → 校准不足)      |
| **PI Width**     | 上界 - 下界的均值                        | 预测区间宽度 (越小越好，给定覆盖)           |

### 朴素基线

朴素基线 = 重复最后一个观测值 N 次。

```
naive_forecast[i] = last_observed_value for i in 0..N-1
```

## 运行精度验证

### 通过基准脚本 (CI)

```bash
# Node.js 基准 (包含精度闸门)
pnpm exec tsx scripts/benchmark-ci.js \
  --json benchmark-report.json \
  --md benchmark-report.md \
  --html benchmark-report.html

# WASM 基准
pnpm exec tsx scripts/web-benchmark-ci.js \
  --json web-benchmark-report.json \
  --md web-benchmark-report.md \
  --html web-benchmark-report.html
```

### 通过 Python vs TypeScript 对比

```bash
# 1. 导出 ONNX 模型
python3 scripts/export-onnx.py \
  --model google/timesfm-2.5-200m-pytorch \
  --output models/timesfm-2.5.onnx

# 2. 运行 Python 推理 (获取参考输出)
python3 scripts/compare-accuracy.py \
  --model google/timesfm-2.5-200m-pytorch \
  --onnx-model models/timesfm-2.5.onnx \
  --dataset air_passengers \
  --context 120 \
  --horizon 24

# 3. 运行 TypeScript 推理
pnpm exec tsx scripts/compare-accuracy.ts \
  --model models/timesfm-2.5.onnx \
  --dataset air_passengers \
  --context 120 \
  --horizon 24
```

## 预期容忍度

由于 ONNX 导出和不同后端的浮点实现差异，轻微数值差异是可预期的：

| 组件     | 预期容忍度     | 说明                             |
| -------- | -------------- | -------------------------------- |
| 点预测   | ±0.5% 相对差异 | 由于 ONNX 算子融合和内存布局差异 |
| 分位数带 | ±1.0% 相对差异 | 分位数头对数值更敏感             |
| MAE 差异 | ≤5% 绝对差异   | 聚合指标应在紧密范围内           |

## CI 精度闸门

基准 CI 作业包含自动精度闸门 (`Accuracy Gate`)：

```
如果 scaled_MAE >= 1.0 → CI 失败
  - TimesFM 的表现不如朴素基线
  - 指示管线回归

如果 scaled_MAE < 1.0 → CI 通过
  - TimesFM 优于朴素基线
```

这确保了任何意外降低预测精度的更改在合并前被捕获。

## 历史精度跟踪

基准结果发布在 [GitHub Pages](https://agentix-e.github.io/agentix-timesfm-ts/benchmark/)。

每次在 `master`/`main` 分支上的 CI 运行都会更新精度报告，提供完整历史记录和回归检测。

---

_最后更新: 2026-06-28_
