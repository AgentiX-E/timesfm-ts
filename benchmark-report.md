# TimesFM 2.5 200M Benchmark Report

> Generated: 2026-06-26T18:11:56.936Z · Git: `8e226fe2` · Node: v22.13.1

## System

| Property     | Value                                |
| ------------ | ------------------------------------ |
| CPU          | AMD EPYC 9K84 96-Core Processor × 32 |
| RAM          | 123.3 GB                             |
| Platform     | linux / x64                          |
| Node.js      | v22.13.1                             |
| ONNX Runtime | 1.27.0                               |

## Model

| Property  | Value  |
| --------- | ------ |
| Size      | 885 MB |
| Load time | 1.72s  |

## Inference Latency

| Context | Patches | Avg (ms) | P50 (ms) | P99 (ms) | Throughput (seq/s) |
| ------- | ------- | -------- | -------- | -------- | ------------------ |
| 128     | 4       | 159.5    | 189.4    | 195.6    | 6.3                |
| 256     | 8       | 149.3    | 184.8    | 192.4    | 6.7                |
| 512     | 16      | 141.7    | 112.4    | 189.4    | 7.1                |

## Memory

| Metric     | Value   |
| ---------- | ------- |
| RSS        | 1468 MB |
| Heap Used  | 5.2 MB  |
| Heap Total | 6 MB    |

## Prediction Accuracy

| Metric         | Value          |
| -------------- | -------------- |
| Naive MAE      | 9.521          |
| TimesFM MAE    | 97.3969        |
| TimesFM RMSE   | 98.1206        |
| **Scaled MAE** | **10.2297** ⚠️ |

---

_Automated benchmark by agentix-timesfm-ts CI_
