# 评测结果汇总

由 `Evaluation/results/*/frames16/eval_*.json` 自动汇总，更新时间：2026-07-07。当前结果均为 `frames16`。

当前测评结果测试了六项benchmark的全集 但是由于测评入口prompt使用标准的<think><answer>输出格式 怀疑影响了ema-v2 van-v2 tgrpo-van的真实得分

## Benchmark 样本数

| Benchmark | 样本数 |
|---|---:|
| LongVideoReason | 1000 |
| TempCompass | 7540 |
| MVBench | 4000 |
| VideoMathQA | 420 |
| MMVU | 625 |
| VideoMMMU | 900 |
| VideoMME | 2700 |
| VSIBench | 5130 |

## 六项完整评测排名

六项指 LongVideoReason、TempCompass、MVBench、VideoMathQA、MMVU、VideoMMMU，数值为简单未加权平均。

| 排名 | 模型 | 六项平均 | LongVideoReason | TempCompass | MVBench | VideoMathQA | MMVU | VideoMMMU |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `TimeThinker-4B-RL-Zero-100-ema` | 58.13% | 71.00% | 73.75% | 63.98% | 25.95% | 65.76% | 48.33% |
| 2 | `TimeThinker-4B-RL-Zero-100-van` | 56.80% | 68.30% | 72.57% | 62.35% | 21.43% | 64.48% | 51.67% |
| 3 | `TimeThinker-4B-SFT-v3-10000-1ep` | 56.22% | 70.30% | 68.70% | 61.10% | 21.19% | 64.80% | 51.22% |
| 4 | `TimeThinker-4B-RL-Zero-100-ema-v2` | 56.08% | 67.50% | 72.81% | 60.80% | 19.05% | 65.12% | 51.22% |
| 5 | `TimeThinker-4B-SFT-v6-10k` | 55.88% | 70.80% | 68.51% | 60.88% | 20.24% | 63.84% | 51.00% |
| 6 | `TimeThinker-4B-RL-Zero-100-tgrpo-van` | 55.37% | 64.50% | 73.09% | 60.65% | 17.62% | 66.56% | 49.78% |
| 7 | `TimeThinker-4B-RL-Zero-100-van-v2` | 55.26% | 66.50% | 72.76% | 60.15% | 17.38% | 64.32% | 50.44% |
| 8 | `Qwen3-VL-4B-Instruct` | 53.80% | 64.90% | 72.39% | 57.95% | 14.29% | 61.92% | 51.33% |

## 所有 Run Overall

数值为对应 benchmark 的 overall accuracy；`-` 表示当前没有该 run。

| 模型 | 完成数 | 六项平均 | LongVideoReason | TempCompass | MVBench | VideoMathQA | MMVU | VideoMMMU | VideoMME | VSIBench |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3-VL-4B-Instruct` | 7 | 53.80% | 64.90% | 72.39% | 57.95% | 14.29% | 61.92% | 51.33% | - | 46.26% |
| `TimeThinker-4B-SFT` | 4 | - | - | 68.73% | - | - | 62.56% | 53.33% | - | 46.03% |
| `TimeThinker-4B-SFT-v2-10000` | 4 | - | - | 68.02% | - | - | 63.20% | 54.22% | - | 43.77% |
| `TimeThinker-4B-SFT-v3-10000` | 4 | - | - | 69.23% | - | - | 64.64% | 53.67% | - | 45.45% |
| `TimeThinker-4B-SFT-v3-10000-1ep` | 6 | 56.22% | 70.30% | 68.70% | 61.10% | 21.19% | 64.80% | 51.22% | - | - |
| `TimeThinker-4B-SFT-v6-10k` | 6 | 55.88% | 70.80% | 68.51% | 60.88% | 20.24% | 63.84% | 51.00% | - | - |
| `TimeThinker-4B-RL-Zero-100-ema` | 8 | 58.13% | 71.00% | 73.75% | 63.98% | 25.95% | 65.76% | 48.33% | 57.96% | 51.39% |
| `TimeThinker-4B-RL-Zero-100-van` | 6 | 56.80% | 68.30% | 72.57% | 62.35% | 21.43% | 64.48% | 51.67% | - | - |
| `TimeThinker-4B-RL-Zero-100-van-v2` | 6 | 55.26% | 66.50% | 72.76% | 60.15% | 17.38% | 64.32% | 50.44% | - | - |
| `TimeThinker-4B-RL-Zero-100-ema-v2` | 6 | 56.08% | 67.50% | 72.81% | 60.80% | 19.05% | 65.12% | 51.22% | - | - |
| `TimeThinker-4B-RL-Zero-100-tgrpo-van` | 6 | 55.37% | 64.50% | 73.09% | 60.65% | 17.62% | 66.56% | 49.78% | - | - |

## 当前结论

- 六项完整评测里，`TimeThinker-4B-RL-Zero-100-ema` 仍是当前最高，六项平均 58.13%。
- 新跑的四个模型中，`TimeThinker-4B-RL-Zero-100-ema-v2` 六项平均最高，为 56.08%；`SFT-v6-10k`、`tgrpo-van`、`van-v2` 分别为 55.88%、55.37%、55.26%。
- `TimeThinker-4B-RL-Zero-100-tgrpo-van` 在 MMVU 上达到当前最高 66.56%，TempCompass 也达到 73.09%，但 LongVideoReason、VideoMathQA 和 VideoMMMU 拉低了六项平均。
- `TimeThinker-4B-RL-Zero-100-ema-v2` 相比旧 `ema` 在所有六项核心 benchmark 上都有回落，尤其 VideoMathQA 从 25.95% 降到 19.05%。
- `TimeThinker-4B-SFT-v6-10k` 比 `SFT-v3-10000-1ep` 在 LongVideoReason 略升，但 MMVU、TempCompass、MVBench、VideoMathQA、VideoMMMU 都略低。

## VideoMMMU 分项

| 模型 | Overall | Multiple choice | Numerical |
|---|---:|---:|---:|
| `TimeThinker-4B-SFT-v2-10000` | 54.22% | 54.99% | 16.67% |
| `TimeThinker-4B-SFT-v3-10000` | 53.67% | 54.54% | 11.11% |
| `TimeThinker-4B-SFT` | 53.33% | 54.31% | 5.56% |
| `TimeThinker-4B-RL-Zero-100-van` | 51.67% | 52.61% | 5.56% |
| `Qwen3-VL-4B-Instruct` | 51.33% | 52.15% | 11.11% |
| `TimeThinker-4B-SFT-v3-10000-1ep` | 51.22% | 52.15% | 5.56% |
| `TimeThinker-4B-RL-Zero-100-ema-v2` | 51.22% | 52.15% | 5.56% |
| `TimeThinker-4B-SFT-v6-10k` | 51.00% | 51.93% | 5.56% |
| `TimeThinker-4B-RL-Zero-100-van-v2` | 50.44% | 51.36% | 5.56% |
| `TimeThinker-4B-RL-Zero-100-tgrpo-van` | 49.78% | 50.57% | 11.11% |
| `TimeThinker-4B-RL-Zero-100-ema` | 48.33% | 49.09% | 11.11% |

## VSIBench 分项

| 模型 | Overall | Multiple choice | Regression | 最强原始题型 | 最弱原始题型 |
|---|---:|---:|---:|---|---|
| `TimeThinker-4B-RL-Zero-100-ema` | 51.39% | 47.11% | 55.42% | object_size_estimation 71.50% | object_rel_direction_hard 31.90% |
| `Qwen3-VL-4B-Instruct` | 46.26% | 47.55% | 45.05% | object_size_estimation 60.73% | object_counting 30.02% |
| `TimeThinker-4B-SFT` | 46.03% | 43.37% | 48.53% | object_size_estimation 65.01% | route_planning 33.51% |
| `TimeThinker-4B-SFT-v3-10000` | 45.45% | 42.33% | 48.40% | object_size_estimation 65.07% | route_planning 28.87% |
| `TimeThinker-4B-SFT-v2-10000` | 43.77% | 43.82% | 43.72% | object_size_estimation 59.49% | object_abs_distance 31.55% |

## TempCompass 分项

| 模型 | Overall | Action | Order | Attribute change | Speed | Direction |
|---|---:|---:|---:|---:|---:|---:|
| `TimeThinker-4B-RL-Zero-100-ema` | 73.75% | 94.95% | 80.27% | 77.49% | 58.84% | 57.54% |
| `TimeThinker-4B-RL-Zero-100-tgrpo-van` | 73.09% | 94.95% | 81.00% | 75.36% | 57.60% | 56.85% |
| `TimeThinker-4B-RL-Zero-100-ema-v2` | 72.81% | 95.13% | 81.36% | 75.28% | 56.88% | 55.78% |
| `TimeThinker-4B-RL-Zero-100-van-v2` | 72.76% | 94.95% | 82.08% | 75.21% | 55.32% | 56.66% |
| `TimeThinker-4B-RL-Zero-100-van` | 72.57% | 94.95% | 81.14% | 73.86% | 57.86% | 55.34% |
| `Qwen3-VL-4B-Instruct` | 72.39% | 94.39% | 81.43% | 75.50% | 55.25% | 55.84% |
| `TimeThinker-4B-SFT-v3-10000` | 69.23% | 91.99% | 76.52% | 70.03% | 56.29% | 51.44% |
| `TimeThinker-4B-SFT` | 68.73% | 91.19% | 75.79% | 68.75% | 57.34% | 50.63% |
| `TimeThinker-4B-SFT-v3-10000-1ep` | 68.70% | 91.99% | 75.79% | 67.97% | 54.79% | 52.83% |
| `TimeThinker-4B-SFT-v6-10k` | 68.51% | 91.56% | 74.13% | 70.10% | 55.45% | 51.32% |
| `TimeThinker-4B-SFT-v2-10000` | 68.02% | 91.74% | 73.27% | 68.82% | 54.34% | 51.76% |

## VideoMME 分项

目前只有 `TimeThinker-4B-RL-Zero-100-ema` 有 VideoMME run。

### 按任务类型

| 类别 | 样本数 | Accuracy |
|---|---:|---:|
| Spatial Reasoning | 56 | 76.79% |
| Attribute Perception | 222 | 74.32% |
| Spatial Perception | 54 | 72.22% |
| Information Synopsis | 323 | 71.83% |
| Object Recognition | 354 | 63.56% |
| OCR Problems | 139 | 61.15% |
| Temporal Perception | 55 | 60.00% |
| Action Recognition | 313 | 56.23% |
| Object Reasoning | 454 | 55.73% |
| Action Reasoning | 285 | 51.58% |
| Temporal Reasoning | 177 | 39.55% |
| Counting Problem | 268 | 36.19% |

### 按领域

| 类别 | 样本数 | Accuracy |
|---|---:|---:|
| Artistic Performance | 360 | 61.94% |
| Film & Television | 360 | 61.39% |
| Knowledge | 810 | 59.75% |
| Sports Competition | 450 | 56.89% |
| Life Record | 630 | 53.17% |
| Multilingual | 90 | 51.11% |
