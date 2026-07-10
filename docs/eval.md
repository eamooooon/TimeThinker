# Evaluation Summary

记录 `Evaluation/results/*/frames16` 当前统一评测结果、耗时和输出稳定性。分数均直接读取各模型的 `_summary.json`，默认使用 8 个 benchmark 的 `answer_acc` 简单平均。

## 结果口径

- `Avg`：8 个 benchmark 的 `answer_acc` 未加权平均，即 `_summary.json` 中的 `macro_avg/by_benchmark`。
- `wall_time`：一个模型完整评测的实际等待时间。
- `benchmark_elapsed_sum`：8 个 benchmark 各自记录的 `elapsed_min` 之和，表示累计 workload/GPU task time，不等于实际等待时间。
- `frames16`：所有结果均为 `MAX_FRAMES=16`。
- `answer_acc` 只统计最终答案是否正确；输出格式问题通过 `extract_rate`、`invalid_rate` 和 `trunc_rate` 单独观察。
- 当前共有 14 个完整 summary，每个 summary 覆盖 8 个 benchmark、总样本数均为 22315。
- 本文只统计当前 `Evaluation/results` 中实际存在的结果，不混入旧目录或历史文档中的已删除 run。

## 最新完整结果

| Model | Avg | LongVideoReason | MMVU | MVBench | TempCompass | VideoMathQA | VideoMME | VideoMMMU | VSIBench |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `TimeThinker-4B-RL-Zero-100-ema-v2` | **55.85** | 70.30 | 64.80 | 61.12 | 72.55 | 23.81 | **55.56** | 52.78 | **45.87** |
| `TimeThinker-4B-SFT-v9-10k-3ep` | 55.76 | **71.50** | 65.44 | **63.38** | 69.10 | 24.52 | 55.15 | 52.89 | 44.12 |
| `TimeThinker-4B-RL-Zero-100-tgrpo-van2` | 55.54 | 68.40 | 65.44 | 61.20 | 72.39 | 24.29 | 55.26 | 52.00 | 45.34 |
| `TimeThinker-4B-RL-Zero-100-van-v2` | 55.40 | 66.80 | **66.24** | 61.58 | 72.89 | 23.10 | 54.00 | 53.11 | 45.50 |
| `TimeThinker-4B-SFT-v3-10000` | 54.89 | 69.10 | 63.68 | 62.80 | 68.71 | 22.14 | 54.89 | **53.67** | 44.11 |
| `TimeThinker-4B-RL-Zero-100-van-bs16` | 54.83 | 68.40 | 65.76 | 61.52 | 72.68 | 20.48 | 54.07 | 51.11 | 44.65 |
| `TimeThinker-4B-RL-v9-100-bs16` | 54.81 | 67.40 | 64.00 | 60.55 | 73.05 | 24.29 | 52.93 | 51.00 | 45.25 |
| `TimeThinker-4B-SFT-v10-50k` | 54.48 | 67.30 | 62.40 | 62.45 | 68.29 | **26.67** | 54.52 | 52.67 | 41.58 |
| `TimeThinker-4B-SFT-v8-10k` | 54.36 | 70.30 | 63.52 | 62.58 | 68.00 | 23.33 | 53.89 | 51.22 | 42.03 |
| `TimeThinker-4B-SFT-v3-10000-1ep` | 54.34 | 69.20 | 63.84 | 62.18 | 67.92 | 24.52 | 53.00 | 52.44 | 41.62 |
| `TimeThinker-4B-SFT-v5-10k` | 54.07 | 69.60 | 62.08 | 62.10 | 67.81 | 23.33 | 54.04 | 51.00 | 42.62 |
| `Qwen3-VL-4B-Instruct` | 53.56 | 65.50 | 64.48 | 60.65 | 71.72 | 19.76 | 52.07 | 50.11 | 44.19 |
| `TimeThinker-4B-SFT-v4-10000` | 53.52 | 69.30 | 63.84 | 60.52 | 67.63 | 20.48 | 53.81 | 51.00 | 41.61 |
| `Qwen3-VL-4B-Thinking` | 49.20 | 61.30 | 56.64 | 60.48 | **74.36** | 18.57 | 52.89 | 36.00 | 33.37 |

## SFT 结果

| Model | Avg | LongVideoReason | MMVU | MVBench | TempCompass | VideoMathQA | VideoMME | VideoMMMU | VSIBench |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `TimeThinker-4B-SFT-v9-10k-3ep` | **55.76** | **71.50** | **65.44** | **63.38** | **69.10** | 24.52 | **55.15** | 52.89 | **44.12** |
| `TimeThinker-4B-SFT-v3-10000` | 54.89 | 69.10 | 63.68 | 62.80 | 68.71 | 22.14 | 54.89 | **53.67** | 44.11 |
| `TimeThinker-4B-SFT-v10-50k` | 54.48 | 67.30 | 62.40 | 62.45 | 68.29 | **26.67** | 54.52 | 52.67 | 41.58 |
| `TimeThinker-4B-SFT-v8-10k` | 54.36 | 70.30 | 63.52 | 62.58 | 68.00 | 23.33 | 53.89 | 51.22 | 42.03 |
| `TimeThinker-4B-SFT-v3-10000-1ep` | 54.34 | 69.20 | 63.84 | 62.18 | 67.92 | 24.52 | 53.00 | 52.44 | 41.62 |
| `TimeThinker-4B-SFT-v5-10k` | 54.07 | 69.60 | 62.08 | 62.10 | 67.81 | 23.33 | 54.04 | 51.00 | 42.62 |
| `TimeThinker-4B-SFT-v4-10000` | 53.52 | 69.30 | 63.84 | 60.52 | 67.63 | 20.48 | 53.81 | 51.00 | 41.61 |

## RL 结果

| Model | Avg | LongVideoReason | MMVU | MVBench | TempCompass | VideoMathQA | VideoMME | VideoMMMU | VSIBench |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `TimeThinker-4B-RL-Zero-100-ema-v2` | **55.85** | **70.30** | 64.80 | 61.12 | 72.55 | 23.81 | **55.56** | 52.78 | **45.87** |
| `TimeThinker-4B-RL-Zero-100-tgrpo-van2` | 55.54 | 68.40 | 65.44 | 61.20 | 72.39 | **24.29** | 55.26 | 52.00 | 45.34 |
| `TimeThinker-4B-RL-Zero-100-van-v2` | 55.40 | 66.80 | **66.24** | **61.58** | 72.89 | 23.10 | 54.00 | **53.11** | 45.50 |
| `TimeThinker-4B-RL-Zero-100-van-bs16` | 54.83 | 68.40 | 65.76 | 61.52 | 72.68 | 20.48 | 54.07 | 51.11 | 44.65 |
| `TimeThinker-4B-RL-v9-100-bs16` | 54.81 | 67.40 | 64.00 | 60.55 | **73.05** | **24.29** | 52.93 | 51.00 | 45.25 |

## Baseline 结果

| Model | Avg | LongVideoReason | MMVU | MVBench | TempCompass | VideoMathQA | VideoMME | VideoMMMU | VSIBench |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3-VL-4B-Instruct` | **53.56** | **65.50** | **64.48** | **60.65** | 71.72 | **19.76** | 52.07 | **50.11** | **44.19** |
| `Qwen3-VL-4B-Thinking` | 49.20 | 61.30 | 56.64 | 60.48 | **74.36** | 18.57 | **52.89** | 36.00 | 33.37 |

## 评测耗时

| Model | wall_time | benchmark_elapsed_sum | schedule |
|---|---:|---:|---|
| `TimeThinker-4B-RL-Zero-100-ema-v2` | 28m53s | 110.5 min | balanced |
| `TimeThinker-4B-SFT-v9-10k-3ep` | 28m21s | 103.9 min | balanced |
| `TimeThinker-4B-RL-Zero-100-tgrpo-van2` | 29m06s | 111.3 min | balanced |
| `TimeThinker-4B-RL-Zero-100-van-v2` | 35m45s | 130.7 min | balanced |
| `TimeThinker-4B-SFT-v3-10000` | 29m31s | 108.7 min | balanced |
| `TimeThinker-4B-RL-Zero-100-van-bs16` | 30m40s | 116.4 min | balanced |
| `TimeThinker-4B-RL-v9-100-bs16` | 36m31s | 132.9 min | balanced |
| `TimeThinker-4B-SFT-v10-50k` | 28m53s | 107.2 min | balanced |
| `TimeThinker-4B-SFT-v8-10k` | 29m24s | 108.1 min | balanced |
| `TimeThinker-4B-SFT-v3-10000-1ep` | 32m41s | 119.3 min | balanced |
| `TimeThinker-4B-SFT-v5-10k` | 31m33s | 117.1 min | balanced |
| `Qwen3-VL-4B-Instruct` | 32m18s | 121.9 min | balanced |
| `TimeThinker-4B-SFT-v4-10000` | 32m17s | 116.4 min | balanced |
| `Qwen3-VL-4B-Thinking` | 51m57s | 171.3 min | balanced |

## 每项最优

| Benchmark | Best overall | Acc | Best SFT | Acc |
|---|---|---:|---|---:|
| LongVideoReason | `TimeThinker-4B-SFT-v9-10k-3ep` | 71.50 | `TimeThinker-4B-SFT-v9-10k-3ep` | 71.50 |
| MMVU | `TimeThinker-4B-RL-Zero-100-van-v2` | 66.24 | `TimeThinker-4B-SFT-v9-10k-3ep` | 65.44 |
| MVBench | `TimeThinker-4B-SFT-v9-10k-3ep` | 63.38 | `TimeThinker-4B-SFT-v9-10k-3ep` | 63.38 |
| TempCompass | `Qwen3-VL-4B-Thinking` | 74.36 | `TimeThinker-4B-SFT-v9-10k-3ep` | 69.10 |
| VideoMathQA | `TimeThinker-4B-SFT-v10-50k` | 26.67 | `TimeThinker-4B-SFT-v10-50k` | 26.67 |
| VideoMME | `TimeThinker-4B-RL-Zero-100-ema-v2` | 55.56 | `TimeThinker-4B-SFT-v9-10k-3ep` | 55.15 |
| VideoMMMU | `TimeThinker-4B-SFT-v3-10000` | 53.67 | `TimeThinker-4B-SFT-v3-10000` | 53.67 |
| VSIBench | `TimeThinker-4B-RL-Zero-100-ema-v2` | 45.87 | `TimeThinker-4B-SFT-v9-10k-3ep` | 44.12 |

## 输出稳定性

以下指标按样本数加权，避免不同 benchmark 样本规模差异影响汇总。

| Model | weighted avg_tokens | weighted trunc_rate | weighted invalid_rate | weighted extract_rate |
|---|---:|---:|---:|---:|
| `TimeThinker-4B-RL-Zero-100-ema-v2` | 139.3 | 2.56% | 1.43% | 98.57% |
| `TimeThinker-4B-SFT-v9-10k-3ep` | 230.0 | 0.13% | 0.05% | 99.95% |
| `TimeThinker-4B-RL-Zero-100-tgrpo-van2` | 137.8 | 2.71% | 1.55% | 98.45% |
| `TimeThinker-4B-RL-Zero-100-van-v2` | 160.2 | 3.28% | 1.62% | 98.38% |
| `TimeThinker-4B-SFT-v3-10000` | 231.3 | 0.19% | 0.05% | 99.95% |
| `TimeThinker-4B-RL-Zero-100-van-bs16` | 148.9 | 2.25% | 1.19% | 98.81% |
| `TimeThinker-4B-RL-v9-100-bs16` | 171.6 | 4.08% | 2.14% | 97.86% |
| `TimeThinker-4B-SFT-v10-50k` | 228.2 | 0.31% | 0.15% | 99.85% |
| `TimeThinker-4B-SFT-v8-10k` | 227.3 | 0.50% | 0.16% | 99.84% |
| `TimeThinker-4B-SFT-v3-10000-1ep` | 234.2 | 0.50% | 0.22% | 99.78% |
| `TimeThinker-4B-SFT-v5-10k` | 229.0 | 0.33% | 0.14% | 99.86% |
| `Qwen3-VL-4B-Instruct` | 179.9 | 5.28% | 2.40% | 97.60% |
| `TimeThinker-4B-SFT-v4-10000` | 236.4 | 0.54% | 0.20% | 99.80% |
| `Qwen3-VL-4B-Thinking` | 598.2 | 37.25% | 4.75% | 95.25% |

## 当前观察

- `ema-v2` 当前总分最高，Avg 55.85；但只领先最强 SFT `v9-10k-3ep` 0.09 点，二者整体能力非常接近。
- `v9-10k-3ep` 是当前最强 SFT，并取得 LongVideoReason 和 MVBench 全局第一；它的输出提取率为 99.95%，稳定性也明显优于 RL 模型。
- RL 组内部 `ema-v2` 最强，领先 `tgrpo-van2` 0.31 点；当前结果仍不足以证明 temporal/T-GRPO 配置稳定优于 EMA-GRPO 对照。
- `van-bs16` Avg 54.83，比 `van-v2` 低 0.57 点；仅从这次完整评测看，增大 batch size 没有带来总体收益，VideoMathQA 下降最明显。
- 以 `v9-10k-3ep` 初始化的 `RL-v9-100-bs16` Avg 54.81，比其 SFT 起点低 0.95 点。它在 TempCompass 和 VSIBench 上上涨，但 LongVideoReason、MVBench、VideoMME、VideoMMMU 均下降。
- `v10-50k` 的 Avg 为 54.48，低于 `v9-10k-3ep` 1.28 点；不过 VideoMathQA 达到全局最高 26.67，说明扩大数据量可能强化了部分数学视频能力，但没有转化为整体最优。
- `Qwen3-VL-4B-Instruct` Avg 53.56，接近较弱 SFT，但仍低于其余主要 SFT/RL run；其截断率和无效答案率也高于所有 SFT。
- `Qwen3-VL-4B-Thinking` 在 TempCompass 达到全局最高 74.36，但 Avg 只有 49.20。其平均输出 598.2 tokens、截断率 37.25%，当前输出长度和答案协议明显限制了整体表现。
- VideoMathQA 仍是所有模型共同短板，当前最高仅 26.67；后续优化应重点分析数学答案提取、视频条件利用和专项训练数据。

## 下一步建议

1. 主线模型优先保留 `ema-v2` 和 `v9-10k-3ep`，前者作为当前最高 Avg，后者作为最强且最稳定的 SFT 起点。
2. 对 `RL-v9-100-bs16` 做逐 benchmark 和 bad case 对比，确认 RL 后 LongVideoReason、MVBench、VideoMME、VideoMMMU 回退的原因。
3. 对 batch-size 消融继续保持相同 checkpoint、采样参数和随机种子；当前单次结果不支持 `bs16` 优于原配置。
4. 单独分析 `v10-50k` 在 VideoMathQA 上的收益来源，判断能否通过数据混合或专项 curriculum 保留该收益，同时恢复 VSIBench 等项目。
5. Thinking 模型应调整 `max_new_tokens`、stop 条件或答案提取协议后再比较，否则 37.25% 的截断率会显著污染结论。
