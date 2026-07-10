# eval-v2 评测汇总

记录 `Evaluation/results/*/frames16` 当前主评测结果。本文档把这套结果记为 `eval-v2`，用于和 `eval-v1-return` 等 prompt/return 口径结果区分。

## 口径说明

- `eval-v2` 来源：`Evaluation/results/<model>/frames16/_summary.json`。
- `Avg`：8 个 benchmark 的 `answer_acc` 简单未加权平均，即 `_summary.json` 里的 `macro_avg/by_benchmark`。
- `frames16`：所有结果均为 `MAX_FRAMES=16`。
- 当前目录下共有 12 个完整模型，每个模型覆盖 8 个 benchmark，总样本数均为 22315。
- 本文档只统计当前 `Evaluation/results` 中实际存在的结果；没有混入旧目录里已经不存在的 v2/v6/v7 等历史 run。

## 总表

| Model | Avg | N | Samples | LongVideoReason | MMVU | MVBench | TempCompass | VideoMathQA | VideoMME | VideoMMMU | VSIBench |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `TimeThinker-4B-RL-Zero-100-tgrpo-van2` | 55.53 | 8 | 22315 | 67.90 | 64.64 | 61.95 | 72.21 | 24.76 | 55.11 | 52.00 | 45.69 |
| `TimeThinker-4B-RL-Zero-100-ema-v2` | 55.29 | 8 | 22315 | 68.70 | 64.32 | 61.98 | 72.14 | 22.62 | 55.22 | 51.78 | 45.55 |
| `TimeThinker-4B-RL-Zero-100-van-v2` | 54.39 | 8 | 22315 | 65.40 | 63.52 | 61.88 | 72.35 | 19.76 | 54.00 | 52.22 | 45.99 |
| `Qwen3-VL-4B-Instruct` | 49.52 | 8 | 22315 | 59.70 | 58.08 | 58.03 | 68.99 | 22.86 | 48.74 | 47.44 | 32.33 |
| `TimeThinker-4B-SFT-v9-10k-3ep` | 47.73 | 8 | 22315 | 58.90 | 54.24 | 56.90 | 64.56 | 23.57 | 48.33 | 44.56 | 30.80 |
| `TimeThinker-4B-SFT-v3-10000` | 46.65 | 8 | 22315 | 55.40 | 52.16 | 56.57 | 64.12 | 22.86 | 47.22 | 45.00 | 29.84 |
| `TimeThinker-4B-SFT-v3-10000-1ep` | 46.56 | 8 | 22315 | 56.50 | 52.00 | 56.30 | 63.86 | 24.05 | 46.37 | 43.22 | 30.18 |
| `TimeThinker-4B-SFT-v8-10k` | 45.91 | 8 | 22315 | 55.10 | 50.56 | 55.90 | 64.26 | 24.29 | 47.11 | 42.78 | 27.32 |
| `Qwen3-VL-4B-Thinking` | 45.70 | 8 | 22315 | 53.10 | 56.00 | 58.55 | 71.76 | 18.57 | 49.11 | 34.67 | 23.85 |
| `TimeThinker-4B-SFT-v4-10000` | 45.11 | 8 | 22315 | 54.60 | 50.40 | 55.17 | 63.04 | 20.48 | 45.89 | 42.44 | 28.84 |
| `TimeThinker-4B-SFT-v5-10k` | 44.82 | 8 | 22315 | 53.90 | 50.56 | 54.75 | 61.88 | 20.71 | 45.78 | 40.56 | 30.42 |
| `TimeThinker-4B-SFT-v10-50k` | 43.10 | 8 | 22315 | 47.70 | 48.80 | 53.20 | 62.59 | 20.71 | 43.67 | 40.00 | 28.11 |

## RL 结果

| Model | Avg | LongVideoReason | MMVU | MVBench | TempCompass | VideoMathQA | VideoMME | VideoMMMU | VSIBench |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `TimeThinker-4B-RL-Zero-100-tgrpo-van2` | 55.53 | 67.90 | 64.64 | 61.95 | 72.21 | 24.76 | 55.11 | 52.00 | 45.69 |
| `TimeThinker-4B-RL-Zero-100-ema-v2` | 55.29 | 68.70 | 64.32 | 61.98 | 72.14 | 22.62 | 55.22 | 51.78 | 45.55 |
| `TimeThinker-4B-RL-Zero-100-van-v2` | 54.39 | 65.40 | 63.52 | 61.88 | 72.35 | 19.76 | 54.00 | 52.22 | 45.99 |

## SFT 结果

| Model | Avg | LongVideoReason | MMVU | MVBench | TempCompass | VideoMathQA | VideoMME | VideoMMMU | VSIBench |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `TimeThinker-4B-SFT-v9-10k-3ep` | 47.73 | 58.90 | 54.24 | 56.90 | 64.56 | 23.57 | 48.33 | 44.56 | 30.80 |
| `TimeThinker-4B-SFT-v3-10000` | 46.65 | 55.40 | 52.16 | 56.57 | 64.12 | 22.86 | 47.22 | 45.00 | 29.84 |
| `TimeThinker-4B-SFT-v3-10000-1ep` | 46.56 | 56.50 | 52.00 | 56.30 | 63.86 | 24.05 | 46.37 | 43.22 | 30.18 |
| `TimeThinker-4B-SFT-v8-10k` | 45.91 | 55.10 | 50.56 | 55.90 | 64.26 | 24.29 | 47.11 | 42.78 | 27.32 |
| `TimeThinker-4B-SFT-v4-10000` | 45.11 | 54.60 | 50.40 | 55.17 | 63.04 | 20.48 | 45.89 | 42.44 | 28.84 |
| `TimeThinker-4B-SFT-v5-10k` | 44.82 | 53.90 | 50.56 | 54.75 | 61.88 | 20.71 | 45.78 | 40.56 | 30.42 |
| `TimeThinker-4B-SFT-v10-50k` | 43.10 | 47.70 | 48.80 | 53.20 | 62.59 | 20.71 | 43.67 | 40.00 | 28.11 |

## Baseline 结果

| Model | Avg | LongVideoReason | MMVU | MVBench | TempCompass | VideoMathQA | VideoMME | VideoMMMU | VSIBench |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3-VL-4B-Instruct` | 49.52 | 59.70 | 58.08 | 58.03 | 68.99 | 22.86 | 48.74 | 47.44 | 32.33 |
| `Qwen3-VL-4B-Thinking` | 45.70 | 53.10 | 56.00 | 58.55 | 71.76 | 18.57 | 49.11 | 34.67 | 23.85 |

## 每项最优

| Benchmark | Best overall | Acc | Best SFT | Acc |
|---|---|---:|---|---:|
| LongVideoReason | `TimeThinker-4B-RL-Zero-100-ema-v2` | 68.70 | `TimeThinker-4B-SFT-v9-10k-3ep` | 58.90 |
| MMVU | `TimeThinker-4B-RL-Zero-100-tgrpo-van2` | 64.64 | `TimeThinker-4B-SFT-v9-10k-3ep` | 54.24 |
| MVBench | `TimeThinker-4B-RL-Zero-100-ema-v2` | 61.98 | `TimeThinker-4B-SFT-v9-10k-3ep` | 56.90 |
| TempCompass | `TimeThinker-4B-RL-Zero-100-van-v2` | 72.35 | `TimeThinker-4B-SFT-v9-10k-3ep` | 64.56 |
| VideoMathQA | `TimeThinker-4B-RL-Zero-100-tgrpo-van2` | 24.76 | `TimeThinker-4B-SFT-v8-10k` | 24.29 |
| VideoMME | `TimeThinker-4B-RL-Zero-100-ema-v2` | 55.22 | `TimeThinker-4B-SFT-v9-10k-3ep` | 48.33 |
| VideoMMMU | `TimeThinker-4B-RL-Zero-100-van-v2` | 52.22 | `TimeThinker-4B-SFT-v3-10000` | 45.00 |
| VSIBench | `TimeThinker-4B-RL-Zero-100-van-v2` | 45.99 | `TimeThinker-4B-SFT-v9-10k-3ep` | 30.80 |

## 输出稳定性

| Model | weighted avg_tokens | weighted trunc_rate | weighted invalid_rate | weighted extract_rate |
|---|---:|---:|---:|---:|
| `TimeThinker-4B-RL-Zero-100-tgrpo-van2` | 134.0 | 2.26% | 1.10% | 98.90% |
| `TimeThinker-4B-RL-Zero-100-ema-v2` | 133.1 | 2.15% | 0.99% | 99.01% |
| `TimeThinker-4B-RL-Zero-100-van-v2` | 156.7 | 3.28% | 1.38% | 98.62% |
| `Qwen3-VL-4B-Instruct` | 182.1 | 5.27% | 2.14% | 97.86% |
| `TimeThinker-4B-SFT-v9-10k-3ep` | 233.3 | 0.15% | 0.04% | 99.96% |
| `TimeThinker-4B-SFT-v3-10000` | 235.3 | 0.29% | 0.12% | 99.88% |
| `TimeThinker-4B-SFT-v3-10000-1ep` | 236.8 | 0.43% | 0.18% | 99.82% |
| `TimeThinker-4B-SFT-v8-10k` | 232.4 | 0.48% | 0.18% | 99.82% |
| `Qwen3-VL-4B-Thinking` | 578.1 | 35.39% | 4.18% | 95.82% |
| `TimeThinker-4B-SFT-v4-10000` | 235.9 | 0.81% | 0.33% | 99.67% |
| `TimeThinker-4B-SFT-v5-10k` | 233.0 | 0.38% | 0.12% | 99.88% |
| `TimeThinker-4B-SFT-v10-50k` | 230.6 | 0.32% | 0.10% | 99.90% |

## 当前观察

- RL 仍是当前 `eval-v2` 主线最强，前三名全部是 RL run，Avg 在 54.39-55.53 之间。
- `tgrpo-van2` 是当前最高分，Avg 55.53；相比 `ema-v2` 高 0.24 点，相比 `van-v2` 高 1.14 点。
- SFT 里 `v9-10k-3ep` 仍然最好，Avg 47.73，并且是 SFT 内 6/8 个 benchmark 第一。
- `Qwen3-VL-4B-Instruct` 在这套结果里达到 49.52，高于所有 SFT run，但仍低于三个 RL run。
- `Qwen3-VL-4B-Thinking` 的 TempCompass 和 MVBench 不低，但整体 Avg 只有 45.70，主要被 VideoMMMU、VSIBench 和 VideoMathQA 拉低。
- `Qwen3-VL-4B-Thinking` 输出非常长，weighted avg tokens 达 578.1，truncation rate 达 35.39%，说明它在当前 max token / 输出协议下不稳定，不能简单理解为 thinking 模型能力更差。
- `v10-50k` 仍然低于 10k 主线 SFT，当前结果不支持“直接扩大到 50k 就提升”的结论。

## 和 eval-v1-return 的关系

`eval-v2` 是当前主结果；`eval-v1-return` 是另一套 prompt/return 口径的诊断结果。二者不应直接混成一个榜。

从当前统计看，`eval-v2` 对 RL 更友好，而 `eval-v1-return` 对 SFT / base 更友好。这说明模型在不同 prompt/return 协议下仍有明显波动，尤其是非 RL 模型更容易受严格输出格式影响。后续如果要报告稳定结论，建议同时报告主分数和 prompt sensitivity gap。

## 下一步建议

1. 继续把 `tgrpo-van2`、`ema-v2`、`van-v2` 作为 RL 主线对照，重点看它们在 VideoMathQA、VideoMMMU、VSIBench 上的差异。
2. SFT 初始化优先看 `v9-10k-3ep`，它仍是当前最强 SFT。
3. 如果继续测试 thinking 模型，应单独调低输出长度或增加 stop/answer 提取约束，否则 truncation 会严重污染分数。
4. 对 prompt sensitivity，继续保留 `eval-v1-return` 作为诊断集，重点分析两套结果互相翻转的样本。
5. 对 `v10-50k` 暂不继续扩大同方向，优先排查数据质量、分布匹配和训练轮数。
