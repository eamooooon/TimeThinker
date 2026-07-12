# Evaluation Summary

记录 `Evaluation/results/<model>/frames16/_summary.json` 中的 canonical prompt 评测结果。本文只统计该目录下现有的完整 run，不与 `Evaluation/results-v4` 的历史 v4 prompt 结果混排。

## 结果口径

- Prompt：使用 `Evaluation/Eval/eval_bench.py` 和 `scripts/prompting/timethinker.py` 的 canonical QA 模板。
- `Avg`：8 个 benchmark 的 `answer_acc` 未加权平均，即 `_summary.json` 的 `macro_avg/by_benchmark`。
- `六项 Avg（去除 LongVideoReason、VideoMathQA）`：其余 6 个 benchmark 的 `answer_acc` 未加权平均。
- `answer_acc`：只判断最终答案正确性；格式服从情况通过 `extract_rate`、`invalid_rate`、`trunc_rate` 单独诊断。
- `frames16`：所有 run 都使用 `MAX_FRAMES=16`，共 8 个 benchmark、22,315 条样本。
- `wall_time`：单模型的实际等待时间；`benchmark_elapsed_sum` 为 8 个 GPU task 时间之和，因此不等于 wall time。
- 所有帧均命中已有 frame cache；本轮结果没有 frame decode fallback。

## 最新完整结果

| Model | Avg | 六项 Avg（去除 LongVideoReason、VideoMathQA） | LongVideoReason | MMVU | MVBench | TempCompass | VideoMathQA | VideoMME | VideoMMMU | VSIBench |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `TimeThinker-4B-RL-bs16-v9-van-500-new` | **57.77** | **60.77** | **71.60** | 66.24 | **66.05** | 70.76 | 25.95 | 57.48 | 52.56 | **51.50** |
| `TimeThinker-4B-RL-bs16-van-500-new` | 57.56 | 60.28 | 71.20 | 65.60 | 64.73 | **72.75** | **27.62** | **58.70** | 53.44 | 46.44 |
| `TimeThinker-4B-SFT-v9-10k-canonical` | 56.43 | 59.15 | 70.30 | **66.40** | 62.70 | 68.78 | 26.19 | 55.19 | 54.11 | 47.74 |
| `TimeThinker-4B-RL-bs16-van-100-new` | 55.66 | 59.35 | 68.70 | 64.48 | 62.18 | 72.52 | 20.48 | 54.63 | 51.44 | **50.86** |
| `TimeThinker-4B-SFT-v8-10k-canonical` | 54.46 | 57.22 | 67.60 | 62.56 | 62.18 | 68.26 | 24.76 | 54.11 | 51.22 | 44.99 |
| `TimeThinker-4B-SFT-v6-10k-canonical` | 54.25 | 56.94 | 70.70 | 60.64 | 61.12 | 67.20 | 21.67 | 54.22 | **54.33** | 44.10 |
| `TimeThinker-4B-SFT-v5-10k-canonical` | 54.14 | 56.95 | 70.50 | 62.08 | 61.68 | 67.80 | 20.95 | 53.67 | 52.44 | 44.02 |
| `TimeThinker-4B-SFT-v7-10k-canonical` | 53.52 | 56.59 | 69.80 | 63.04 | 61.70 | 67.19 | 18.81 | 53.04 | 50.89 | 43.70 |
| `Qwen3-VL-4B-Instruct` | 50.53 | 54.46 | 57.20 | 60.16 | 56.85 | 66.25 | 20.24 | 45.85 | 50.67 | 47.00 |
| `Qwen3-VL-4B-Thinking` | 49.01 | 52.98 | 55.20 | 59.36 | 60.70 | 68.61 | 19.05 | 50.44 | 41.44 | 37.30 |

`TimeThinker-4B-RL-bs16-v9-van-500-new` 使用
`models/TimeThinker-4B-RL-bs16-v9-van-500-new/global_step_300/actor/huggingface`。
8 个结果 JSON 均已写入；启动脚本在生成 `_summary` 前中断，因此本表根据这 8 个结果文件重建，未记录可靠的本轮 `wall_time`。

## 训练模型结果

| Model | Avg | 相对 canonical SFT v5 |
|---|---:|---:|
| `TimeThinker-4B-RL-bs16-v9-van-500-new` | **57.77** | +3.63 |
| `TimeThinker-4B-RL-bs16-van-500-new` | 57.56 | +3.42 |
| `TimeThinker-4B-SFT-v9-10k-canonical` | 56.43 | +2.29 |
| `TimeThinker-4B-RL-bs16-van-100-new` | 55.66 | +1.52 |
| `TimeThinker-4B-SFT-v8-10k-canonical` | 54.46 | +0.32 |
| `TimeThinker-4B-SFT-v6-10k-canonical` | 54.25 | +0.11 |
| `TimeThinker-4B-SFT-v5-10k-canonical` | 54.14 | — |
| `TimeThinker-4B-SFT-v7-10k-canonical` | 53.52 | -0.62 |

RL v9 500 是当前总分最高模型，并在 LongVideoReason、MVBench、VSIBench 三项最佳；此前的 RL 500 仍保持 TempCompass、VideoMathQA、VideoMME 最好。v9 SFT 保持 MMVU 最好，v6 SFT 保持 VideoMMMU 最好。RL v9 500 相较 canonical SFT v5 总分提升 3.63 点、相较此前 RL 500 再提升 0.21 点。

## Baseline 结果

| Model | Avg | 相对 RL v9 500 |
|---|---:|---:|
| `Qwen3-VL-4B-Instruct` | 50.53 | -7.24 |
| `Qwen3-VL-4B-Thinking` | 49.01 | -8.76 |

`Qwen3-VL-4B-Instruct` 已有稳定的基线能力，但在 VideoMME 和 LongVideoReason 上与训练模型差距较大。`Qwen3-VL-4B-Thinking` 在 TempCompass（68.61）和 MVBench（60.70）不弱，但长输出使整体效率和答案稳定性明显下降。

## 评测耗时

| Model | wall_time | benchmark_elapsed_sum | Schedule |
|---|---:|---:|---|
| `TimeThinker-4B-RL-bs16-v9-van-500-new` | 未记录（收尾中断） | 130.3m | balanced |
| `TimeThinker-4B-RL-bs16-van-500-new` | 37m30s | 134.9m | balanced |
| `TimeThinker-4B-SFT-v9-10k-canonical` | 37m11s | 131.6m | balanced |
| `TimeThinker-4B-RL-bs16-van-100-new` | 33m32s | 121.2m | balanced |
| `TimeThinker-4B-SFT-v8-10k-canonical` | 40m33s | 143.6m | balanced |
| `TimeThinker-4B-SFT-v6-10k-canonical` | 41m32s | 140.0m | balanced |
| `TimeThinker-4B-SFT-v5-10k-canonical` | 37m11s | 133.0m | balanced |
| `TimeThinker-4B-SFT-v7-10k-canonical` | 44m04s | 154.6m | balanced |
| `Qwen3-VL-4B-Instruct` | 34m45s | 121.3m | balanced |
| `Qwen3-VL-4B-Thinking` | 51m22s | 169.1m | balanced |

## 每项最优

| Benchmark | Best model | Acc |
|---|---|---:|
| LongVideoReason | `TimeThinker-4B-RL-bs16-v9-van-500-new` | 71.60 |
| MMVU | `TimeThinker-4B-SFT-v9-10k-canonical` | 66.40 |
| MVBench | `TimeThinker-4B-RL-bs16-v9-van-500-new` | 66.05 |
| TempCompass | `TimeThinker-4B-RL-bs16-van-500-new` | 72.75 |
| VideoMathQA | `TimeThinker-4B-RL-bs16-van-500-new` | 27.62 |
| VideoMME | `TimeThinker-4B-RL-bs16-van-500-new` | 58.70 |
| VideoMMMU | `TimeThinker-4B-SFT-v6-10k-canonical` | 54.33 |
| VSIBench | `TimeThinker-4B-RL-bs16-v9-van-500-new` | 51.50 |

## 输出稳定性

以下指标按样本数加权。

| Model | weighted avg_tokens | weighted trunc_rate | weighted invalid_rate | weighted extract_rate |
|---|---:|---:|---:|---:|
| `TimeThinker-4B-RL-bs16-v9-van-500-new` | 211.2 | 0.04% | 0.03% | 99.97% |
| `TimeThinker-4B-RL-bs16-van-500-new` | 209.3 | 0.07% | 0.05% | 99.95% |
| `TimeThinker-4B-SFT-v9-10k-canonical` | 230.2 | 0.17% | 0.08% | 99.92% |
| `TimeThinker-4B-RL-bs16-van-100-new` | 87.2 | 2.09% | 1.44% | 98.56% |
| `TimeThinker-4B-SFT-v8-10k-canonical` | 225.6 | 0.52% | 0.17% | 99.83% |
| `TimeThinker-4B-SFT-v6-10k-canonical` | 227.3 | 0.34% | 0.14% | 99.86% |
| `TimeThinker-4B-SFT-v5-10k-canonical` | 226.3 | 0.35% | 0.18% | 99.82% |
| `TimeThinker-4B-SFT-v7-10k-canonical` | 227.1 | 0.32% | 0.15% | 99.85% |
| `Qwen3-VL-4B-Instruct` | 100.2 | 3.20% | 5.46% | 94.54% |
| `Qwen3-VL-4B-Thinking` | 487.9 | 26.32% | 3.57% | 96.43% |

RL v9 500 的格式稳定性最好：加权提取率 99.97%，截断率和无效率分别仅 0.04% 与 0.03%。相较 RL 100，它的输出更长（211.2 vs. 87.2 tokens），但 VideoMathQA 的截断率从 34.29% 降至 1.43%、无效率从 19.76% 降至 0.95%。Thinking baseline 的加权截断率达到 26.32%，不宜在当前 `max_tokens=1024` 设定下直接以总分判断其能力。

## 当前观察

- RL v9 500 是当前 `Evaluation/results` 中的最佳总分模型，Avg 为 57.77，较此前 RL 500 高 0.21 点、较 canonical SFT v9 高 1.34 点、较 RL 100 高 2.11 点。
- 去除 LongVideoReason 和 VideoMathQA 后，RL v9 500 的六项 Avg 为 60.77，较此前 RL 500 高 0.49 点、较 canonical SFT v9 高 1.62 点。
- v8 是新增 v6–v8 中总分最高的模型（54.46），较 v5 高 0.32 点，但仍低于 v9 1.97 点；其 VideoMathQA（24.76）和 VSIBench（44.99）优于 v5。
- RL v9 500 在 LongVideoReason、MVBench、VSIBench 取得当前最优；此前 RL 500 在 TempCompass、VideoMathQA、VideoMME 仍然最好，v9 SFT 保持 MMVU 最优，v6 SFT 保持 VideoMMMU 最优。
- VideoMathQA 仍是所有模型的最弱项目；RL v9 500 的 25.95 低于此前 RL 500 的 27.62，但其低截断率与无效率表明差异不是主要由格式或答案抽取造成。
- v6 的 VideoMMMU（54.33）和 v9 的 MMVU（66.40）仍是 RL 500 未覆盖的优势项目，应作为后续 RL 对照的重点。

## 下一步建议

1. 以 `TimeThinker-4B-RL-bs16-v9-van-500-new` 作为当前总分最佳候选，同时保留此前 RL 500 作为 TempCompass、VideoMathQA、VideoMME 的对照，v9 SFT 作为 MMVU、v6 SFT 作为 VideoMMMU 的对照。
2. 对两个 RL 500 run 做逐 benchmark 和 case study，重点解释 RL v9 500 在 MVBench/VSIBench 的收益以及在 TempCompass/VideoMathQA/VideoMME 的回退。
3. 以 RL v9 500 为初始化 checkpoint 复训或对照 RL，重点验证能否保留其三项新增收益，同时恢复 TempCompass、VideoMathQA、VideoMME 的优势。
4. 对 Thinking baseline 可单独提高 `MAX_TOKENS` 后复跑，避免 26.32% 的截断率混入模型能力比较。

## Evaluator and prompt history

`Evaluation/Eval/eval_bench.py` is the only active evaluator. It is the default in `scripts/eval/run_bench.sh` and imports the canonical QA prompt from `scripts/prompting/timethinker.py`.

| Version | Prompt regime | Material behavior | Historical result scope |
|---|---|---|---|
| v1 | Natural instruction with fixed answer examples such as `A`, `3.14`, and `Paris` | Basic `<answer>` extraction and accuracy | `Evaluation/results-v1-rerun/` only |
| v2 | Strict full `<think>...</think><answer>...</answer>` contract; still fixed examples | Added robust multiple-choice/numeric extraction, validity, extract/invalid/truncation diagnostics, category metrics, bootstrap CI, and `--rescore_existing` | Strict-prompt history only |
| v3 | Natural outer instruction; retained v2 examples | Same v2 scoring and diagnostics | Prompt-ablation history only |
| v4 | Natural instruction with no concrete answer examples | Same v2/v3 scoring and diagnostics | `Evaluation/results-v4/` only |
| canonical (current) | Shared template with no fixed answer examples | Mature v2+ scoring; one prompt implementation shared by SFT, RL, eval, and inference | `Evaluation/results/` only |

Do not mix scores across result roots/prompt regimes. Fixed type-specific examples were not valid few-shot demonstrations: they supplied answer priors without task-specific visual evidence and measurably biased multiple-choice predictions. The old source snapshots were removed after this record was consolidated; historical JSON and markdown reports remain in [archive/](archive/README.md). If exact old code is ever needed, restore it from the Git revision before the cleanup instead of creating another `eval_bench_v*.py` copy.
