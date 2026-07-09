# Evaluation Summary

记录当前统一评测结果、耗时口径和调度状态。分数均来自 `Evaluation/results/<model>/frames16/_summary.json`，默认使用 8 个 benchmark 的 `answer_acc` 简单平均。

## 结果口径

- `Avg`：8 个 benchmark 的 `answer_acc` 未加权平均，即 `_summary.json` 里的 `macro_avg/by_benchmark`。
- `wall_time`：一个模型完整评测的实际等待时间，由新版 `run_bench.sh` 写入 `_summary.json`；早期 summary 没有该字段则记为 `-`。
- `benchmark_elapsed_sum`：各 benchmark 自己记录的 `elapsed_min` 相加，表示 workload/GPU task time，不等于实际等待时间。
- `frames16`：所有结果均为 `MAX_FRAMES=16`。
- 当前 `answer_acc` 只看答案是否正确，不再要求输出严格 `<think>/<answer>` 格式；格式相关问题通过 `extract_rate` / `invalid_rate` / `strict_format` 等诊断指标单独观察。
- 当前共有 15 个完整 summary，每个 summary 覆盖 8 个 benchmark、总样本数 22315。
- `van-v2` / `ema-v2` 是早期 summary，没有把 `wall_time` 写入 `_summary.json`；表里的 wall time 从对应 `logs/eval_list_*.log` 的 `DATE_SUFFIX` 和日志修改时间还原。

## 最新完整结果

| Model | Avg | LongVideoReason | MMVU | MVBench | TempCompass | VideoMathQA | VideoMME | VideoMMMU | VSIBench |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `TimeThinker-4B-RL-Zero-100-van-v2` | **56.57** | **69.60** | 67.04 | 62.85 | **72.89** | 23.57 | **55.96** | **53.78** | **46.90** |
| `TimeThinker-4B-RL-Zero-100-ema-v2` | 56.31 | 69.00 | **67.36** | **63.18** | 72.37 | 23.81 | 55.89 | 52.44 | 46.42 |
| `TimeThinker-4B-RL-Zero-100-tgrpo-van2` | 56.08 | 69.50 | **67.36** | 62.98 | 72.18 | 22.62 | 55.48 | 52.11 | 46.38 |
| `TimeThinker-4B-RL-Zero-100-tgrpo-van` | 55.38 | 66.90 | 65.76 | 63.10 | 72.33 | 23.10 | 54.37 | 52.44 | 45.04 |
| `TimeThinker-4B-SFT-v9-10k-3ep` | **51.93** | 65.40 | 59.20 | 61.27 | 67.93 | **26.19** | 51.96 | 51.11 | 32.34 |
| `TimeThinker-4B-SFT-v7-10k` | 50.15 | 63.30 | 57.12 | 60.22 | 68.26 | 20.71 | 51.44 | 45.89 | 34.28 |
| `TimeThinker-4B-SFT-v3-10000` | 50.08 | 63.80 | 57.92 | 60.95 | 66.59 | 23.10 | 50.70 | 47.33 | 30.21 |
| `TimeThinker-4B-SFT-v6-10k` | 49.95 | 64.70 | 58.24 | 59.20 | 66.23 | 22.38 | 51.22 | 45.89 | 31.73 |
| `TimeThinker-4B-SFT-v3-10000-1ep` | 48.91 | 61.70 | 55.52 | 58.20 | 66.09 | 21.43 | 49.44 | 46.89 | 32.05 |
| `TimeThinker-4B-SFT-v5-10k` | 45.21 | 54.10 | 50.56 | 55.53 | 62.82 | 22.14 | 44.52 | 41.78 | 30.19 |
| `TimeThinker-4B-SFT-v2-10000` | 44.72 | 54.30 | 48.96 | 54.67 | 62.72 | 23.10 | 44.00 | 41.00 | 28.98 |
| `TimeThinker-4B-SFT-v10-50k` | 43.67 | 52.40 | 47.68 | 54.07 | 62.40 | 20.71 | 43.63 | 37.89 | 30.58 |
| `TimeThinker-4B-SFT-v4-10000` | 42.69 | 51.30 | 47.36 | 52.28 | 60.78 | 20.24 | 41.85 | 39.67 | 28.06 |
| `TimeThinker-4B-SFT-v8-10k` | 40.70 | 51.30 | 51.04 | 57.43 | 52.79 | 15.24 | 43.63 | 27.44 | 26.75 |
| `Qwen3-VL-4B-Instruct` | 35.94 | 44.70 | 47.84 | 42.25 | 50.37 | 15.95 | 38.44 | 33.67 | 14.28 |

## SFT 结果

| Model | Avg | LongVideoReason | MMVU | MVBench | TempCompass | VideoMathQA | VideoMME | VideoMMMU | VSIBench |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `TimeThinker-4B-SFT-v9-10k-3ep` | **51.93** | **65.40** | **59.20** | **61.27** | 67.93 | **26.19** | **51.96** | **51.11** | 32.34 |
| `TimeThinker-4B-SFT-v7-10k` | 50.15 | 63.30 | 57.12 | 60.22 | **68.26** | 20.71 | 51.44 | 45.89 | **34.28** |
| `TimeThinker-4B-SFT-v3-10000` | 50.08 | 63.80 | 57.92 | 60.95 | 66.59 | 23.10 | 50.70 | 47.33 | 30.21 |
| `TimeThinker-4B-SFT-v6-10k` | 49.95 | 64.70 | 58.24 | 59.20 | 66.23 | 22.38 | 51.22 | 45.89 | 31.73 |
| `TimeThinker-4B-SFT-v3-10000-1ep` | 48.91 | 61.70 | 55.52 | 58.20 | 66.09 | 21.43 | 49.44 | 46.89 | 32.05 |
| `TimeThinker-4B-SFT-v5-10k` | 45.21 | 54.10 | 50.56 | 55.53 | 62.82 | 22.14 | 44.52 | 41.78 | 30.19 |
| `TimeThinker-4B-SFT-v2-10000` | 44.72 | 54.30 | 48.96 | 54.67 | 62.72 | 23.10 | 44.00 | 41.00 | 28.98 |
| `TimeThinker-4B-SFT-v10-50k` | 43.67 | 52.40 | 47.68 | 54.07 | 62.40 | 20.71 | 43.63 | 37.89 | 30.58 |
| `TimeThinker-4B-SFT-v4-10000` | 42.69 | 51.30 | 47.36 | 52.28 | 60.78 | 20.24 | 41.85 | 39.67 | 28.06 |
| `TimeThinker-4B-SFT-v8-10k` | 40.70 | 51.30 | 51.04 | 57.43 | 52.79 | 15.24 | 43.63 | 27.44 | 26.75 |

## RL 结果

| Model | Avg | LongVideoReason | MMVU | MVBench | TempCompass | VideoMathQA | VideoMME | VideoMMMU | VSIBench |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `TimeThinker-4B-RL-Zero-100-van-v2` | **56.57** | **69.60** | 67.04 | 62.85 | **72.89** | 23.57 | **55.96** | **53.78** | **46.90** |
| `TimeThinker-4B-RL-Zero-100-ema-v2` | 56.31 | 69.00 | **67.36** | **63.18** | 72.37 | **23.81** | 55.89 | 52.44 | 46.42 |
| `TimeThinker-4B-RL-Zero-100-tgrpo-van2` | 56.08 | 69.50 | **67.36** | 62.98 | 72.18 | 22.62 | 55.48 | 52.11 | 46.38 |
| `TimeThinker-4B-RL-Zero-100-tgrpo-van` | 55.38 | 66.90 | 65.76 | 63.10 | 72.33 | 23.10 | 54.37 | 52.44 | 45.04 |

## Baseline 结果

| Model | Avg | LongVideoReason | MMVU | MVBench | TempCompass | VideoMathQA | VideoMME | VideoMMMU | VSIBench |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3-VL-4B-Instruct` | 35.94 | 44.70 | 47.84 | 42.25 | 50.37 | 15.95 | 38.44 | 33.67 | 14.28 |

## 评测耗时

| Model | wall_time | benchmark_elapsed_sum | Schedule | Summary updated |
|---|---:|---:|---|---|
| `TimeThinker-4B-RL-Zero-100-van-v2` | `1h42m13s` | `357.75m` | listed/old | 2026-07-07 17:59 |
| `TimeThinker-4B-RL-Zero-100-ema-v2` | `35m08s` | `109.56m` | listed/old | 2026-07-07 18:34 |
| `TimeThinker-4B-RL-Zero-100-tgrpo-van2` | `34m32s` | `109.28m` | listed | 2026-07-08 03:27 |
| `TimeThinker-4B-RL-Zero-100-tgrpo-van` | `32m07s` | `112.66m` | balanced | 2026-07-08 02:19 |
| `TimeThinker-4B-SFT-v9-10k-3ep` | `36m42s` | `133.45m` | balanced | 2026-07-08 11:58 |
| `TimeThinker-4B-SFT-v7-10k` | `41m32s` | `147.66m` | balanced | 2026-07-08 10:44 |
| `TimeThinker-4B-SFT-v3-10000` | `34m53s` | `123.58m` | balanced | 2026-07-08 03:29 |
| `TimeThinker-4B-SFT-v6-10k` | `39m15s` | `133.93m` | balanced | 2026-07-08 05:28 |
| `TimeThinker-4B-SFT-v3-10000-1ep` | `38m17s` | `133.29m` | balanced | 2026-07-08 04:08 |
| `TimeThinker-4B-SFT-v5-10k` | `33m35s` | `122.60m` | balanced | 2026-07-08 13:27 |
| `TimeThinker-4B-SFT-v2-10000` | `35m23s` | `127.24m` | balanced | 2026-07-08 02:55 |
| `TimeThinker-4B-SFT-v10-50k` | `36m47s` | `131.38m` | balanced | 2026-07-08 12:35 |
| `TimeThinker-4B-SFT-v4-10000` | `41m21s` | `147.35m` | balanced | 2026-07-08 04:49 |
| `TimeThinker-4B-SFT-v8-10k` | `37m49s` | `134.31m` | balanced | 2026-07-08 11:21 |
| `Qwen3-VL-4B-Instruct` | `38m27s` | `141.28m` | balanced | 2026-07-08 13:24 |

## 当前观察

- RL 仍然整体最强，四个 RL run 都在 55.38-56.57；最佳 SFT `v9-10k-3ep` 是 51.93，距离最佳 RL 还差 4.64 点。
- SFT 里 `v9-10k-3ep` 明显最好，相比 `v3-10000` 提升 1.85 点，并且拿到 SFT 内 6/8 个 benchmark 第一；它也是全表里 VideoMathQA 最高的模型。
- `v5-10k` 只提高 `image_max_pixels`，Avg 为 45.21，仅略高于 `v2-10000` 的 44.72，明显低于 v3/v6/v7/v9；单独提高图像分辨率没有带来视频 benchmark 的主线收益。
- `v7-10k` 和 `v3-10000` 几乎持平，主要涨在 TempCompass、VideoMME、VSIBench，掉在 VideoMathQA、VideoMMMU；vision tower + projector 都解冻没有形成稳定全面收益。
- `v8-10k` video-only 明显失败，Avg 只有 40.70，且 weighted invalid rate 达 13.16%，优先排查数据格式/答案抽取/训练样本构造，不建议作为当前主线。
- `v10-50k` 低于 10k 主线模型，说明扩大到 50k 这次没有直接带来收益；需要先确认数据分布、采样质量和训练配置，再决定是否继续加大数据。
- `Qwen3-VL-4B-Instruct` baseline 已按“只看答案、不看格式”的口径重算，Avg 从严格格式口径下的 12.20 修正到 35.94；weighted extract rate 从 24.73% 回升到 85.05%。它仍明显低于 SFT/RL 主线，但不再把缺失 `<answer>` 开标签这类格式问题直接计为能力错误。

## 每项最优

| Benchmark | Best overall | Acc | Best SFT | Acc |
|---|---|---:|---|---:|
| LongVideoReason | `TimeThinker-4B-RL-Zero-100-van-v2` | 69.60 | `TimeThinker-4B-SFT-v9-10k-3ep` | 65.40 |
| MMVU | `TimeThinker-4B-RL-Zero-100-ema-v2` | 67.36 | `TimeThinker-4B-SFT-v9-10k-3ep` | 59.20 |
| MVBench | `TimeThinker-4B-RL-Zero-100-ema-v2` | 63.18 | `TimeThinker-4B-SFT-v9-10k-3ep` | 61.27 |
| TempCompass | `TimeThinker-4B-RL-Zero-100-van-v2` | 72.89 | `TimeThinker-4B-SFT-v7-10k` | 68.26 |
| VideoMathQA | `TimeThinker-4B-SFT-v9-10k-3ep` | 26.19 | `TimeThinker-4B-SFT-v9-10k-3ep` | 26.19 |
| VideoMME | `TimeThinker-4B-RL-Zero-100-van-v2` | 55.96 | `TimeThinker-4B-SFT-v9-10k-3ep` | 51.96 |
| VideoMMMU | `TimeThinker-4B-RL-Zero-100-van-v2` | 53.78 | `TimeThinker-4B-SFT-v9-10k-3ep` | 51.11 |
| VSIBench | `TimeThinker-4B-RL-Zero-100-van-v2` | 46.90 | `TimeThinker-4B-SFT-v7-10k` | 34.28 |

## 输出稳定性

| Model | weighted avg_tokens | weighted trunc_rate | weighted invalid_rate | weighted extract_rate |
|---|---:|---:|---:|---:|
| `TimeThinker-4B-RL-Zero-100-van-v2` | 104.8 | 0.96% | 1.04% | 99.04% |
| `TimeThinker-4B-RL-Zero-100-ema-v2` | 85.0 | 0.58% | 0.59% | 99.42% |
| `TimeThinker-4B-RL-Zero-100-tgrpo-van2` | 89.6 | 0.67% | 0.70% | 99.33% |
| `TimeThinker-4B-RL-Zero-100-tgrpo-van` | 100.3 | 0.96% | 1.03% | 99.04% |
| `TimeThinker-4B-SFT-v9-10k-3ep` | 230.3 | 0.24% | 0.25% | 99.76% |
| `TimeThinker-4B-SFT-v7-10k` | 224.2 | 0.44% | 0.45% | 99.56% |
| `TimeThinker-4B-SFT-v3-10000` | 230.3 | 0.23% | 0.27% | 99.74% |
| `TimeThinker-4B-SFT-v6-10k` | 221.8 | 0.44% | 0.48% | 99.52% |
| `TimeThinker-4B-SFT-v3-10000-1ep` | 231.7 | 0.58% | 0.58% | 99.42% |
| `TimeThinker-4B-SFT-v5-10k` | 225.7 | 0.44% | 0.55% | 99.46% |
| `TimeThinker-4B-SFT-v2-10000` | 228.3 | 0.79% | 0.79% | 99.22% |
| `TimeThinker-4B-SFT-v10-50k` | 231.3 | 0.52% | 0.93% | 99.08% |
| `TimeThinker-4B-SFT-v4-10000` | 238.5 | 3.53% | 3.54% | 96.47% |
| `TimeThinker-4B-SFT-v8-10k` | 217.6 | 0.50% | 13.16% | 86.85% |
| `Qwen3-VL-4B-Instruct` | 121.8 | 3.77% | 14.95% | 85.05% |

## T-GRPO 诊断

当前结果并不支持 “T-GRPO 必然优于 GRPO” 这个结论。更像是 temporal reward 改变了训练信号和输出行为，但没有稳定转化成最终 8 项平均分。

### 配置检查

| Model | adv_estimator | temporal | temporal_reward | shuffled_rollout_ratio |
|---|---|---:|---:|---:|
| `van-v2` | `grpo` | false | 0.3 | 0.5 |
| `ema-v2` | `ema_grpo` | false | 0.3 | 0.5 |
| `tgrpo-van` | `grpo` | true | 0.3 | 0.5 |
| `tgrpo-van2` | `grpo` | true | 0.3 | 0.5 |

注意：原先记作 `tgrpo-ema` 的这组结果已经更名为 `tgrpo-van2`。它的 `experiment_config.json` 里 `adv_estimator=grpo`，不是 `ema_grpo`，所以应归类为第二个 `grpo + temporal` run，不能当作 T-GRPO + EMA 结论。

### T-GRPO vs 对照组

| Pair | Avg diff | 主要上涨项 | 主要下降项 |
|---|---:|---|---|
| `tgrpo-van2` - `ema-v2` | -0.23 | LongVideoReason +0.50, VideoMME Spatial Perception +9.26, TempCompass direction +1.07, VSIBench object_counting +4.88 | VideoMathQA -1.19, VideoMMMU open -5.00, VideoMME Temporal Perception -5.45, VSIBench route_planning -5.15 |
| `tgrpo-van` - `van-v2` | -1.19 | MVBench +0.25, VSIBench route_planning +1.03, VideoMMMU open +5.00 | LongVideoReason -2.70, VideoMME Temporal Perception -7.27, VSIBench object_counting -5.43, VideoMME OCR -5.76 |

细分类别里有局部收益，但 temporal 相关项并没有整体变好。特别是 VideoMME 的 `Temporal Perception` 两个 T-GRPO 都下降，这和预期相反，值得优先排查 reward 设计、shuffle 样本构造和答案抽取是否对齐。

## 下一步建议

1. SFT 主线优先看 `v9-10k-3ep`，它是当前最强 SFT，也是最值得作为下一阶段候选 init 的 SFT。
2. `v5` 暂不作为高分辨率方向的正向证据；如果继续分辨率实验，应优先测视频侧 `video_max_pixels` / `video_maxlen`，而不是只提高 `image_max_pixels`。
3. `v8` 和 `v10` 暂时不要继续扩展同方向，先做数据/格式排查。
4. 如果继续 SFT 消融，优先做 checkpoint selection，而不是再扫 scheduler、weight decay、LoRA、batch/GA。
5. VideoMathQA 仍是短板，但 `v9` 在该项超过所有 RL run，值得单独看它的正确样例和输出格式。
