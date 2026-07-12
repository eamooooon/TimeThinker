# eval-v1-return 评测汇总

记录 `Evaluation/results-v1-rerun/*/frames16` 的复测汇总结果。本文档把这套结果记为 `eval-v1-return`，用于和主评测 `Evaluation/results-v4/*/frames16` 做 prompt/return 口径对照。

## 口径说明

- `eval-v1-return` 来源：从 `Evaluation/results-v1-rerun/<model>/frames16/eval_*.json` 解析并汇总而来。
- 单模型汇总文件：`Evaluation/results-v1-rerun/<model>/frames16/_summary.json`。
- `Avg`：各 benchmark 的 `answer_acc` 简单未加权平均。
- `frames16`：所有结果均为 `MAX_FRAMES=16`。
- 这套结果不是主榜替代，而是用于观察不同 prompt/return 格式下模型是否稳定。
- 当前有 15 个完整 8 项模型，以及 1 个不完整模型 `TimeThinker-4B-SFT-v9-10k-3ep`。
- `TimeThinker-4B-RL-Zero-100-van` 来自 `Evaluation/results-v1-rerun/TimeThinker-4B-RL-Zero-100-van/frames16`；主 bench 对照时使用对应目录名 `TimeThinker-4B-RL-Zero-100-van-bs16`。
- `TimeThinker-4B-SFT-v9-10k-3ep` 只有 5 项有效结果：LongVideoReason、MVBench、VideoMME、VideoMMMU、VSIBench。MMVU / VideoMathQA 当前没有有效汇总，TempCompass JSON 截断，因此不能和完整 8 项模型直接比较。

## 总表

| Model | Avg | N | Samples | LongVideoReason | MMVU | MVBench | TempCompass | VideoMathQA | VideoMME | VideoMMMU | VSIBench |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `TimeThinker-4B-RL-Zero-100-van` | 55.68 | 8 | 22315 | 69.10 | 65.76 | 62.73 | 72.20 | 20.95 | 54.89 | 52.00 | 47.78 |
| `TimeThinker-4B-SFT-v3-10000` | 55.35 | 8 | 22315 | 69.30 | 65.12 | 62.58 | 68.41 | 23.81 | 55.37 | 52.67 | 45.53 |
| `TimeThinker-4B-SFT` | 55.15 | 8 | 22315 | 71.00 | 63.20 | 62.80 | 67.52 | 24.52 | 54.63 | 52.44 | 45.07 |
| `TimeThinker-4B-RL-Zero-100-ema-v2` | 55.15 | 8 | 22315 | 68.80 | 65.60 | 61.22 | 72.49 | 21.90 | 54.52 | 51.22 | 45.40 |
| `TimeThinker-4B-SFT-v10-50k` | 54.74 | 8 | 22315 | 68.60 | 63.20 | 63.40 | 68.85 | 22.86 | 55.33 | 52.78 | 42.94 |
| `TimeThinker-4B-SFT-v6-10k` | 54.47 | 8 | 22315 | 70.20 | 64.00 | 60.55 | 68.14 | 24.76 | 53.41 | 51.78 | 42.89 |
| `TimeThinker-4B-SFT-v8-10k` | 54.41 | 8 | 22315 | 70.20 | 64.80 | 61.62 | 68.37 | 22.14 | 54.26 | 51.78 | 42.14 |
| `TimeThinker-4B-RL-Zero-100-tgrpo-van` | 54.40 | 8 | 22315 | 66.30 | 66.24 | 61.35 | 72.31 | 19.76 | 52.30 | 52.11 | 44.81 |
| `TimeThinker-4B-SFT-v5-10k` | 54.24 | 8 | 22315 | 69.10 | 63.68 | 62.12 | 68.33 | 20.24 | 53.52 | 53.22 | 43.69 |
| `TimeThinker-4B-SFT-v3-10000-1ep` | 54.09 | 8 | 22315 | 70.10 | 63.84 | 61.30 | 68.10 | 22.86 | 52.56 | 52.44 | 41.53 |
| `TimeThinker-4B-SFT-v2-10000` | 53.81 | 8 | 22315 | 70.50 | 58.08 | 60.65 | 67.59 | 23.81 | 53.70 | 52.22 | 43.96 |
| `TimeThinker-4B-RL-Zero-100-van-v2` | 53.72 | 8 | 22315 | 66.00 | 64.16 | 61.02 | 72.56 | 18.10 | 52.04 | 51.00 | 44.89 |
| `TimeThinker-4B-SFT-v4-10000` | 53.19 | 8 | 22315 | 69.10 | 62.88 | 60.17 | 67.20 | 21.67 | 52.07 | 49.56 | 42.90 |
| `TimeThinker-4B-SFT-v7-10k` | 53.16 | 8 | 22315 | 67.90 | 61.92 | 60.82 | 68.94 | 20.71 | 53.00 | 50.00 | 42.00 |
| `TimeThinker-4B-SFT-v9-10k-3ep` | 52.75 | 5 | 9268 | 71.40 | - | 61.98 | - | - | 55.93 | 28.65 | 45.78 |
| `Qwen3-VL-4B-Instruct` | 52.12 | 8 | 22315 | 64.50 | 62.08 | 59.10 | 71.21 | 16.19 | 49.70 | 49.33 | 44.83 |

## 和主 bench 整体对照

下表只比较同时存在完整 8 项结果的模型。`Diff = eval-v1-return Avg - 主 bench Avg`。

| Model | eval-v1-return Avg | 主 bench Avg | Diff |
|---|---:|---:|---:|
| `Qwen3-VL-4B-Instruct` | 52.12 | 35.94 | +16.18 |
| `TimeThinker-4B-SFT-v8-10k` | 54.41 | 40.70 | +13.71 |
| `TimeThinker-4B-SFT-v10-50k` | 54.74 | 43.67 | +11.07 |
| `TimeThinker-4B-SFT-v4-10000` | 53.19 | 42.69 | +10.50 |
| `TimeThinker-4B-SFT-v2-10000` | 53.81 | 44.72 | +9.10 |
| `TimeThinker-4B-SFT-v5-10k` | 54.24 | 45.21 | +9.03 |
| `TimeThinker-4B-SFT-v3-10000` | 55.35 | 50.08 | +5.27 |
| `TimeThinker-4B-SFT-v3-10000-1ep` | 54.09 | 48.91 | +5.18 |
| `TimeThinker-4B-SFT-v6-10k` | 54.47 | 49.95 | +4.52 |
| `TimeThinker-4B-SFT-v7-10k` | 53.16 | 50.15 | +3.01 |
| `TimeThinker-4B-RL-Zero-100-van` / `van-bs16` | 55.68 | 54.83 | +0.84 |
| `TimeThinker-4B-RL-Zero-100-tgrpo-van` | 54.40 | 55.38 | -0.98 |
| `TimeThinker-4B-RL-Zero-100-ema-v2` | 55.15 | 56.31 | -1.16 |
| `TimeThinker-4B-RL-Zero-100-van-v2` | 53.72 | 56.57 | -2.85 |

这个差异说明 `eval-v1-return` 对 SFT 和基座模型更友好，尤其是原本缺少严格 `<answer>` 格式或在主 bench strict-ish prompt 下容易格式/抽取失败的模型。RL 模型之间并不完全一致：`van` 这次在 `eval-v1-return` 下略高于主 bench，`tgrpo-van`、`ema-v2`、`van-v2` 则仍然是主 bench 更高，说明不同 RL checkpoint 对 prompt/return 口径的敏感性不一样。

## ema-v2 对照主 bench

`TimeThinker-4B-RL-Zero-100-ema-v2` 在主 bench 和 `eval-v1-return` 上总体接近，但主 bench 略高：

- 主 bench Avg：56.31
- `eval-v1-return` Avg：55.15
- 差值：-1.16

逐项对比如下：

| Benchmark | eval-v1-return | bench | Diff | v1-return 对/bench 错 | bench 对/v1-return 错 |
|---|---:|---:|---:|---:|---:|
| LongVideoReason | 68.80 | 69.00 | -0.20 | 59 | 61 |
| MMVU | 65.60 | 67.36 | -1.76 | 35 | 46 |
| MVBench | 61.22 | 63.18 | -1.95 | 279 | 357 |
| TempCompass | 72.49 | 72.37 | +0.12 | 351 | 342 |
| VideoMathQA | 21.90 | 23.81 | -1.90 | 35 | 43 |
| VideoMME | 54.52 | 55.89 | -1.37 | 157 | 194 |
| VideoMMMU | 51.22 | 52.44 | -1.22 | 71 | 82 |
| VSIBench | 45.40 | 46.42 | -1.02 | 346 | 380 |

合计：

- `eval-v1-return` 对、主 bench 错：1333 个样本。
- 主 bench 对、`eval-v1-return` 错：1505 个样本。

这说明两套 prompt/return 口径不是简单的强弱关系，而是会改变样本级推理轨迹。主 bench 整体略高，但 `eval-v1-return` 在不少具体样本上会答对主 bench 答错的题。

## 为什么会出现 v1-return 对但 bench 错

主要原因是 prompt 约束改变了模型的生成方式。

主 bench 更强调严格的 `<think>...</think><answer>...</answer>` 结构，模型输出更规整，整体解析更稳定。但它有时会让模型更快给出短判断，导致局部样本上出现漏看、漏数或选项映射错误。

`eval-v1-return` 的约束相对弱一些，模型更容易自然描述视觉内容。这样在一些需要细看时间变化、物体计数、读图数值、选项映射的题上，反而可能答对。

真实例子：

- MVBench `problem_id=1036`：题目问开头字母顺序，GT 是 `A. bpx`。主 bench 看出了 `b, p, x`，但错误映射成 `C`；`eval-v1-return` 写出 `bpx` 并选择 `A`。
- TempCompass `problem_id=5858`：题目问门是否一直关闭，GT 是 `B. no`。主 bench 判断 0:00-0:06 都关闭；`eval-v1-return` 注意到 0:04 开始打开、0:05 完全打开。
- VideoMMMU `problem_id=801`：题目问正视图三个宽度，GT 是 `C. 25,25,25`。主 bench 把第一个数读成 `20`；`eval-v1-return` 读成三个 `25`。
- VSIBench `problem_id=98`：题目问房间里几张桌子，GT 是 `2`。主 bench 只看到一张主桌；`eval-v1-return` 区分了 desk 和 dining table。

## 当前结论

- `eval-v1-return` 不应替代主 bench 排名。它更像 prompt/return 口径敏感性诊断。
- 完整 8 项里，`TimeThinker-4B-RL-Zero-100-van` 是当前 `eval-v1-return` 最强模型，Avg 为 55.68。
- `eval-v1-return` 对 SFT / base 明显更友好；RL checkpoint 的方向不完全一致，其中 `van` 在 `eval-v1-return` 高出主 bench 0.84 点，而 `van-v2` 在主 bench 高出 `eval-v1-return` 2.85 点。
- `eval-v1-return` 更适合作为 prompt sensitivity 诊断：它能暴露哪些样本是 prompt 触发型正确，哪些样本是稳定能力。
- `ema-v2` 的样本级翻转很多，说明模型在时间变化、计数、读图和选项映射上仍不够 prompt-invariant。
- 如果后续要降低 prompt 差异带来的波动，可以考虑多模板评测、prompt ensemble，或者在 SFT/RL 中加入多 prompt 模板训练。

## 数据问题

`TimeThinker-4B-SFT-v9-10k-3ep` 当前不是完整 8 项结果：

- `eval_tempcompass.json` 解析失败：

```text
JSONDecodeError("Expecting ',' delimiter: line 78914 column 800 (char 8380587)")
```

- MMVU / VideoMathQA 当前没有有效汇总文件。
- 因此 `TimeThinker-4B-SFT-v9-10k-3ep` 在本表只有 5 项有效结果，不能和完整 8 项模型直接比较。
