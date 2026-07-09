# SFT 消融实验记录

本文档只记录 SFT 系列实验的关键变化、观察结论和下一步。共同配置不重复展开；每个版本只写它相对前一个主要参照变了什么。

## 共同基线

除非单独说明，SFT 实验默认使用：

- Base model：`Qwen/Qwen3-VL-4B-Instruct`
- Dataset：`timethinker_sft_image, timethinker_sft_video`
- Template：`qwen3_vl`
- Training：full finetune, ZeRO-2, bf16, flash attention 2
- Data scale：`max_samples: 10000`
- Vision config：`image_max_pixels: 100352`, `video_max_pixels: 100352`, `video_maxlen: 16`, `video_fps: 2`
- Default freeze：vision tower 冻结，multi-modal projector 可训练
- Scheduler：cosine

当前配置文件已移动到：

```text
config/sft/
```

## 版本变化

### SFT v1：全量 cold start 参照

- 输出目录：`models/TimeThinker-4B-SFT`
- 主要变化：使用接近全量 SFT 数据，不设 10k 采样上限。
- projector：冻结。
- 备注：从 `checkpoint-4000` resume，HF 的 `train_loss` 口径不太可靠，实际后段 logged loss 大约在 `0.6-0.7`。

### SFT v2：10k 小样本 + projector 解冻

- 输出目录：`models/TimeThinker-4B-SFT-v2-10000`
- 相比 v1：
  - `max_samples` 改为 `10000`。
  - 增加 `val_size: 0.1`。
  - multi-modal projector 改为可训练。
  - `learning_rate` 提到 `3e-6`。
- 观察：作为 10k 小样本基线可用，eval loss 约 `0.708`。

### SFT v3：更高 LR + 2 epochs

- 输出目录：`models/TimeThinker-4B-SFT-v3-10000`
- 相比 v2：
  - `learning_rate: 5e-6`
  - `num_train_epochs: 2`
  - `warmup_ratio: 0.03`
- 观察：eval loss 约 `0.703`，略好于 v2；但 2 epochs 和更高 LR 同时变化，不能单独归因。

### SFT v3-10000-1ep：一轮训练参照

- 输出目录：`models/TimeThinker-4B-SFT-v3-10000-1ep`
- 主要用途：作为 v3 训练早期 / 约一轮训练的评测参照。
- 观察：六项视频评测平均约 `56.22%`，是当前 SFT 里较强的参照之一。
- 注意：旧名 `TimeThinker-4B-SFT-v3-5k` 已统一改为 `TimeThinker-4B-SFT-v3-10000-1ep`，对应评测结果目录也已同步。

### SFT v4：低 LR 参照

- 输出目录：`models/TimeThinker-4B-SFT-v4-10000`
- 相比 v3：
  - `learning_rate` 降到 `1e-6`
  - epoch 回到 `1`
- 观察：
  - eval loss 约 `0.733`，弱于 v2/v3。
  - loss 大约在 step `100-150` 已经进入平台，当时 LR 仍接近 `9e-7`，所以不是 cosine 尾部衰减导致。
  - `1e-6` 对当前可训练参数集合偏保守，可作为 under-training 参照。

### SFT v5：提高图像分辨率

- 配置：`config/sft/qwen3_sft-v5.yaml`
- 输出目录：`models/TimeThinker-4B-SFT-v5-10k`
- 相比 v4/v6 类 10k 参照：
  - `image_max_pixels: 200704`
  - `video_max_pixels` 保持 `100352`
  - `learning_rate: 5e-6`
- 状态：当前模型目录没有 `trainer_state.json` / `train_results.json`，未形成可分析的有效训练结果。

### SFT v6：解冻 vision tower，只训练视觉塔侧适配

- 配置：`config/sft/qwen3_sft-v6.yaml`
- 输出目录：`models/TimeThinker-4B-SFT-v6-10k`
- 相比默认 10k 参照：
  - `freeze_vision_tower: false`
  - `freeze_multi_modal_projector: true`
  - 添加 `use_reentrant_gc: false`，避免 ZeRO-2 + reentrant checkpointing 的重复 reduce 问题。
  - `learning_rate: 5e-6`
- 观察：
  - eval loss 约 `0.702`，和 v3 接近。
  - 六项视频评测平均约 `55.88%`，略低于 `SFT-v3-10000-1ep`。

### SFT v7：vision tower + projector 都解冻，LR 提高

- 配置：`config/sft/qwen3_sft-v7.yaml`
- 输出目录：`models/TimeThinker-4B-SFT-v7-10k`
- 相比 v6：
  - `freeze_multi_modal_projector: false`
  - `learning_rate: 1e-5`
- 观察：
  - eval loss 约 `0.701`，略好于 v6。
  - 目前还需要补完整 benchmark 后再判断是否真的优于 v6。

### SFT v8：video-only 数据

- 配置：`config/sft/qwen3_sft-v8.yaml`
- 输出目录：`models/TimeThinker-4B-SFT-v8-10k`
- 相比默认 mixed SFT：
  - `dataset: timethinker_sft_video`
  - `max_samples: 20000`
  - `learning_rate: 5e-6`
  - `tokenized_path` 改为 `LLaMA-Factory/cache/timethinker_sft_10k_video`
- 目的：验证只用视频 SFT 是否能更直接提升视频 benchmark。
- 状态：当前未看到对应模型结果目录。

### SFT v9：10k mixed 训练到 3 epochs

- 配置：`config/sft/qwen3_sft-v9.yaml`
- 输出目录：`models/TimeThinker-4B-SFT-v9-10k-3ep`
- 相比 v3：
  - `num_train_epochs: 3.0`
  - 其它核心设置保持 `10k mixed + lr=5e-6 + vision frozen + projector trainable`
- 目的：验证 10k mixed 数据继续多训一个 epoch 是否还有收益，还是开始过拟合 SFT 分布。
- 状态：待跑。

### SFT v10：50k mixed 数据规模

- 配置：`config/sft/qwen3_sft-v10.yaml`
- 输出目录：`models/TimeThinker-4B-SFT-v10-50k`
- 相比默认 10k mixed 参照：
  - `max_samples: 50000`
  - `tokenized_path` 改为 `LLaMA-Factory/cache/timethinker_sft_50k`
  - `learning_rate: 5e-6`
  - `num_train_epochs: 1.0`
- 目的：验证扩大 SFT 数据覆盖面是否比在 10k 上重复训练更有效。
- 状态：待跑。

## 当前结论

- `1e-6` 学习率偏低，v4 很早进入 loss 平台，效果弱于 v2/v3。
- `3e-6 -> 5e-6` 在 10k mixed SFT 上是合理方向，但 v3 同时改了 epoch，后续要避免多个变量混在一起。
- 解冻 vision tower 后，v6 的 eval loss 不差，但 benchmark 没明显超过 `SFT-v3-10000-1ep`，说明“训练 loss/val loss 更低”不一定等于下游视频评测更好。
- v7 在 loss 上略好于 v6，但需要完整评测确认。
- 高分辨率（v5）和 video-only（v8）目前还缺有效训练/评测结果，不能下结论。

## 待验证计划

当前目标是尽快选出一个可作为后续训练起点的 SFT，而不是把 SFT 超参空间完整扫完。下面按信息增益排序，优先保留会改变模型选择的实验；训练系统类超参和必然会在 RL 阶段验证的内容先不单独消融。

### P0. 先补齐已有/已配置实验

| 实验 | 目的 | 当前动作 |
|---|---|---|
| `SFT-v7-10k` 完整 benchmark | 判断 vision tower + projector 都解冻是否真的优于 v6/v3 | 只补评测，不新增训练变量 |
| `SFT-v9-10k-3ep` | 判断 10k mixed 多训是否还有收益 | 可跑 |
| `SFT-v10-50k` | 判断扩大数据覆盖是否比重复 10k 更有效 | 可跑 |

优先级判断：

- v9 和 v10 是目前最直接的方向：一个回答“多训是否有用”，一个回答“多数据是否有用”。
- v7 已经有 loss 结果，只差 benchmark；补评测成本低，应该先完成。
- v5 目前缺有效训练结果，且只改 `image_max_pixels`，对视频 benchmark 的直接价值不如 v9/v10。

### P1. 只保留少量数据与视频侧实验

不要完整扫数据配比和视频采样网格。先做最可能改变结论的少数点。

| 实验 | 目的 | 是否保留 |
|---|---|---|
| `video-only` / v8 | 看纯视频数据是否显著提升视频 benchmark | 保留一个点 |
| `video_maxlen=32` | 看更多帧是否提升长视频/时序任务 | 保留一个点 |
| `video_max_pixels=200704` | 看视频分辨率是否提升细节类任务 | 保留一个点 |
| `video-70` / `video-90` | 精细数据配比 | 暂不做 |
| `temporal-heavy` | 专门服务 T-GRPO 的采样权重 | 暂不做，等 RL 误差分析后再说 |
| `math-ocr-balanced` | 针对 VideoMathQA/OCR 短板 | 暂不做，VideoMathQA 当前整体都低，先别为单项开分支 |
| low-res / fewer-frames / low-fps / high-fps | 成本或采样网格 | 暂不做 |

### P1. CoT 长度先统计，不急着训练新版本

SFT/RL 的输出长度、截断率和 answer extraction 很关键，但现在先做数据统计和结果归因，不立刻开 `short-cot` / `answer-focused` / `answer-only` 三个训练分支。

先统计：

- target answer token 长度分布。
- `<think>` token 长度分布。
- `<answer>` token 长度分布。
- 被 `cutoff_len` 截断的比例。
- benchmark 结果里的 `avg_output_tokens`、`truncation_rate`、`answer_extract_rate`、`invalid_answer_rate`。

只有当统计显示某个候选模型明显有输出过长、截断或抽取失败问题时，再做一个 `answer-focused` 版本。`answer-only` 暂不做，风险是牺牲 reasoning 信号，而且和当前 TimeThinker/RL 目标不完全一致。

### P2. 只保留 checkpoint selection，不扩训练系统超参

不要只评估 final checkpoint。每个重要 SFT run 尽量保存并评估：

```text
checkpoint-250
checkpoint-500
final
```

重点看：

- final 是否已经过拟合。
- eval loss 最低点是否等于 benchmark 最好点。
- 1ep / 2ep / 3ep 哪个 checkpoint 更适合作为后续训练起点。

### 暂不单独做的实验

这些先从 SFT 消融里移出，除非后面出现明确故障信号。

| 类别 | 暂不做原因 |
|---|---|
| Scheduler / Warmup | 当前主要瓶颈不是训练不稳定；信息增益低，容易消耗大量 run。 |
| Weight Decay / Gradient Norm | 这类优化器细节一般收益小，且很难用少量 benchmark 稳定归因。 |
| Batch size / Gradient Accumulation | 只在 OOM、吞吐太差或 loss 明显抖动时调整；不作为质量消融。 |
| `cutoff_len=8192/32768` 训练消融 | 先统计截断率；没有明显截断问题就不跑。 |
| Full Finetune vs LoRA | 当前主线是 full finetune，LoRA 对最终路线帮助有限，先不分叉。 |
| SFT -> RL smoke 作为 SFT 消融项 | RL 本来就是下一阶段要做，不需要在 SFT 文档里单列一组额外实验。 |

## 后续建议

优先做少量清晰实验，不要继续叠变量：

1. 补齐 `SFT-v7-10k` 的完整 benchmark，和 `SFT-v6-10k`、`SFT-v3-10000-1ep` 对比。
2. 跑 `SFT-v9-10k-3ep`，确认 10k 数据是否还能从更多 epoch 受益。
3. 跑 `SFT-v10-50k`，确认扩大数据覆盖面是否比重复 10k 更有效。
4. 如需补视频侧视觉配置，只跑 `video_maxlen=32` 和 `video_max_pixels=200704` 两个点，先不扫 fps/resolution/frame 网格。
5. 如果验证 video-only，建议和 mixed 数据保持相近 step 数，而不只是比较 `max_samples`。
6. 统计 SFT CoT 长度、target 长度和 `cutoff_len` 截断率，再决定是否做 short-CoT / answer-focused。
7. 暂不单独做 scheduler/warmup、weight decay/grad norm、LoRA、batch/GA、cutoff_len 训练消融。
8. 每次训练前把实际使用的 YAML 复制到模型输出目录，例如 `train_config.yaml`，避免后续只能从 SwanLab 或当前配置文件反推。

## 结果速记

- v2：eval loss `0.708`
- v3：eval loss `0.703`
- v4：eval loss `0.733`
- v6：eval loss `0.702`，六项视频评测平均 `55.88%`
- v7：eval loss `0.701`
- `SFT-v3-10000-1ep`：六项视频评测平均 `56.22%`

注意：不同版本的训练 runtime、samples/sec、steps/sec 不宜直接横向比较。v1 有 resume 口径问题，v2/v3/v4/v6/v7 又包含 eval 开销。
