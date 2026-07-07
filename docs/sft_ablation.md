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

## 后续建议

优先做少量清晰实验，不要继续叠变量：

1. 补齐 `SFT-v7-10k` 的完整 benchmark，和 `SFT-v6-10k`、`SFT-v3-10000-1ep` 对比。
2. 跑 `SFT-v9-10k-3ep`，确认 10k 数据是否还能从更多 epoch 受益。
3. 跑 `SFT-v10-50k`，确认扩大数据覆盖面是否比重复 10k 更有效。
4. 如果继续跑高分辨率实验，以 v5 为准，保持 LR/epoch 不变，只改 `image_max_pixels`，否则难以归因。
5. 如果验证 video-only，建议和 mixed 数据保持相近 step 数，而不只是比较 `max_samples`。
6. 每次训练前把实际使用的 YAML 复制到模型输出目录，例如 `train_config.yaml`，避免后续只能从 SwanLab 或当前配置文件反推。

## 结果速记

- v2：eval loss `0.708`
- v3：eval loss `0.703`
- v4：eval loss `0.733`
- v6：eval loss `0.702`，六项视频评测平均 `55.88%`
- v7：eval loss `0.701`
- `SFT-v3-10000-1ep`：六项视频评测平均 `56.22%`

注意：不同版本的训练 runtime、samples/sec、steps/sec 不宜直接横向比较。v1 有 resume 口径问题，v2/v3/v4/v6/v7 又包含 eval 开销。
