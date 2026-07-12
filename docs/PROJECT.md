# Project Understanding: TimeThinker Codebase with Video-R1 Data

## 1. 当前项目状态

这个仓库主体是 TimeThinker 训练与评测代码，内部集成了三套主要组件：

- `LLaMA-Factory`：用于 SFT cold start。
- `EasyR1`：用于 GRPO / EMA-GRPO 类 RL 训练。
- `Evaluation` / `VLMEvalKit`：用于 TimeThinker 风格的评测。

当前本地下载的数据并不是 TimeThinker README 中声明的 `TimeThinker-train-data`，而是 Video-R1 数据集。

本地 `data/README.md` 明确说明该数据来自：

> Video-R1: Reinforcing Video Reasoning in MLLMs

当前主要数据文件是：

- `data/Video-R1-COT-165k.json`
  - 165,575 条样本。
  - 用于 SFT cold start。
  - 比 RL 数据多 `process` 字段，也就是 `<think>...</think>` 推理过程。
- `data/Video-R1-260k.json`
  - 263,071 条样本。
  - 用于 RL training。
  - 保留问题、答案、媒体路径、任务类型和数据来源。

当前媒体目录包括：

- 视频类：`CLEVRER`, `LLaVA-Video-178K`, `NeXT-QA`, `PerceptionTest`, `STAR`
- 图像类：`Chart`, `General`, `Knowledge`, `Math`, `OCR`, `Spatial`

## 2. 和 TimeThinker README 的差异

TimeThinker 原始 README 期望的是：

- `EasyR1/data/timethinker_rl_train.json`
- `timethinker_sft_image.json`
- `timethinker_sft_video.json`
- 以及 `_unsampled` 版本

同时 TimeThinker README 描述的训练语料是 `TimeThinker-600k` 和 `TimeThinker-SFT-340k`，任务覆盖：

- rule-based QA
- open-ended QA
- captioning
- spatial grounding
- temporal grounding
- spatio-temporal grounding
- tracking
- segmentation

但当前 Video-R1 数据实际覆盖的 `problem_type` 主要是：

- `multiple choice`
- `numerical`
- `free-form`
- `OCR`
- `regression`

因此，Video-R1 可以支撑图像/视频 QA reasoning 训练，但不能完整替代 TimeThinker-600k。特别是 grounding、tracking、segmentation 这些结构化视觉定位任务，当前数据基本不具备对应标注。

当前仓库已经完成本地格式适配，实际训练入口使用的是：

- SFT image：`LLaMA-Factory/data/timethinker_sft_image.json`
- SFT video：`LLaMA-Factory/data/timethinker_sft_video.json`
- RL train：`EasyR1/data/timethinker_rl_train_split.json`
- RL val：`EasyR1/data/timethinker_rl_val_512.json`

## 3. 当前训练目标的合理定位

使用 Video-R1 数据训练 TimeThinker 代码，合理目标应该是：

> 基于 Qwen3-VL-Instruct，训练一个具备图像/视频问答、视觉数学、OCR、图表理解、空间估计和 CoT reasoning 格式能力的多模态推理模型。

它更接近：

```text
Qwen3-VL-Instruct
+ Video-R1 SFT CoT 数据
+ Video-R1 RL 可验证答案数据
+ TimeThinker / EasyR1 训练框架
= Video-R1-style multimodal reasoning model
```

而不是完整复现：

```text
TimeThinker-4B all-in-one image/video reasoning model
```

预期能增强的能力：

- 视频多选推理。
- 图像问答。
- 图像数学、几何、数值推理。
- 图表、科学图、文档类理解。
- OCR 和公式/手写内容识别。
- 简单空间关系和距离估计。
- `<think>...</think><answer>...</answer>` 格式遵循。

预期较弱或不会显著获得的能力：

- temporal grounding。
- spatial grounding。
- spatio-temporal grounding。
- object tracking。
- segmentation prompt generation。
- RefCOCO / GOT10K / ReasonVOS / MeViS 一类任务。

## 4. Video-R1 为什么这样选数据

Video-R1 的核心问题是：视频推理数据稀缺，并且直接对视频模型做 RL 容易不稳定。

因此它没有只使用视频数据，而是构造了 image-video mixed reasoning data：

- 视频样本提供动态场景、时间顺序、事件变化和视频理解能力。
- 图像样本提供更高密度、更稳定的推理监督，例如数学、图表、OCR、知识和空间关系。
- 多数样本采用可验证答案，例如多选题和数值题，以便 RL 阶段可以用 rule-based reward。
- 少量开放题、OCR、回归题用于补充真实任务多样性。

本地 `Video-R1-260k.json` 统计为：

```text
total: 263071
image: 146823
video: 116248
```

题型分布为：

```text
multiple choice: 168769
free-form:         38722
numerical:         34354
OCR:               15886
regression:         5340
```

这说明它的配比策略是：

> 以可规则判分的 QA / numerical 任务为主体，用视频数据提供时序能力，用图像数据补足通用视觉推理能力。

## 5. Video-R1 数据如何构造成最终样式

Video-R1 最终形成两个阶段的数据。

### 5.1 RL 数据：`Video-R1-260k.json`

RL 文件的样本形态大致是：

```json
{
  "problem_id": 2,
  "problem": "What appears on the screen in Russian during the missile's ascent?",
  "data_type": "video",
  "problem_type": "multiple choice",
  "options": [
    "A. A YouTube subscription notification",
    "B. A military command",
    "C. A warning message",
    "D. A weather update"
  ],
  "solution": "<answer>A</answer>",
  "path": "./LLaVA-Video-178K/...",
  "data_source": "LLaVA-Video-178K/30_60_s_youtube_v0_1"
}
```

这个文件服务于 RL。模型生成回答后，`EasyR1/verl/reward_function/timethinker_reward.py` 根据 `problem_type` 选择对应 reward：

- `multiple choice`：答案选项匹配。
- `numerical`：数值匹配。
- `OCR`：WER 相关得分。
- `regression`：相对误差得分。
- `open-ended`：ROUGE 或外部 reward model。

### 5.2 SFT 数据：`Video-R1-COT-165k.json`

SFT 文件比 RL 文件多一个 `process` 字段：

```json
{
  "problem": "...",
  "process": "<think>...</think>",
  "solution": "<answer>...</answer>",
  "path": "...",
  "data_type": "image",
  "problem_type": "numerical",
  "data_source": "Multimath-300k"
}
```

它的作用是 cold start：

- 先教模型按 `<think>...</think><answer>...</answer>` 输出。
- 先教模型遵守不同 `problem_type` 的答案格式。
- 先让模型具备基本视觉推理轨迹。
- 再进入 RL，让 reward 优化答案正确性。

## 6. 数据适配策略

当前采用：

> 把 Video-R1 数据改造成 TimeThinker 训练代码期望的格式，而不是大改训练脚本。

原因：

- TimeThinker 的 SFT 和 RL 脚本已经围绕固定数据接口写好。
- `LLaMA-Factory` 期望 `timethinker_sft_image` 和 `timethinker_sft_video` 两个注册数据集。
- `scripts/train/run_rl.sh` 通过 `config/rl/qwen3_rl.yaml` 读取 `EasyR1/data/timethinker_rl_train_split.json` 和 `EasyR1/data/timethinker_rl_val_512.json`。
- reward function 已经支持 Video-R1 当前的 `problem_type`。
- 改数据是一次性适配；改训练脚本容易把路径、字段、reward、prompt builder 的兼容问题扩散到多个地方。

当前产物：

- `LLaMA-Factory/data/timethinker_sft_image.json`
- `LLaMA-Factory/data/timethinker_sft_video.json`
- `EasyR1/data/timethinker_rl_train.json`：完整转换文件。
- `EasyR1/data/timethinker_rl_train_split.json`：当前训练使用的 train split。
- `EasyR1/data/timethinker_rl_val_512.json`：当前验证集。

转换时应保留：

- `problem`
- `data_type`
- `problem_type`
- `options`
- `solution`
- `process`，仅 SFT 需要
- `path`
- `data_source`

同时需要确保媒体路径可访问。当前 zip 包未完全解压时，JSON 中的 `path` 不能直接被常规 loader 读取。

## 7. 为什么仍然需要高质量 CoT

即使 Qwen3-VL 已经有 Thinking 版本，高质量 CoT 仍然有必要。

核心原因是：

> 模型具备通用思考能力，不等于它会按照当前任务、当前格式和当前 reward function 思考。

CoT cold start 解决的是训练对齐问题：

- 让模型学会在视觉任务中组织推理过程。
- 让模型学会输出严格的 `<think>...</think><answer>...</answer>`。
- 让模型学会不同任务的答案 schema，例如多选只输出选项、数值题只输出数字、OCR 只输出文本。
- 降低 RL 初期全是零 reward 的概率。
- 给小模型蒸馏强模型的视觉推理轨迹。

没有 SFT cold start，RL 容易先被格式问题拖住，而不是优化真正的答案正确性。

## 8. 为什么基座选 Qwen3-VL-Instruct 而不是 Thinking

当前配置中 SFT 基座是：

```yaml
model_name_or_path: Qwen/Qwen3-VL-4B-Instruct
```

选择 Instruct 而不是 Thinking，主要是出于可控性和归因考虑。

### 8.1 Instruct 更干净

Thinking 模型已经经过额外 reasoning post-training，带有自己的思考模板、长度偏好和输出习惯。继续用 Video-R1 / TimeThinker CoT 训练，可能是在改造一个已经有强偏好的模型。

Instruct 模型更适合作为可塑性较强的起点。

### 8.2 输出格式更可控

本项目的 reward 明确依赖：

```text
<think>...</think><answer>...</answer>
```

以及不同 `problem_type` 的答案格式。Thinking 模型未必天然匹配这个协议，可能会产生过长推理、额外说明或格式漂移。

### 8.3 RL 更稳定

RL 阶段需要大量可判分输出。Instruct + SFT cold start 的路线可以先把输出格式压稳，再用 reward 优化答案正确性。

如果直接从 Thinking 模型开始，模型可能已经倾向长思考，导致 token 成本更高、答案边界更模糊、reward 噪声更大。

### 8.4 实验归因更清楚

如果直接使用官方 Thinking 模型，很难区分最终提升来自：

- 官方 Thinking 模型已有能力；
- Video-R1 数据；
- SFT CoT；
- RL 训练。

使用 Instruct 作为基座，更容易证明当前数据构造和训练流程本身的效果。

一句话总结：

> Thinking 模型解决“模型是否已有通用推理能力”；高质量 CoT 解决“模型是否按当前任务格式和奖励协议推理”；Instruct 基座解决“训练是否可控、干净、容易归因”。

## 9. 后续实施建议

当前基础链路已经跑通，后续推进重点是：

1. 每次新实验固定 config、模型目录和 SwanLab run name，避免命名漂移。
2. SFT 继续按 `docs/archive/sft_ablation.md` 做单变量消融。
3. RL 继续按 `docs/archive/rl_ablation.md` 比较 GRPO、EMA-GRPO、T-GRPO、online filtering 和 KL。
4. 评测先用 `MAX_SAMPLES` 快速验证，再对候选模型跑完整 benchmark。
5. 对慢 benchmark 使用 frame cache，并记录 `_summary.md` 中的耗时和 cache 命中。
6. bad case 统一记录到 `docs/archive/bad_case.md` 的模板中。

当前最重要的边界是：

> Video-R1 适合作为 TimeThinker 代码里的 QA reasoning 训练数据；不要期待它单独训练出完整 TimeThinker 的 grounding、tracking 和 segmentation 能力。
