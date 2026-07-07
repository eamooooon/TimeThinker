# TimeThinker Daily Record

记录训练、评测、数据和工程优化的日常推进。日期按 UTC 工作区时间记录。

## 2026-07-07

### 数据侧整理与对齐

- 梳理了当前项目实际使用的数据不是完整 TimeThinker-600k，而是基于 Video-R1 的图像/视频混合推理数据。
- 记录了两条主要数据链路：
  - SFT：`Video-R1-COT-165k.json` 转成 LLaMA-Factory ShareGPT 格式，拆成 `timethinker_sft_image` 和 `timethinker_sft_video`。
  - RL：`Video-R1-260k.json` 转成 EasyR1 可读格式，训练集为 `EasyR1/data/timethinker_rl_train_split.json`，验证集为 `EasyR1/data/timethinker_rl_val_512.json`。
- 梳理了数据转换脚本 `scripts/data/convert_data.py`：
  - 校验媒体文件是否存在。
  - 将 SFT 数据拆成 image/video 两份。
  - 将 RL 数据统一成 EasyR1 所需字段。
  - 将 `free-form` 归一到 `open-ended`，将 `ocr` 归一到 `OCR`。
  - 输出转换后样本数、模态分布、题型分布和 skipped reason。
- 明确了当前训练能力边界：
  - 数据主要覆盖视频理解、图像数学、图表、OCR、知识、空间推理和通用视觉 QA。
  - 不覆盖 dense grounding、tracking、segmentation 这类 box/mask/trajectory 能力。
  - 因此评测和 prompt 不应继续强化 grounding/tracking/segmentation 输出格式。
- 在数据文档中补充了 Video-R1 数据来源、bucket 配比、题型含义和 source 内部细分：
  - General Video
  - Math Image
  - Chart Image
  - OCR Image
  - Knowledge Image
  - Spatial Image
  - General Image
- 形成了两份数据说明文档：
  - `docs/data.md`
  - `docs/qa.md`

### RL 数据读取与视频处理优化

- 修改 `EasyR1/verl/utils/dataset.py`，让 EasyR1 可以直接读取本地 `.json` / `.jsonl` 文件，而不是只能走 HuggingFace `load_dataset` 逻辑。
- 本地 JSON 兼容以下结构：
  - 顶层 list
  - `{"train": [...]}`
  - `{"data": [...]}`
  - `{"instances": [...]}`
- 对本地 records 做字段补齐，避免不同样本字段不一致导致 Dataset 构建失败。
- 数据 prompt 端和 eval prompt 端统一为严格 `<think>...</think><answer>...</answer>`。
- 数据侧删除 grounding/tracking/segmentation 专用 prompt 模板，只保留当前训练真正使用的题型：
  - `multiple choice`
  - `numerical`
  - `OCR`
  - `open-ended`
  - `regression`
  - `math`
- 原始数据中的 `free-form` 在转换脚本里归一为 `open-ended`，避免 reward 和 prompt 路由分裂。
- 改进图像/视频占位符插入逻辑：
  - 如果 prompt 中没有 `<image>` / `<video>`，自动在文本前插入对应数量的媒体块。
  - 如果 prompt 中有占位符，则按占位符位置插入，并补齐剩余媒体。
- 增加视频读取 fallback：
  - 默认用 `qwen_vl_utils.fetch_video`。
  - 失败时 fallback 到 PyAV 解码。
  - 解决部分视频在 decord/torchvision 路径下读不了的问题。
- 修复 Qwen3-VL 视频输入打包：
  - 读取 `video_metadata`。
  - `processor(..., do_resize=False, video_metadata=...)`。
  - 在 rollout 侧传递 `mm_processor_kwargs`，避免 video token 和 visual features 数量不对齐。

### SFT 训练配置与消融

- 将 SFT 配置集中到 `config/sft/`：
  - `qwen3_sft.yaml`
  - `qwen3_sft-v5.yaml`
  - `qwen3_sft-v6.yaml`
  - `qwen3_sft-v7.yaml`
  - `qwen3_sft-v8.yaml`
  - `qwen3_sft-v9.yaml`
- 建立 SFT 消融记录：`docs/sft_ablation.md`。
- 当前 SFT 共同基线：
  - Base model：`Qwen/Qwen3-VL-4B-Instruct`
  - full finetune
  - ZeRO-2
  - bf16
  - flash attention 2
  - cutoff_len `16384`
  - video_maxlen `16`
  - video_fps `2`
  - max_samples `10000`
- 已整理的 SFT 变量：
  - 学习率：`1e-6` / `5e-6` / `1e-5`
  - 是否解冻 vision tower
  - 是否解冻 multi-modal projector
  - 图像分辨率：`image_max_pixels=100352` vs `200704`
  - mixed image+video vs video-only
  - `use_reentrant_gc: false` 用于规避 ZeRO-2 + reentrant checkpointing 的重复 reduce 问题。
- 当前观察：
  - `1e-6` 偏保守，v4 很早进入 loss 平台。
  - `5e-6` 是 10k mixed SFT 的较合理方向。
  - 解冻 vision tower 后 loss 不差，但 benchmark 不一定明显更好。
  - v7 loss 略好于 v6，但还需要完整 benchmark 判断。
  - high-res 和 video-only 还缺有效完整评测。

### SFT 训练脚本优化

- `scripts/train/run_sft.sh` 改为统一从 `config/sft/` 读取配置。
- 增加 resume 控制：
  - `RESUME_FROM_CHECKPOINT=auto`
  - `RESUME_FROM_CHECKPOINT=none`
  - `RESUME_FROM_CHECKPOINT=/path/to/checkpoint`
- 新增 `scripts/train/run_sft_list.sh`：
  - 支持一次串行跑多个 SFT 配置。
  - 每个 run 使用独立 `MASTER_PORT`，避免端口冲突。
  - 日志写入 `logs/train/`。
- 目的：减少手动改配置、手动换端口、手动记录日志带来的实验污染。

### RL 训练配置整理

- 将 RL 配置集中到 `config/rl/`：
  - `qwen3_rl.yaml`
  - `qwen3_rl_t.yaml`
- GRPO 基础配置：
  - rollout batch size `32`
  - rollout n `8`
  - max prompt length `16384`
  - max response length `768`
  - KL loss enabled，`kl_coef=4e-2`
  - online filtering enabled
  - filter key `accuracy`
  - filter range `[0.01, 0.99]`
  - save freq `50`
  - max steps `100`
- 新增/整理 RL 启动脚本：
  - `scripts/train/run_rl.sh`
  - `scripts/train/run_rl_t.sh`
  - `scripts/train/run_rl_list.sh`
- `run_rl_list.sh` 支持串行跑 GRPO 和 EMA-GRPO 两组实验，并分别写到不同 checkpoint 目录：
  - `models/TimeThinker-4B-RL-Zero-100-van-v2`
  - `models/TimeThinker-4B-RL-Zero-100-ema-v2`

### RL checkpoint 转 HuggingFace

- 确认 EasyR1 的 FSDP checkpoint 需要通过 `EasyR1/scripts/model_merger.py` 合并成 HuggingFace 格式后再用于评测。
- 常用命令：

```bash
.venv_rl/bin/python EasyR1/scripts/model_merger.py \
  --local_dir models/TimeThinker-4B-RL-Zero-100-van-v2/global_step_100/actor
```

- 合并后会生成：

```text
models/.../global_step_100/actor/huggingface
```

- 评测脚本使用该 `huggingface` 目录作为 `MODEL_PATH`。

### RL reward 简化与对齐

- 大幅简化 `EasyR1/verl/reward_function/timethinker_reward.py`：
  - 去掉 grounding/tracking/segmentation 相关 IoU、点匹配、结构奖励逻辑。
  - 保留当前训练数据真正需要的 QA reward。
- reward 输出统一包含：
  - `overall`
  - `format`
  - `accuracy`
- 格式奖励要求严格匹配 `<think>...</think><answer>...</answer>`。
- accuracy 按 `problem_type` 路由：
  - 多选：选项字母精确匹配。
  - 数值：数值解析。
  - OCR：文本/字符层面指标。
  - regression：相对误差类指标。
  - math：优先使用 math equivalence。
  - open-ended：保留外部 RM / ROUGE fallback 的接口。
- 这样 reward 和实际训练数据边界一致，减少未训练任务类型对 RL 的干扰。

### T-GRPO / Temporal Reward 实现

- 在 `EasyR1/verl/trainer/config.py` 增加 T-GRPO 和长度控制相关配置：
  - `temporal`
  - `shuffled_rollout_ratio`
  - `temporal_reward`
  - `temporal_compare_ratio`
  - `temporal_correct_threshold`
  - `len_control`
  - `len_reward`
  - `len_min`
  - `len_max`
- 在 `EasyR1/verl/trainer/ray_trainer.py` 实现 Video-R1 风格时序奖励：
  - 找出 batch 中的视频样本。
  - 对视频帧构造 shuffled 版本。
  - 正常视频 rollout 和 shuffled 视频 rollout 分别打分。
  - 如果正常顺序表现不弱于乱序视频，并且样本本身正确，则给 temporal bonus。
  - bonus 加到 response 最后一个有效 token 上。
- 同时预留长度控制奖励：
  - 正确样本如果 response length 落在 `[len_min, len_max]`，可加 `len_reward`。
- 额外记录 reward metrics：
  - `final_overall`
  - `temporal_bonus`
  - `temporal_applied`
  - `shuffled_accuracy`
  - `length_bonus`

### Rollout / vLLM 多模态兼容

- 修改 `EasyR1/verl/workers/rollout/vllm_rollout_spmd.py`：
  - 对多模态数据调用 `_process_multi_modal_data`。
  - 将 video 场景需要的 `mm_processor_kwargs` 注入 vLLM input。
  - 支持逐样本 fps / video metadata 传递。
- 修改 `EasyR1/verl/workers/sharding_manager/fsdp_vllm.py`：
  - 兼容不同 vLLM 版本中 tensor parallel group API 的差异。
  - `wake_up(tags=...)` 做签名检测，兼容带 tags 和不带 tags 的版本。
- 这些改动主要为了解决 Qwen3-VL + vLLM + FSDP rollout 时的视频输入和权重同步兼容问题。

### 评测 prompt 对齐

- 将评测端 prompt 对齐到训练期望的严格格式：
  - `<think>...</think>`
  - `<answer>...</answer>`
- 约束模型不能在 `<think>` 前或 `</answer>` 后输出额外内容。
- 删除评测 prompt 中和当前训练目标不一致的 grounding / tracking / segmentation 类模板，避免测评端诱导模型输出未训练能力。
- 保留底层 grounding/tracking metric 兼容逻辑，防止历史数据或特殊数据集读取时报错。

### 评测指标扩展

在 `Evaluation/Eval/eval_bench.py` 中新增/规范了以下指标：

- `answer_acc`
- `macro_avg/by_benchmark`
- `per_category_acc`
- `answer_extract_rate`
- `invalid_answer_rate`
- `avg_output_tokens`
- `truncation_rate`
- `bootstrap_ci`
- `format/has_think_rate`
- `format/strict_rate`

样本级结果新增字段：

- `answer_extracted`
- `invalid_answer`
- `output_tokens`
- `finish_reason`
- `stop_reason`
- `truncated`
- `has_think`
- `strict_format`
- `category`

说明：

- `answer_acc` 仍是主性能指标。
- `answer_extract_rate` / `invalid_answer_rate` 用于检查格式和答案抽取是否稳定。
- `avg_output_tokens` / `truncation_rate` 用于判断输出长度、截断和推理成本。
- `per_category_acc` 用于定位 benchmark 内部不同题型或能力维度的强弱。
- `bootstrap_ci` 只在最终写盘时计算，避免每个 batch checkpoint 都重采样导致额外开销。

### 跨 benchmark 汇总

新增 `scripts/eval/summarize_results.py`：

- 读取一个模型目录下的多个 `eval_*.json`。
- 每个 benchmark 汇总一行。
- 计算 `macro_avg/by_benchmark`，即各 benchmark `answer_acc` 的简单未加权平均。
- 输出 `_summary.json` 和 `_summary.md`。

`scripts/eval/run_bench.sh` 评测结束后会自动调用该汇总脚本：

```bash
bash scripts/eval/run_bench.sh
```

如果设置了 `RESULT_SUFFIX`，summary 默认只汇总对应后缀的结果，避免 smoke run 和 full run 混在一起：

```bash
RESULT_SUFFIX=_strict800 MAX_SAMPLES=800 bash scripts/eval/run_bench.sh
```

### 多模型串行评测脚本

新增 `scripts/eval/run_bench_list.sh`：

- 支持一次传入多个模型路径。
- 模型之间串行执行，避免多模型同时抢 GPU。
- 每个模型内部是否并行 benchmark 仍由 `RUN_PARALLEL` 控制。
- 支持 `CONTINUE_ON_ERROR=1`，某个模型失败后继续跑后续模型。

示例：

```bash
bash scripts/eval/run_bench_list.sh \
  models/TimeThinker-4B-SFT-v3-10000 \
  models/TimeThinker-4B-RL-Zero-100-van-v2/global_step_100/actor/huggingface \
  models/TimeThinker-4B-RL-Zero-100-ema-v2/global_step_100/actor/huggingface
```

### 评测速度分析

确认当前主要慢 benchmark 的原因：

| Benchmark | 样本数 | 唯一视频数 | 平均视频时长 | p90 时长 | 视频总大小 | 主要瓶颈 |
|---|---:|---:|---:|---:|---:|---|
| LongVideoReason | 1000 | 991 | 426s | 746s | 184GB | 几乎每题一个长视频，IO + seek + 抽帧重 |
| VideoMMMU | 900 | 300 | 507s | 871s | 13GB | 视频很长，每个视频约复用 3 题 |
| VideoMME | 2700 | 900 | 1021s | 2681s | 97GB | 超长视频，打开后会非常慢 |
| VSIBench | 5130 | 288 | 97s | 163s | 3.6GB | 单视频短，但题目数多且重复处理同视频 |

结论：

- 慢主要来自视频容器打开、seek、抽帧、PIL 转换和 processor 打包，不只是模型生成。
- `MAX_FRAMES=16` 不代表只花 16 帧的时间，长视频 seek 和解码成本仍然明显。
- LongVideoReason 的 decord 第一重失败率约为 `57/1000 = 5.7%`。
- VideoMMMU、VideoMME、VSIBench 当前日志中 decord fallback 基本为 `0%`。
- 因此主要瓶颈不是 decord 失败后的二重解码，而是长视频本身的预处理成本。

### 运行时 frame cache

新增运行时 on-disk frame cache：

- 默认路径：`Evaluation/data/.cache/eval_frames`
- 可通过 `FRAME_CACHE_DIR` 修改。
- 可通过 `DISABLE_FRAME_CACHE=1` 关闭。
- `auto/decord/pyav` 视频读取都会先查 cache。
- miss 后按原逻辑解码抽帧，并将抽好的帧写入磁盘。
- 后续模型、后续 benchmark run 可以复用同一批帧。

示例：

```bash
# 默认使用 Evaluation/data/.cache/eval_frames
bash scripts/eval/run_bench.sh

# 指定 cache 路径
FRAME_CACHE_DIR=/path/to/eval_frames bash scripts/eval/run_bench.sh

# 关闭 cache
DISABLE_FRAME_CACHE=1 bash scripts/eval/run_bench.sh
```

cache key 包含：

- 视频绝对路径
- 文件大小
- 文件 mtime
- `MAX_FRAMES`
- `FPS`
- `video_start/video_end`
- cache 格式版本

因此：

- 先跑 `MAX_SAMPLES=800`，再跑全集，可以复用前 800 条涉及的视频帧 cache。
- 改 `RESULT_SUFFIX` 不影响 frame cache 复用。
- 改 `MAX_FRAMES` / `FPS` / 视频文件后，会自动生成新的 cache，不会误用旧帧。

### 时间统计

新增每个 benchmark 的结构化耗时统计：

- `meta.elapsed_seconds`
- `meta.frame_cache.hit`
- `meta.frame_cache.miss`
- `meta.frame_cache.write`
- `meta.frame_cache.fallback_to_pyav`

`_summary.md` 新增列：

- `elapsed_min`
- `cache_hit`
- `cache_miss`
- `cache_write`
- `fallback_pyav`

用于后续对比：

```bash
# 第一次：建 cache
RESULT_SUFFIX=_first MAX_SAMPLES=800 bash scripts/eval/run_bench.sh

# 第二次：复用 cache
RESULT_SUFFIX=_cache_hit MAX_SAMPLES=800 bash scripts/eval/run_bench.sh
```

### 当前推荐的快速验证方式

快速看趋势：

```bash
DATASETS=eval_mmvu.json,eval_videomathqa.json,eval_tempcompass.json \
MAX_SAMPLES=200 \
RESULT_SUFFIX=_smoke200 \
bash scripts/eval/run_bench.sh
```

单独看长视频能力：

```bash
DATASETS=eval_longvideoreason.json,eval_videommmu.json \
MAX_SAMPLES=100 \
RESULT_SUFFIX=_long100 \
bash scripts/eval/run_bench.sh
```

完整跑之前建议先用 `RESULT_SUFFIX`，避免 smoke 结果和 full eval 的输出 JSON resume 混淆。

## 2026-07-06

### 数据与 SFT 文档化

- 整理 `docs/qa.md`，按面试追问方式说明：
  - 数据到底是什么。
  - SFT 和 RL 分别用哪份数据。
  - 为什么图像/视频混合，而不是纯视频。
  - 为什么多选题占比较高。
  - 视频是否切分、一个视频是否对应多个问题。
- 整理 `docs/sft_ablation.md`：
  - 记录 SFT v1-v9 的关键变量。
  - 将 loss、评测结果和实验目的分开，避免只凭 loss 判断模型好坏。

### SFT 消融结论阶段性整理

- `SFT-v3-10000-1ep` 作为当前较强 SFT 参照，六项视频评测平均约 `56.22%`。
- `SFT-v6-10k` 解冻 vision tower 后，eval loss 接近 v3，但六项平均约 `55.88%`，没有明显超过 `SFT-v3-10000-1ep`。
- 说明当前下游 benchmark 表现不能只靠训练 loss 判断，需要完整评测闭环。

### 新模型评测与异常现象定位

- 评测并对比了 `TimeThinker-4B-RL-Zero-100-ema-v2`、`TimeThinker-4B-RL-Zero-100-van-v2`、`TimeThinker-4B-SFT-v6-10k`、`TimeThinker-4B-RL-Zero-100-tgrpo-van` 等模型。
- 发现 `TimeThinker-4B-RL-Zero-100-ema` 和 `TimeThinker-4B-RL-Zero-100-van` 在旧 prompt 下没有输出 `<think>`，但准确率反而高。
- 初步判断不是模型真实性能突然变强，而是评测 prompt 和训练 prompt/输出格式未对齐导致的测评偏差。
- 因此后续将评测 prompt 调整为严格 `<think><answer>` 格式，并新增格式类诊断指标。

### Benchmark 速度瓶颈初步统计

从日志中统计出部分 benchmark 的近似耗时：

| Benchmark | 样本数 | 近似耗时 |
|---|---:|---:|
| LongVideoReason | 1000 | 97.5m |
| VideoMMMU | 900 | 95.8m |
| MVBench | 4000 | 86.8m |
| TempCompass | 7540 | 52.3m |
| VideoMathQA | 420 | 24.4m |
| MMVU | 625 | 8.1m |

结论：

- LongVideoReason 和 VideoMMMU 样本数不多但耗时很高，优先怀疑长视频预处理。
- MVBench 样本多且部分视频处理不稳定，也会拖慢整体。
- MMVU 相对适合作为快速 smoke benchmark。

## 2026-07-05

### RL 实验组织

- 开始将 RL 实验分成 GRPO、EMA-GRPO、T-GRPO 等可复现实验配置。
- 明确 100-step RL Zero 实验的 checkpoint 命名方式：
  - `TimeThinker-4B-RL-Zero-100-van`
  - `TimeThinker-4B-RL-Zero-100-ema`
  - 后续 v2 / tgrpo 命名继续沿用该规则。
- 形成“训练输出目录即实验名”的习惯，便于后续 eval 脚本自动生成 model tag。

### 评测结果整理

- 汇总了已有模型在 LongVideoReason、TempCompass、MVBench、VideoMathQA、MMVU、VideoMMMU 等六项核心 benchmark 上的结果。
- 将六项平均作为临时总指标，采用简单未加权平均。
- 注意到 VideoMME 和 VSIBench 有历史结果，但由于评测成本较高、题型和训练目标不完全一致，暂时没有纳入六项核心平均。

### 当前结果观察

- `TimeThinker-4B-RL-Zero-100-ema` 在六项核心 benchmark 上暂时最高，六项平均约 `58.13%`。
- `TimeThinker-4B-RL-Zero-100-van` 次之，六项平均约 `56.80%`。
- 新一批 v2 模型在部分 benchmark 上低于旧版，需要结合 prompt 对齐问题重新判断。

## 2026-07-03

### 数据/训练管线问题定位

- 记录了一批 Qwen3-VL 视频输入不对齐错误到 `EasyR1/bad_samples.txt`：
  - 典型形式：`[NOT ALIGN][video] tokens=... features=...`
- 这类错误说明训练/rollout 视频路径中，文本侧 video token 数和视觉侧 features 数不一致。
- 后续围绕这个问题做了：
  - 数据侧 `process_video(..., return_fps=True)`。
  - processor 传入 `video_metadata`。
  - vLLM rollout 注入 `mm_processor_kwargs`。
  - PyAV fallback。

### Eval 基础设施整理

- 仓库提交记录显示新增 eval 相关内容：`5c8be1b [add] eval`。
- 初步建立 benchmark 数据下载、评测入口和结果目录结构。
- 清理 pyc 文件：`b07d4a6 [del] .pyc`。

### 评测数据和结果目录约定

当前主要约定：

- 数据目录：`Evaluation/data`
- 结果目录：`Evaluation/results/<model_tag>/frames<MAX_FRAMES>/`
- 单 benchmark 输出：`eval_<benchmark>.json`
- 模型级汇总：`_summary.json` / `_summary.md`

## 待办

- 用新 prompt、新指标和 frame cache 重新跑一轮小规模 smoke，对比 cache 前后 `elapsed_min`。
- 对 LongVideoReason、VideoMME、VSIBench 这种重复或长视频 benchmark，观察第二轮 cache hit 后节省的比例。
- 根据新指标检查：
  - `answer_extract_rate`
  - `invalid_answer_rate`
  - `truncation_rate`
  - `format/strict_rate`
- 如果 frame cache 效果明显，再考虑是否增加离线预热脚本，用于提前 materialize 全部 benchmark frames。
