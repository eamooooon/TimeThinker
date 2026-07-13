# TimeThinker Daily Record

记录训练、评测、数据和工程优化的日常推进。日期按 UTC 工作区时间记录。

## 2026-07-12

### 仓库、文档与磁盘整理

- 将 Quick Start、环境安装、数据下载、SFT/RL 训练和评测入口集中到根目录 `README.md`，减少首次使用时在多份文档之间跳转。
- 将运行维护说明整理为中文 `docs/OPERATIONS.md`，覆盖磁盘排查、checkpoint 清理、模型合并、日志定位和常见恢复操作。
- 精简 `docs/` 主目录，将历史消融、旧评测口径和阶段性问题记录移动到 `docs/archive/`；主目录只保留当前仍需维护的项目、数据、评测和运维文档。
- 评测入口收敛到 `Evaluation/Eval/eval_bench.py`，旧 v1/v2/v3 实现不再作为活跃入口；历史差异和结果保留在 archive 文档中。
- 清理训练输出中的过期 optimizer、extra-state 和重复 checkpoint，并将需要长期保留的 FSDP 模型分片合并为 Hugging Face `safetensors`，避免一个实验同时保留多份 40+ GiB 的可续训状态。
- 对 `/tianyuesong` 可见目录、隐藏目录、`.git`、`.ipynb_checkpoints`、deleted-open 文件和常见临时/回收站位置做过排查；容器内 `du` 与平台 GPFS 配额口径并不等价，服务端 fileset quota、快照和其他节点打开文件仍需管理员侧命令才能彻底闭环。

### SFT-v9 起点的 200-step RL 验证

- 新增并运行 `config/rl/qwen3_rl_sftv9_kl200.yaml`：
  - 从 `models/TimeThinker-4B-SFT-v9-10k-canonical` 初始化。
  - `max_steps=200`、`rollout_batch_size=16`、`rollout.n=8`。
  - 正确启用 reference policy 和 loss-level KL：`disable_kl=false`、`use_kl_loss=true`、`kl_coef=0.01`。
  - 将 format reward 降为 `0.05`、length reward 降为 `0.02`，固定验证每 25 step 执行一次。
- 训练正常完成 `200/200`，耗时约 `4h56m`；无 OOM、NaN、worker death 或持续梯度/entropy/长度异常。
- 固定 512 条验证集结果：

| Step | Accuracy | Overall |
|---:|---:|---:|
| 0 | 0.672995 | 0.688368 |
| 25 | 0.642536 | 0.659238 |
| 50 | 0.659791 | 0.675923 |
| 75 | 0.667859 | 0.683196 |
| 100 | 0.665694 | 0.681433 |
| 125 | 0.670031 | 0.685845 |
| 150 | 0.671597 | 0.687236 |
| 175 | **0.673945** | **0.689565** |
| 200 | 0.667156 | 0.683017 |

- 最佳观测点 step 175 仅比 SFT 基线高 `0.095 pp`，最终 step 200 低 `0.584 pp`；当前 reward 没有带来显著、稳定的泛化提升。
- `val_freq=25` 与 `save_freq=50` 不一致导致 step 175 没有 checkpoint；后续应对齐验证和保存频率，并按固定验证指标保存 best。
- 全程监控、异常阈值、checkpoint 选择与最终审计记录在 `docs/archive/rl_kl200_monitor.md`。
- 修复 `EasyR1/verl/workers/actor/dp_actor.py` 的 KL 指标聚合：此前每卡只保留最后一个 microbatch，现在会聚合所有 microbatch；该缺陷只影响日志精度，不影响已经执行的 KL 反向传播。

### Reward 与 online filtering 审计

- 当前训练日志的 `reward/*` 是所有生成重试的过滤前候选全局均值，不应被当作单调学习曲线；固定验证才是可比较口径。
- 训练集与 val-512 的题型比例基本一致：multiple choice `64.2%`、open-ended `14.7%`、numerical `13.1%`、OCR `6.0%`、regression `2.0%`。
- 现有日志没有按 `problem_type` 保存 reward 趋势，也没有区分过滤前、过滤后和最终 update batch；选择题占 64%，全局均值可能掩盖小题型变化。
- online filtering 当前按组平均 accuracy 过滤；这对二值 reward 可排除全对/全错组，但不能保证连续/多档 reward 的组内有效方差，也没有直接检查 GRPO 实际使用的 `overall + length bonus`。
- 题型 reward 的主要问题：
  - open-ended 使用单参考 ROUGE，语义等价答案可能被误罚。
  - OCR 使用空格级 WER，不适合无空格文本和公式等价表达。
  - regression 只有有限档位，同档回答缺少排序信号。
  - numerical 固定按两位小数完全相等，缺少合理的绝对/相对容差。
- 已完成第一轮 P0 修复：
  - validation 新增每个 `problem_type` 的 accuracy/overall/format/count 及 macro 指标。
  - train 新增逐题型 pre-filter、post-filter、实际 update reward，以及 filter-key/final-outcome 两套 group std、range、零方差率和保留率。
  - online filtering 新增 `filter_key_min_std` 与 `filter_min_std`；活跃 GRPO 配置均设为 `1e-3`，同时排除答案分数无差异和最终 outcome 无差异的组。
  - 某次生成没有留下任何组时不再立即报错，而是继续生成并受 `max_try_make_batch` 上限保护。
  - RL dataset 改为通过共享 `build_prompt()` 构造 prompt，统一规范化 `problem_type`，修复大写 `OCR` 无法命中小写模板键的问题。
- 新增下一轮独立配置 `config/rl/qwen3_rl_sftv9_reward_v2.yaml`，保持上一轮 KL/LR/reward shaping 不变，仅启用双重方差过滤、逐题型指标，并将 `save_freq` 与 `val_freq` 对齐为 25；使用新的输出目录，避免覆盖上一轮模型。
- open-ended/OCR/regression 的 reward 公式重做仍需基于新增分桶指标做短程对照，暂不在没有 per-type baseline 的情况下同时改变多套 reward 尺度。

## 2026-07-11

### 新模型评测与结果归档

- 完成 `TimeThinker-4B-RL-bs16-v9-van-500-new` 的主评测并将最新结果补入 `docs/eval.md`。
- 对 `Evaluation/Eval` 中 v1/v2/v3/当前版本的差异做完整梳理：旧版本只用于历史结果解释，新实验统一走当前 `eval_bench.py`，避免多个相似入口继续分叉。
- 整理 `Evaluation/results`、训练日志和模型命名，强调不能只凭目录名判断初始化模型或算法配置，必须同时核对 `experiment_config.json`。

### RL reward 停滞问题定位

- 检查 `TimeThinker-4B-RL-bs16-v9-van-500-new` 后确认：训练批 reward 不增长不能单独证明模型完全没学习，因为每步样本不同且日志包含过滤前候选。
- 发现旧实验中部分配置虽然设置了 `kl_coef`，但 `disable_kl=true` 会使 reference worker 不存在，导致预期的 KL loss 实际没有生效。
- 初步将效果瓶颈从“单纯学习率/KL 不合适”转向 reward 可辨别性、online filtering 有效组比例、题型评分规则和固定验证设计，为 7 月 12 日的受控 200-step 实验建立对照基线。

## 2026-07-10

### Eval 汇总与 Prompt 敏感性定位

- 基于 `Evaluation/results/*/frames16/_summary.json` 重新生成 `docs/eval.md`，当前包含 14 个完整模型、每个模型覆盖 8 个 benchmark 和 22315 个样本。
- 当前无具体答案示例口径下：
  - `TimeThinker-4B-RL-Zero-100-ema-v2` Avg 为 `55.85`，仍是最高分。
  - `TimeThinker-4B-SFT-v9-10k-3ep` Avg 为 `55.76`，只落后 `0.09`。
  - 不能将此解释为 SFT 突然超过最强 RL；主要变化是移除固定答案示例后，SFT 不再受到严重 prompt bias。
- 对比保留的 `Evaluation/results-v3` 样本级结果：
  - SFT-v9 在旧模板下多选预测 `A` 的比例为 `55.91%`，而 GT 为 `A` 的比例约为 `30.75%`。
  - 删除具体 `A` 示例后，SFT-v9 的 `A` 预测比例降为 `27.95%`，多选正确率从 `55.51%` 升至 `61.10%`。
  - `ema-v2` 的 `A` 预测比例约 `30.69% -> 30.87%`，多选正确率基本不变；说明主榜排序变化主要来自 SFT 恢复，而不是 RL 在无示例下明显退化。
- 检查 RL 运行配置后发现：目录名含 `v9` 的部分旧 RL run 实际仍从 `Qwen3-VL-4B-Instruct` 初始化，不能直接当作 “SFT-v9 -> RL” 的对照；后续必须以 `experiment_config.json` 中的 `worker.actor.model.model_path` 为准。

### Prompt 历史记录与统一迁移

- 新增 `docs/prompt-history.md`，记录以下模板的演变和适用范围：
  - 旧 SFT / 旧 RL 的自然式模板及 answer-only 示例。
  - 2026-07-07 后 strict RL / `eval_bench_v2.py` 的完整 `<think>/<answer>` 模板。
  - `eval_bench_v3.py`、迁移前 `eval_bench-v4.py` 和当前统一模板。
- 明确固定 `A`、`3.14`、`Paris` 等格式示例不是有效 few-shot：它们没有提供与当前题目配对的视觉证据和推理，却会向最终答案注入先验。
- 新增唯一 QA prompt 定义：`scripts/prompting/timethinker.py`。
  - 统一格式为自然式 `Respond exactly in this format:` 加完整 `<think>...</think><answer>...</answer>`。
  - 保留按题型约束 `<answer>` 内内容的 `TYPE_TEMPLATE`。
  - 删除具体答案示例以及 “标签前后不得出现任何文本” 的额外硬约束。
- 当前活跃入口均复用该模板：
  - RL：`EasyR1/verl/utils/dataset.py`
  - 主评测：`Evaluation/Eval/eval_bench.py`
  - 单样本 QA 推理：`Evaluation/inference_single/inference.py`
  - 主评测 runner 默认调用 `eval_bench.py`；`eval_bench-v4.py` 保留为历史 v4 复现入口。
- 新增 `scripts/train/normalize_timethinker_sft_prompts.py`，并已将忽略版本控制的 SFT 数据 user prompt 全部迁移：
  - `timethinker_sft_video.json`：86217 条。
  - `timethinker_sft_image.json`：79355 条。
  - 共 165572 条，二次 dry-run 为 `0` 条待更新，逐条等值校验通过。
- 保留 `eval_bench_v1.py`、`eval_bench_v2.py`、`eval_bench_v3.py` 作为历史结果复现快照；新训练和新主评测使用统一模板，不与历史榜单直接混排。

### 验证

- 对统一 prompt 模块、SFT 标准化脚本、RL 数据集、`eval_bench.py` 和单样本推理运行 `py_compile`，均通过。
- `bash -n scripts/eval/run_bench.sh` 与 `git diff --check` 均通过。

## 2026-07-09

### 无具体答案示例的新一轮主评测

- 在 `Evaluation/results` 完成 14 个模型的完整 `frames16` 主评测；所有 summary 覆盖 8 个 benchmark、22315 个样本。
- 本轮包含此前已有的 SFT、RL、Qwen3-VL baseline，以及新增的：
  - `TimeThinker-4B-RL-v9-100-bs16`
  - `TimeThinker-4B-RL-Zero-100-van-bs16`
- 评测结果写入各模型的 `_summary.json` / `_summary.md`，并记录 balanced 调度下的 `wall_time`、累计 benchmark elapsed time 与输出诊断指标。
- 新版主评测模板移除类型示例中的固定答案，只保留 `<think>/<answer>` 指令和 answer type 约束；该结果集成为后续 prompt bias 分析和统一模板迁移的基线。

## 2026-07-08

### Eval 结果汇总与文档更新

- 汇总 `Evaluation/results` 下的新一轮 8 项 benchmark 结果，并更新 `docs/eval.md`。
- 当前主表改为 8 项 benchmark 的 `answer_acc` macro average：
  - LongVideoReason
  - MMVU
  - MVBench
  - TempCompass
  - VideoMathQA
  - VideoMME
  - VideoMMMU
  - VSIBench
- 明确 `Avg` 是 benchmark 级简单未加权平均，不是按样本数加权。
- 补充 weighted 诊断指标含义：
  - `weighted avg_tokens`：按样本数加权的平均输出 token 数。
  - `weighted trunc_rate`：按样本数加权的生成截断比例。
  - `weighted invalid_rate` / `weighted extract_rate`：用于判断答案抽取和格式稳定性。
- 将 `TimeThinker-4B-SFT-v5-10k` 和 `Qwen3-VL-4B-Instruct` baseline 的新测评结果写入 `docs/eval.md`。
- 从日志和 summary 记录补回 v2 RL 模型的实际墙钟时间：
  - `TimeThinker-4B-RL-Zero-100-van-v2`：约 `1h42m13s`，benchmark elapsed 累加 `357.75m`。
  - `TimeThinker-4B-RL-Zero-100-ema-v2`：约 `35m08s`，benchmark elapsed 累加 `109.56m`。

### 当前 Eval 口径修正

- 发现新版严格格式评测会把部分基座/旧模型的答案能力和格式错误混在一起，尤其 `Qwen3-VL-4B-Instruct` 分数被严重压低。
- 修改 `Evaluation/Eval/eval_bench.py` 的答案抽取逻辑：
  - `answer_acc` 只看答案是否正确，不再要求严格 `<think>/<answer>` 格式。
  - 多选题支持从 `<answer>`、句式 `"answer is A"`、尾部单字母等形式中抽取选项。
  - 数值题支持从自然语言答案中抽取最后/显式数值。
  - `<think>`、strict format、invalid answer、extract rate 改为诊断指标，而不是直接决定 `answer_acc`。
- 增加 `--rescore_existing` 思路/能力，用当前抽取逻辑重算已有输出，避免为了改 scoring 重跑昂贵生成。
- 在 `docs/eval.md` 中说明当前 `answer_acc` 已经是 answer-only 口径。

### v1 Prompt 复原与复跑

- 从历史版本恢复 `Evaluation/Eval/eval_bench_v1.py`，用于对比旧 prompt 评测口径。
- 新增/整理 `scripts/eval/run_bench_list_v1.sh`，支持一次传入多个模型串行跑 v1 eval。
- 删除单模型版 `run_bench_v1.sh`，避免脚本入口分裂。
- 给 v1 eval 补上 frame cache 支持，使其能复用 `Evaluation/data/.cache/eval_frames`，避免每次从头解码视频。
- 修复 v1 list runner 的 cache/resume 语义：
  - 已有完整输出时跳过。
  - 有部分输出时继续 resume。
  - cache 只负责复用视频帧，不等于结果文件可以自动跳过。
- 后续又清理了 v1 prompt 中已经不属于当前 benchmark 的 grounding / tracking / segmentation JSON 输出模板，只保留当前可能用到的 QA 类型：
  - `multiple choice`
  - `numerical`
  - `OCR`
  - `open-ended`
  - `free-form`
  - `regression`
  - `math`
- 统计当前 `Evaluation/data` 实际 problem type：
  - `multiple choice`：19657
  - `regression`：2640
  - `numerical`：18

### v1-rerun 结果与 prompt 敏感性分析

- 汇总 `Evaluation/results-v1-rerun`：
  - 完整 8 项模型共 12 个。
  - v1-rerun 最强完整模型为 `TimeThinker-4B-SFT-v3-10000`，Avg8 约 `55.35`。
  - `TimeThinker-4B-RL-Zero-100-ema-v2` Avg8 约 `55.15`。
  - `TimeThinker-4B-RL-Zero-100-van-v2` Avg8 约 `53.72`。
- 发现当前 strict-ish prompt 和 v1 prompt 会改变模型排序：
  - 当前 `Evaluation/results` 更偏向 RL，RL v2 模型约 `56+`。
  - `results-v1-rerun` 中 SFT 更强，部分 RL 下降明显。
- 对比样本后发现差异主要来自 prompt regime：
  - 当前 prompt 更强制最后输出单个选项字母，减少 option mapping 和尾部抽取错误。
  - v1 prompt 更自然，允许解释后再给答案，对没有严格学 `<think>` 的模型更友好。
- 找了两类典型样本：
  - SFT 在 v1 下正确但 RL 错：动作顺序、yes/no、计数、空间估计等。
  - RL 在 v1 下错但当前 prompt 下正确：选项字母映射、`Not available`、yes/no 反转、最后答案未收敛等。
- 结论：v1 更像 answer-only benchmark 能力测试；当前 prompt 更像 TimeThinker 接口服从 + 答题测试。两者都不能单独作为全部结论。

### Video-R1 原仓库 Prompt 对齐

- 查找 `/tianyuesong/zy/videor1` 原仓库评测 prompt，定位到：
  - `/tianyuesong/zy/videor1/src/eval_bench.py`
  - `/tianyuesong/zy/videor1/src/r1-v/src/open_r1/sft_video.py`
- 原仓库 benchmark prompt 核心是：
  - 要求模型像人一样深入思考。
  - 鼓励使用 `"let me think"`、`"wait"`、`"Hmm"`、`"let's break it down"` 等自然思考表达。
  - 要求 reasoning 放在 `<think>...</think>`，final answer 放在 `<answer>...</answer>`。
  - 多选题要求 `<answer>` 内只给单个选项字母。
- 判断原仓库 prompt 更接近 v1，而不是当前严格结构 prompt。
- 但原仓库比我们的 v1 更强调自然长推理和 self-reflection，因此后续若要复现 Video-R1，应单独建立 `original prompt eval`，不要只在 v1/current 二选一。

### 格式奖励与 RL 目标反思

- 统计发现：
  - 旧 `TimeThinker-4B-RL-Zero-100-ema` 在 v1 下 `<think>` 率为 `0%`，但 answer accuracy 很高。
  - v1-rerun `ema-v2` 也基本不输出 `<think>`，但能得到较高分。
  - SFT 模型几乎稳定输出 `<think>/<answer>`，但在某些 v1 口径下不一定排名最高。
- 这说明强制 `<think>` 格式和 benchmark answer accuracy 不完全一致。
- 阶段性判断：
  - `<answer>` 对抽取和评测很重要，应保留。
  - `<think>` 不宜作为过强硬约束或高权重 reward，尤其在单选题占比很高时可能优化到“格式更好但答案不更强”。
  - 后续 RL reward 应考虑把格式奖励降权，或者只奖励答案可抽取，而不是强制完整 `<think>...</think><answer>...</answer>`。

### SFT 消融与实验优先级调整

- 分析 `SFT-v9-10k-3ep` 为什么明显强：
  - 相比 1 epoch，3 epoch 让模型在固定 10k 高质量数据上反复学习，可能更充分掌握目标 benchmark 相关能力。
  - 训练 loss 阶段性下降，说明 1 epoch 可能还没学透。
- 分析 `SFT-v10-50k` 数据更多但效果反而不如 v2/v9：
  - 更大数据量不等于更强，可能引入与 benchmark 不对齐的分布。
  - 在同样学习率下，更多数据可能导致每类能力都学到一点，但目标能力不够扎实。
  - 学习率相对数据规模/训练步数也可能偏大，需要结合 loss 曲线和输出质量判断。
- 明确数据分布不对齐的解决方向：
  - 优先做目标 benchmark 相关数据筛选/重加权。
  - 保留小而准的 10k 主线。
  - 用 checkpoint selection 和 epoch 数控制扎实程度，而不是盲目扩大数据。
- 精简 `docs/sft_ablation.md` 待验证项：
  - Scheduler / warmup 暂不作为优先实验。
  - Weight decay / gradient norm 暂不作为优先实验。
  - SFT -> RL 迁移验证本来就是主线，不需要作为独立待验证项堆在消融列表里。
- 当前 SFT 主线优先级：
  - 以 `SFT-v9-10k-3ep` 作为最强 SFT 候选。
  - 优先做 checkpoint/epoch selection。
  - 暂缓大规模 scheduler、weight decay、LoRA、batch/GA 网格。

### 当前评测策略结论

- 后续建议同时保留三类口径：
  - `current answer-only`：主表，答案正确为主，格式指标单独诊断。
  - `v1/original prompt`：用于判断自然 prompt 下真实 answer ability。
  - `format diagnostics`：用于判断模型是否适合 TimeThinker 接口和后续 RL。
- 不再简单用单一分数决定模型优劣，尤其不能把“格式服从”误当成“视觉推理能力提升”。
- 对后续主线的影响：
  - 如果目标是 benchmark answer accuracy，强 `<think>` reward 可能不是必要条件。
  - 如果目标是可控推理接口，则必须同时报告 strict/extract/invalid，而不是只看 accuracy。

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

#### 模型级墙钟时间口径

注意：`_summary.md` 里的 `elapsed_min` 是单个 benchmark 的耗时。多个 benchmark 并行跑时，不能把各行 `elapsed_min` 直接相加当作模型评测的真实等待时间。

- 各 benchmark `elapsed_min` 相加：更接近 GPU task time / workload total。
- 模型级实际等待时间：应按 `logs/eval_list_*.log` 的 `DATE_SUFFIX` 到结果目录 `_summary.json` 写出时间计算。
- `run_bench.sh` 在 `RUN_PARALLEL=1` 时按 GPU 数并行调度 benchmark，某个 GPU 空出来后继续接下一项，因此墙钟时间由最长的 GPU 任务链决定。

已完成的两组 v2 full eval 墙钟时间：

| 模型 | eval list 开始 | summary 写出 | 实际墙钟时间 | benchmark elapsed 累加 |
|---|---:|---:|---:|---:|
| `TimeThinker-4B-RL-Zero-100-van-v2` | `2026-07-07 16:16:55` | `2026-07-07 17:59:08` | `1h42m53s` | `357.75m` |
| `TimeThinker-4B-RL-Zero-100-ema-v2` | `2026-07-07 17:59:08` | `2026-07-07 18:34:15` | `35m08s` | `109.56m` |

结论：后续记录“一个模型评测花了多久”时优先写墙钟时间；`elapsed_min` 累加只用于分析 benchmark workload 和 cache/preprocess 成本。

#### 并行 benchmark 调度优化

`scripts/eval/run_bench.sh` 新增 `EVAL_SCHEDULE`：

- `EVAL_SCHEDULE=balanced`：默认值。按近期 cache-hit full eval 的历史耗时从长到短重排 benchmark 队列，再交给动态 GPU 调度器。这样 4 路并行时，长任务先占住 GPU，短任务会在 GPU 空闲后补上，近似实现“最长 + 最短”配平。
- `EVAL_SCHEDULE=listed`：保留 `BENCHMARK_DATASETS` / `DATASETS` 给出的原始顺序，便于复现旧调度。

默认 8 项 full eval 的 balanced 入队顺序：

```text
eval_mvbench.json
eval_tempcompass.json
eval_videomme.json
eval_longvideoreason.json
eval_vsibench.json
eval_videommmu.json
eval_videomathqa.json
eval_mmvu.json
```

按 `ema-v2` cache-hit 耗时估算，原调度的最长 GPU 任务链约 `35m`，balanced 后预计可降到约 `30m` 左右。实际收益仍取决于模型输出长度、cache 命中、vLLM 初始化和视频 IO 抖动。

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
