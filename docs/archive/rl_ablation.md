# RL 消融实验记录

本文档记录 TimeThinker RL 系列实验的变量、结果和后续计划。原则是：每次只改一个主变量，训练配置、模型目录、评测 prompt 和结果 summary 必须能对应起来，否则不做强结论。

## 共同基线

除非单独说明，RL 实验默认使用：

- Base model：`Qwen/Qwen3-VL-4B-Instruct`
- Framework：EasyR1 / verl
- Train file：`EasyR1/data/timethinker_rl_train_split.json`
- Val file：`EasyR1/data/timethinker_rl_val_512.json`
- Reward：`EasyR1/verl/reward_function/timethinker_reward.py:compute_score`
- Prompt / answer format：`<think>...</think><answer>...</answer>`
- Rollout：`worker.rollout.n: 8`
- Temperature：`1.0`
- Top-p：`1.0`
- Max prompt length：`16384`
- Max response length：`768`
- KL：`use_kl_loss: true`, `kl_coef: 4.0e-2`, `kl_penalty: low_var_kl`
- Online filtering：`filter_key: accuracy`, `filter_low: 0.01`, `filter_high: 0.99`
- Optimizer：`lr: 1.0e-6`, `weight_decay: 1.0e-2`, `max_grad_norm: 5.0`
- Vision tower：`freeze_vision_tower: false`
- Save freq：`50`
- Max steps：`100`

当前配置文件：

```text
config/rl/qwen3_rl.yaml
config/rl/qwen3_rl_t.yaml
```

启动脚本：

```text
scripts/train/run_rl.sh
scripts/train/run_rl_t.sh
scripts/train/run_rl_list.sh
```

注意：当前磁盘模型目录已经使用 `van` / `ema` / `tgrpo-van` / `tgrpo-van2` 命名；真正的 `tgrpo-ema` 需要后续用 `adv_estimator=ema_grpo` 重跑。部分 yaml 和 list 脚本里仍可能保留旧的 `grpo` 字样。新实验开始前要先确认 `trainer.experiment_name` 和 `trainer.save_checkpoint_path`，避免结果写进旧目录。

## 命名约定

| 名称 | 含义 | 主要变量 |
|---|---|---|
| `van` | vanilla GRPO | `algorithm.adv_estimator=grpo` |
| `ema` | EMA-GRPO | `algorithm.adv_estimator=ema_grpo` |
| `tgrpo-van` | T-GRPO + vanilla GRPO advantage | `temporal=true`, `adv_estimator=grpo` |
| `tgrpo-van2` | 第二个 T-GRPO + vanilla GRPO run | `temporal=true`, `adv_estimator=grpo` |
| `tgrpo-ema` | T-GRPO + EMA-GRPO advantage | `temporal=true`, `adv_estimator=ema_grpo` |
| `v2` | 第二轮同类实验 | 需要在备注里写清楚和第一轮差异 |

建议新实验目录使用：

```text
models/TimeThinker-4B-RL-Zero-100-<variant>
models/TimeThinker-4B-RL-Zero-100-<variant>-v2
```

## 当前已跑版本

### RL Zero 100 ema

- 模型目录：`models/TimeThinker-4B-RL-Zero-100-ema`
- 主要变量：`algorithm.adv_estimator=ema_grpo`
- 状态：已有完整评测结果。
- 当前观察：
  - 六项核心 benchmark 平均约 `58.13%`。
  - 目前是已记录结果里最强的 RL 参照。
  - 旧评测中 prompt 曾出现不对齐问题，需要以新 prompt 复测结果为准。

### RL Zero 100 van

- 模型目录：`models/TimeThinker-4B-RL-Zero-100-van`
- 主要变量：`algorithm.adv_estimator=grpo`
- 状态：已有完整评测结果。
- 当前观察：
  - 六项核心 benchmark 平均约 `56.80%`。
  - 略弱于 `ema`。
  - 旧结果中也受 prompt 对齐问题影响，后续结论要看新评测。

### RL Zero 100 ema-v2

- 模型目录：`models/TimeThinker-4B-RL-Zero-100-ema-v2`
- 主要变量：`algorithm.adv_estimator=ema_grpo`
- 状态：已有新一轮评测结果。
- 当前观察：
  - 六项核心 benchmark 平均约 `56.08%`。
  - 相比旧 `ema` 回落，尤其 VideoMathQA 回落明显。
  - 需要检查 v2 和第一轮是否只改了 prompt / 训练配置 / checkpoint 命名，不能直接归因为 EMA-GRPO 本身变差。

### RL Zero 100 van-v2

- 模型目录：`models/TimeThinker-4B-RL-Zero-100-van-v2`
- 主要变量：`algorithm.adv_estimator=grpo`
- 状态：已有新一轮评测结果。
- 当前观察：
  - 六项核心 benchmark 平均约 `55.26%`。
  - 弱于旧 `van`，也弱于 `ema-v2`。
  - 需要复核训练时 prompt 和评测 prompt 是否完全一致。

### RL Zero 100 tgrpo-van

- 模型目录：`models/TimeThinker-4B-RL-Zero-100-tgrpo-van`
- 主要变量：
  - `algorithm.temporal=true`
  - `algorithm.adv_estimator=grpo`
  - shuffled video rollout 用于 temporal reward。
- 状态：已有新一轮评测结果。
- 当前观察：
  - 六项核心 benchmark 平均约 `55.37%`。
  - MMVU 上达到当前较高结果，TempCompass 也接近强参照。
  - LongVideoReason、VideoMathQA、VideoMMMU 拉低平均。
  - temporal reward 只应该重点看时间顺序/视频推理类指标，不要用 OCR、数学、空间类下降直接否定 T-GRPO。

### RL Zero 100 tgrpo-van2

- 模型目录：`models/TimeThinker-4B-RL-Zero-100-tgrpo-van2`
- 主要变量：
  - `algorithm.temporal=true`
  - `algorithm.adv_estimator=grpo`
- 状态：已有新一轮评测结果。该结果原先误记为 `tgrpo-ema`，现已更正为 `tgrpo-van2`。
- 当前观察：
  - 8 项平均约 `56.08%`。
  - 比 `tgrpo-van` 更稳，但不能作为 T-GRPO + EMA 结论。
  - 和 `tgrpo-van` 的差异主要应视为同类 T-GRPO + GRPO run 的随机轨迹/输出分布差异。

### RL Zero 100 tgrpo-ema

- 模型目录：待定，建议使用 `models/TimeThinker-4B-RL-Zero-100-tgrpo-ema`
- 主要变量：
  - `algorithm.temporal=true`
  - `algorithm.adv_estimator=ema_grpo`
- 状态：待重跑严格版本。
- 当前待确认：
  - 配置和保存目录必须同时指向 `tgrpo-ema`。
  - `experiment_config.json` 中必须是 `adv_estimator=ema_grpo`。
  - 评测结果不能复用 `tgrpo-van2`。

## 消融维度

### 1. Advantage estimator

目标：比较 `grpo` 和 `ema_grpo` 在相同 reward、相同数据、相同步数下的稳定性和最终准确率。

对比组：

| 对比 | A | B | 控制变量 |
|---|---|---|---|
| vanilla vs EMA | `van` | `ema` | 非 temporal、同 max_steps、同数据、同评测 |
| T-GRPO vanilla vs EMA | `tgrpo-van` | `tgrpo-ema` | temporal 配置一致 |

重点看：

- `answer_acc`
- `macro_avg/by_benchmark`
- `avg_output_tokens`
- `truncation_rate`
- SwanLab 中 reward、KL、response length 是否更稳。

判断标准：

- 如果 EMA 只提升平均分但某些 benchmark 明显掉，要看 per-benchmark 和 per-category。
- 如果 EMA 输出更短或更规整，要检查是否只是格式收益。
- 如果 GRPO 更高但波动大，要补 bootstrap CI 或多 seed。

### 2. Temporal reward / T-GRPO

目标：验证 shuffled video temporal reward 是否真的提升视频时序推理，而不是只改变输出风格。

对比组：

| 对比 | A | B | 关键变量 |
|---|---|---|---|
| no temporal vs temporal | `van` | `tgrpo-van` | `algorithm.temporal` |
| no temporal vs temporal | `ema` | `tgrpo-ema` | `algorithm.temporal` |

重点 benchmark：

- `eval_tempcompass`
- `eval_mvbench`
- `eval_longvideoreason`
- `eval_videomme`
- `eval_videommmu`

不适合作为主要判断的 benchmark：

- 纯 OCR 子类。
- 纯数学/图表推理子类。
- grounding / tracking / segmentation 这类当前不训练的能力。

需要额外记录：

- shuffled rollout 是否成功生成。
- temporal reward 被触发的比例。
- 原视频正确、打乱帧错误的样本比例。
- 原视频和打乱帧都正确的比例。
- 原视频和打乱帧都错误的比例。

如果 temporal reward 触发率很低，T-GRPO 训练信号会很弱，结果不涨不一定说明方法无效。

### 3. Online filtering

目标：判断 DAPO-style group filtering 是否过滤掉太多样本，导致训练信号不足。

当前配置：

```yaml
online_filtering: true
filter_key: accuracy
filter_low: 0.01
filter_high: 0.99
```

建议对比：

| 实验 | online_filtering | filter_low | filter_high | 目的 |
|---|---:|---:|---:|---|
| baseline | true | 0.01 | 0.99 | 当前默认 |
| no-filter | false | - | - | 看过滤是否伤害覆盖面 |
| loose-filter | true | 0.0 | 1.0 | 近似不过滤，但保留代码路径 |
| strict-filter | true | 0.1 | 0.9 | 只保留中等难度样本 |

重点看：

- 每 step 实际有效 batch 数。
- `max_try_make_batch` 是否频繁触发。
- reward 方差是否过低。
- 一组 `n=8` 是否经常全对或全错。

### 4. KL / update strength

目标：判断 RL 是否更新过强或过弱。

当前配置：

```yaml
use_kl_loss: true
kl_coef: 4.0e-2
disable_kl: true
lr: 1.0e-6
max_grad_norm: 5.0
```

建议只做小范围：

| 实验 | kl_coef | lr | 目的 |
|---|---:|---:|---|
| baseline | `4e-2` | `1e-6` | 当前默认 |
| lower-kl | `2e-2` | `1e-6` | 放大策略更新 |
| higher-kl | `8e-2` | `1e-6` | 更保守更新 |
| lower-lr | `4e-2` | `5e-7` | 看是否更稳 |

观察：

- KL 是否飙升。
- response length 是否漂移。
- format rate 是否下降。
- answer_acc 是否提升但 invalid/truncation 变差。

### 5. Response length / length reward

目标：控制模型是否输出过短、过长或被截断。

当前配置：

```yaml
max_response_length: 768
len_control: false
len_reward: 0.2
len_min: 320
len_max: 512
```

建议先不要急着开 length reward，先用评测指标判断：

- `avg_output_tokens`
- `truncation_rate`
- `answer_extract_rate`
- `invalid_answer_rate`

如果出现大量截断，再考虑：

| 实验 | max_response_length | len_control | 目的 |
|---|---:|---:|---|
| baseline | 768 | false | 当前默认 |
| shorter | 512 | false | 降低成本，观察是否伤害推理 |
| len-control | 768 | true | 控制 CoT 长度 |

### 6. Rollout n

目标：比较训练信号质量和训练成本。

当前配置：

```yaml
worker.rollout.n: 8
```

建议对比：

| 实验 | n | 目的 |
|---|---:|---|
| baseline | 8 | 当前默认 |
| cheap | 4 | 降低成本，快速验证 |
| stronger | 16 | 提高组内差异，但成本显著上升 |

注意：

- `n` 变化会影响 GRPO advantage 质量，也会影响显存、吞吐和过滤通过率。
- 如果 online filtering 打开，`n` 太小可能更容易全对/全错，导致有效训练样本更少。

## 评测协议

### 快速验证

用于训练刚结束后的 sanity check，不用于最终结论：

```bash
DATASETS=eval_mvbench.json,eval_tempcompass.json,eval_videomathqa.json \
MAX_SAMPLES=800 \
RESULT_SUFFIX=_smoke800 \
FRAME_CACHE_DIR=Evaluation/data/.cache/eval_frames \
bash scripts/eval/run_bench_list.sh \
  models/TimeThinker-4B-RL-Zero-100-van-v2/global_step_100/actor/huggingface \
  models/TimeThinker-4B-RL-Zero-100-ema-v2/global_step_100/actor/huggingface
```

长视频专项快速验证：

```bash
DATASETS=eval_longvideoreason.json,eval_videommmu.json \
MAX_SAMPLES=100 \
RESULT_SUFFIX=_long100 \
FRAME_CACHE_DIR=Evaluation/data/.cache/eval_frames \
bash scripts/eval/run_bench_list.sh \
  models/TimeThinker-4B-RL-Zero-100-tgrpo-van/global_step_100/actor/huggingface
```

### 完整评测

完整结论必须使用统一 prompt、统一 `MAX_FRAMES`、统一 benchmark 列表：

```bash
RESULT_SUFFIX=_full \
FRAME_CACHE_DIR=Evaluation/data/.cache/eval_frames \
bash scripts/eval/run_bench_list.sh \
  models/TimeThinker-4B-RL-Zero-100-van-v2/global_step_100/actor/huggingface \
  models/TimeThinker-4B-RL-Zero-100-ema-v2/global_step_100/actor/huggingface \
  models/TimeThinker-4B-RL-Zero-100-tgrpo-van/global_step_100/actor/huggingface
```

每个模型目录会生成：

```text
Evaluation/results/<model_tag>/frames<MAX_FRAMES>/_summary.md
Evaluation/results/<model_tag>/frames<MAX_FRAMES>/_summary.json
```

如果设置了 `RESULT_SUFFIX`，summary 默认只汇总对应后缀的结果。

## 结果表模板

### 总表

| Model | Variant | Step | Avg | LongVideoReason | VideoMMMU | MVBench | TempCompass | VideoMathQA | MMVU | VideoMME | VSIBench | Note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `TimeThinker-4B-RL-Zero-100-ema` | EMA-GRPO | 100 | 58.13 | 71.00 | 63.98 | 65.76 | 73.75 | 25.95 | 48.33 | 57.96 | 51.39 | old prompt risk |
| `TimeThinker-4B-RL-Zero-100-van` | GRPO | 100 | 56.80 | 68.30 | 62.35 | 64.48 | 72.57 | 21.43 | 51.67 | - | - | old prompt risk |
| `TimeThinker-4B-RL-Zero-100-ema-v2` | EMA-GRPO | 100 | 56.08 | 67.50 | 60.80 | 65.12 | 72.81 | 19.05 | 51.22 | - | - | new run |
| `TimeThinker-4B-RL-Zero-100-van-v2` | GRPO | 100 | 55.26 | 66.50 | 60.15 | 64.32 | 72.76 | 17.38 | 50.44 | - | - | new run |
| `TimeThinker-4B-RL-Zero-100-tgrpo-van` | T-GRPO + GRPO | 100 | 55.37 | 64.50 | 60.65 | 66.56 | 73.09 | 17.62 | 49.78 | - | - | new run |
| `TimeThinker-4B-RL-Zero-100-tgrpo-van2` | T-GRPO + GRPO | 100 | 56.08 | 69.50 | 52.11 | 62.98 | 72.18 | 22.62 | 67.36 | 55.48 | 46.38 | was mislabeled as tgrpo-ema |
| `TimeThinker-4B-RL-Zero-100-tgrpo-ema` | T-GRPO + EMA | 100 | | | | | | | | | | pending strict rerun |

### 训练稳定性表

| Model | Reward Mean | Format Reward | Accuracy Reward | KL | Avg Resp Len | Clip Ratio | Filter Pass | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| | | | | | | | | |

### 输出质量表

| Model | answer_acc | extract_rate | invalid_rate | avg_tokens | trunc_rate | category_macro | bootstrap_ci |
|---|---:|---:|---:|---:|---:|---:|---|
| | | | | | | | |

## 当前结论

- 旧结果里 `ema` 暂时最强，但旧 prompt 和新 prompt 没完全对齐，不能把它当成最终论文式结论。
- 新一轮结果里 `ema-v2` 略强于 `van-v2`，说明 EMA-GRPO 可能更稳，但差距不大，需要看 bootstrap CI 和更多 benchmark。
- `tgrpo-van` 没有提高六项平均，但在 MMVU、TempCompass 上有亮点，说明 temporal reward 的收益可能是局部能力收益，而不是全 benchmark 平均收益。
- VideoMathQA 普遍偏低，是 RL 后训练和评测都需要重点排查的短板。
- LongVideoReason、VideoMMMU 很慢，快速验证时建议先小样本跑；完整结论再跑全集。

## 待验证实验计划

这一轮实验要避免只看总平均分。每个实验都需要同时记录训练配置、reward 曲线、response length、invalid/truncation、per-benchmark 和 per-category 结果。

### 1. 对比实验

#### 1.1 EMA 和 Value/Critic 对比

目标：确认 advantage estimator 本身的收益，而不是被 temporal reward、batch size 或 prompt 差异混淆。

建议对比：

| 实验 | adv_estimator | temporal | 目的 |
|---|---|---:|---|
| GRPO | `grpo` | false | 当前 vanilla baseline |
| EMA-GRPO | `ema_grpo` | false | 验证 EMA 是否更稳 |
| Value/Critic | `gae` 或当前框架对应 value estimator | false | 验证 critic/value baseline 是否优于无 critic |

控制变量：

- 同一 base model。
- 同一 train/val 数据。
- 同一 `max_steps`。
- 同一 `rollout_batch_size` / `global_batch_size`。
- 同一 `max_response_length`。
- 同一 eval prompt 和 benchmark 列表。

重点看：

- `reward/accuracy`
- `reward/format`
- KL 曲线
- response length 漂移
- `invalid_answer_rate`
- `truncation_rate`
- eval 的 bootstrap CI

#### 1.2 T-GRPO 和普通 GRPO 对比

目标：确认 temporal reward 是否真的带来时序理解收益。

建议对比：

| 实验 | adv_estimator | temporal | 目的 |
|---|---|---:|---|
| GRPO | `grpo` | false | 普通 baseline |
| T-GRPO | `grpo` | true | 验证 temporal reward |
| EMA-GRPO | `ema_grpo` | false | EMA baseline |
| T-GRPO + EMA | `ema_grpo` | true | 验证 temporal + EMA 是否叠加 |

注意：原先的 `tgrpo-ema` 结果已经更名为 `tgrpo-van2`，因为其 `experiment_config.json` 中实际是 `adv_estimator=grpo`。后续真正的 `tgrpo-ema` 必须重跑并确认 `adv_estimator=ema_grpo`。

### 2. T-GRPO 效果分析与优化

#### 2.1 排查 T-GRPO 为什么更差

需要补充数据分布统计：

- ordered video accuracy。
- shuffled video accuracy。
- ordered 正确、shuffled 错误的比例。
- ordered 和 shuffled 都正确的比例。
- ordered 和 shuffled 都错误的比例。
- `temporal_bonus` 触发比例。
- 触发 bonus 的样本类型分布。
- 触发 bonus 后 response length 是否变长。
- 触发 bonus 的样本在 eval temporal 子类上是否真的收益更高。

重点排查：

- `temporal_compare_ratio=0.8` 是否过松，导致 shuffled 也能答对的样本被奖励。
- `temporal_reward=0.3` 是否过大，压过原始 accuracy reward。
- T-GRPO 是否提高了 temporal 子类，但损伤 VideoMathQA、VideoMMMU、OCR、空间推理等非 temporal 项。
- 是否出现更高 `invalid_answer_rate` 或 `truncation_rate`。

#### 2.2 T-GRPO 超参数实验

建议先小网格，不要一次扫太大。

| 参数 | 当前值 | 候选值 | 目的 |
|---|---:|---|---|
| `temporal_reward` | 0.3 | `0.05 / 0.1 / 0.2 / 0.3` | 判断 temporal bonus 是否过强 |
| `temporal_compare_ratio` | 0.8 | `0.8 / 1.0 / 1.2` | 提高 ordered 必须优于 shuffled 的门槛 |
| `temporal_correct_threshold` | 0.1 | `0.1 / 0.3 / 0.5` | 避免弱正确样本拿 bonus |
| `shuffled_rollout_ratio` | 0.5 | `0.25 / 0.5 / 1.0` | 判断 shuffled 对比样本数是否足够 |

优先级：

1. `temporal_reward=0.1`
2. `temporal_compare_ratio=1.0`
3. `temporal_reward=0.1 + temporal_compare_ratio=1.0`
4. `shuffled_rollout_ratio=1.0`

#### 2.3 长度奖励设计

当前没有启用长度奖励：

```yaml
len_control: false
```

设计长度奖励前，先统计 response length 和 accuracy 的关系。

建议分桶：

| Length bucket | 需要统计 |
|---|---|
| `0-64` | accuracy / format / invalid |
| `64-128` | accuracy / format / invalid |
| `128-256` | accuracy / format / invalid |
| `256-384` | accuracy / format / invalid |
| `384-512` | accuracy / format / invalid |
| `512-768` | accuracy / format / truncation |

如果发现正确率最高区间集中在某个范围，再启用：

```yaml
len_control: true
len_min: <best_lower>
len_max: <best_upper>
len_reward: 0.05
```

长度奖励建议从 `0.05` 或 `0.1` 开始，不建议直接用 `0.2`，避免模型为了拿长度奖励硬凑 token。

### 3. 模型训练与配置

#### 3.1 Batch size

需要验证 batch size 对 reward 方差和最终 eval 的影响。

| 实验 | rollout_batch_size | global_batch_size | 目的 |
|---|---:|---:|---|
| bs16 | 16 | 16 | 当前小 batch 配置 |
| bs32 | 32 | 32 | 当前 T-GRPO 配置 |
| bs64 | 64 | 64 | 如果资源允许，验证稳定性 |

重点看：

- 每 step reward 方差。
- online filtering 后实际有效 batch。
- response length 是否更稳定。
- eval bootstrap CI 是否变窄。

#### 3.2 Step 数

只看 step100 不够，需要看训练是否还在上升或已经过拟合。

建议保存并评估：

| Step | 目的 |
|---:|---|
| 50 | 早期 checkpoint |
| 100 | 当前默认 |
| 200 | 看是否继续提升 |
| 400 | 看是否过拟合或分布漂移 |

评估时同一个 run 要比较不同 checkpoint，避免不同 seed/run 的方差干扰。

#### 3.3 图像 pixel

当前配置：

```yaml
max_pixels: 100352
```

建议对比：

| 实验 | max_pixels | 目的 |
|---|---:|---|
| low-res | 50176 | 降低成本，检查是否明显掉分 |
| baseline | 100352 | 当前默认 |
| high-res | 200704 | 看视觉细节类 benchmark 是否提升 |

重点看：

- 显存占用。
- 训练吞吐。
- VideoMME / MVBench / VSIBench 是否提升。
- 长视频任务是否因为 pixel 增加导致吞吐明显下降。

#### 3.4 Max response length

当前：

```yaml
max_response_length: 768
```

建议对比：

| 实验 | max_response_length | 目的 |
|---|---:|---|
| short | 512 | 降低截断前的生成成本，约束输出 |
| baseline | 768 | 当前默认 |
| long | 1024 | 验证 VideoMathQA / VideoMMMU 是否受截断影响 |

如果 `1024` 提升 VideoMathQA 但输出变长、耗时变高，需要配合 length reward 或更强 KL。

#### 3.5 Vision Tower / Projector 冻结策略

目标：确认 RL 阶段是否需要继续训练视觉模块。SFT 阶段的结论不能完全直接迁移到 RL，因为 RL reward 更稀疏、更 noisy，更容易造成视觉表征或视觉-语言对齐漂移。

当前 RL 配置：

```yaml
worker:
  actor:
    model:
      freeze_vision_tower: false
```

当前代码只支持 `freeze_vision_tower`，没有单独的 `freeze_multi_modal_projector` 开关。因此：

| 配置 | Vision Tower | Projector / Merger | LLM |
|---|---|---|---|
| `freeze_vision_tower: false` | train | train | train |
| `freeze_vision_tower: true` | freeze | train | train |

推荐默认先试：

```yaml
freeze_vision_tower: true
```

原因：

- Vision Tower 参数量大，RL reward 噪声可能破坏底层视觉表征。
- Projector / Merger 是视觉特征到 LLM hidden states 的适配层，参数更小，保留可训练通常更稳。
- 只冻结 Vision Tower 仍允许模型调整视觉信息进入语言模型的方式，比只训 LLM 更灵活。

建议实验：

| 实验 | Vision Tower | Projector / Merger | LLM | 目的 |
|---|---|---|---|---|
| train-all | train | train | train | 当前配置，对照组 |
| freeze-vision | freeze | train | train | 推荐优先验证 |
| llm-only | freeze | freeze | train | 判断 projector 是否也被 RL reward 带偏 |

注意：`llm-only` 当前需要新增代码支持单独冻结 projector / merger，不能只靠现有 yaml 完成。

重点看：

- `answer_acc`
- `format`
- `avg_output_tokens`
- `invalid_answer_rate`
- `truncation_rate`
- VideoMME / MVBench / VSIBench 等视觉相关 benchmark
- TempCompass / LongVideoReason 等时序相关 benchmark

如果 `freeze-vision` 比 `train-all` 更稳或分数更高，则后续 RL 默认冻结 Vision Tower。只有在高分辨率、长视频或视觉细节类任务明显受限时，再考虑继续训练 Vision Tower。

### 4. KL 超参数

当前配置存在两个容易混淆的字段：

```yaml
disable_kl: true
use_kl_loss: true
kl_penalty: low_var_kl
kl_coef: 4.0e-2
```

需要先确认代码里：

- `disable_kl` 是否只关闭 reward-level KL。
- `use_kl_loss` 是否仍然启用 actor loss 中的 KL。
- 实际训练日志中 KL 是否有非零记录。

建议实验：

| 实验 | kl_coef | kl_penalty | 目的 |
|---|---:|---|---|
| no-kl | 0.0 | `low_var_kl` | 看无 KL 是否漂移严重 |
| low-kl | 1e-2 | `low_var_kl` | 放松约束 |
| baseline | 4e-2 | `low_var_kl` | 当前默认 |
| high-kl | 8e-2 | `low_var_kl` | 更保守 |
| very-high-kl | 1e-1 | `low_var_kl` | 强约束，观察是否保住格式和长度 |

如果框架支持，也可以对比：

| 实验 | kl_type | 目的 |
|---|---|---|
| fixed | `fixed` | 当前固定 KL |
| adaptive | `adaptive` | 根据目标 KL 自动调整 |

重点看：

- KL 曲线是否飙升。
- answer accuracy 是否提升。
- format 是否下降。
- avg output tokens 是否漂移。
- invalid/truncation 是否升高。

### 5. 额外必须补的控制项

#### 5.1 Seed variance

当前 0.5 到 1 分的差距可能落在随机方差内。关键对比至少跑：

```text
seed=1
seed=2
seed=3
```

如果资源不够，至少对最关键的 `GRPO`、`EMA-GRPO`、`T-GRPO` 跑 2 个 seed。

#### 5.2 Checkpoint selection

不要只评估最后一步。每个 run 至少记录：

```text
global_step_50
global_step_100
global_step_200
```

已有日志显示最后一步 batch 波动可能影响结论，所以 checkpoint selection 很重要。

#### 5.3 Per-category eval

T-GRPO 不能只看总平均，要重点看：

- VideoMME `Temporal Perception`
- VideoMME `Temporal Reasoning`
- TempCompass `order`
- TempCompass `speed`
- TempCompass `direction`
- LongVideoReason
- MVBench temporal 子类

#### 5.4 Answer extraction / invalid rate

每次 eval 都要记录：

- `answer_extract_rate`
- `invalid_answer_rate`
- `avg_output_tokens`
- `truncation_rate`
- `bootstrap_ci`

如果模型总分下降但 invalid/truncation 上升，需要先判断是不是输出格式/长度问题，而不是能力问题。

## 下一步

1. 先把 `config/rl/*.yaml` 和 `scripts/train/run_rl_list.sh` 的旧 `grpo` 命名同步到 `van` 规则，避免后续新实验写错目录。
2. 重跑严格的 `tgrpo-ema`，确认 `experiment_config.json` 中使用 `adv_estimator=ema_grpo`。
3. 补 response length 与 accuracy 的分桶统计，再决定是否启用 length reward。
4. 增加 temporal reward 触发率、ordered/shuffled accuracy 对比统计，否则很难解释 temporal 实验为什么涨或不涨。
5. 优先跑小规模 T-GRPO 超参：`temporal_reward=0.1`、`temporal_compare_ratio=1.0`、两者组合。
6. 做 KL ablation：`kl_coef=1e-2 / 4e-2 / 8e-2`。
7. 做 step ablation：评估 `global_step_50 / 100 / 200`。
8. 重新用新 prompt 复测旧 `ema` / `van`，确认旧高分是否真实。
9. 如果训练资源紧张，优先做 `MAX_SAMPLES=800` 快速验证，再对最有希望的 1-2 个模型跑完整评测。
10. 每个 RL 结果目录保存实际训练配置，例如 `train_config.yaml`，不要依赖当前 yaml 反推历史实验。
