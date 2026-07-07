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

注意：当前磁盘模型目录已经使用 `van` / `ema` / `tgrpo-van` / `tgrpo-ema` 命名，但部分 yaml 和 list 脚本里仍可能保留旧的 `grpo` 字样。新实验开始前要先确认 `trainer.experiment_name` 和 `trainer.save_checkpoint_path`，避免结果写进旧目录。

## 命名约定

| 名称 | 含义 | 主要变量 |
|---|---|---|
| `van` | vanilla GRPO | `algorithm.adv_estimator=grpo` |
| `ema` | EMA-GRPO | `algorithm.adv_estimator=ema_grpo` |
| `tgrpo-van` | T-GRPO + vanilla GRPO advantage | `temporal=true`, `adv_estimator=grpo` |
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

### RL Zero 100 tgrpo-ema

- 模型目录：`models/TimeThinker-4B-RL-Zero-100-tgrpo-ema`
- 主要变量：
  - `algorithm.temporal=true`
  - `algorithm.adv_estimator=ema_grpo`
- 状态：目录存在，待补统一评测记录。
- 当前待确认：
  - 是否已完成 `global_step_100/actor/huggingface` 转换。
  - 是否已用新 eval prompt 跑完整 benchmark。
  - 和 `tgrpo-van` 的差异是否只来自 advantage estimator。

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
| `TimeThinker-4B-RL-Zero-100-tgrpo-ema` | T-GRPO + EMA | 100 | | | | | | | | | | pending |

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

## 下一步

1. 先把 `config/rl/*.yaml` 和 `scripts/train/run_rl_list.sh` 的旧 `grpo` 命名同步到 `van` 规则，避免后续新实验写错目录。
2. 补 `tgrpo-ema` 的转换和统一评测。
3. 重新用新 prompt 复测旧 `ema` / `van`，确认旧高分是否真实。
4. 做 `van-v2` vs `ema-v2` 的同 prompt、同 benchmark、同 suffix 对比，优先看 `_summary.md`。
5. 对 T-GRPO 增加 temporal reward 触发率统计，否则很难解释 temporal 实验为什么涨或不涨。
6. 如果训练资源紧张，优先做 `MAX_SAMPLES=800` 快速验证，再对最有希望的 1-2 个模型跑完整评测。
7. 每个 RL 结果目录保存实际训练配置，例如 `train_config.yaml`，不要依赖当前 yaml 反推历史实验。
