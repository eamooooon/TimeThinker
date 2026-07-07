# Bad Case Analysis

本文档用于分析 TimeThinker / Video-R1 风格 SFT、RL、Benchmark 评测中的坏例。目标不是只记录“错了”，而是把错误归因到可行动的配置、数据、模型能力或评测问题上。

## 1. 分析对象

优先分析以下来源：

- `Evaluation/results/<model_tag>/frames<MAX_FRAMES>/eval_*.json`：benchmark 推理与打分结果。
- `Evaluation/results/<model_tag>/frames<MAX_FRAMES>/_summary.md`：当前模型的一页式评测概览。
- `Evaluation/results/<model_tag>/frames<MAX_FRAMES>/_summary.json`：可脚本化读取的聚合指标。
- `logs/eval_*.log`：评测运行日志，重点看视频解码、fallback、batch 耗时、ERROR。
- `swanlog/` / SwanLab：训练 loss、reward、KL、response length、format reward、accuracy reward 曲线。
- `EasyR1/data/timethinker_rl_train_split.json`：RL 样本与 reward 可验证性。
- `LLaMA-Factory/data/`：SFT 样本、媒体路径、任务类型。

一个 bad case 至少需要包含：

- 样本 ID / benchmark / 数据集来源。
- 媒体路径。
- 问题、选项、标准答案。
- 模型完整输出。
- 解析出来的 `<answer>`。
- 评测分数或错误信息。

## 2. 快速分桶

先把 bad case 分成下面几类，不要一开始就猜模型“不会”。

| 类型 | 现象 | 常见原因 | 优先动作 |
| --- | --- | --- | --- |
| 格式错误 | 没有 `<answer>`、标签顺序错、输出多余解释 | SFT 格式学习不足；RL format reward 太弱；prompt 不一致 | 检查 SFT 模板、reward format 权重、生成 stop 配置 |
| 选项映射错误 | 推理内容对，但最终选错 A/B/C/D | 选项拼接顺序、答案抽取、模型最后一步不稳 | 看完整输出和 options，必要时加 answer-only reward |
| 视觉感知错误 | 看错物体、文字、图表、空间关系 | 分辨率低、帧数低、OCR 弱、视觉塔冻结/未适配 | 提高 `max_pixels` / `video_maxlen`，加入同类 SFT/RL 样本 |
| 时间理解错误 | 事件顺序、动作变化、前后状态判断错 | 视频帧太少、采样 fps 不合适、长视频截断 | 提高 `MAX_FRAMES` 或 `video_maxlen`，检查 fps 与片段范围 |
| 数学/数值错误 | 看对题但计算错 | 语言推理或数值 reward 不够细 | 增加数学/数值样本，检查 reward 容差 |
| OCR 错误 | 文字识别错、公式漏读 | 分辨率不足、压缩、OCR 数据不够 | 提高图像上限，单独统计 OCR 子集 |
| 开放题语义错 | 答案大意不对，ROUGE 低 | 评测指标不适合；答案多样性 | 人工复核，必要时使用 judge/RM |
| 解码/工程错误 | `<answer>ERROR</answer>` 或日志异常 | 视频文件缺失、decord/torchvision/PyAV 问题 | 查 log、media path、reader fallback |
| 数据标注问题 | GT 本身错、选项不唯一 | benchmark 或转换错误 | 标记为 data issue，不用于训练结论 |

## 3. 单例分析模板

```markdown
## Case ID

- Date:
- Model:
- Stage: SFT / RL / Eval
- Benchmark:
- Result file:
- Sample index / problem_id:
- Media path:
- Data type:
- Problem type:

### Input

Question:

Options:

Ground truth:

### Model Output

Raw output:

Extracted answer:

Score:

### Error Type

- [ ] Format
- [ ] Option mapping
- [ ] Visual perception
- [ ] Temporal reasoning
- [ ] Math / numerical
- [ ] OCR
- [ ] Open-ended metric
- [ ] Decode / infra
- [ ] Data label
- [ ] Other:

### Observation

模型实际看到了什么 / 推理了什么：

和 GT 的关键差异：

### Root Cause Hypothesis

最可能原因：

证据：

反证或不确定点：

### Fix

短期修复：

长期修复：

是否需要加入训练数据：

是否需要改 reward：

是否需要改评测脚本：

### Follow-up

- [ ] 复现单样本推理
- [ ] 检查媒体是否正确
- [ ] 检查 prompt / template
- [ ] 检查 answer extraction
- [ ] 加入同类 case 统计
```

## 4. 结果级统计

每次评测后先做聚合，再挑样本。建议至少统计：

- benchmark 级准确率：`answer_acc`。
- benchmark 宏平均：`macro_avg/by_benchmark`。
- problem_type / category 级准确率：`per_category_acc`、`per_category/macro_acc`。
- data_type 级准确率：image / video / video-image。
- 答案抽取成功率：`answer_extract_rate`。
- 无效答案率：`invalid_answer_rate`。
- 输出长度：`avg_output_tokens`。
- 截断率：`truncation_rate`。
- bootstrap 置信区间：`bootstrap_ci`。
- `<answer>ERROR</answer>` 数量。
- 每个 benchmark 耗时：`elapsed_min`。
- frame cache 统计：`cache_hit` / `cache_miss` / `cache_write`。
- 视频 fallback 次数：`fallback_pyav`。

推荐表格：

| Benchmark | Samples | Acc | Extract | Invalid | Trunc | Avg Tokens | Time | Cache Hit | Main Failure Type |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MVBench | | | | | | | | | temporal / option |
| TempCompass | | | | | | | | | temporal |
| VideoMMMU | | | | | | | | | knowledge / OCR / long video |
| VideoMME | | | | | | | | | temporal / perception |
| VSI-Bench | | | | | | | | | spatial |
| MMVU | | | | | | | | | open-ended / reasoning |

优先先看 summary：

```bash
sed -n '1,160p' Evaluation/results/<model_tag>/frames16/_summary.md
```

如果想重新生成 summary：

```bash
python scripts/eval/summarize_results.py Evaluation/results/<model_tag>/frames16
```

## 5. SFT 相关归因

如果 bad case 大量集中在格式或答案风格：

- 检查 `config/sft/qwen3_sft*.yaml` 的 `template` 是否和评测 prompt 一致。
- 检查 SFT 样本里 `<think>`、`<answer>` 是否完整。
- 看 `train/loss` 和 `eval/loss` 是否同步下降。
- 如果 train loss 降但 eval loss 不降，优先考虑数据分布或泄漏/验证集问题。

如果视觉问题明显：

- 当前 SFT 配置中的 `image_max_pixels`、`video_max_pixels`、`video_maxlen` 会直接限制视觉信息。
- SFT 与 eval/RL 的分辨率、帧数应尽量对齐。
- 如果 SFT 冻结视觉塔，只能主要调语言侧和 projector，视觉 gap 不一定能补上。

## 6. RL 相关归因

RL bad case 重点看 reward 是否真的奖励了你想要的行为：

- `format` 高但 `accuracy` 低：模型学会格式，但没有学会任务。
- `accuracy` 高但人工看不对：reward / parser / GT 可能有问题。
- KL 飙升：策略更新太猛，降低 LR、提高 KL、减小 rollout 随机性。
- response length 失控：需要 length reward 或更严格 `max_response_length`。
- 一组 `n=8` 回答全对或全错：GRPO advantage 信息少，online filtering 会过滤。

当前 RL 配置重点关注：

- `config/rl/qwen3_rl.yaml`
- `config/rl/qwen3_rl_t.yaml`
- `worker.rollout.n`
- `data.rollout_batch_size`
- `worker.actor.global_batch_size`
- `algorithm.kl_coef`
- `algorithm.online_filtering`
- `algorithm.filter_low` / `algorithm.filter_high`
- `algorithm.temporal`
- `algorithm.shuffled_rollout_ratio`
- `algorithm.temporal_reward`
- `algorithm.temporal_compare_ratio`
- `algorithm.len_control`
- `worker.actor.optim.lr`
- `worker.reward.reward_function`

T-GRPO 相关 bad case 要额外看：

- 原视频回答对、打乱帧回答也对：说明这个样本未必能提供时间顺序训练信号。
- 原视频回答错、打乱帧回答错：advantage 可能很弱，容易被 online filtering 过滤。
- 打乱帧反而答对：需要检查采样帧是否本身不依赖顺序，或题目/答案是否有泄漏线索。
- temporal reward 只适合训练时间顺序能力；如果 benchmark 是 OCR、数学、空间 grounding，不要把提升或下降都归因到 T-GRPO。

## 7. 评测工程归因

如果结果异常差，先排除工程问题：

- 媒体路径是否存在。
- 视频 reader 是否大量 fallback。
- 是否出现 `<answer>ERROR</answer>`。
- `MAX_FRAMES` 是否过低。
- `max_pixels_video` 是否和训练差异太大。
- 是否 resume 到旧结果文件，导致新配置没有重新评测。
- `RESULT_SUFFIX` 是否区分了 smoke / full，避免 800 条快速验证结果污染全量结果。
- `FRAME_CACHE_DIR` 是否一致；如果想对比 cache 前后耗时，要记录 cache 是否命中。
- `DISABLE_FRAME_CACHE=1` 是否被误设。

检查日志：

```bash
tail -f logs/eval_*.log
rg "ERROR|fallback|video_reader|No frames|not found" logs/eval_*.log
```

检查结果：

```bash
python - <<'PY'
import json
from pathlib import Path

for p in sorted(Path("Evaluation/results").glob("*/frames*/eval_*.json")):
    obj = json.load(open(p))
    rows = obj.get("results", obj if isinstance(obj, list) else [])
    if not rows:
        continue
    errors = sum("<answer>ERROR</answer>" in str(x.get("response", x.get("prediction", ""))) for x in rows)
    metrics = obj.get("metrics", {})
    meta = obj.get("meta", {})
    cache = meta.get("frame_cache", {})
    print(
        p,
        "samples=", len(rows),
        "answer_acc=", metrics.get("answer_acc"),
        "extract=", metrics.get("answer_extract_rate"),
        "invalid=", metrics.get("invalid_answer_rate"),
        "trunc=", metrics.get("truncation_rate"),
        "errors=", errors,
        "elapsed_min=", round(meta.get("elapsed_seconds", 0) / 60, 2) if meta.get("elapsed_seconds") else None,
        "cache=", {k: cache.get(k) for k in ("hit", "miss", "write", "fallback_to_pyav")},
    )
PY
```

检查 summary：

```bash
find Evaluation/results -path '*/frames*/_summary.md' -print
sed -n '1,120p' Evaluation/results/<model_tag>/frames16/_summary.md
```

## 8. 修复优先级

按成本从低到高：

1. 修 answer parser / prompt / template。
2. 清理明显错误数据和路径。
3. 调 eval 的帧数、分辨率、reader。
4. 加同类 SFT 数据。
5. 调 RL reward。
6. 调 RL KL / LR / batch。
7. 解冻 projector / vision tower 或提高视觉分辨率。

不要一次改太多。每次只改一个主变量，并保留对应 bad case 集合做回归测试。
