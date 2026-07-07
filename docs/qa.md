# Data Deep Dive Q&A

这份文档按面试追问的节奏整理，不追求把每个数据源都列全，而是把“我知道数据里有什么、为什么这么配、SFT/RL 怎么用”讲清楚。

```text
第一节：数据
第二节：SFT
第三节：RL
```

## 第一节：数据

### 1.1 这个项目的数据到底是什么？

当前本地数据不是完整的 TimeThinker-600k，而是 Video-R1 的图像/视频混合推理数据。

主要有两份原始文件：

- `data/Video-R1-COT-165k.json`：约 16.6 万条，用于 SFT cold start，含 `<think>...</think>` 推理过程。
- `data/Video-R1-260k.json`：约 26.3 万条，用于 RL，主要保留问题、答案、媒体路径、题型和来源。

媒体数据在 `data/` 下，视频包括 LLaVA-Video、STAR、CLEVRER、NeXT-QA、PerceptionTest；图像包括 Math、Chart、OCR、Knowledge、Spatial、General 等。

### 1.2 每条样本长什么样？

本质是“一张图/一个视频 + 一个问题 + 可选选项 + 标准答案”。

RL 样本大概是：

```json
{
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
  "path": "./LLaVA-Video-178K/.../ytb_7nRmsEw7nsE.mp4",
  "data_source": "LLaVA-Video-178K/30_60_s_youtube_v0_1"
}
```

SFT 样本比 RL 多一个 `process` 字段，也就是 teacher 生成的 CoT：

```json
{
  "problem": "...",
  "process": "<think>...</think>",
  "solution": "<answer>...</answer>",
  "path": "...",
  "data_type": "image",
  "problem_type": "numerical"
}
```

### 1.3 数据具体覆盖什么能力？

可以概括成：图像/视频 QA reasoning。

覆盖比较多的是：

- 视频动作、事件、场景理解。
- 多选问答。
- 图像数学、几何、数值题。
- 图表和表格读数。
- OCR、手写文字、手写公式。
- 科学图、医学图、教材图问答。
- 空间关系和尺寸/距离估计。
- 少量开放式短答。

它不包含 dense grounding、tracking、segmentation 这类 box/mask/trajectory 标注，所以不能期待它单独训出完整 TimeThinker 的定位、跟踪、分割能力。

### 1.4 数据配比大概是什么？

RL 主数据 `Video-R1-260k` 里：

- 图像约 14.7 万，占 56%。
- 视频约 11.6 万，占 44%。
- 多选题约 64%，是最大头。
- 其他主要是 free-form、numerical、OCR、regression。

能力上大概是：

- 视频理解是最大块，主要来自 LLaVA-Video、STAR、CLEVRER、NeXT-QA、PerceptionTest。
- 图像侧用 Math、Chart、OCR、Knowledge、Spatial 补静态视觉推理。
- Math 和 Knowledge 各自大约占 14% 左右。
- Chart、OCR、Spatial、General 分别补图表、文本、空间和通用视觉问答。

不用背每个数据源的精确比例，面试里讲清楚“图像略多于视频、多选占主、按能力 bucket 配平”就够了。

### 1.5 为什么这样配，而不是纯视频？

原因主要有三个：

1. 高质量视频推理数据相对稀缺。
2. 纯视频训练成本高，长视频采帧会带来更高 token 和显存压力。
3. RL 需要可验证答案，很多视频开放问答不好做稳定 reward。

图像数据里的数学、图表、OCR、知识和空间题更密集、更容易判分，可以提供稳定的视觉推理监督；视频数据则补动作、时序和事件变化。

所以这套数据是 image-video mixed reasoning data，而不是纯视频数据。

### 1.6 为什么多选题这么多？

因为 RL 需要稳定 reward。

多选题最终答案通常是：

```text
<answer>A</answer>
```

这种可以直接精确匹配，reward 很稳定。开放式答案同义表达太多，字符串匹配容易误判。

所以多选占比较高不是偶然，而是为了让 RL 初期更容易得到非零、低噪声的奖励。同时数据里还保留了数值、OCR、regression 和 free-form，避免模型只学会选项猜测。

### 1.7 为什么不是按原始数据集大小直接拼接？

因为原始规模不等于训练价值。

比如 LLaVA-Video 和 Multimath 原始规模都很大，如果按原始大小直接拼，会让单一数据源支配训练分布。Video-R1 更像是先按能力分桶：

```text
video / math / chart / OCR / knowledge / spatial / general
```

再从每个桶里选互补数据源，并限制大数据源上限。这样模型不会只学到某一个数据集的模板。

### 1.8 视频一般多长？都切分了吗？

视频已经是 clip 级别，但不是所有 clip 都很短。

大概分布：

- CLEVRER：约 5 秒的合成短视频。
- 0-30 秒、30-60 秒：占很多，是常见视频长度。
- 1-2 分钟、2-3 分钟：也有，尤其 LLaVA academic 里的长一点视频。

所以“切分了”不等于“全部切到十几秒”。有些数据源本身就保留了 2-3 分钟 clip，用来训练更长时间跨度的动作流程、事件顺序和因果理解。

### 1.9 一个视频会对应多个问题吗？

会，而且很常见。

`Video-R1-260k` 里大概：

- 11.6 万个视频问题。
- 3.5 万个独立视频。
- 平均每个视频 3.35 个问题。
- 中位数 2 个问题。
- 最多一个视频 20 个问题。

问题最多的是 LLaVA academic 里的 ActivityNet / 2-3 分钟视频，经常一个视频挂十几道题。原因是这类视频信息量更高，可以从主活动、地点、人物、物体、文字、动作顺序、结果等多个角度提问。

### 1.10 举几个视频样本例子

视频动作理解：

```text
视频：YouCook2 cooking clip
问题：What cooking action does the person perform with the black frying pan on the right burner?
答案：<answer>The person cracks an egg into the black frying pan on the right burner.</answer>
```

视频多选：

```text
视频：ActivityNet knitting clip
问题：What attribute of the knitting piece changes as the person continues their work?
选项：color / needles / yarn type / size
答案：<answer>D</answer>
```

视频计数：

```text
视频：CLEVRER synthetic video
问题：How many moving rubber objects are there when the video ends?
答案：<answer>1</answer>
```

### 1.11 举几个图像样本例子

数学图像：

```text
图片：数轴题
问题：B 和 F 是相反数时 D 表示什么？D 和 H 是相反数时 C 表示什么？
答案：<answer>0</answer>
```

OCR：

```text
图片：手写文字
问题：Convert the handwriting in this image to text.
答案：<answer>" You should.</answer>
```

图表读数：

```text
图片：bar chart
问题：What is the total value for the category 'summer'?
答案：<answer>16</answer>
```

空间估计：

```text
图片：室内货架/空间图
问题：Estimate the shelf height in feet.
答案：<answer>13.1</answer>
```

## 第二节：SFT

### 2.1 SFT 用哪份数据？

SFT 使用 `Video-R1-COT-165k.json` 转换后的数据，拆成 image/video 两份：

- `LLaMA-Factory/data/timethinker_sft_image.json`：约 7.9 万条。
- `LLaMA-Factory/data/timethinker_sft_video.json`：约 8.6 万条。

SFT 里视频略多于图像，大概 52% vs 48%。

### 2.2 SFT 数据格式是什么？

转换后是 LLaMA-Factory 的 ShareGPT 格式：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image>\nQuestion...\nOptions...\nPlease answer with <think> and <answer>."
    },
    {
      "role": "assistant",
      "content": "<think>Reasoning...</think>\n<answer>D</answer>"
    }
  ],
  "images": ["data/...jpg"]
}
```

视频样本则用 `videos` 字段。

### 2.3 SFT 的目标是什么？

SFT 主要是 cold start，不是最终优化正确率。

它要先教会模型：

- 按 `<think>...</think><answer>...</answer>` 输出。
- 不同题型的最终答案应该是什么格式。
- 多模态问题的基本推理轨迹。

这样后续 RL 才不会一开始就因为格式错、答案抽取不到，导致 reward 大量为 0。

### 2.4 SFT 对不同题型怎么处理？

没有为不同题型设计不同 loss，也没有不同 head。主要是 prompt 和答案格式对齐：

- 多选：最终只输出选项字母。
- 数值/regression：最终只输出数字。
- OCR：最终只输出转写文本。
- free-form：最终输出简洁文本答案。

这样做的目的，是让后续 RL reward 能按 `problem_type` 稳定判分。

### 2.5 SFT 中为什么多选更多？

SFT 中多选题占比比 RL 还高，大概 76%。

合理原因是：多选题更容易生成和过滤出可靠 CoT，答案格式也最稳定。regression 这类估计题则更难过滤，保留下来的比例较低。

所以 SFT 数据不是 260k 的均匀抽样，而是经过 teacher CoT 生成和规则过滤后的子集。

### 2.6 SFT 样本中的 CoT 长什么样？

以论文图里的石英圆柱题为例，SFT 会教模型这样回答：

```text
<think>
图里 quartz cylinder 有 laser passage hole，LDV 用激光测振。
它不像重量，也不像绝缘壳，更像是让激光通过用于测量的介质。
</think>
<answer>D</answer>
```

重点不是这条 CoT 本身多完美，而是让模型稳定学到“先推理、再给可抽取答案”的协议。

## 第三节：RL

### 3.1 RL 用哪份数据？

当前 RL 脚本：

```text
scripts/train/run_rl_t.sh
```

使用配置：

```text
scripts/train/qwen3_rl_t.yaml
```

关键数据配置：

```yaml
data:
  train_files: EasyR1/data/timethinker_rl_train_split.json
  val_files: EasyR1/data/timethinker_rl_val_512.json
  prompt_key: problem
  answer_key: answer
  image_key: images
  video_key: videos
  image_dir: .
  video_fps: 2.0
```

### 3.2 RL 数据规模和格式是什么？

原始 RL 数据是 `Video-R1-260k.json`，约 26.3 万条。

转换后：

- train：`EasyR1/data/timethinker_rl_train_split.json`，约 26.25 万条。
- val：`EasyR1/data/timethinker_rl_val_512.json`，512 条。

EasyR1 读取的字段主要是：

```json
{
  "problem": "...",
  "answer": "<answer>D</answer>",
  "data_type": "image",
  "problem_type": "multiple choice",
  "options": ["A...", "B...", "C...", "D..."],
  "images": ["data/...jpg"]
}
```

视频样本则用：

```json
{
  "videos": ["data/...mp4"]
}
```

### 3.3 RL reward 怎么算？

当前配置：

```yaml
worker:
  reward:
    reward_type: batch
    reward_function: EasyR1/verl/reward_function/timethinker_reward.py:compute_score
```

reward 会按 `problem_type` 路由：

- `multiple choice`：选项字母匹配。
- `numerical`：数值解析和匹配。
- `OCR`：转写文本相关得分。
- `regression`：相对误差。
- `open-ended`：文本匹配或模型判分逻辑。

这就是为什么数据里大量保留可验证答案：RL 需要自动、稳定、低噪声的 reward。

### 3.4 RL 训练时会再切视频吗？

不会把一个长视频再切成多个子 clip。

EasyR1 按 JSON 里的视频路径读取整段 clip，然后采样帧：

```yaml
video_fps: 2.0
```

代码默认最多：

```python
max_frames: int = 128
```

所以 2-3 分钟视频不会逐帧全部送进去，而是稀疏采样，最多约 128 帧。

### 3.5 当前 temporal RL 是怎么利用视频时序的？

当前配置中有：

```yaml
algorithm:
  temporal: true
  shuffled_rollout_ratio: 0.5
  temporal_reward: 0.3
  temporal_compare_ratio: 0.8
  temporal_correct_threshold: 0.1
```

直观理解：

- 正常视频生成一组回答。
- 对一部分 rollout 构造打乱帧顺序的视频。
- 比较正常视频和乱序视频的表现。
- 如果模型真的利用了时序，正常视频应该比乱序视频更容易答对。

这个 temporal reward 的目标，是减少模型只靠单帧或语言先验答题的问题。

### 3.6 RL 阶段主要风险是什么？

主要风险：

- 多选题多，模型可能学到选项偏置。
- 长视频最多采 128 帧，细节可能丢失。
- open-ended reward 不如多选/数值稳定。
- 当前数据没有 grounding、tracking、segmentation 标注，RL 不会凭空训出这些能力。

对应缓解：

- 保留 numerical、OCR、regression、open-ended，避免全是多选。
- 用 temporal shuffle reward 强化时序利用。
- 先用 SFT cold start 稳定输出格式。
- 明确目标是 Video-R1-style multimodal reasoning，而不是完整复现 TimeThinker 全能力。
