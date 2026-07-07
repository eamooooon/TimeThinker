# TimeThinker：多模态视频推理后训练项目

这份文档用于简历和面试表达，重点讲清楚我在项目里做了什么、为什么这样做、遇到追问时怎么回答。

---

## 一句话项目描述

我做了一个面向图像/视频推理的多模态后训练项目：基于 Qwen-VL 系列模型，构建 image-video mixed reasoning 数据，先用 SFT 做 CoT cold-start，再用 GRPO / rule-based reward 做强化学习，让模型稳定输出 `<think>...</think><answer>...</answer>`，并提升视频时序、图表、OCR、数学和空间推理能力。

简历可写：

> 构建并适配 26 万级图像/视频多模态推理数据，覆盖视频时序、数学图像、图表、OCR、空间关系和知识问答等能力；基于 LLaMA-Factory + EasyR1 搭建 SFT + GRPO 后训练流程，设计按题型路由的 rule-based reward，并完成 Qwen3-VL-4B 的 SFT cold-start 训练适配。

---

## 项目背景

开源 instruct VLM 在视频推理上有几个明显问题：

1. **视频时序推理弱**  
   模型能描述视频内容，但对事件顺序、动作变化、因果关系、长视频上下文经常不稳。

2. **推理格式不可控**  
   多模态模型直接生成答案时，容易混入解释、格式不统一，不利于自动评估和 RL reward。

3. **高质量视频推理数据稀缺**  
   纯视频数据数量少、标注成本高，而且很多问题只是识别/描述，不是真正推理。

4. **RL 需要可验证答案**  
   如果答案不能稳定自动判分，GRPO 训练会很不稳定。因此需要把数据整理成多选、数值、OCR、回归等可规则判分的形式。

我的路线是：

```text
多源图像/视频推理数据构建
-> 统一 schema 和答案格式
-> SFT CoT cold-start
-> rule-based reward / GRPO
-> 视频推理 benchmark 评估
```

---

## 我的主要工作

### 1. 设计 image-video mixed data mixture

我没有只用纯视频数据，而是把训练数据分成视频和图像两大类。

视频数据负责：

- 开放域视频理解。
- 动作识别和人物/物体交互。
- 场景变化和状态变化。
- 时序关系理解。
- 因果推理 / why-how 问答。
- 物理推理、计数和连续动作理解。

图像数据负责：

- 数学图像：几何、图形数学、视觉计数。
- 图表/表格：chart、plot、map、table。
- OCR：场景文字、手写文字、公式识别。
- 空间关系：位置、方向、距离、尺寸估计。
- 知识图像：科学图、医学图、论文图、教材图。
- 通用图像：常识问答、图像描述、开放式 QA。

这样设计的原因是：纯视频数据稀缺且监督稀疏，而图像推理数据更密集、可验证，可以补足数学、OCR、图表、空间关系等底层推理能力；视频数据则补动态场景和时序理解。

---

## 数据来源和能力覆盖

### 视频数据

| 数据源 | 规模 | 主要能力 | 独特价值 |
|---|---:|---|---|
| LLaVA-Video-178K | 约 82k | 开放域视频理解、视频问答 | 覆盖面最大，真实公开视频多样性强 |
| STAR | 约 11.5k | 动作识别、人物/物体交互 | 强调动作和状态变化 |
| CLEVRER | 约 8k | 物理因果、碰撞、计数 | 合成场景可控，答案可验证 |
| NeXT-QA | 约 7.5k | 时序、why/how 因果问答 | 适合训练视频因果和前后事件理解 |
| PerceptionTest | 约 6k | 记忆、抽象、物理、语义 | 综合视频认知能力 |

### 图像数据

| Bucket | 规模 | 数据源 | 主要能力 |
|---|---:|---|---|
| Math | 约 37k | Multimath、UniGeo、Geometry3K、GeoQA+、CLEVR-Math | 几何、图形数学、视觉计数 |
| Chart | 约 21k | FigureQA、DVQA、PlotQA、ChartQA、MapQA、TabMWP | 图表读数、趋势比较、表格推理 |
| Knowledge | 约 37k | AI2D、ScienceQA、TQA、ArxivQA、PMC-VQA | 科学图、医学图、论文图、教材图 |
| OCR | 约 16k | TextVQA、HME100K、IAM、TextOCR、TextCaps | 场景文字、手写、公式识别 |
| Spatial | 约 20k | OpenSpaces、SpaceLLaVA | 位置、方向、距离、尺寸估计 |
| General | 约 15k | A-OKVQA、IconQA、Visual7W、ShareGPT4V | 通用视觉问答和开放表达 |

---

## 为什么采用多数据集少采样

我的核心原则是：**覆盖能力空间，而不是堆单一数据集规模**。

如果从少数大数据集多采样，会有几个问题：

- 模型容易学到特定数据集模板。
- 视觉形态单一，比如全是合成图表或全是几何题。
- 题型失衡，比如多选题过多或数值题过多。
- 同源样本重复度高，新增样本边际收益低。
- 大数据源会压过小但独特的数据源。

所以我采用多源混合：

```text
大源限量采样
小源保留独特能力
按能力 bucket 控制比例
按题型保证 reward 可用
按视频/图片分布避免重复
```

一句话：

> 少数据集多采样是在同一个分布里加厚；多数据集少采样是在能力空间里铺开。

---

## 数据筛选策略

### 1. 按能力 bucket 控制比例

我先把数据分成 video、math、chart、OCR、knowledge、spatial、general image 等 bucket，再给每个 bucket 分配目标量级。

这样能避免某个能力过强、另一个能力缺失。例如：

- 如果视频太少，模型学不到时序和动作变化。
- 如果 chart/math 太多，模型会偏考试题和图表模板。
- 如果 OCR 太少，模型遇到图中文字会读不出来。
- 如果 spatial 太少，距离、位置、方向类问题会弱。

### 2. 对大数据源设置 cap

很多原始数据集非常大，比如 Multimath-300K、PlotQA、DVQA、LLaVA-Video-178K。如果全量使用，会让它们主导训练。

我的策略是：

- 大源只取一部分，保留代表性能力。
- 小而独特的源尽量保留。
- source 内部再看题型、时长、学科等子分布。

### 3. 优先保留可验证题型

RL 阶段需要 reward 稳定，所以我会优先保留容易规则判分的数据：

| 题型 | 说明 | reward |
|---|---|---|
| multiple choice | 多选题 | 选项字母精确匹配 |
| numerical | 数值题 | 数值解析后比较 |
| OCR | 文字识别 | 文本相似度 / WER |
| regression | 连续数值估计 | `1 - 相对误差` |
| free-form | 开放回答 | ROUGE / 文本相似度，权重受控 |

对于 yes/no 这种二值题，我会倾向于过滤或降低比例，因为它信息量低，模型容易学捷径。

---

## 数据源内部细分

很多数据源内部还有更细的子分布，不是一个整体随机抽样。

### LLaVA-Video-178K

按视频来源和时长拆：

```text
0_30_s_academic
30_60_s_youtube
2_3_m_academic
```

不同视频长度对应不同难度：短视频偏瞬时动作，长视频更考察时序和上下文。

### NeXT-QA

按视频时长拆：

```text
0_30_s_nextqa
30_60_s_nextqa
1_2_m_nextqa
2_3_m_nextqa
```

并且原始数据是一个视频对应多个问题。我会控制单视频问题数，避免某个视频贡献过多相似问题。

### EXAMS-V

按学科拆：

```text
Chemistry / Geography / Mathematics / Physics / Biology / History / Science
```

这样能覆盖不同学科图像和推理方式。

### OpenSpaces / SpaceLLaVA

按题型拆：

```text
regression / multiple choice / free-form
```

这类数据不是普通问答，而是空间关系、位置、距离、尺寸估计。

---

## NExT-QA 处理案例

NExT-QA 是一个很典型的 source-level 处理案例。

原始 NExT-QA 是一个视频对应多个问题。以 `OE/train` 为例：

```text
videos: 3,870
questions: 37,523
平均每个视频约 9.7 个问题
```

我的处理思路是：

1. 不直接全量使用每个视频的所有问题。
2. 过滤信息量较低的二值判断题，例如 yes/no。
3. 每个视频尽量只保留 1-2 个问题，扩大视频覆盖面。
4. 将适合构造选项的 open-ended 问题转成五选一多选题。
5. 平衡正确选项位置，避免模型学到选项偏置。

### 为什么从开放题转多选题

NExT-QA 的开放答案通常很短，但同义表达很多。例如：

```text
hit the bird
hit the bird with swinging object
knock it down
```

如果用字符串精确匹配，会误判；如果用语义模型判分，又引入额外 reward model 噪声。转成多选后，reward 可以稳定地判断：

```text
<answer>A</answer>
```

这更适合 rule-based RL。

### NExT-QA type

| type | 含义 | 例子 |
|---|---|---|
| CW | causal why，原因/动机 | `why did ...` |
| CH | causal how，方式/手段 | `how did ...` |
| TN | temporal next，之后发生什么 | `what did ... after ...` |
| TP | temporal previous，之前发生什么 | `what did ... before ...` |
| TC | 当前动作/状态 | `what is ... doing` |
| DB | 二值判断 | `is ...` / `are ...` |
| DC | 计数 | `how many ...` |
| DL | 地点/位置 | `where is ...` |
| DO | 其他描述，如颜色、物体、关系 | `what color ...` / `who ...` |

我会过滤或降低 `DB` 这类二值判断题，因为它对推理能力提升有限。

---

## 数据格式统一

不同数据集字段差异很大，所以我统一到内部 schema：

```json
{
  "problem": "...",
  "solution": "<answer>A</answer>",
  "data_type": "image | video",
  "problem_type": "multiple choice | numerical | OCR | regression | free-form",
  "options": ["A. ...", "B. ..."],
  "path": "...",
  "data_source": "..."
}
```

统一动作包括：

- 不同字段名映射成统一 schema。
- 多选题选项统一成 A/B/C/D/E。
- 文本答案反查选项，转换成字母。
- 数值题抽取数字，去掉单位和货币符号。
- OCR 答案做大小写、空格和标点归一化。
- 媒体路径统一到本地目录。
- 答案统一包进 `<answer>...</answer>`。

SFT 数据进一步转成 LLaMA-Factory ShareGPT 格式：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image>\n问题\nOptions...\n请在 <think> 和 <answer> 中回答"
    },
    {
      "role": "assistant",
      "content": "<think>...</think>\n<answer>A</answer>"
    }
  ],
  "images": ["..."]
}
```

---

## CoT cold-start 数据

原始数据通常只有问题和答案，没有推理过程。为了让模型学会稳定输出推理链，我用强模型生成 CoT，再做规则过滤。

流程：

```text
Video-R1-260k
-> teacher model 生成 <think>...</think><answer>...</answer>
-> 抽取 answer
-> 按题型计算 reward
-> 过滤错误或格式不合规样本
-> 得到 SFT cold-start 数据
```

过滤标准：

- 必须能解析出 `<answer>`。
- 答案要和 ground truth 基本一致。
- `<think>` 不能为空或明显异常。
- 输出格式必须符合训练模板。

这样得到的 SFT 数据不只是“原始 260k 的随机子集”，而是经过 teacher CoT 质量过滤后的子集。

---

## 训练流程

### SFT

使用 LLaMA-Factory 做 SFT：

- base：Qwen3-VL-4B-Instruct。
- 数据：image/video SFT 数据拆成两个注册数据集。
- 输出格式：`<think>...</think><answer>...</answer>`。
- 目标：教模型遵守推理和答案格式，降低 RL 初期 reward 全 0 的概率。

### SFT 对不同题型的处理

SFT 阶段没有给不同题型设计不同的 loss，也没有单独训练不同 head；我主要做的是**题型感知的 prompt 和答案格式对齐**。这样模型在进入 RL 之前，已经知道每类题应该怎么读题、怎么组织推理、最终答案应该以什么形式落到 `<answer>` 里。

| 题型 | SFT 输入侧处理 | SFT 输出侧约束 | 目的 |
|---|---|---|---|
| multiple choice | 在问题后拼接 `Options`，统一成 A/B/C/D/E 选项 | `<answer>` 里只输出选项字母 | 让 reward 可以直接做字母精确匹配 |
| numerical | 保留原问题，提示模型只输出数值 | `<answer>` 里只输出数字，如 `42` / `3.14` | 避免单位、解释文本影响数值解析 |
| regression | 和数值题一样要求输出连续数值 | `<answer>` 里只输出估计值 | 方便 RL 阶段按相对误差给连续分 |
| OCR | 提示模型转写图像/视频中的文字 | `<answer>` 里输出转写文本 | 对齐 OCR 的 WER / 文本相似度 reward |
| free-form | 保留开放问题，不强行转多选 | `<answer>` 里输出短文本答案 | 保留开放表达能力，但控制比例和格式 |

所以如果面试官问“不同题型 SFT 有没有特殊处理”，我会这样回答：

> 有，但不是改训练目标，而是在数据构造和 prompt schema 上做题型对齐。比如多选题会显式拼接 options，并要求最终只输出选项字母；数值题和 regression 会要求只输出数字；OCR 要求输出转写文本；free-form 则保留短文本答案。这样 SFT 先把模型的输出格式训稳定，后续 GRPO 才能按 `problem_type` 路由到不同 reward，否则模型即使语义上答对，也可能因为格式不稳定拿不到分。

### RL / GRPO

使用 EasyR1 做 GRPO 类训练：

- reward 按 `problem_type` 路由。
- 多选题判断字母。
- 数值题判断数字。
- OCR 判断文本。
- regression 用相对误差。
- free-form 用弱文本相似度。

---

## 外部依据

这个 mixture 思路和 Qwen 系列技术报告是一致的。

Qwen3-VL / Qwen2.5-VL 都不是用单一数据训练，而是混合：

- image caption
- interleaved image-text
- OCR
- document parsing
- VQA
- STEM / math
- grounding
- video
- long-document / long-video
- agent / GUI 数据

Qwen2.5-VL 还明确提到，会在不同训练阶段 carefully adjust data composition and proportions。这说明多模态训练本身就需要按能力调 mixture，而不是按原始数据集大小直接拼接。

所以我的数据设计原则是：

> 基于目标能力来配数据，而不是基于原始数据量来配数据。

---

## 面试 2-3 分钟讲法

> 我这个项目主要做的是图像/视频多模态推理后训练。数据上我没有只用纯视频，因为高质量视频推理数据稀缺，而且 RL 需要可验证答案。所以我设计了 image-video mixed data mixture：视频侧负责动作、时序、因果和场景变化；图像侧用 math、chart、OCR、knowledge、spatial 等数据补静态推理能力。
>
> 数据构建时，我不是按原始数据集大小直接拼接，而是先按能力 bucket 分配比例，再从每个 bucket 选互补数据源。比如 LLaVA-Video 提供开放域视频覆盖，NExT-QA 提供视频因果问答，CLEVRER 提供物理因果，DVQA/PlotQA 提供图表读数，TextVQA/HME100K 提供 OCR，OpenSpaces 提供空间距离估计。
>
> 对大数据源我会做 cap，避免某个模板或题型支配训练。对一个视频多个问题的数据，我会控制单视频问题数，比如 NExT-QA 里每个视频原来平均有接近 10 个问题，我会筛掉 yes/no 这种信息量低的问题，尽量每个视频保留 1-2 个更有价值的问题，并把适合的开放问答构造成多选题，方便 rule-based reward。
>
> 最后我把不同数据源统一成内部 schema，统一答案格式和题型标签，再用 teacher model 生成 `<think>` 推理过程，通过规则过滤得到 SFT cold-start 数据。训练上先用 SFT 教格式，再用 GRPO 和按题型路由的 reward 优化答案正确性。

---

## 追问预案

### Q: 为什么不用纯视频？

纯视频数据稀缺，而且很多视频问题只是描述或识别，不适合 RL。图像侧的 math、chart、OCR、spatial 数据更密集、可验证，可以给模型提供稳定推理监督；视频侧再补时序和动态理解。

### Q: 为什么不用少数大数据集多采样？

少数大源会带来模板偏置。比如 PlotQA/DVQA 太多会让模型偏合成图表，Multimath 太多会偏数学，LLaVA-Video 太多会压过图像推理。多源少采样可以覆盖更多视觉形态和问题类型。

### Q: 为什么过滤 yes/no？

yes/no 信息量低，模型容易不看视频也能猜；而且不适合构造成高质量多选题。对于视频推理，我更想保留 why/how、after/before、counting、location、action 这类更有监督价值的问题。

### Q: 为什么把 NExT-QA open-ended 转成多选？

因为 RL 需要稳定 reward。开放答案同义表达很多，字符串匹配不可靠；转成多选后可以直接判断 `<answer>A</answer>` 是否正确，reward 更稳定。

### Q: 多选题会不会太简单？

多选题确实比开放题更容易判分，但它在 RL 里更稳定。我的做法不是只保留多选，而是同时保留数值题、OCR、regression 和部分 free-form，保证题型多样性。

### Q: regression 是什么？

连续数值估计题，主要来自空间数据。比如问两个物体距离多少厘米、某个物体宽度多少英寸。reward 不是完全匹配，而是按相对误差给连续分数。

### Q: 这个比例有没有理论依据？

不是理论最优比例，而是经验型 mixture design。依据是目标能力覆盖、数据源互补性、可验证性和分布平衡。Qwen3-VL/Qwen2.5-VL 的技术报告也说明，多模态训练需要混合 OCR、document、VQA、STEM、video 等多类数据，并按阶段调整比例。

### Q: 你怎么保证数据质量？

主要从几方面做：统一 schema，统一答案格式，过滤低信息量问题，控制单视频重复度，按题型路由 reward，SFT 数据再通过 teacher CoT 生成和规则过滤，确保格式和答案都可用。

---

## 简历 bullet 版本

- 构建 26 万级 image-video multimodal reasoning 数据，覆盖视频时序、数学图像、图表、OCR、空间关系、科学知识和通用视觉问答等能力。
- 设计多源数据 mixture 策略，对大规模数据源进行 cap，下采样重复模板，同时保留小而独特的数据源以提升能力覆盖。
- 针对一个视频多问题的数据源设计去冗余策略，控制单视频问题数，并过滤 yes/no 等低信息量问题。
- 将 open-ended 视频问答转换为 multiple-choice 形式，提升 rule-based reward 的稳定性和可复现性。
- 统一 20+ 数据源 schema、媒体路径、答案格式和题型标签，适配 LLaMA-Factory SFT 与 EasyR1 GRPO 训练。
- 使用 teacher model 生成 CoT，并通过按题型 reward 过滤构建 SFT cold-start 数据。
- 设计 multiple choice / numerical / OCR / regression / free-form 多题型 reward 路由，支持图像和视频混合 RL 训练。
