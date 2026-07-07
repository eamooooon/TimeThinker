# Video-R1 数据收集与取样说明

本文说明当前仓库所用 `data/Video-R1-260k.json` 的数据来源、取样规模和题型标记含义。

结论先说：Video-R1-260k 不是把每个公开视频/图像数据集全量拼起来，而是从大量公开数据源中按能力 bucket 做下采样和配平。很多源只用了原数据的一小部分，尤其是 LLaVA-Video-178K、Multimath-300K、PlotQA、DVQA、PMC-VQA 这类原始规模很大的数据集。

## 题型缩写

图中括号里的 `mc / free / num / reg / ocr` 是题目类型。

| 缩写 | 本地字段 | 含义 | 典型答案 |
|---|---|---|---|
| `mc` | `multiple choice` | 多选题，通常给 A/B/C/D 选项 | `<answer>A</answer>` |
| `free` | `free-form` | 开放式文本回答 | `<answer>the man is cooking</answer>` |
| `num` | `numerical` | 数值题，要求输出离散或确定数值 | `<answer>3</answer>` |
| `reg` | `regression` | 连续数值回归题，reward 按相对误差给分 | `<answer>2.5</answer>` |
| `ocr` | `OCR` | 文字识别/转写题 | `<answer>STOP</answer>` |

这些类型不是装饰标签，而是训练和 RL reward 路由的关键字段。比如多选题用选项字母精确匹配，数值题做数值解析，OCR 用 WER 类指标，regression 用相对误差。

## Video-R1-260k 总体分布

本地统计来自 `data/Video-R1-260k.json`。

| Bucket | 本地样本数 | 占比 | 作用 |
|---|---:|---:|---|
| General Video | 116,248 | 44.2% | 视频事件、动作、时序、因果和日常场景理解 |
| Knowledge Image | 37,214 | 14.1% | 科学图、医学图、论文图、常识/学科知识 |
| Math Image | 36,924 | 14.0% | 几何、图形数学、表格/公式数值推理 |
| Chart Image | 21,528 | 8.2% | 图表、地图、表格、科学 plot 的读数和推理 |
| Spatial Image | 20,284 | 7.7% | 空间关系、距离/方位/位置判断 |
| OCR Image | 15,886 | 6.0% | 场景文字、手写公式、文档/图片文字识别 |
| General Image | 14,987 | 5.7% | 通用图像问答、caption、常识问答 |

## 各数据源主要覆盖的能力

可以把 Video-R1 的数据混合理解成一张能力拼图：不同数据源不是等价样本，而是分别补不同短板。视频源主要补时序、动作、事件变化；图像源主要补密集可验证的静态推理能力，例如数学、图表、OCR、知识和空间关系。

### General Video：视频时序与事件推理

| 数据源 | 主要能力 | 数据类型特点 | 为什么需要它 |
|---|---|---|---|
| LLaVA-Video-178K | 开放域视频理解、动作/事件识别、场景变化、长短视频问答 | 真实公开视频，覆盖日常、教学、活动、YouTube 等多来源；既有多选也有开放题 | 提供最大规模的视频语言监督，让模型见到足够多的真实视频场景和自然问题 |
| STAR | 行为推理、状态变化、动作前后关系 | 真实视频 clip，多选题，问题围绕人物动作和物体交互 | 补“某人做了什么、接下来发生什么、动作如何改变状态”这类 situated reasoning |
| CLEVRER | 物理推理、因果、计数、预测 | 合成视频，物体运动/碰撞，答案通常可精确验证 | 补真实视频中较少见的可控物理因果题，也给 numerical/counting reward 提供样本 |
| NeXT-QA | 时序、因果、意图、why/how 推理 | 日常视频 QA，多选为主 | 补视频中“为什么这么做”“事件之间有什么因果关系”的问题 |
| PerceptionTest | 记忆、抽象、物理、语义、动作理解 | 人类拍摄视频，任务覆盖多个认知维度 | 补更细粒度的视频感知和泛化测试风格问题 |

### General Image：通用视觉理解与开放表达

| 数据源 | 主要能力 | 数据类型特点 | 为什么需要它 |
|---|---|---|---|
| A-OKVQA | 图像常识、外部知识推理 | 图像 + 多选/短答，需要超出像素本身的常识 | 让模型不要只做表面识别，还能结合常识回答 |
| IconQA | 抽象图形、图标、符号化视觉推理 | 图标/抽象图示上的问答 | 补非自然图像场景，训练模型理解简单符号和抽象视觉关系 |
| ShareGPT4V | 图片描述、开放式视觉对话 | GPT-4V 风格高质量 caption/instruction | 补开放回答和自然语言表达能力 |
| Visual7W | 物体、属性、位置、关系问答 | COCO 图像上的 who/what/where/when/why/how 问题 | 补基础图像 QA 覆盖面 |
| ShareGPT4o | 通用多模态描述/问答 | GPT-4o 风格开放式样本 | 补更自然的 instruction-following 和开放表达 |

### Chart：图表、表格与结构化数值推理

| 数据源 | 主要能力 | 数据类型特点 | 为什么需要它 |
|---|---|---|---|
| FigureQA | 图表关系判断、趋势比较 | 合成 figure，多选判断 | 训练模型读坐标、比较曲线/柱状关系 |
| DVQA | 柱状图读数、离散数值提取 | 合成 bar chart，数值答案 | 补图表中的 OCR-like label 读取和数值查询 |
| PlotQA | 科学 plot 读数、比较、计算 | 大规模合成 plot，数值题 | 补折线图/散点图等科学图表上的精确数值推理 |
| ChartQA | 人写图表问题、真实图表推理 | human + augmented chart QA | 提供更接近真实用户提问的图表理解题 |
| MapQA | 地图/区域图理解 | choropleth map 等地图问答 | 补地图颜色、区域、地理分布类视觉推理 |
| TabMWP | 表格数学、文本+表格联合推理 | 表格 + 数学文字题，多选/数值 | 补 tabular reasoning，尤其是从表格找数再计算 |
| Chart2Text | 图表摘要、结构化信息转文字 | 图表到文本描述 | 补图表解释和开放式描述能力 |
| RoBUT-SQA | 表格/结构化问答 | 表格或半结构化数据问答 | 补结构化信息查询和文本化回答 |
| VisualWebInstruct(filtered) | 网页/图文综合理解 | 网页截图、视觉 instruction | 补复杂版面、网页信息和开放式图文推理 |

### OCR：文字识别与文字参与推理

| 数据源 | 主要能力 | 数据类型特点 | 为什么需要它 |
|---|---|---|---|
| TextVQA | 场景文字问答 | 图片中有文字，答案依赖读字 | 让模型学会“先读图中文字，再回答问题” |
| HME100K | 手写数学公式识别 | 手写公式图片 | 补公式 OCR，和数学推理有交叉 |
| ChromeWriting | 手写/字体文字识别 | 文字图片 | 补不同字体、手写风格和短文本转写 |
| IAM | 英文手写识别 | 手写行/句/page | 补自然手写英文 OCR |
| Rendered Text | 合成渲染文字识别 | 合成文字图 | 补干净可控的文字识别样本 |
| TextCaps | 带文字的图像 caption | caption 需要读取场景文字 | 让 OCR 结果能融入自然描述，而不是只转写 |
| TextOCR | 场景文字检测/识别 | 图片文字标注丰富 | 补自然场景中的多文字区域识别 |

### Math：图形数学与数值计算

| 数据源 | 主要能力 | 数据类型特点 | 为什么需要它 |
|---|---|---|---|
| Multimath-300K | 多模态数学、算术、几何、图表数学 | 大规模图文数学题，多选/数值 | 提供图像侧最密集的可验证推理监督，是数学 bucket 主体 |
| UniGeo | 几何计算与证明 | 几何图 + 题干 | 补几何逻辑链和图形约束推理 |
| Geometry3K | 几何图题 | 标准几何问题 | 补几何 diagram 理解和选择题推理 |
| GeoQA+ | 几何问答 | 几何图 + 多选 | 补中文/几何题风格的视觉数学推理 |
| Super-CLEVR | 合成视觉计数/属性组合 | CLEVR-like 合成图像 | 补可控的组合推理和数值计数 |
| CLEVR-Math | 视觉数学、计数、简单运算 | CLEVR 风格图像数学 | 补从视觉对象集合到数值答案的映射 |
| GEOS | 小规模几何题 | 几何图 + 多选 | 小而精，补经典几何 benchmark 风格 |

### Knowledge：科学、医学、学科知识与论文图理解

| 数据源 | 主要能力 | 数据类型特点 | 为什么需要它 |
|---|---|---|---|
| ArxivQA | 论文图、科学图表、实验图理解 | arXiv figure + 问题 | 补科研图、实验装置图、论文图 caption 相关理解 |
| PMC-VQA | 医学图像问答 | 医学论文/临床图像 | 补医学视觉知识和专业术语场景 |
| AI2D | 科学 diagram 理解 | 教材式科学示意图 | 补箭头、标签、部件关系等 diagram reasoning |
| AI2D-gpt4v | 开放式科学图问答 | AI2D 的 GPT-4V 风格生成/改写 | 补科学图上的自然语言解释能力 |
| ScienceQA | 学科知识、多模态科学选择题 | 小学到高中科学题，图文混合 | 补基础科学知识和选择题推理 |
| GVLQA | 视觉知识问答 | 图像 + 知识型问题 | 补视觉实体和外部知识结合 |
| TQA | 教材图文问答 | textbook lesson、diagram、问题 | 补教材式图文联合理解 |
| EXAMS-V | 学科考试视觉题 | 考试题，覆盖多学科 | 补考试/学科推理风格 |
| VQA-RAD | 放射医学图像问答 | radiology images + QA | 补小规模但专业的医学问答 |

### Spatial：空间关系、位置和距离判断

| 数据源 | 主要能力 | 数据类型特点 | 为什么需要它 |
|---|---|---|---|
| OpenSpaces | 空间关系、方位、距离、位置估计 | reg/mc/free 混合 | 补“左/右/前/后/远/近/位置坐标/距离”一类空间推理 |
| SpaceLLaVA | 空间视觉指令跟随 | reg/mc/free 混合 | 补空间描述、空间问答和开放式解释 |

## 为什么要按能力配比

你的理解是对的：每个源聚焦的能力不同，所以不能只按数据集大小来混。Video-R1 的配比更像 curriculum/mixture design：

- 视频源负责把模型拉向动态理解：动作、事件顺序、因果、物理变化。
- 数学和 chart 源负责提供高密度、可验证的推理监督。
- OCR 源解决“看见文字但读不出来”的瓶颈。
- Knowledge 源补科学图、医学图、论文图和学科常识。
- Spatial 源补方位、距离、位置等视觉空间能力。
- General image 源保持基础视觉 QA 和开放表达能力，避免模型过度偏向考试题/图表题。

所以某些大数据集只取一部分，是为了让它们承担自己的能力角色，而不是支配整个训练分布。

## 数据源内部是否还有更细分类型

是的。很多公开数据集内部并不是一个均质分布，而是还有更细的 split、任务类型、题型、学科、视频时长或数据来源。Video-R1 取到的样本通常只是这些内部子分布中的一部分，或者是按内部子分布下采样后的结果。

本地 `data/Video-R1-260k.json` 里能直接看到一部分细分信息，主要来自 `data_source` 字段和 `problem_type` 字段。

| 数据源 | 本地可见的内部细分 | Video-R1 取样情况 | 说明 |
|---|---|---:|---|
| LLaVA-Video-178K | `0_30_s_academic_v0_1`、`2_3_m_academic_v0_1`、`30_60_s_youtube_v0_1`；题型有 `mc/free` | 82,676 | 它不是均匀取整个 178K，而是取了 academic source 和 YouTube source 中不同视频时长段的部分样本。 |
| NeXT-QA | `0_30_s_nextqa`、`30_60_s_nextqa`、`1_2_m_nextqa`、少量 `2_3_m_nextqa` | 7,549 | 本地样本按视频时长分桶，主要集中在 0-2 分钟视频，2-3 分钟只保留 22 条。 |
| EXAMS-V | Chemistry、Geography、Mathematics、Physics、Biology、History、Science | 2,000 | 明确按学科细分，Video-R1 从不同学科考试题里取样。 |
| CLEVRER | `multiple choice` 和 `numerical` | 8,220 | 同一数据源里既有多选因果/预测题，也有计数类数值题。 |
| TabMWP | `multiple choice` 和 `numerical` | 4,471 | 同样是表格数学题，但答案形式分为选择题和数值题。 |
| OpenSpaces | `regression`、`free-form`、`multiple choice` | 10,284 | 空间数据内部包含距离/尺寸估计、开放式空间描述、多选空间判断。 |
| SpaceLLaVA | `multiple choice`、`free-form`、`regression` | 10,000 | 和 OpenSpaces 类似，但更偏空间 instruction-following。 |
| AI2D / AI2D-gpt4v | AI2D 是 `mc`，AI2D-gpt4v 是 `free` | 8,000 | 同一科学 diagram 领域，分别保留选择题版本和 GPT-4V 风格开放问答版本。 |
| Multimath-300K | `numerical` 和 `multiple choice` | 27,000 | 本地字段没有保留更细年级/题型标签，但至少能看到数值题占主体。 |
| Spatial bucket | OpenSpaces + SpaceLLaVA；题型覆盖 `reg/mc/free` | 20,284 | 不是一个单一空间数据，而是把空间回归、空间选择和空间开放问答混起来。 |
| OCR bucket | TextVQA、HME100K、IAM、Rendered Text、TextCaps、TextOCR 等 | 15,886 | 每个源对应不同文字形态：场景文字、手写公式、英文手写、合成文字、caption 中的文字。 |
| Chart bucket | FigureQA、DVQA、PlotQA、ChartQA、MapQA、TabMWP、Chart2Text 等 | 21,528 | 同属 chart/table，但内部能力差很多：关系判断、读数、地图、表格数学、图表摘要。 |

因此，“用了某个数据源”不等于“用了这个数据源的全部分布”。更准确的理解是：

> Video-R1 先按能力 bucket 选源，再在 source 内部按题型、时长、学科、答案形式或数据质量做筛选/下采样。

### 本地能看到的几个具体例子

`LLaVA-Video-178K` 在本地不是一个整体标签，而是三个子来源：

```text
LLaVA-Video-178K/0_30_s_academic_v0_1      34,076
LLaVA-Video-178K/2_3_m_academic_v0_1       28,895
LLaVA-Video-178K/30_60_s_youtube_v0_1      19,705
```

这说明它既区分了视频来源，也区分了视频时长。短视频、30-60 秒视频、2-3 分钟视频提供的时序难度不同，所以不应该简单看成同一种样本。

`NeXT-QA` 也按时长拆：

```text
NeXT-QA/30_60_s_nextqa     3,017
NeXT-QA/0_30_s_nextqa      2,523
NeXT-QA/1_2_m_nextqa       1,987
NeXT-QA/2_3_m_nextqa          22
```

这说明 Video-R1 并没有均匀保留所有时长段，长视频样本明显更少，可能是出于训练成本、帧数限制和数据质量的考虑。

`EXAMS-V` 按学科拆：

```text
Chemistry      421
Geography      413
Mathematics    388
Physics        353
Biology        189
History        134
Science        102
```

这类数据的价值不只是“考试题”，还在于覆盖不同学科的图像、术语和推理方式。

`OpenSpaces` 和 `SpaceLLaVA` 按题型拆：

```text
OpenSpaces: regression 3,657 / free-form 3,574 / multiple choice 3,053
SpaceLLaVA: multiple choice 5,030 / free-form 3,287 / regression 1,683
```

这说明 spatial 数据不是单纯选择题，而是同时训练空间判断、空间描述和连续距离/尺寸估计。

### 面试时可以怎么说

如果面试官问“是不是每个数据集都只取了一部分”，可以这样说：

> 是的，而且不只是 source-level 的一部分，很多 source 内部还有更细分的子分布。比如 LLaVA-Video 会按视频来源和时长拆，NeXT-QA 也按视频时长拆，EXAMS-V 按学科拆，OpenSpaces/SpaceLLaVA 按 regression、multiple-choice、free-form 题型拆。所以取样不是简单随机切一刀，而是为了保留不同能力信号，同时控制大源和重复模板的占比。

如果继续追问“为什么不直接从一个大源里多取”，可以接着说：

> 因为同一个大源内部即使样本很多，也可能集中在某种视觉形态或问法。多源少采样能覆盖更多细分分布；source 内再按子类型控制比例，可以避免某个时长段、某个题型、某个学科或某种答案形式过度占比。

## 外部技术报告能提供什么支撑

Video-R1 的具体 source-level 配比，例如 `Multimath-300K` 取 27k、`LLaVA-Video-178K` 取 82k，并没有被论文证明为理论最优比例。但它的设计原则和 Qwen 系列、通用 VLM 技术报告里的数据 mixture 思路是一致的：多模态模型不是靠单一数据源训练出所有能力，而是通过多类型、多来源、分阶段的数据组合覆盖不同能力。

### Qwen3-VL 的相关证据

Qwen3-VL 技术报告里明确提到，预训练数据是 vision-language data 和 text-only data 的 mixture。VL 数据又包括 interleaved image-text documents、visual grounding、VQA、STEM data，以及 video data。

它的训练还会按阶段调整数据构成：

| 阶段 | 数据 mixture 思路 | 对 Video-R1 的启发 |
|---|---|---|
| Stage 1 | VL 数据包含 VQA、STEM、grounding 和少量 video，用于引入 temporal understanding | 说明视频数据并不是越多越好，早期可以作为特定能力信号加入 |
| Stage 2 | 增加 video 和 agent-oriented instruction-following 数据，用于 long-context 和复杂任务 | 说明不同阶段要根据目标能力调整数据类型比例 |
| Stage 3 | 更聚焦 long-video 和 long-document understanding | 说明当目标变成长视频/长文档时，数据分布需要进一步偏向对应能力 |

这可以支撑一个原则：如果目标能力不同，数据配比也应该不同。Video-R1 是后训练阶段，目标更窄，聚焦 image-video reasoning，所以它把数据切成 video、math、chart、OCR、knowledge、spatial、general image 这些能力 bucket。

### Qwen2.5-VL 的相关证据

Qwen2.5-VL 技术报告对数据构成说得更直接。它的预训练数据包含：

- image captions
- interleaved image-text data
- OCR data
- visual knowledge
- multi-modal academic questions
- localization data
- document parsing data
- video descriptions
- video localization
- agent-based interaction data

报告还明确说，训练过程中会 carefully adjust the composition and proportions of these data types at different stages。这说明大型 VLM 训练本身就会调数据类别比例，而不是把所有公开数据按原始规模拼起来。

Qwen2.5-VL 还提到对 interleaved image-text 数据做 scoring 和 cleaning，标准包括：

| 质量维度 | 含义 |
|---|---|
| text-only quality | 文本本身是否高质量 |
| image-text relevance | 图像和文本是否真的相关 |
| image-text complementarity | 图像和文本是否提供互补信息 |
| information density balance | 信息是否在图像和文本之间分布均衡 |

这也能支撑 Video-R1 里的一个判断：数据不是越多越好，大规模原始数据需要清洗、筛选和限量采样，否则会引入噪声、模板偏置和低价值重复样本。

### Qwen2-VL 的相关证据

Qwen2-VL 技术报告同样使用多类型数据，包括 image-text pairs、OCR、interleaved image-text articles、VQA datasets、video dialogues 和 image knowledge datasets。它还采用多阶段训练：先做视觉-语言基础对齐，再加入更广泛的数据进行综合学习，最后用 instruction 数据做微调。

这说明“多能力数据混合 + 分阶段训练”不是 Video-R1 独有做法，而是通用 VLM 训练中的常见设计。

### 能支撑什么，不能支撑什么

这些外部报告能支撑的是：

- 多模态训练需要多来源、多类型数据。
- OCR、document、chart、STEM、video、grounding、knowledge 等数据分别服务不同能力。
- 数据 mixture 会按训练阶段和目标能力调整。
- 大规模数据通常需要清洗、打分和筛选。

这些外部报告不能直接支撑的是：

- `Video-R1` 每个 source 的取样量是最优解。
- `Video-R1` 中 `General Video 44% / Math 14% / Knowledge 14%` 是理论推导出来的比例。
- 每个原始数据集内部具体按什么规则选中这些样本。

所以更准确的说法是：Qwen 系列报告支撑 Video-R1 的 mixture design 原则，但不证明 Video-R1 的每个 source cap 是唯一正确比例。

### 面试时的稳妥表述

如果被问“这个配比有没有依据”，可以这样回答：

> 它不是理论最优比例，更像经验型 mixture design。依据主要来自两个层面：第一，Qwen3-VL、Qwen2.5-VL 这类通用 VLM 的技术报告都说明，多模态模型训练需要混合 OCR、document、VQA、STEM、video、grounding、knowledge 等不同能力数据，并且会按阶段调整 composition and proportions。第二，Video-R1 自己的目标是 image-video reasoning，所以它把后训练数据按 video、math、chart、OCR、knowledge、spatial 等能力 bucket 做配平，再限制大数据源的上限，避免单一数据集支配训练。

如果被继续追问“那为什么每个数据集都只取一部分”，可以这样回答：

> 因为很多原始数据集内部模板重复很高，而且规模差异极大。比如 PlotQA、DVQA、Multimath-300K、LLaVA-Video-178K 如果全量使用，会让模型偏向某类题型或视觉风格。多数据集少采样可以覆盖更多能力分布；少数据集多采样则是在同一个分布里加厚，边际收益低，还容易过拟合数据集模板。

如果被问“有没有数据清洗”，可以这样回答：

> 已公开信息能确认至少有筛选和过滤。Video-R1 论文说他们 carefully sample and balance 各 subset，并用 Qwen2.5-VL-72B 生成 CoT，再通过 rule-based filtering 得到 165k SFT 数据。原仓库的 `generate_cot_vllm.py` 里也能看到，会解析 `<answer>`、按题型计算 reward，并用阈值筛选 CoT。至于从每个原始数据集到 260k 的完整 adapter 和清洗细节，官方没有完全开源，所以不能说完全可复现。

## 各数据源的独到之处

下面这张表更细地回答“为什么同一类能力还要用多个数据集”。核心判断标准不是它们是否都属于 Math、Chart 或 Video，而是它们提供的视觉形态、问题方式、答案格式和推理难点是否不同。

### 视频数据源

| 数据源 | 独到之处 | 和同类数据的区别 |
|---|---|---|
| LLaVA-Video-178K | 覆盖面最大，视频来源杂，问题表达更接近通用视频助手场景 | 它像“视频通用底座数据”，覆盖广但质量和题型更杂；适合提供真实视频多样性 |
| STAR | 强调人物动作、物体交互和状态变化 | 比 LLaVA-Video 更集中在 situated action reasoning，问题常要求理解动作前后关系 |
| CLEVRER | 合成物理世界，答案可控，因果链明确 | 和真实视频不同，它专门训练碰撞、运动、计数、反事实/预测类物理推理 |
| NeXT-QA | 强调 why/how 类型的视频因果和时序问答 | 比 STAR 更偏“解释为什么发生”，而不是只识别动作发生了什么 |
| PerceptionTest | 覆盖记忆、抽象、物理、语义等认知维度 | 更像视频理解综合测试集，补模型在多认知维度上的泛化 |

如果只用 LLaVA-Video，视频场景多但时序/因果信号可能不够集中；如果只用 CLEVRER，又会过于合成。几个视频源合在一起，是为了同时覆盖真实开放域、动作交互、因果解释和可控物理推理。

### 通用图像数据源

| 数据源 | 独到之处 | 和同类数据的区别 |
|---|---|---|
| A-OKVQA | 图像答案往往依赖外部常识 | 不是单纯看见物体，而是要结合生活常识、世界知识 |
| IconQA | 抽象图标和符号化图形 | 视觉形式不像自然图片，能补抽象符号和图示理解 |
| Visual7W | 基础图像 QA 覆盖 who/what/where/when/why/how | 更像通用视觉问答基本功，问题类型覆盖全面 |
| ShareGPT4V | 高质量开放式描述和视觉对话 | 主要补自然语言表达、详细描述和 instruction-following |
| ShareGPT4o | GPT-4o 风格开放式多模态响应 | 补更现代的开放式回答风格，让模型输出不只会短答案 |

这些源的差别在于：A-OKVQA 练常识，IconQA 练抽象符号，Visual7W 练基础 QA，ShareGPT4V/4o 练开放表达。能力相邻，但不重复。

### 图表和表格数据源

| 数据源 | 独到之处 | 和同类数据的区别 |
|---|---|---|
| FigureQA | 判断图表关系，如更大/更小、趋势是否一致 | 偏关系判断，适合训练视觉比较 |
| DVQA | 柱状图，常需要读 label 和数值 | 更强调 bar chart OCR-like 读取和精确数值抽取 |
| PlotQA | 折线图、散点图等 plot 上的数值和趋势推理 | 比 DVQA 更接近科学 plot，问题规模大但高度模板化，所以只取少量 |
| ChartQA | 真实图表 + 人写问题 | 比合成图表更贴近真实用户问题，语言和图表样式更自然 |
| MapQA | 地图/区域图问答 | 图表类里唯一偏地理空间分布和区域颜色理解 |
| TabMWP | 表格数学文字题 | 不是单纯看 chart，而是要从表格取数再做数学计算 |
| Chart2Text | 图表到文本摘要 | 训练把结构化视觉信息组织成自然语言，而不是只输出一个选项/数字 |
| RoBUT-SQA | 表格/结构化信息问答 | 补表格查询、跨单元格定位和结构化 QA |
| VisualWebInstruct(filtered) | 网页截图和复杂版面理解 | 图表/表格之外，补真实网页布局、混合文本和视觉组件 |

Chart bucket 里最容易误以为重复，但其实差异很大：DVQA/PlotQA 偏精确读数，FigureQA 偏关系比较，ChartQA 偏真实问题，TabMWP 偏表格数学，Chart2Text 偏解释和摘要。

### OCR 数据源

| 数据源 | 独到之处 | 和同类数据的区别 |
|---|---|---|
| TextVQA | 读场景文字并回答问题 | 文字只是中间证据，最终要完成 QA |
| HME100K | 手写数学公式 | OCR 内容是公式，不是普通自然语言 |
| ChromeWriting | 手写/字体文字样本 | 补不同书写风格和短文本识别 |
| IAM | 真实英文手写行/句/page | 比合成文字更自然，适合手写英文转写 |
| Rendered Text | 合成渲染文字 | 干净、可控，适合补基础文字识别覆盖 |
| TextCaps | 含文字图片的 caption | 训练“读到文字后融入描述”，不是只转写 |
| TextOCR | 场景文字密集标注 | 适合补自然场景中多文字区域识别 |

OCR 不是一种能力那么简单：普通场景文字、手写英文、手写公式、合成文字、文字参与问答、文字参与 caption，都会触发不同错误模式。

### 数学数据源

| 数据源 | 独到之处 | 和同类数据的区别 |
|---|---|---|
| Multimath-300K | 大规模多模态数学，题型覆盖广 | 数量最大，是数学能力主干，但需要下采样避免支配训练 |
| UniGeo | 几何计算和证明风格 | 更强调几何约束、定理关系和多步推导 |
| Geometry3K | 标准几何图题 | 题目更接近经典 benchmark，适合补图形几何基本功 |
| GeoQA+ | 几何问答 | 与 Geometry3K/UniGeo 互补，提供另一套几何题风格 |
| Super-CLEVR | 合成对象组合、计数和属性推理 | 比几何题更偏组合推理和视觉计数 |
| CLEVR-Math | CLEVR 风格视觉数学 | 从对象集合、属性、数量到数值答案的映射更直接 |
| GEOS | 小规模经典几何题 | 规模小但质量集中，保留比例高，用来补经典几何分布 |

数学 bucket 不是只有“算数”。它拆成几何图推理、对象计数、组合属性、公式/图文数学和 benchmark 风格题。Multimath 提供广度，Geo/Geometry 系列提供几何深度，CLEVR 系列提供可控计数和组合推理。

### 知识数据源

| 数据源 | 独到之处 | 和同类数据的区别 |
|---|---|---|
| ArxivQA | 论文图、实验装置图、科研 figure | 补模型理解论文图和科学可视化的能力 |
| PMC-VQA | 医学图像和医学知识 | 专业术语多，视觉分布和自然图像差别很大 |
| AI2D | 教材科学示意图 | 强调箭头、标签、部件、过程图等 diagram reasoning |
| AI2D-gpt4v | 开放式科学图解释 | 和 AI2D 的选择题互补，补自然语言解释能力 |
| ScienceQA | 学科知识选择题 | 覆盖基础科学常识和多模态考试题风格 |
| GVLQA | 视觉实体与知识问答 | 补视觉识别和知识检索结合 |
| TQA | 教材 lesson + diagram + 问题 | 比 ScienceQA 更接近教材上下文和教学图 |
| EXAMS-V | 多学科考试视觉题 | 补正式考试题语言和学科跨度 |
| VQA-RAD | 放射医学问答 | 小规模但专业，补 radiology 分布 |

Knowledge bucket 的独特性在于“视觉内容本身携带专业知识”：科学图、医学图、教材图、论文图、考试题的视觉风格和语言风格都不一样。

### 空间数据源

| 数据源 | 独到之处 | 和同类数据的区别 |
|---|---|---|
| OpenSpaces | 空间关系、位置、距离估计 | 更偏空间判断和可量化位置/距离问题 |
| SpaceLLaVA | 空间 instruction-following | 更偏把空间关系用自然语言问答和解释出来 |

Spatial bucket 和普通图像 QA 的区别是，它不只问“是什么”，而是问“在哪里、离多远、相对谁、朝哪个方向”。这类问题对视觉定位和几何关系更敏感。

## 原始数据量与 Video-R1 取用量

说明：

- `原始公开规模` 尽量按原论文/数据卡写 QA 对数；如果原数据主要按图片/视频数发布，会在括号里注明。
- `Video-R1 取用量` 是本地 `Video-R1-260k.json` 实扫数量，不是图片里的四舍五入数。
- `取用比例` 只在原始规模和取用量口径大致一致时给出；口径不一致时用 `-`。
- `待核验` 表示公开来源版本较多，或 Video-R1 论文没有给出它采用的精确原始池大小。

### General Video

| 数据源 | 原始公开规模 | Video-R1 取用量 | 取用比例 | 题型 | 说明 |
|---|---:|---:|---:|---|---|
| LLaVA-Video-178K | 178,510 videos；约 960,792 open-ended QA；196,198 MC QA | 82,676 | - | mc/free | 最大视频源。Video-R1 只取其中约 83k QA，按时长子集分为 `0_30_s`、`30_60_s`、`2_3_m` 等。 |
| STAR | 约 60k questions；22k video clips | 11,455 | 19.1% | mc | 真实世界行为/事件 situated reasoning。 |
| CLEVRER | 20k synthetic videos；超过 300k QA | 8,220 | 约 2.7% | mc/num | 合成物理碰撞场景，偏因果、计数、预测。 |
| NeXT-QA | 5,440 videos；52,044 open-ended QA；47,692 MC QA | 7,549 | 约 14.5% vs 52k QA | mc | 日常视频因果/时序/描述问答。 |
| PerceptionTest | 11.6k videos；38k mc-vQA；6k grounded vQA | 6,348 | 16.7% vs 38k mc-vQA | mc | 记忆、抽象、物理、语义等视频感知题。 |

### General Image

| 数据源 | 原始公开规模 | Video-R1 取用量 | 取用比例 | 题型 | 说明 |
|---|---:|---:|---:|---|---|
| A-OKVQA | 约 25k questions | 3,999 | 16.0% | mc | 需要常识/世界知识的图像问答。 |
| IconQA | 107,439 questions | 2,000 | 1.9% | mc | 抽象图标/图示推理。 |
| ShareGPT4V | 100k GPT-4V captions；扩展版 1.2M captions | 1,988 | 2.0% vs 100k | free | 高质量图片描述/开放回答。 |
| Visual7W | 327,939 QA；47,300 images | 1,000 | 0.3% | mc | COCO 图像上的 7W 问答。 |
| ShareGPT4o | 公开版本规模不稳定，HF 标为 10K-100K | 6,000 | - | free | GPT-4o 生成的高质量多模态描述/问答。 |

### Chart

| 数据源 | 原始公开规模 | Video-R1 取用量 | 取用比例 | 题型 | 说明 |
|---|---:|---:|---:|---|---|
| FigureQA | 超过 1M QA；超过 100k figures | 1,996 | 约 0.2% | mc | 合成科学图表的关系判断。 |
| DVQA | 3,487,194 QA；300k charts | 1,744 | 0.05% | num | bar chart 读数和推理。 |
| PlotQA | 28.9M QA；224,377 plots | 953 | 0.003% | num | 科学 plot 上的读数/比较/推理。 |
| ChartQA | 9.6k human questions + 23.1k generated questions | 1,033 | 3.2% | num | 人写/生成图表问答。 |
| MapQA | 约 800k QA；约 60k choropleth maps | 1,331 | 0.17% | mc | 地图/分区图理解。 |
| TabMWP | 38,431 tabular math word problems | 4,471 | 11.6% | mc/num | 表格数学文字题。 |
| Chart2Text | 待核验 | 2,000 | - | free | 图表到文本摘要。 |
| RoBUT-SQA | 待核验 | 2,000 | - | free | 表格/结构化 QA，Video-R1 中归入 chart/table 类。 |
| VisualWebInstruct(filtered) | 待核验 | 6,000 | - | free | 网页/图文 instruction 数据中过滤出的视觉推理样本。 |

### OCR

| 数据源 | 原始公开规模 | Video-R1 取用量 | 取用比例 | 题型 | 说明 |
|---|---:|---:|---:|---|---|
| TextVQA | 45,336 questions；28,408 images | 1,886 | 4.2% | ocr | 场景文字理解问答。 |
| HME100K | 约 100k handwritten expression images | 4,000 | 4.0% | ocr | 手写数学公式识别。 |
| ChromeWriting | 待核验 | 2,000 | - | ocr | 手写/文字识别数据。 |
| IAM | 1,539 pages；5,685 sentences；13,353 lines | 2,000 | - | ocr | 英文手写识别。 |
| Rendered Text | 待核验 | 2,000 | - | ocr | 合成渲染文字识别。 |
| TextCaps | 142k-145k captions；28k images | 2,000 | 约 1.4% | ocr | 需要读图中文字的 caption。 |
| TextOCR | 28,134 images；约 903k word annotations | 2,000 | - | ocr | 场景文字检测/识别。 |

### Math

| 数据源 | 原始公开规模 | Video-R1 取用量 | 取用比例 | 题型 | 说明 |
|---|---:|---:|---:|---|---|
| Multimath-300K | 约 300k samples | 27,000 | 9.0% | mc/num | 最大图像数学源，明显下采样以避免主导训练。 |
| UniGeo | 4,998 calculation + 9,543 proving problems | 3,357 | 23.1% | mc | 几何逻辑推理。 |
| Geometry3K | 3,002 geometry problems | 2,000 | 66.6% | mc | 几何图题。 |
| GeoQA+ | 待核验 | 2,000 | - | mc | 几何问答。 |
| Super-CLEVR | 待核验 | 1,288 | - | num | 合成视觉数学/计数推理。 |
| CLEVR-Math | 待核验 | 1,118 | - | num | CLEVR 风格视觉数学。 |
| GEOS | 约 186 geometry problems | 161 | 86.6% | mc | 小规模几何题，基本保留。 |

### Knowledge

| 数据源 | 原始公开规模 | Video-R1 取用量 | 取用比例 | 题型 | 说明 |
|---|---:|---:|---:|---|---|
| ArxivQA | 待核验 | 10,000 | - | mc | 论文图/科学图问答。本地该源 `data_source` 为空，需要按路径识别。 |
| PMC-VQA | 227k VQA；149k medical images | 5,982 | 2.6% | mc | 医学图像问答。 |
| AI2D | 约 5k diagrams；约 15k questions；官方页也列 4,903 images / 4,563 questions | 4,000 | - | mc | 科学 diagram 理解。 |
| AI2D-gpt4v | 基于 AI2D 的 GPT-4V 风格改写/生成子集 | 4,000 | - | free | 开放式科学图问答。 |
| ScienceQA | 21,208 multimodal MC questions | 3,999 | 18.9% | mc | 小学到高中科学题，带图/文本上下文。 |
| GVLQA | 待核验 | 3,811 | - | mc | 视觉知识问答。 |
| TQA | 1,076 lessons；26,260 questions；6,229 images | 3,000 | 11.4% | mc | textbook QA，多模态教材题。 |
| EXAMS-V | 待核验 | 2,000 | - | mc | 学科考试视觉题。 |
| VQA-RAD | 315 radiology images；3,515 visual questions | 422 | 12.0% | mc | 医学 radiology QA，小数据集。 |

### Spatial

| 数据源 | 原始公开规模 | Video-R1 取用量 | 取用比例 | 题型 | 说明 |
|---|---:|---:|---:|---|---|
| OpenSpaces | 待核验 | 10,284 | - | reg/mc/free | 空间关系、位置、距离等视觉空间推理。 |
| SpaceLLaVA | 待核验 | 10,000 | - | reg/mc/free | 空间推理 instruction 数据。 |

## 为什么只取一部分

Video-R1 的取样不是按原始数据集大小成比例采样，而是按训练目标反推配比：

1. 避免大源支配训练。PlotQA、DVQA、PMC-VQA、Multimath-300K、LLaVA-Video-178K 都很大，如果全量使用，会让模型偏向少数题型。
2. 保持能力 bucket 大致平衡。图像侧分成 Math、Chart、OCR、Knowledge、Spatial、General；视频侧统一作为 General Video。
3. 优先保留可规则判分样本。RL 阶段需要 rule-based reward，所以 `mc`、`num`、`ocr`、`reg` 占比较高。
4. SFT 还会再经过 CoT teacher 过滤。`Video-R1-COT-165k` 是从 260k 用 Qwen2.5-VL-72B 生成 CoT，再按答案正确性/格式过滤后的子集，不是 260k 的均匀抽样。

## 本地统计命令

```bash
python - <<'PY'
import json
from collections import Counter

rows = json.load(open("data/Video-R1-260k.json", encoding="utf-8"))

def source(row):
    raw = row.get("data_source") or ""
    parts = row.get("path", "").lstrip("./").split("/")
    if raw:
        return raw.split("/")[0]
    return parts[1] if len(parts) >= 2 else "<blank>"

print("total", len(rows))
print(Counter(row["data_type"] for row in rows))
print(Counter(row["problem_type"] for row in rows))
for name, count in Counter(source(row) for row in rows).most_common():
    print(name, count)
PY
```

## 参考来源

- Video-R1 paper: https://arxiv.org/html/2503.21776v2
- Video-R1 data card / local mirror: `data/README.md`
- Qwen3-VL Technical Report: https://arxiv.org/pdf/2511.21631
- Qwen2.5-VL Technical Report: https://arxiv.org/pdf/2502.13923
- Qwen2-VL Technical Report: https://arxiv.org/pdf/2409.12191
- LLaVA-Video-178K data card: https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K
- NExT-QA paper: https://openaccess.thecvf.com/content/CVPR2021/papers/Xiao_NExT-QA_Next_Phase_of_Question-Answering_to_Explaining_Temporal_Actions_CVPR_2021_paper.pdf
- Perception Test paper: https://proceedings.neurips.cc/paper_files/paper/2023/file/8540fba4abdc7f9f7a7b1cc6cd60e409-Paper-Datasets_and_Benchmarks.pdf
- STAR paper: https://arxiv.org/html/2405.09711v1
- CLEVRER paper: https://ar5iv.labs.arxiv.org/html/1910.01442
- A-OKVQA paper: https://arxiv.org/pdf/2206.01718
- IconQA paper: https://arxiv.org/abs/2110.13214
- Visual7W paper: https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhu_Visual7W_Grounded_Question_CVPR_2016_paper.pdf
- ShareGPT4V paper: https://arxiv.org/html/2311.12793
- FigureQA: https://www.microsoft.com/en-us/research/publication/figureqa-an-annotated-figure-dataset-for-visual-reasoning/
- DVQA: https://kushalkafle.com/projects/dvqa.html
- PlotQA: https://github.com/NiteshMethani/PlotQA
- MapQA: https://arxiv.org/abs/2211.08545
- TextVQA: https://textvqa.org/
- TextCaps: https://arxiv.org/abs/2003.12462
- TextOCR: https://textvqa.org/textocr/dataset/
- AI2D: https://prior.allenai.org/projects/diagram-understanding
- ScienceQA: https://scienceqa.github.io/
- PMC-VQA: https://www.nature.com/articles/s43856-024-00709-2
- VQA-RAD: https://www.nature.com/articles/sdata2018251
- TQA: https://registry.opendata.aws/allenai-tqa/
- TabMWP: https://promptpg.github.io/
- UniGeo: https://github.com/chen-judge/UniGeo
- Geometry3K: https://arxiv.org/pdf/2105.04165
