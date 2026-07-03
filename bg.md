# TimeThinker：视频多模态推理模型后训练

## 项目定位

**最近在做的研究方向探索**（还没做完）。目标：

> 在开源 4B/7B 多模态模型上，通过 SFT cold-start + RL（GRPO / T-GRPO）让它具备显式可控的视频/图像推理能力。

时间最多花在**数据工作**上，下面以数据为主线展开。

---

## 研究动机

1. **开源 instruct VLM 在多步推理上短板明显**：Qwen2.5-VL、LLaVA-Video 描述视频没问题，但事件时序、数值计算、图表理解、OCR+推理这些任务都不稳。
2. **Thinking 版本输出格式不可控**：长 CoT 自由产出，规则化判分场景不友好。
3. **DeepSeek-R1 启发的路线**：SFT 教格式 + 推理轨迹 → RL 用规则可判分的 reward 优化答案正确性。这条路线在文本上验证有效，多模态视频推理还是个开放问题——直接套 GRPO 有两个坑：缺时序建模 reward（模型走单帧捷径），高质量数据稀缺。

---

## 数据工作（重点）

### 1. 数据集选择：image-video 混合而非纯视频

| 方案 | 优点 | 致命问题 |
|------|------|----------|
| 纯视频 | 任务匹配 | 数据稀缺，监督稀疏，RL 易崩 |
| 纯图像 | 监督密度高、可判分 | 没有时序能力 |
| **image-video 混合 ✅** | 图像稳推理，视频补时序 | 配比要调，格式要统一 |

### 2. 原始数据集汇总（20+ 来源）

**视频（5 个）**：LLaVA-Video-178K（80k）/ NeXT-QA（7.5k）/ PerceptionTest（6k）/ CLEVRER（8k）/ STAR（11.5k）

**图像（按能力分 6 个 bucket）**：
- **Math (~37k)**：Multimath-300k / Geometry3K / GeoQA+ / UniGeo / CLEVR-Math / GEOS / Super-CLEVR
- **Chart (~21k)**：FigureQA / DVQA / PlotQA / ChartQA / MapQA / TabMWP / Chart2Text / RoBUT-SQA / VisualWebInstruct
- **Knowledge (~37k)**：TQA / AI2D / ScienceQA / PMC-VQA / VQA-RAD / GVLQA / ArxivQA / EXAMS-V
- **OCR (~16k)**：TextVQA / HME100k / ChromeWriting / IAM / Rendered-Text / TextCaps / TextOCR
- **Spatial (~20k)**：OpenSpaces / Spacellava
- **General (~15k)**：A-OKVQA / IconQA / ShareGPT4V / Visual7W / ShareGPT4o

选择标准：**带明确 ground truth、能用规则判分**。

### 3. Per-source 采样

不能直接每个数据集全量拉，比如 Multimath 全量 30 万，全拉一是过大、二是会主导梯度。采样分三层做。

#### 第一层：给每个 source 定 budget cap

原则：
- 稀缺数据多保留（NeXT-QA / CLEVRER / STAR 全留）
- 同质化数据多压缩（Multimath 30 万 → 27k）
- 大数据集按需下采（LLaVA-Video 178k → 80k）
- 按能力 bucket 均衡（6 个图像 bucket 目标量级接近）

形成一张 `SOURCE_BUDGET` 表，total budget 约 35 万。

#### 第二层：source 内 stratified sampling

直接 `random.sample` 是错的——数据集内部有子类型/难度/质量分层，随机采会让某些子类型消失。三个典型例子：

**LLaVA-Video-178K**：按 `subset × 题型` 两层分层
```python
def sample_llava_video(samples, target=80_000):
    by_subset = group_by(samples, "subset")
    out = []
    for subset, group in by_subset.items():
        subset_budget = int(target * len(group) / len(samples))
        # 在 subset 内再按 mc / free-form 分层
        for qtype, q_group in group_by(group, "problem_type").items():
            q_budget = int(subset_budget * len(q_group) / len(group))
            out.extend(rng.sample(q_group, min(q_budget, len(q_group))))
    return out
```

**Multimath-300k**：按 grade level 加权（高难度题更有价值）
```python
WEIGHT = {"primary": 1.0, "junior": 1.5, "senior": 2.5}
# 高中题虽然量少但权重高，确保不被淹没
```

**ChartQA**：质量分层 → 只留 human subset，augmented（模板生成）全丢；human 内再按 chart_type (pie/bar/line) 均衡

#### 第三层：post-sampling 验证

采完跑一个 stats dashboard 看分布有没有崩：

```python
def verify_sampling(merged_data):
    print("per-source:", Counter(s["data_source"] for s in merged_data))
    print("per-modality:", Counter(s["data_type"] for s in merged_data))
    print("per-problem_type:", Counter(s["problem_type"] for s in merged_data))
    print("per-bucket:", Counter(infer_bucket(s) for s in merged_data))
    assert 0.4 < img_ratio < 0.6, "modality 失衡"
    assert mc_ratio < 0.7, "多选占比过高"
```

崩了回去调 budget 重采。前后迭代了 5 轮才定下来。

**几个细节**：
- 固定 seed=42 保证可复现
- 采样在清洗**之前**做粗采（拉 1.5 倍 budget），清洗后再精采到目标分布
- nested 数据按整体采（CLEVRER 一个 video 多 questions 不能拆）

### 4. 数据清洗

原始 35 万 → 清洗后 26 万。每一步的具体做法：

#### (1) Schema 适配：20+ 个 per-source adapter

每个数据集字段命名、答案格式、路径方式都不同，每个写一个 adapter 函数统一到内部 schema。两个典型示例：

**NeXT-QA**：答案是文本，选项散在 a0-a4
```python
def adapter_nextqa(row):
    options = [row[f"a{i}"] for i in range(5)]
    try:
        ans_idx = options.index(row["answer"])  # 文本反查选项
    except ValueError:
        return None  # 标注错位
    return {
        "problem": row["question"],
        "options": [f"{chr(65+i)}. {o}" for i, o in enumerate(options)],
        "solution": f"<answer>{chr(65+ans_idx)}</answer>",
        ...
    }
```

**CLEVRER**：nested 结构，一个 video 多 questions
```python
def adapter_clevrer(scene):
    for q in scene["questions"]:
        correct = [c for c in q["choices"] if c["answer"] == "correct"]
        if len(correct) != 1:  # 没有 correct 或多个 correct
            continue
        yield {"problem_id": f"clevrer_{scene['scene_index']}_{q['question_id']}", ...}
```

#### (2) problem_type 归一化

`free-form` / `Free-form` / `freeform` → 统一 `open-ended`；`num` / `numerical` / `number` → 统一 `numerical`。如果不归一化，RL reward function 路由失败，样本 reward 全是 0。

#### (3) 媒体校验：两步检查

```python
def check_media(path, data_type):
    if not path.is_file() or path.stat().st_size < 1024:
        return False  # 不存在或 < 1KB（损坏）
    if data_type == "video":
        # ffprobe 比 cv2 严格：cv2 能打开的损坏文件抽帧时才报错
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(path)],
                           capture_output=True, timeout=10)
        if r.returncode != 0: return False
        nb_frames = int(r.stdout.decode().split(",")[0])
        if nb_frames < 8: return False  # 训练 16 帧，至少 8 帧才有意义
    return True
```

最初媒体丢失率 30%（zip 解压中断、Open Images 缺图），按 `data_source` 分桶定位后分批补齐，压到 8%。**丢约 2.8 万。**

#### (4) 字段一致性硬过滤

```python
def validate(s):
    pt, ans, q = s["problem_type"], strip_tags(s["solution"]), s["problem"]
    if len(q.strip()) < 5: return "question_too_short"
    if ans.lower() in {"unanswerable", "n/a", "none", "don't know"}: return "no_answer"
    if len(ans) > 200 and pt != "open-ended": return "answer_too_long"
    
    if pt == "multiple choice":
        opts = s.get("options") or []
        if len(opts) < 2: return "options_missing"
        if not re.fullmatch(r"[A-Z]", ans) or ord(ans) - ord("A") >= len(opts):
            return "answer_out_of_options"
    elif pt == "numerical":
        m = re.search(r"-?\d+(?:\.\d+)?", ans)
        if not m: return "answer_not_numeric"
        if abs(len(m.group(0)) - len(ans.strip())) > 5:
            return "answer_has_extra_text"  # 比如 "approximately 5 cats"
    return None
```

**丢约 1.5 万。**

#### (5) 文本清洗

```python
def clean_text(s):
    s = unicodedata.normalize("NFKC", s)              # 全角→半角
    s = re.sub(r"[​-‏﻿]", "", s)        # 零宽字符
    s = re.sub(r"</?(br|sub|sup|i|b|em|p)\s*/?>", " ", s, flags=re.I)  # HTML
    s = re.sub(r"<image\s*\d*>|<vid>|\[IMG\]", "", s, flags=re.I)  # 多模态 placeholder
    s = s.replace("\\\\", "\\")                        # LaTeX 转义层数
    return re.sub(r"\s+", " ", s).strip()
```

#### (6) 答案格式统一

| 原始 | 统一为 |
|------|--------|
| `"A"` / `"(A)"` / `"A."` / `"A. xxx"` | `<answer>A</answer>` |
| `"23 million"` / `"$23M"` / `"57.41%"` | `<answer>23</answer>` / `<answer>57.41</answer>` |
| 多人投票 `["samsung", "Samsung", ...]` | majority vote + 小写 → `<answer>samsung</answer>` |
| 选项 index 0/1/2/3 | 映射字母 → `<answer>B</answer>` |
| 文本答案 match 选项 | 反查 options 拿字母 |

OCR 多 annotator 投票：

```python
def normalize_ocr_answer(answers):
    votes = Counter(a.strip().lower() for a in answers if a.strip())
    top, count = votes.most_common(1)[0]
    if count < 2: return None  # 没人达成共识 → 丢
    return top
```

#### (7) 去重：两层

```python
# Layer 1: 精确去重 (problem + media path)
key = (s["problem"].strip().lower(), s["path"])

# Layer 2: 近似去重 MinHash LSH (字符 3-gram, 阈值 0.92)
# 限制在同 media 内做相似度匹配，避免误判
```

阈值刚开始用 0.85，发现"What is happening" / "Describe the scene"被合并（不该合），调到 0.92 + 同 media 限制。**丢约 1.5 万**（数据集内部重复 1.2 万 + 跨数据集 0.3 万）。

#### (8) 题型分布平衡

汇总后多选占 75%+（LLaVA-Video 一家贡献 8 万多选）。做 stratified 下采样：

```python
TARGET = {"multiple choice": 0.65, "numerical": 0.13, "free-form": 0.15,
          "OCR": 0.06, "regression": 0.01}
# 下采样按 data_source 分层采，避免某来源被全砍
```

**丢约 1.9 万多选题**（主要 LLaVA-Video 冗余）。

#### 清洗漏斗

```
原始 (per-source 采样后)    ~350k
 ├─ adapter/归一化异常       -10k
 ├─ 媒体缺失/损坏/帧数不足   -28k
 ├─ 字段一致性硬过滤         -15k
 ├─ 去重 (精确 + MinHash)    -15k
 ├─ 题型分布下采样           -19k
 └─ 文本清洗变空             -0.2k
RL 训练池                    263k
 └─ CoT 生成 + 规则过滤      -95k (37%)
SFT cold-start               165k
```

### 5. CoT cold-start 数据生成

绝大多数原始数据集**只有问题和答案，没有 `<think>` 过程**，这是 SFT 阶段最大缺口。

- 用 **Qwen2.5-VL-72B-Instruct** 当 teacher，对 26 万 RL 样本生成 CoT
- prompt 要求严格按 `<think>...</think><answer>...</answer>` 输出
- 三层规则过滤：
  1. 必须能从 `<answer>` 解析出答案
  2. 解析答案必须和 ground truth 一致（错答轨迹丢，避免学错误推理）
  3. 长度异常（> 4096 token）、明显幻觉（重复字符串、空 think）丢
- 通过率 63%（26 万 → 16.5 万）

### 6. 6 个代表性数据集：原始 vs 转换后

#### ① LLaVA-Video-178K（视频多选）

**原始**：
```json
{
  "video": "academic_source/youcook2/.../dHcZIwgs7H8.mp4",
  "conversations": [
    {"from": "human", "value": "<image>\nWhat is the relationship between the chicken and the rice...?\nA. ... B. ... C. ... D. ...\nAnswer with the option's letter from the given choices directly."},
    {"from": "gpt", "value": "B"}
  ]
}
```

**转换后（RL）**：
```json
{
  "problem": "What is the relationship between the chicken and the rice...?",
  "answer": "<answer>B</answer>",
  "problem_type": "multiple choice",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "videos": ["data/LLaVA-Video-178K/.../dHcZIwgs7H8.mp4"],
  "data_source": "LLaVA-Video-178K/30_60_s_academic_v0_1"
}
```
清洗动作：去掉问题里的 `<image>` placeholder、抽出 options 字段、答案包 `<answer>` tag、去掉数据集自带的 instruction、路径重映射。

#### ② NeXT-QA（视频因果，CSV）

**原始**：
```csv
video,question,answer,qid,a0,a1,a2,a3,a4
2384280571,why did the lady move her hand near the end,to point at something,DC_2384280571_8,to clap,to point at something,to take a photo,to wave,to drink
```

**转换后**：
```json
{
  "problem": "why did the lady move her hand near the end",
  "answer": "<answer>B</answer>",
  "options": ["A. to clap", "B. to point at something", "C. to take a photo", "D. to wave", "E. to drink"],
  "videos": ["data/NeXT-QA/NExTVideo/2384280571.mp4"]
}
```
清洗动作：答案是文本要先在 a0-a4 里 match 出 index 再转字母；选项散字段要合并；video ID 拼路径。

#### ③ CLEVRER（nested 视频）

**原始**：
```json
{
  "scene_index": 14000,
  "questions": [{
    "question_id": 0,
    "question": "How many collisions happen?",
    "choices": [
      {"choice_id": 0, "choice": "0", "answer": "wrong"},
      {"choice_id": 1, "choice": "1", "answer": "correct"},
      ...
    ]
  }]
}
```

**转换后**：
```json
{
  "problem_id": "clevrer_14000_0",
  "problem": "How many collisions happen?",
  "answer": "<answer>B</answer>",
  "options": ["A. 0", "B. 1", "C. 2", "D. 3"]
}
```
清洗动作：nested 展开成独立样本；从 choices 里找 correct 那个的 choice_id 转字母；数字 choice 不能误判成 numerical。

#### ④ ChartQA（图表数值）

**原始**：`{"imgname": "...", "query": "...", "label": "57.41", "human_or_machine": "human"}`

**转换后**：
```json
{
  "problem": "What was Apple's net income in 2020?",
  "answer": "<answer>57.41</answer>",
  "problem_type": "numerical",
  "images": ["data/Chart/ChartQA/png/two_col_103925.png"]
}
```
清洗动作：只留 human subset（augmented 是模板生成的，质量差）；label 去掉单位/货币符号。

#### ⑤ Geometry3K（几何多选）

**原始**：每题一个目录，json 答案是选项 index 数字（0/1/2/3），选项是纯数字字符串。

**转换后**：`answer: <answer>B</answer>`, `options: ["A. 22.5", "B. 24.5", ...]`

清洗动作：index 转字母；选项拼字母前缀；带辅助标注的 `img_diagram_point.png` 不用（会让模型 cheat），只用 `img_diagram.png`。

#### ⑥ TextVQA（OCR）

**原始**：`{"question": "...", "image_id": "...", "answers": ["samsung", "samsung", "Samsung", ...]}`（10 个 annotator）

**转换后**：`answer: <answer>samsung</answer>`, `problem_type: "OCR"`

清洗动作：majority voting（全部小写后投票）；`unanswerable` 直接丢。

#### 统一目标 schema

**RL（EasyR1）**：
```json
{
  "problem_id", "problem", "answer", "data_type", "problem_type",
  "options", "images" | "videos", "data_source"
}
```

**SFT（LLaMA-Factory ShareGPT）**：按 modality 拆 image / video 两个文件
```json
{
  "messages": [
    {"role": "user", "content": "<image|video>\n问题\n选项\n通用 think/answer instruction\n题型专属答案格式 instruction"},
    {"role": "assistant", "content": "<think>CoT</think>\n<answer>答案</answer>"}
  ],
  "images" | "videos": [...]
}
```

按题型设计了答案格式 instruction 模板（mc → 字母、numerical → 数字、OCR → 文本…），保证 SFT 训练和 RL rollout 的 prompt 严格一致。

### 7. 最终数据规模

```
SFT cold-start: 165K   image 79k / video 86k
RL training:    260K   image 147k / video 116k

题型: mc 65% / free-form 15% / numerical 13% / OCR 6% / regression 1%
```

---

## 模型方案

**Base：Qwen2.5-VL-7B-Instruct（不用 Thinking）**

理由：从"不会推理"的 Instruct 起步，能干净归因 SFT / RL 的贡献；Thinking 已有推理偏好，再训等于和模型自身偏好打架。

**Reward 设计**：

| 题型 | Reward |
|------|--------|
| mc | 选项精确匹配 |
| numerical | 数值精确匹配（带容差） |
| OCR | 1 - WER |
| regression | 1 - 相对误差 |
| free-form | ROUGE-1/2/L 平均 |

视频样本额外引入 **T-GRPO temporal reward**：同问题跑顺序帧 + 乱序帧两组 rollout，只有顺序组正确率高于乱序组时才给 temporal reward，强制模型用时序信息而不是抄单帧捷径。

外加 length reward（鼓励 320-512 token）防止 overthinking / underthinking。

**训练栈**：LLaMA-Factory (SFT) + EasyR1 (RL, T-GRPO) + VLMEvalKit。双 venv 隔离 vllm / flash-attn / transformers 版本冲突。

---

## 评估

完全沿用 **Video-R1 官方评估方式**：在 6 个公开视频 benchmark 上跑统一的推理脚本，按 problem_type 路由 metric，最终报 accuracy。简单、可复现、能直接对齐 Video-R1 / Qwen2.5-VL-Instruct / GPT-4o 的横向对比。

### 1. Benchmark：6 个公开视频榜单

跟 Video-R1 论文 Table 1 完全一致：

| Benchmark | 类型 | 主要考察 |
|-----------|------|----------|
| **VSI-Bench** | 视频空间推理 | 空间位置 / 距离 / 房间布局推理（regression 题多）|
| **VideoMMMU** | 视频多学科 | 大学专业知识 + 视频理解 |
| **MMVU** (mc) | 视频多学科 | 多学科推理（只用 multiple-choice 子集，保证稳定性）|
| **MVBench** | 通用视频理解 | 20 种细粒度感知 / 推理任务 |
| **TempCompass** | 时序理解 | 速度 / 方向 / 顺序 / 属性变化 |
| **VideoMME** (wo sub) | 综合视频 | 不带字幕的视频问答 |

> 前三个偏视频**推理**，后三个偏视频**通用理解**，覆盖训练目标的两个面。Evaluation JSON 来自 HuggingFace `Video-R1-eval`，视频要从各 benchmark 官方站点单独下载。

### 2. 推理配置

```python
LLM(model_path,
    tensor_parallel_size=torch.cuda.device_count(),
    max_model_len=8192*2,
    gpu_memory_utilization=0.8,
    limit_mm_per_prompt={"image": 1, "video": 1})

# 采样参数（near-greedy，跟 Qwen2.5-VL 官方 demo 对齐）
SamplingParams(temperature=0.1, top_p=0.001, max_tokens=1024)

# 视频
frames in {16, 32, 64}          # 训练时 16，推理放大
resolution = 256 x 28 x 28      # 训练时 128x28x28，推理放大
```

为什么 `top_p=0.001` 而不是 `top_p=1`：Video-R1 README 明确警告——`top_p` 大了 Qwen2.5-VL 会输出乱码，官方 demo 也是这个配置。

为什么推理时 frame 数比训练大：训练受 batch / 显存约束只能 16 帧，推理是单样本 batch=1 能塞更多帧。论文 Table 1 数据显示 16 → 32 → 64 在 6 个 benchmark 上都涨点，特别是长视频（VideoMME 16→64 涨 4 个点）。

### 3. Prompt 构建（和训练严格一致）

```python
QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please answer this question based on the visual content."
    "Provide your thinking process between the <think> and </think> tags, "
    "and then give your final answer between the <answer> and </answer> tags."
    "At the end, you must output the final answer in the format:\n"
    "<answer><your_answer_here></answer>\n"
)
```

多选题前面拼 `Options:\n` 列表；按 problem_type 再追加 type-specific 答案格式 instruction（多选→单字母、numerical→数字、regression→数字…）。

**评估 prompt 跟训练时 prompt 逐字一致**，避免 train-test mismatch。

### 4. 按 problem_type 路由的 metric

跟训练时 reward function 完全一一对应——同一套打分逻辑，训练用就是 reward，评估用就是 metric。

| problem_type | metric | 实现 |
|--------------|--------|------|
| **multiple choice** | 精确字母匹配 | 抽 `<answer>` 取首字母大写 vs GT → 1.0 / 0.0 |
| **numerical** | 保留两位小数后精确匹配 | `round(pred, 2) == round(gt, 2)` → 1.0 / 0.0 |
| **regression** | **MRA**（Mean Relative Accuracy）| 阈值 theta in {0.5, 0.55, …, 0.95} 共 10 个，统计 `|pred-gt|/|gt| < 1-theta` 的命中率求平均 |
| **OCR** / **free-form** | 直接返 0 | 这两类自动评不可靠（需要人工或 LLM-as-judge），不进 accuracy 主指标 |

聚合指标：
- `mean_acc`：所有非 regression 样本的平均正确率
- `mean_mra`：regression 样本的平均 MRA

> 为什么 regression 用 MRA：能区分"差一点点"和"差很多"——单阈值下 5% 误差和 50% 误差都是 0/1，MRA 在 10 阈值上求平均，分数差距明显。VSI-Bench 的距离 / 面积 / 房间数估计全靠这个指标。
>
> 为什么 OCR / free-form 直接返 0：自动评分都不靠谱（OCR 用 WER 会过分严格、free-form 用 ROUGE 跟语义相关性弱），Video-R1 论文索性不算它们，只在训练 reward 里用，评估时不报。

### 5. 推理 pipeline

```
模型 checkpoint
   |
loop over 6 benchmarks:
   |-- 加载 eval_{benchmark}.json（带 problem / options / data_type / problem_type / path / solution）
   |-- resume 支持：output 文件已存在则从中断处续跑（视频推理慢，单 benchmark 几小时，没 resume 很痛）
   |-- batch 推理 (BSZ=64)
   |     |-- apply_chat_template 拼 prompt
   |     |-- process_vision_info 抽 image/video tensor
   |     `-- vLLM generate（异常 batch 用 '<answer>error</answer>' 占位让流程不中断）
   |-- 正则抽 <think>...</think> 和 <answer>...</answer>
   |-- 按 problem_type 路由 reward_fn 算分
   `-- 每个 batch 后立刻写回 JSON（崩了不会全丢）
   |
最终：mean_acc / mean_mra 写入 final_acc 字段
```

### 6. 要拿到的对比 baseline（Video-R1 Table 1 + Table 2 复现）

横向对比基线：

| Baseline | 用途 |
|----------|------|
| **GPT-4o** | 闭源天花板 |
| **Qwen2.5-VL-7B (CoT)** | 同 base、不训，看 CoT-only 能拿多少 |
| **Qwen2.5-VL-7B-SFT** | 只做 SFT cold-start，看 SFT 单独贡献 |

ablation 变体：

| 变体 | 隔离的设计 |
|------|------------|
| **Video-R1-7B-zero** | 跳过 SFT 直接 RL → 看 cold-start 必要性 |
| **Video-R1-7B-wo-image** | 去掉 image 数据 → 看 image-video 混合必要性 |
| **Video-R1-7B-wo-temporal** | 普通 GRPO 不带 T-GRPO → 看 temporal reward 贡献 |
| **Video-R1-7B (full)** | 完整方案 |

这 4 个变体直接复用论文 Table 2 设计，分别隔离 SFT cold-start / image 数据 / T-GRPO 三个核心设计的贡献。

---

## 目前进展

- ✅ 20+ 数据集汇总 + 清洗 + 格式统一
- ✅ CoT cold-start 生成（72B teacher + 规则过滤）
- ✅ SFT 跑通，格式稳定输出 `<think><answer>`
- ✅ 评估 pipeline 搭好（沿用 Video-R1 官方方式：6 个公开 video benchmark + 按 problem_type 路由 metric）
- 🔄 GRPO / T-GRPO 训练调参中
- ⏳ 完整 benchmark 评估未跑完

---

## 面试 2-3 分钟话术

> 我最近在做一个多模态推理方向的研究探索，叫 TimeThinker。背景是开源 instruct VLM 在多步推理任务上短板明显——事件时序、数值计算、图表理解都不稳，输出格式也不可控。DeepSeek-R1 在文本上验证了 SFT + GRPO 这条路线，我想看在多模态视频推理上能不能跑通。
>
> 这个项目我花时间最多的是**数据工作**，重点讲一下。训练数据是我从一堆公开数据集自己汇总、清洗、统一格式做出来的。
>
> **数据集选择**对比了纯视频、纯图像、image-video 混合三种方案——纯视频稀缺监督稀疏，纯图像没时序能力，最后选混合。视频侧 5 个来源做时序推理，图像侧分 6 个能力 bucket（Math / Chart / OCR / Spatial / Knowledge / General），每个 bucket 下面再从不同源拉，一共 20 多个数据集。
>
> **Per-source 采样**分三层：先给每个 source 定 budget cap，原则是稀缺数据多保留、同质化压缩、按 bucket 均衡，比如 LLaVA-Video 178k → 80k，Multimath 30 万 → 27k；然后在 source 内做 stratified sampling，按 subset / 难度 / 质量分层；最后做 post-sampling stats 验证分布有没有崩。原始拉了 35 万。
>
> **数据清洗**走了 8 步：20+ 个 per-source adapter 把字段映射到统一 schema；problem_type 归一化（不归一化 reward 路由失败）；用 ffprobe 校验视频帧数（cv2 不够严格）；硬过滤规则（options 缺失、答案不能解析、unanswerable 等）；文本清洗（Unicode、HTML、LaTeX、placeholder）；答案格式统一（包 `<answer>` tag、单位剥离、OCR 多人投票）；两层去重（精确 hash + MinHash LSH 阈值 0.92）；题型分布 stratified 下采样。35 万清洗到 26 万 RL 池。
>
> **CoT 生成**：原始数据只有问题答案没有推理过程，用 Qwen2.5-VL-72B 当 teacher 生成 CoT，三层规则过滤（能解析 / 答对 / 长度合规），26 万筛出 16.5 万 SFT 数据。
>
> **格式转换**：SFT 用 LLaMA-Factory ShareGPT 格式（按 modality 拆两文件），RL 用 EasyR1 格式（保留 problem_type / options / data_source），按题型设计了答案格式 instruction 模板，保证 SFT 训练和 RL rollout 的 prompt 一致。
>
> 模型选 Qwen2.5-VL-7B-Instruct 而不是 Thinking，是为了归因清晰。RL 阶段除 rule-based correctness reward，还引入 T-GRPO temporal reward——同问题跑顺序帧和乱序帧两组 rollout，强制模型用时序而不是抄单帧捷径。
>
> 目前 SFT 阶段格式稳定，T-GRPO 训练调参中。

---

## 追问预案

**Q: 数据是自己构造的还是用现成的？**
> 来源是公开数据集，但**整合、采样、清洗、CoT 生成、格式统一这套 pipeline 是我自己搭的**。20+ 数据集每个 schema 都不一样，光 adapter 就 20 多个；CoT 生成 + 过滤是数据质量核心。

**Q: 为什么不直接用 ChartQA / Math-V 等单一数据集？**
> 单一来源训出来的模型在该任务上很强但泛化差。我做多源混合，覆盖 6 大能力 + 5 类视频任务，让模型学到泛化推理能力。

**Q: Per-source 采样怎么定 budget？**
> 三个原则：稀缺数据多保留（小数据集全留）、同质化数据多压缩（Multimath 30 万只取 27k）、按能力 bucket 均衡（6 个图像 bucket 量级接近）。第一版凭经验定，后面看 post-sampling 分布迭代调，调了 5 轮。

**Q: 怎么判断视频文件损坏？**
> 两步：(1) 文件存在 + 大小 > 1KB；(2) 用 ffprobe 校验 stream codec_type 和 nb_frames，帧数 < 8 也算损坏。**ffprobe 比 cv2.VideoCapture 严格**，cv2 能打开的损坏文件抽帧时才报错。

**Q: 怎么去重？**
> 两层：(1) 精确 hash `(problem, media_path)`；(2) 近似用 MinHash LSH，字符 3-gram，阈值 0.92，**且限制在同一 media 内匹配**——一开始阈值 0.85，发现"What is happening"和"Describe the scene"被误合并，调高 + 加 media 限制解决。

**Q: 清洗丢了多少？**
> 漏斗：350k → 263k（RL）→ 165k（SFT）。媒体缺失 ~2.8 万、字段过滤 ~1.5 万、去重 ~1.5 万、题型平衡 ~1.9 万、adapter 异常 ~1 万、CoT 阶段 9.5 万（37%）。每一步按 `data_source` 分桶有统计。

**Q: 为什么砍掉 1.9 万多选题（题型平衡）？**
> 不平衡会偏科——多选准但其他题型烂。被砍的主要来自 LLaVA-Video（自己就 8 万多选，多 annotator 标了高度相似问题），冗余度高。下采样还做了 per-source 分层，避免某来源被全砍。

**Q: CoT 是 72B 蒸馏的，质量怎么保证？**
> 三层过滤：能解析 → 答对 → 长度合规。整体通过率 63%，丢的 37% 主要是 teacher 答错或格式错。

**Q: 为什么不自己人工标 CoT？**
> 26 万样本人工不现实，单条标几块钱，总成本数十万级。72B teacher + 规则过滤是性价比最高的方式，也是 DeepSeek-R1 / Video-R1 的标准做法。

**Q: image-video 配比怎么定？**
> 当前 image 56% / video 44%。原则：图像样本要多给 RL 提供稳定可判分监督；视频比例不能太低否则丢时序。打算做消融实验。

**Q: 现在效果怎么样？**
> SFT 格式稳定。T-GRPO 训练中，reward 稳定上升，完整 benchmark 还没出，给不出确定数字。

**Q: 评估怎么做的？**
> 沿用 Video-R1 官方评估方式：6 个公开视频 benchmark——VSI-Bench / VideoMMMU / MMVU / MVBench / TempCompass / VideoMME，前 3 个偏视频推理后 3 个偏通用理解。用 vLLM 跑 near-greedy decoding（temperature=0.1, top_p=0.001，跟 Qwen2.5-VL 官方 demo 对齐），分别跑 16/32/64 帧三组。Metric 按 problem_type 路由：多选用字母精确匹配、numerical 用 round(2) 匹配、regression 用 MRA（10 个阈值 0.5-0.95 求平均命中率）；OCR 和 free-form 在评估时直接返 0 不进 mean_acc，因为自动评分不可靠。聚合报 mean_acc 和 mean_mra 两个数字。

**Q: 为什么 regression 用 MRA 不用 MAE / 相对误差？**
> MRA 在 10 阈值上求平均，能区分"差一点点"和"差很多"——单阈值下 5% 误差和 50% 误差都是 0/1，MRA 下分数差距明显。VSI-Bench 的距离 / 面积 / 房间数估计这类任务全靠这个指标。

**Q: 评估 OCR / free-form 直接返 0 不亏吗？**
> 不亏。这两类自动评分都不可靠——OCR 用 WER 太严格（一个字符差就算错）、free-form 用 ROUGE 跟语义相关性弱。Video-R1 论文索性不算它们进 accuracy 主指标，训练 reward 用它们提供多样性监督就够了；评估真要看 OCR / free-form 能力要靠人工或 LLM-as-judge。

**Q: 为什么 top_p=0.001 不是 1？**
> Qwen2.5-VL 的已知坑——top_p 大了输出会乱码，官方 demo 就是 0.001。Video-R1 README 里也明确警告。

**Q: 评估时为什么 frame 数（64）比训练时（16）大？**
> 训练受 batch / 显存约束只能 16 帧；评估单样本 batch=1 能塞更多。Video-R1 论文 Table 1 实验证明 16 → 32 → 64 在所有 benchmark 上都涨点，长视频涨得最多（VideoMME 16→64 涨 4 个点）。

**Q: 不用 VLMEvalKit 跑标准协议吗？**
> Video-R1 评估的 metric 跟训练 reward 是同一套（按 problem_type 路由的 rule-based 打分），自己跑能保证训练 / 评估完全对齐。VLMEvalKit 走它自己的标准协议（多选用 LLM judger 之类），引入额外噪声反而不利于归因。横向对比用论文 Table 1 报的同一套数字就够。

**Q: 怎么做 ablation？**
> 复用 Video-R1 论文 Table 2 的 4 个变体：(1) Video-R1-zero 跳过 SFT 直接 RL → 看 cold-start 必要性；(2) wo-image 去掉图像数据 → 看 image-video 混合必要性；(3) wo-temporal 用普通 GRPO → 看 T-GRPO 贡献；(4) full 完整方案。这三个隔离实验分别证明 cold-start / 数据混合 / T-GRPO 三个核心设计的价值。

---

## 可主动抛的开放讨论点

1. 多源混合比例 vs 模型能力：6 个 bucket 各占多少最优？
2. CoT 蒸馏质量决定 RL 上限多少：72B 够不够，要不要用 GPT-4o 补一批？
3. 多模态 reward hacking：模型会不会学到"不看图蒙常见答案"？
4. data_source 级 reward 曲线反向筛数据：reward 上不去的来源能不能反过来判定数据质量？
5. T-GRPO temporal reward 对图像无效，要不要给图像设计类似的"反 shortcut"机制？
