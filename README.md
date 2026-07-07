# TimeThinker

这是一个基于 Qwen3-VL-4B 的本地多模态后训练项目，当前主要围绕 Video-R1 风格的图像/视频推理数据做 SFT、RL 和 benchmark 评测。

当前目标不是完整复现 TimeThinker-600k，而是训练和评测一个面向 image/video QA、视觉数学、OCR、图表理解、空间推理、视频时序推理的多模态推理模型，并稳定输出：

```text
<think>...</think><answer>...</answer>
```

由于当前数据没有 box、mask、trajectory 等监督，grounding、tracking、segmentation 不是当前训练目标。

## 目录结构

| 路径 | 作用 |
|---|---|
| `config/sft/` | LLaMA-Factory SFT 配置 |
| `config/rl/` | EasyR1 / verl RL 配置 |
| `scripts/data/` | Video-R1 到本地训练格式的转换脚本 |
| `scripts/train/` | SFT 和 RL 启动脚本 |
| `scripts/eval/` | benchmark、串行多模型评测、结果汇总脚本 |
| `LLaMA-Factory/` | SFT 后端 |
| `EasyR1/` | RL 后端 |
| `Evaluation/` | benchmark 数据、推理和结果目录 |
| `docs/` | 项目说明、数据说明、消融实验、评测分析 |

## 数据

原始数据在 `data/`：

- `Video-R1-COT-165k.json`：SFT cold-start 数据，包含 CoT 推理过程。
- `Video-R1-260k.json`：RL 数据，答案可规则判分。
- 媒体目录：`CLEVRER`、`LLaVA-Video-178K`、`NeXT-QA`、`PerceptionTest`、`STAR`、`Chart`、`General`、`Knowledge`、`Math`、`OCR`、`Spatial`。

当前配置实际使用的转换后文件：

- `LLaMA-Factory/data/timethinker_sft_image.json`
- `LLaMA-Factory/data/timethinker_sft_video.json`
- `EasyR1/data/timethinker_rl_train_split.json`
- `EasyR1/data/timethinker_rl_val_512.json`

数据转换入口：

```bash
.venv_sft/bin/python scripts/data/convert_data.py
```

注意：转换脚本默认会生成完整 RL 文件；当前 RL 配置使用的是已经切分好的 train/val 文件。

## 环境

当前工作区使用三个本地虚拟环境：

```text
.venv_sft
.venv_rl
.venv_eval
```

训练和评测脚本默认会使用这些路径；如果环境已经存在，通常不需要手动 activate。

## 训练

运行默认 SFT 配置：

```bash
bash scripts/train/run_sft.sh
```

指定 SFT 配置或 resume 行为：

```bash
CONFIG=config/sft/qwen3_sft-v6.yaml RESUME_FROM_CHECKPOINT=none bash scripts/train/run_sft.sh
```

运行默认 RL 配置：

```bash
bash scripts/train/run_rl.sh
```

运行 T-GRPO 配置：

```bash
bash scripts/train/run_rl_t.sh
```

常用 RL 配置：

```text
config/rl/qwen3_rl.yaml
config/rl/qwen3_rl_t.yaml
```

## 评测

对单个模型运行默认 benchmark：

```bash
MODEL_PATH=models/TimeThinker-4B-SFT-v3-10000-1ep bash scripts/eval/run_bench.sh
```

快速验证：

```bash
DATASETS=eval_mvbench.json,eval_tempcompass.json,eval_videomathqa.json \
MAX_SAMPLES=800 \
RESULT_SUFFIX=_smoke800 \
bash scripts/eval/run_bench.sh
```

串行评测多个模型：

```bash
bash scripts/eval/run_bench_list.sh \
  models/TimeThinker-4B-RL-Zero-100-van-v2/global_step_100/actor/huggingface \
  models/TimeThinker-4B-RL-Zero-100-ema-v2/global_step_100/actor/huggingface
```

评测结果写入：

```text
Evaluation/results/<model_tag>/frames<MAX_FRAMES>/
```

每个模型结果目录会自动生成：

```text
_summary.json
_summary.md
```

默认开启抽帧缓存：

```text
Evaluation/data/.cache/eval_frames
```

可用 `FRAME_CACHE_DIR=...` 修改缓存目录，或用 `DISABLE_FRAME_CACHE=1` 关闭缓存。

## 文档导航

| 文档 | 用途 |
|---|---|
| `docs/PROJECT.md` | 项目范围、当前定位和技术路线 |
| `docs/data.md` | Video-R1 数据来源、bucket、取样逻辑 |
| `docs/qa.md` | 面试追问风格的数据 / SFT / RL 问答 |
| `docs/sft_ablation.md` | SFT 消融实验记录 |
| `docs/rl_ablation.md` | RL 消融实验记录 |
| `docs/eval-v1.md` | 当前评测结果汇总 |
| `docs/bad_case.md` | bad case 分析模板 |
| `docs/issues.md` | 已遇到的问题和修复记录 |
| `docs/daily.md` | 日常推进记录 |
