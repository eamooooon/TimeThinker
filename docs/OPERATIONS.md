# 运行与资产管理

本文是数据、缓存、模型和 checkpoint 的操作手册。首次运行的完整命令见仓库 [README](../README.md)。

## 训练数据

当前训练集使用公开的 [Video-R1 数据发布页](https://huggingface.co/datasets/Video-R1/Video-R1-data)。本地目录应为：

```text
data/
├── Video-R1-COT-165k.json       # 含 CoT 的 SFT 源数据
├── Video-R1-260k.json           # 含可验证答案的 RL 源数据
├── CLEVRER/  Chart/  General/  Knowledge/  Math/  OCR/  Spatial/
└── LLaVA-Video-178K/  NeXT-QA/  PerceptionTest/  STAR/
```

使用 `scripts/data/download_videor1.py` 下载元数据、指定 bucket 或完整数据。上游数据由大型 ZIP 分片组成，应先用 `--dry-run` 查看下载清单；确认有充足磁盘空间后才使用 `--all`。

```bash
.venv_sft/bin/python scripts/data/download_videor1.py --components Chart Math --dry-run
.venv_sft/bin/python scripts/data/download_videor1.py --components Chart Math
.venv_sft/bin/python scripts/data/convert_data.py
.venv_rl/bin/python scripts/data/split_rl_data.py
```

后两条命令会生成 LLaMA-Factory 的两份 SFT JSON，以及 EasyR1 的 RL train/validation JSON。转换时会跳过媒体缺失的记录并打印原因。切分脚本固定使用 seed 42 留出 512 条验证样本；只有明确需要重建时才使用 `--overwrite`。

评测媒体与训练媒体相互独立：

```bash
.venv_eval/bin/python scripts/eval/download_dataset.py tempcompass
```

## 存储与缓存清理

| 内容 | 位置 | 可重新生成？ |
|---|---|---|
| 原始训练媒体 | `data/` | 是，可从 Video-R1 下载 |
| 训练下载缓存 | `data/.hf_cache/` | 是 |
| RL 抽帧缓存 | `data/.cache/rl_frames/` | 是 |
| 评测媒体与下载缓存 | `Evaluation/data/`, `Evaluation/data/.hf_cache/` | 是 |
| 评测抽帧缓存 | `Evaluation/data/.cache/eval_frames/` | 是 |
| 转换后的 JSON | `LLaMA-Factory/data/`, `EasyR1/data/` | 是，可由原始数据生成 |

转换、训练或评测正在使用某个媒体 bucket 或缓存时，不要删除它。

## 模型与 checkpoint

| 路径 | 含义 | 正确用途 |
|---|---|---|
| `models/` | 当前优先使用的资产与 canonical 导出 | 新推理、评测和实验 |
| `models-v1/` | 历史实验记录 | 仅用于对照或恢复 |
| `models/<SFT>/checkpoint-*` | Hugging Face 格式的 SFT checkpoint | 将该 checkpoint 目录作为 `MODEL_PATH` |
| `models/<RL>/global_step_*/actor/huggingface/` | Hugging Face 格式的 RL actor 导出 | 将最终 `huggingface/` 目录作为 `MODEL_PATH` |
| DeepSpeed `global_step_*` state | 分布式训练状态 | 仅在优化器状态仍存在时用于续训 |

`models-v1/` 中的优化器状态已被有意删除，因此其中 checkpoint 不再适合精确续训；保留的 Hugging Face 导出仍可用于评测和推理。

新实验应在目录名中体现阶段、来源、算法、batch size 与步数，避免继续使用含义不清的 `-new`、`-v2` 后缀。每次运行都应记录 config 路径、源模型、数据版本和评测摘要。
