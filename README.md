# TimeThinker

基于 Qwen3-VL-4B 的多模态后训练项目。当前实现使用 Video-R1 风格的图像/视频推理数据进行 SFT、RL 与 benchmark 评测，目标输出为：

```text
<think>...</think><answer>...</answer>
```

当前数据不包含 box、mask、trajectory 监督；grounding、tracking、segmentation 不属于本项目现阶段目标。项目范围和技术背景见 [docs/PROJECT.md](docs/PROJECT.md)。

## Quick start

```bash
# 0. 新机器仅需初始化一次；已有 .venv_* 可跳过
python -m venv .venv_sft && .venv_sft/bin/pip install -e ./LLaMA-Factory
python -m venv .venv_rl && .venv_rl/bin/pip install -r EasyR1/requirements.txt && .venv_rl/bin/pip install -e ./EasyR1
python -m venv .venv_eval && .venv_eval/bin/pip install -r Evaluation/VLMEvalKit/requirements.txt

# 1. 下载训练索引；媒体按 bucket 按需下载，不要默认全量下载
.venv_sft/bin/python scripts/data/download_videor1.py --metadata-only
.venv_sft/bin/python scripts/data/download_videor1.py --components Chart Math

# 2. 全部所需媒体到位后，构建 SFT/RL 文件和固定的 512 条 RL 验证集
.venv_sft/bin/python scripts/data/convert_data.py
.venv_rl/bin/python scripts/data/split_rl_data.py

# 3. 启动训练。默认脚本按 4 GPU 配置；可覆盖 CUDA_VISIBLE_DEVICES/NPROC_PER_NODE
CONFIG=config/sft/qwen3_sft-v10.yaml RESUME_FROM_CHECKPOINT=none bash scripts/train/run_sft.sh
CONFIG=config/rl/qwen3_rl_bs16.yaml bash scripts/train/run_rl.sh

# 4. 下载一个评测集并做 smoke test
.venv_eval/bin/python scripts/eval/download_dataset.py tempcompass
MODEL_PATH=models/TimeThinker-4B-SFT-v10-50k-canonical/checkpoint-1000 \
DATASETS=eval_tempcompass.json MAX_SAMPLES=32 RUN_PARALLEL=0 \
bash scripts/eval/run_bench.sh
```

完整的数据布局、存储清理、模型/checkpoint 使用规则见 [docs/OPERATIONS.md](docs/OPERATIONS.md)。

## Directory map

| Path | Purpose |
|---|---|
| `config/sft/`, `config/rl/` | Versioned SFT and RL experiment configs |
| `scripts/data/` | Download, conversion, splitting, and preprocessing entry points |
| `scripts/train/`, `scripts/eval/` | Runnable training and evaluation entry points |
| `data/` | Ignored raw Video-R1 training data and caches |
| `Evaluation/data/` | Ignored evaluation data and caches |
| `models/` | Preferred current exports and experiments |
| `models-v1/` | Legacy model history; optimizer state has been removed |
| `LLaMA-Factory/`, `EasyR1/`, `Evaluation/` | SFT, RL, and benchmark backends |
| `docs/` | Operational guides, experiment records, and analysis |

## Documentation

| Document | Use it for |
|---|---|
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | Data download/conversion, cache cleanup, and model/checkpoint lifecycle |
| [docs/eval.md](docs/eval.md) | Current canonical leaderboard, evaluator/prompt history, and comparability rules |
| [docs/PROJECT.md](docs/PROJECT.md) | Scope and current Video-R1-based project positioning |
| [docs/data.md](docs/data.md) | Data mixture and capability analysis |
| [docs/archive/README.md](docs/archive/README.md) | Historical ablations, daily logs, legacy evaluation reports, and case notes |
