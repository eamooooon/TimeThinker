# TimeThinker

TimeThinker is a local training project adapted to use Video-R1 data for Qwen3-VL-4B multimodal reasoning training.

The current goal is to train a Qwen3-VL-4B based multimodal reasoning model with image/video QA, visual math, OCR, chart understanding, spatial reasoning, and `<think>...</think><answer>...</answer>` style outputs.

The local data is Video-R1, so grounding, tracking, and segmentation abilities are not expected unless corresponding data is added later.

## Data

Current local data lives in `data/`:

- `Video-R1-COT-165k.json`: SFT cold-start data with CoT reasoning.
- `Video-R1-260k.json`: RL data with verifiable answers.
- Media folders: `CLEVRER`, `LLaVA-Video-178K`, `NeXT-QA`, `PerceptionTest`, `STAR`, `Chart`, `General`, `Knowledge`, `Math`, `OCR`, `Spatial`.

Expected TimeThinker training files after conversion:

- `LLaMA-Factory/data/timethinker_sft_image.json`
- `LLaMA-Factory/data/timethinker_sft_video.json`
- `timethinker_rl_train.json`

## Setup

```bash
# SFT environment
uv venv --python 3.11 .venv-llamafactory
uv pip install setuptools wheel --python .venv-llamafactory/bin/python
cd LLaMA-Factory
uv pip install -e ".[torch,metrics]" --no-build-isolation --python ../.venv-llamafactory/bin/python
cd ..

# RL environment
uv venv --python 3.11 .venv-easyr1
cd EasyR1
uv pip install -e . --python ../.venv-easyr1/bin/python --no-build-isolation-package flash-attn
cd ..
```

## Training

Run SFT:

```bash
source .venv-llamafactory/bin/activate
bash ./LLaMA-Factory/local_scripts/run_timethinker_sft.sh
```

Run RL:

```bash
source .venv-easyr1/bin/activate
bash ./EasyR1/local_scripts/run_timethinker_rl.sh
```

## Evaluation

```bash
bash ./Evaluation/Eval/eval_bench_all.sh
```

For a single example:

```bash
python ./Evaluation/inference_single/inference.py
```

## Notes

- See `PROJECT.md` for the current project understanding, data scope, and model selection rationale.
