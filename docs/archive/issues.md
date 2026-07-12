# Issues

本文档记录前期训练、评测、数据下载、环境与 Git 操作中遇到的问题，以及当时的判断和处理方式。后续遇到类似现象时，优先在这里查。

## 1. SFT Loss 很快进入平台

**现象**

- `train/loss` 前期快速从较高值下降到约 `0.7` 左右。
- 后续几百 step 波动很大，但整体下降不明显。

**判断**

- 这不一定是训练坏了，更像模型先学会了输出格式、答案模板、`<think>/<answer>` 风格，然后进入能力提升较慢的平台。
- 之前 10k 样本、1 epoch、`learning_rate=3e-6` 较保守，训练步数偏少。
- 如果视觉塔冻结，视觉 gap 很难只靠语言侧完全补上。

**处理**

- SFT 配置改为更充分的下一轮实验：
  - `max_samples: 30000`
  - `num_train_epochs: 2.0`
  - `learning_rate: 5.0e-6`
  - `tokenized_path: LLaMA-Factory/cache/timethinker_sft_30k`
  - `output_dir: models/TimeThinker-4B-SFT-v3-30000`

**后续建议**

- 不只看 `train/loss`，要看 `eval/loss` 和 benchmark。
- 如果 train loss 下降而 eval loss 不降，优先排查数据分布、验证集泄漏、过拟合。
- 如果视觉理解类 benchmark 不涨，再考虑提高帧数/分辨率或解冻更多视觉模块。

## 2. `val_size` 和 eval dataset 报错

**现象**

```text
ValueError: You have set args.eval_strategy to STEPS but you didn't pass an eval_dataset to Trainer.
```

**原因**

- `eval_strategy: steps` 要求有验证集。
- `val_size` 表示从训练数据中切出一部分作为 validation，不是额外的测试集文件数量。

**处理**

- 开启：

```yaml
val_size: 0.1
eval_strategy: steps
eval_steps: 200
```

**注意**

- 如果 validation 是从原 train 数据里切出来的，它适合观察 eval loss，但不等于严格无泄漏 benchmark。

## 3. Tokenized Cache 和 `max_samples`

**现象**

- 设置：

```yaml
max_samples: 10000
tokenized_path: LLaMA-Factory/cache/timethinker_sft_tokenized_val
```

但日志仍显示：

```text
Num examples = 165,572
```

**原因**

- 如果 `tokenized_path` 已经存在，训练会优先加载旧 cache。
- `max_samples` 不会自动裁剪已经保存好的全量 tokenized cache。

**处理**

- 小样本测试要使用新的 cache 路径，或删除旧 cache。
- 全量 train / valid cache 建议分开保存，避免误用。

## 4. 改 `image_max_pixels` 后 token/features 不匹配

**现象**

```text
ValueError: Image features and image tokens do not match: tokens: 356, features 624
```

**原因**

- 多模态 token 数与视觉特征数不一致，常见于：
  - tokenized cache 是旧分辨率生成的。
  - 训练时改了 `image_max_pixels` / `video_max_pixels`。
  - cache 和当前 processor 配置不一致。

**处理**

- 改视觉分辨率后必须重新生成 tokenized cache。
- cache 路径最好带上配置含义，例如 `sft_30k_px100352`。

## 5. Tokenizer / Cache 生成很慢

**现象**

- 前 2/3 较快，后 1/3 明显变慢。

**判断**

- 可能不是 tokenizer 本身慢，而是后面样本更长、视频/图像更多、IO 更重。
- 多进程预处理和磁盘 IO 会互相影响。

**建议**

- 使用稳定的 `preprocessing_num_workers`，不要盲目加太大。
- 避免同时跑评测下载和大规模 tokenize。
- 改分辨率或帧数后重新 cache，要预留时间。

## 6. Video-R1 原仓库配置差异

**结论**

- 当前项目使用的是 LLaMA-Factory 做 SFT、EasyR1/verl 做 RL。
- Video-R1 原仓库使用 `open_r1/grpo.py`，并有 `temporal true`、`len_control true` 等 trainer 逻辑。
- 早期 `config/rl/qwen3_rl.yaml` 只参考了 Video-R1 的一部分超参，不是严格复刻原仓库。
- 后续已经补了 Video-R1 风格的 T-GRPO temporal reward 路径，当前主要在 `config/rl/qwen3_rl_t.yaml` 和 `EasyR1/verl/trainer/ray_trainer.py` 中实现。

**已对齐的主要超参**

- `learning_rate=1e-6`
- `weight_decay=0.01`
- `kl_coef=0.04`
- `max_prompt_length=16384`
- `max_response_length=768`
- `num_generations=8`
- `temperature=1.0`
- `top_p=1.0`
- `max_grad_norm=5`

**主要差异**

- 标准 RL 配置 `config/rl/qwen3_rl.yaml` 仍是 GRPO / EMA-GRPO 路线，reward 是 `timethinker_reward.py:compute_score`。
- T-GRPO 配置 `config/rl/qwen3_rl_t.yaml` 开启：
  - `algorithm.temporal: true`
  - `algorithm.shuffled_rollout_ratio: 0.5`
  - `algorithm.temporal_reward: 0.3`
  - `algorithm.temporal_compare_ratio: 0.8`
  - `algorithm.len_control: false`
- 当前训练框架仍是 EasyR1/verl，不是直接运行 Video-R1 原仓库。

## 7. 评测数据下载问题

### VideoMMMU gated

**现象**

```text
GatedRepoError: 403
Access to dataset lmms-lab/VideoMMMU is restricted
```

**原因**

- 登录 HuggingFace 不等于已获得 gated dataset 权限。

**处理**

- 需要在 HuggingFace 页面申请访问权限。

### MVBench 文件不齐

**现象**

- MVBench 初始只下到部分视频。

**处理**

- 补充了下载和整理逻辑：
  - ssv2 `.mp4 -> .webm`
  - star/clevrer 路径修正
  - TVQA frame folder 转 mp4
  - NTURGBD avi 补齐并做 symlink

### OneThinker / VideoMathQA 下载慢

**现象**

- HuggingFace 下载 `VideoMathQA_part1.zip` 卡住，`.incomplete` 长时间不增长。

**处理**

- 停掉 HF 下载。
- 改用 ModelScope 下载：

```text
MBZUAI/VideoMathQA
```

**注意**

- `eval_videomathqa.json` 只是索引文件，不包含视频本体。
- 视频 zip 下载并解压后，覆盖率才会从 `0/420` 增长。

## 8. Eval 依赖和视频读取问题

### 缺少 `rouge_score`

**现象**

```text
ModuleNotFoundError: No module named 'rouge_score'
```

**处理**

- 在 `.venv_eval` 中安装缺失依赖。

### torchvision 无 `read_video`

**现象**

```text
AttributeError: module 'torchvision.io' has no attribute 'read_video'
```

**原因**

- 当前 `torchvision 0.26` 已没有 `torchvision.io.read_video`。
- qwen-vl-utils 在 decord 失败后 fallback 到 torchvision，导致崩溃。

**处理**

- 在 `Evaluation/Eval/eval_bench.py` 中加入 PyAV fallback。
- 默认 `VIDEO_READER=auto`：
  - 优先项目内 decord 路径。
  - decord 失败后 fallback 到 PyAV。
- 加入运行时 frame cache：
  - 默认路径：`Evaluation/data/.cache/eval_frames`
  - 可用 `FRAME_CACHE_DIR=...` 改 cache 目录。
  - 可用 `DISABLE_FRAME_CACHE=1` 关闭 cache。
  - 结果 JSON 的 `meta.frame_cache` 会记录 `hit` / `miss` / `write` / `fallback_to_pyav`。

**注意**

- MVBench decord 失败较多时，`auto` 可能比直接 `pyav` 慢，因为会先失败一次。
- frame cache 缓的是抽帧后的图片，不是最终 processor tensor；同一个视频、采样参数、文件 mtime、片段范围一致时，800 条 smoke 和后续全集可以复用同一 cache。

## 9. 评测只用一张卡

**现象**

- 跑 benchmark 时只有 GPU0 有显存占用。

**原因**

- 默认串行时 `CUDA_VISIBLE_DEVICES=0`。
- 单个 eval 进程只会看到一张卡。

**处理**

- `run_bench.sh` 加入并行模式：

```bash
RUN_PARALLEL=1 EVAL_GPUS=0,1,2,3 bash scripts/eval/run_bench.sh
```

**注意**

- 并行模式是不同 benchmark 分到不同 GPU。
- 单个 benchmark 本身不会自动切成 4 份。

## 10. 评测并行后反而更慢

**现象**

- `MAX_FRAMES=16` 并行跑 4 个 benchmark，感觉比单独跑更慢。

**原因**

- 主要瓶颈不是 GPU，而是 CPU/视频解码/预处理。
- 4 个 eval 进程同时解码视频，每个进程又开大量线程，导致线程过载和上下文切换。
- 当时每个进程可到数百线程，4 个进程合计两千多线程。

**处理**

- 给评测脚本加线程限制：

```bash
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
OPENBLAS_NUM_THREADS=8
OPENCV_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false
```

- `eval_bench.py` 中限制 torch / OpenCV 线程。

**建议**

- 不一定 4 个 benchmark 同时跑最快。
- 对长视频 benchmark，可以先 2 并发。
- MVBench decord fallback 多时，可以单独 `VIDEO_READER=pyav`。

## 11. 虚拟环境重命名问题

**操作**

- `.venvs -> .venv_sft`
- `.venvr -> .venv_rl`

**问题**

- `.venv_rl` 原本 Python 解释器路径坏了，`pyvenv.cfg` 指向不存在的 uv Python home。

**处理**

- 修正 `.venv_rl/bin/python` symlink。
- 更新 `.venv_rl/pyvenv.cfg`。
- 验证：
  - `torch`
  - `verl`
  - `vllm`
  - `transformers`
  - `qwen_vl_utils`

## 12. Git Push 卡住

**现象**

```bash
git push -u origin main
```

一直加载。

**原因**

- remote 是 HTTPS，GitHub 返回 401，需要用户名和 token。
- 当前环境无法交互读取 username/password。

**SSH 尝试**

```text
git@github.com: Permission denied (publickey)
```

说明 SSH 通了，但本机公钥没有加到 GitHub。

**处理建议**

- 将 `~/.ssh/id_rsa.pub` 加到 GitHub SSH keys。
- 或使用 HTTPS + GitHub PAT。

## 13. Git 临时大包和 pycache

**现象**

```text
.git/objects/pack/tmp_pack_xxx 1.2G
```

**原因**

- Git pack/push 中断留下的临时垃圾包，不代表环境目录已被正常提交。

**处理**

```bash
git gc --prune=now
```

清理后 `.git/objects/pack` 从约 `1.2G` 降到约 `22M`。

**pycache 处理**

- `.gitignore` 增加：

```gitignore
__pycache__/
*.py[cod]
*$py.class
```

- 已经跟踪的 `.pyc` 用 `git rm --cached` 从索引移除。

## 14. RL 配置整理

**问题**

- `run_rl.sh` 里覆盖了很多配置，`qwen3_rl.yaml` 里也有同名配置，容易出现“改 yaml 不生效”。
- 初始复制的 EasyR1 配置残留了 `math12k`、`qwen2_5_7b_math_grpo`。

**处理**

- 新建并整理：

```text
config/rl/qwen3_rl.yaml
```

- `run_rl.sh` 精简为只负责启动：

```bash
python -m verl.trainer.main config="${CONFIG}" "$@"
```
- `scripts/train/run_rl.sh` 默认读取 `config/rl/qwen3_rl.yaml`。
- `scripts/train/run_rl_t.sh` 默认读取 `config/rl/qwen3_rl_t.yaml`。

**当前 RL 关键设置**

- `train_files: EasyR1/data/timethinker_rl_train_split.json`
- `val_files: EasyR1/data/timethinker_rl_val_512.json`
- `model_path: Qwen/Qwen3-VL-4B-Instruct`
- `logger: ["file", "swanlab"]`
- `kl_coef: 4.0e-2`
- `rollout.n: 8`
- `max_response_length: 768`
- `gpu_memory_utilization: 0.7`
- 标准配置保存路径当前仍写在 yaml 里：
  - `config/rl/qwen3_rl.yaml`: `models/TimeThinker-4B-RL-Zero-100-grpo`
  - `config/rl/qwen3_rl_t.yaml`: `models/TimeThinker-4B-RL-Zero-100-tgrpo-van2`
- 如果磁盘目录已经改名为 `van` / `van-v2` / `tgrpo-van` / `tgrpo-van2`，需要同步检查 `config/rl/*.yaml` 和 `scripts/train/run_rl_list.sh`，否则新训练会继续写到旧目录名。

**SwanLab**

- `.venv_rl` 安装了 `swanlab`。
- `run_rl.sh` 设置：

```bash
SWANLAB_DIR=swanlog
```

## 15. RL smoke test 修复记录

**背景**

- RL 配置整理后，不能只看脚本和 yaml 是否能解析，还要实际跑一次最小 smoke test。
- 这次用 `max_steps=0`、单卡、单样本验证集测试，目标是至少跑通：
  - 本地 RL 数据加载。
  - reward function import。
  - SFT checkpoint 加载。
  - FSDP/vLLM 初始化和权重同步。
  - vLLM 多模态 generation。
  - reward 计算。

**测试命令**

```bash
timeout 420 bash scripts/train/run_rl.sh \
  trainer.max_steps=0 \
  trainer.val_before_train=false \
  trainer.val_freq=-1 \
  trainer.save_freq=1 \
  trainer.find_last_checkpoint=false \
  trainer.logger='["file"]' \
  trainer.n_gpus_per_node=1 \
  worker.rollout.tensor_parallel_size=1 \
  worker.rollout.n=1 \
  data.val_files=/tmp/timethinker_rl_smoke_val.json \
  data.rollout_batch_size=1 \
  data.val_batch_size=1 \
  worker.actor.global_batch_size=1 \
  worker.actor.micro_batch_size_per_device_for_update=1 \
  worker.actor.micro_batch_size_per_device_for_experience=1
```

**问题 1：本地 json 数据加载失败**

```text
NotImplementedError: Loading a dataset cached in a LocalFileSystem is not supported
```

**原因**

- `datasets.load_dataset("json", data_files=local_file)` 在当前环境/版本组合下走到了不支持的 LocalFileSystem cache 路径。

**处理**

- 在 `EasyR1/verl/utils/dataset.py` 增加本地 `.json` / `.jsonl` loader。
- 本地文件直接 `json.load` / 逐行 `json.loads` 后用 `datasets.Dataset.from_list(records)` 构造数据集。

**验证**

- `.venv_rl` 下成功加载 `EasyR1/data/timethinker_rl_train.json`。
- 训练集长度为 `263071`。

**问题 2：reward 依赖缺失**

```text
ModuleNotFoundError: No module named 'rouge_score'
ModuleNotFoundError: No module named 'math_verify'
```

**处理**

- 用 uv 给 `.venv_rl` 安装：

```bash
/tianyuesong/zy/.uv/uv pip install --python .venv_rl/bin/python rouge-score math-verify
```

**验证**

- `rouge_score`、`math_verify`、`mathruler` import 正常。
- `EasyR1.verl.reward_function.timethinker_reward.compute_score` import 正常。

**问题 3：vLLM 0.23 EngineArgs 参数变化**

```text
TypeError: EngineArgs.__init__() got an unexpected keyword argument 'disable_mm_preprocessor_cache'
```

**原因**

- 旧 EasyR1 代码使用 `disable_mm_preprocessor_cache`。
- 当前 vLLM 0.23 已改为 `mm_processor_cache_gb`。

**处理**

- 在 `EasyR1/verl/workers/rollout/vllm_rollout_spmd.py` 里检查 `EngineArgs` 签名：
  - 旧版本使用 `disable_mm_preprocessor_cache=True`。
  - 新版本使用 `mm_processor_cache_gb=0`。

**问题 4：vLLM tensor parallel group API 变化**

```text
AttributeError: module 'vllm.distributed.parallel_state' has no attribute 'get_tensor_model_parallel_group'
```

**原因**

- 当前 vLLM 0.23 使用 `get_tp_group()`。

**处理**

- 在 `EasyR1/verl/workers/sharding_manager/fsdp_vllm.py` 增加兼容函数：
  - 有 `get_tensor_model_parallel_group()` 就走旧 API。
  - 否则走 `get_tp_group()`。
  - 如果返回对象有 `device_group`，取 `device_group`。

**问题 5：SamplingParams 的 `eos_token_id` 变成只读**

```text
AttributeError: property 'eos_token_id' of 'SamplingParams' object has no setter
```

**原因**

- EasyR1 在 generation 前用 `setattr(self.sampling_params, key, value)` 更新 meta info。
- vLLM 0.23 的 `SamplingParams.eos_token_id` 是只读 property，实际构造/存储字段是 `_eos_token_id`。

**处理**

- 在 `vllm_rollout_spmd.py` 增加 sampling params key 映射：
  - `eos_token_id` 如果没有 setter，则映射为 `_eos_token_id`。
  - 上下文退出时也按实际字段回滚。

**验证**

```text
{'mapped_key': '_eos_token_id', 'roundtrip_ok': True}
```

**问题 6：图片样本缺少视觉占位符**

```text
AssertionError: Failed to apply prompt replacement for mm_items['image'][0]
```

**原因**

- 数据样本有 `images` 字段，但 `problem` 文本不一定包含字面量 `<image>`。
- 原 `_build_messages()` 只有在文本中遇到 `<image>` 时才插入 `{"type": "image"}`。
- 结果传给 vLLM 时有 `multi_modal_data`，但 prompt token 里没有 `<|vision_start|><|image_pad|><|vision_end|>` 占位符。

**处理**

- 修改 `RLHFDataset._build_messages()`：
  - 有图片字段但文本没有 `<image>` 时，自动在 user content 前插入对应数量的 `{"type": "image"}`。
  - 视频同理，缺少 `<video>` 时自动插入 `{"type": "video"}`。
  - 如果文本里已有 `<image>` / `<video>`，继续按原位置插入；数量不足时补齐。

**验证**

- 单条图片样本的 `raw_prompt_ids` 中已出现：

```text
151652 1
151655 1
151653 1
```

对应：

```text
<|vision_start|><|image_pad|><|vision_end|>
```

**问题 7：reward 缺少原始题目字段**

```text
KeyError: 'problem_reserved_text'
```

**原因**

- `BatchFunctionRewardManager.compute_reward()` 会读取：

```python
data.non_tensor_batch["problem_reserved_text"]
```

- 但 dataset 在 `__getitem__()` 中 `example.pop(self.prompt_key, None)` 后没有保留原始题目文本。

**处理**

- 在 `RLHFDataset.__getitem__()` 中 pop 前保存：

```python
example["problem_reserved_text"] = example.get(self.prompt_key, "")
```

**问题 8：本地 loader 丢失 `videos` 列导致 `multi_modal_data` 长度不一致**

```text
AssertionError: key multi_modal_data length 22 is not equal to bsz 32.
```

**最终结论**

- 不是 RL 数据集中混入了 text-only 样本。
- 原始数据每条都有模态：
  - image 样本有 `images`。
  - video 样本有 `videos`。
- 报错里的 `multi_modal_data length 22` 是 loader/schema 问题造成的：
  - 本地 json 转 HF Dataset 时没有先补齐所有字段。
  - 第一条 image 样本没有 `videos`，导致后续 video 列没有被正确保留下来。
  - 于是部分 video 样本在 dataset 里读出来像“无模态样本”，最终 `multi_modal_data` 数量少于 batch size。
- `None` 只应该作为列补齐和兜底占位使用：
  - image 样本的 `videos=None` 是正常的。
  - video 样本的 `images=None` 是正常的。
  - 但 video 样本的 `videos=None` 就是不正常的，说明 loader 丢列或数据损坏。

**原因**

- 原始 `timethinker_rl_train.json` 里没有 text-only 样本：
  - image 样本：`146823`
  - video 样本：`116248`
  - 无图片/视频样本：`0`
- 真实问题出在本地 json loader：
  - 之前用 `datasets.Dataset.from_list(records)` 直接建数据集。
  - 第一条样本是 image 样本，只有 `images`，没有 `videos`。
  - HF Dataset 会按输入记录推 schema；如果没有先补齐所有 key，后续 video 样本的 `videos` 列可能被丢掉。
- 结果是很多 video 样本读出来变成：

```text
data_type='video', images=None, videos=None
```

- 这些样本无法命中 image/video 分支，因此不会写入有效的 `multi_modal_data`。
- `DataProto.from_single_dict()` 会严格检查所有 non-tensor 字段长度必须等于 batch size，因此直接报错。

**处理**

- 在 `RLHFDataset._load_local_file()` 中，构造 HF Dataset 前先收集所有记录的 key，并给每条记录补齐缺失字段：

```python
all_keys = set()
for record in records:
    all_keys.update(record.keys())

for record in records:
    for key in all_keys:
        record.setdefault(key, None)
```

- 这样 `images` / `videos` 两列都会保留下来，image 样本的 `videos=None`，video 样本的 `images=None`。
- 同时保留 `RLHFDataset.__getitem__()` 的兜底默认值：

```python
example["multi_modal_data"] = None
```

- 正常图片/视频分支会覆盖为：

```python
example["multi_modal_data"] = {"images": images}
example["multi_modal_data"] = {"videos": videos}
```

- `vllm_rollout_spmd.py` 也兼容了兜底情况：
  - `multi_modal_data is None` 时只传 `prompt_token_ids`。
  - `multi_modal_data` 是 dict 时才调用 `_process_multi_modal_data()` 并传给 vLLM。

**验证**

- 修复前扫描 `RLHFDataset.dataset` 前 5000 条：

```text
Counter({'image_branch': 2790, 'miss_branch': 2210})
```

- 典型 miss 样本：

```text
data_type='video', images=None, videos=None
keys=['problem_id', 'problem', 'answer', 'data_type', 'problem_type', 'options', 'data_source', 'images']
```

- 修复后扫描前 5000 条：

```text
Counter({'image_branch': 2790, 'video_branch': 2210})
examples []
```

- `dataset.py` 通过 `py_compile`。

**问题 9：RL 视频读取落到 torchvision 后失败**

```text
qwen-vl-utils using torchvision to read video.
AttributeError: module 'torchvision.io' has no attribute 'read_video'
```

**原因**

- `.venv_rl` 原本没有 `decord`。
- 当前 `qwen_vl_utils` 支持的 video backend 是 `decord` / `torchvision` / `torchcodec`。
- 没有强制指定 backend 时，它落到了 `torchvision`。
- 当前环境的 `torchvision 0.26.0+cu130` 已没有 `torchvision.io.read_video`。

**处理**

- 给 `.venv_rl` 安装 `decord`：

```bash
/tianyuesong/zy/.uv/uv pip install --python .venv_rl/bin/python decord
```

- 在 `scripts/train/run_rl.sh` 中设置默认 video reader：

```bash
export FORCE_QWENVL_VIDEO_READER=${FORCE_QWENVL_VIDEO_READER:-decord}
```

**验证**

- 单条 video 样本验证：

```text
qwen-vl-utils using decord to read video.
problem_id 2
data_type video
multi_modal_data {'videos': ['./data/LLaVA-Video-178K/liwei_youtube_videos/videos/youtube_video_2024/ytb_7nRmsEw7nsE.mp4']}
raw has video token True
raw len 152
```

- 小 batch 验证：

```text
input_ids bsz 4
multi_modal_data len 4
modal keys [['images'], ['videos'], ['videos'], ['images']]
DataProto len 4 ok
```

**问题 10：decord 对个别视频解码失败后仍 fallback 到坏掉的 torchvision**

```text
decord._ffi.base.DECORDError:
Check failed: avcodec_send_packet(dec_ctx_.get(), pkt.get()) >= 0 (-11 vs. 0)
Thread worker: Error sending packet.

During handling of the above exception, another exception occurred:

AttributeError: module 'torchvision.io' has no attribute 'read_video'
```

**原因**

- 设置 `FORCE_QWENVL_VIDEO_READER=decord` 只能让首选 reader 变成 decord。
- 但 `qwen_vl_utils.fetch_video()` 内部逻辑是：
  - 先用当前 backend 读取视频。
  - 如果当前 backend 报错，就硬编码 fallback 到 `torchvision`。
- 个别视频 decord 会因为编码/损坏 packet 报 `DECORDError`。
- fallback 到 torchvision 后，又撞上当前环境 `torchvision.io.read_video` 不存在，最终 dataloader worker 崩溃。

**处理**

- 在 `EasyR1/verl/utils/dataset.py` 的 `process_video()` 外层 catch 住整个 `fetch_video()` 失败。
- 失败后不再让它继续依赖 torchvision，而是走自定义 PyAV fallback：
  - 用 `av.open()` 解码视频帧。
  - 用 `qwen_vl_utils.smart_nframes()` 按相同 `fps/max_frames` 规则采样。
  - 把采样后的 PIL frame list 交回 `fetch_video()` 的 list-of-frames 分支处理 resize 和 metadata。
- fallback 时会打印失败视频路径，方便后续定位坏样本：

```text
fetch_video failed for <video_path>, falling back to pyav: <error>
```

**验证**

- 直接测试 PyAV fallback：

```text
pyav (88, 3, 384, 192) 1.9681810726586848 dict_keys(['fps', 'frames_indices', 'total_num_frames'])
```

- 正常 `process_video()` 入口仍可走 decord：

```text
qwen-vl-utils using decord to read video.
process_video (88, 3, 416, 224) 1.9681810726586848 dict_keys(['fps', 'frames_indices', 'total_num_frames', 'video_backend'])
```

- 强制模拟字符串视频读取失败时，`process_video()` 能进入 PyAV fallback：

```text
fetch_video failed for ./data/LLaVA-Video-178K/liwei_youtube_videos/videos/youtube_video_2024/ytb_7nRmsEw7nsE.mp4, falling back to pyav: forced string-video decode failure
forced fallback ok (88, 3, 384, 192) 1.9681810726586848 dict_keys(['fps', 'frames_indices', 'total_num_frames'])
```

**最终 smoke 结果**

- `models/TimeThinker-4B-SFT-v3-10000` checkpoint 成功加载。
- vLLM 初始化、权重同步、图片 generation 成功。
- reward 成功计算。

```text
Finish validation.
Final validation metrics:
val:
  accuracy_reward: 1.0
  format_reward: 1.0
  overall_reward: 1.5
  reward_score: 1.5
  structure_reward_reward: 0.5
val_prompt_length:
  clip_ratio: 0.0
  max: 287.0
  mean: 287.0
  min: 287.0
val_response_length:
  clip_ratio: 0.0
  max: 231.0
  mean: 231.0
  min: 231.0
```

**注意**

- 这个 smoke test 是单卡、单样本、`max_steps=0` 的最小路径验证，不等价于完整 RL 训练。
- 由于 `ray_trainer.py` 在 `max_steps=0` 后仍会做 final validation，所以测试时把 `data.val_files` 指向了 `/tmp/timethinker_rl_smoke_val.json`，避免跑完整验证集。
- Ray 的 `/dev/shm` 警告、Transformers v4 deprecation、FlashAttention dtype warning 在本次测试中不是阻塞错误。

## 16. 评测结果、summary 和 cache 对齐

**当前输出结构**

- `scripts/eval/run_bench.sh` 默认输出到：

```text
Evaluation/results/<model_tag>/frames<MAX_FRAMES>/eval_<benchmark>.json
```

- `SUMMARY_RESULTS=1` 时会自动在每个模型结果目录生成：

```text
_summary.json
_summary.md
```

- 这是“一个模型目录一个 summary”，summary 内按 benchmark 聚合；不是每个 benchmark 单独一个 summary。

**当前默认 benchmark**

- `eval_longvideoreason`
- `eval_videommmu`
- `eval_mvbench`
- `eval_tempcompass`
- `eval_videomathqa`
- `eval_mmvu`
- `eval_videomme`
- `eval_vsibench`

grounding / tracking / segmentation 这类题型已经不放在默认评测链路里，因为当前训练目标不是这类能力。

**当前核心指标**

- `answer_acc`
- `macro_avg/by_benchmark`
- `per_category_acc`
- `answer_extract_rate`
- `invalid_answer_rate`
- `avg_output_tokens`
- `truncation_rate`
- `bootstrap_ci`
- 每个 benchmark 的 `elapsed_min`
- frame cache 的 `hit` / `miss` / `write` / `fallback_to_pyav`

**smoke / 全量复用关系**

- frame cache 和结果 JSON 是两套东西：
  - frame cache 复用抽帧结果。
  - 结果 JSON 负责 resume 已评过的样本。
- 先跑 `MAX_SAMPLES=800`、只跑两个 benchmark，后续跑全集时可以继续用同一个 `FRAME_CACHE_DIR`。
- 但是建议用 `RESULT_SUFFIX` 区分 smoke 和 full，例如：

```bash
RESULT_SUFFIX=_smoke800 MAX_SAMPLES=800 bash scripts/eval/run_bench.sh
RESULT_SUFFIX=_full bash scripts/eval/run_bench.sh
```

- 这样可以避免 full eval 误 resume 到 smoke 结果，同时 frame cache 仍然能复用。

**慢 benchmark 判断**

- 长视频 benchmark 慢主要来自视频 IO、seek、解码、PIL/frame 转换、processor 打包和模型生成共同叠加。
- `LongVideoReason`、`VideoMMMU` 这类长视频/长上下文 benchmark 更容易被抽帧和预处理拖慢。
- 对比 cache 前后耗时，优先看 `_summary.md` 里的 `elapsed_min`、`cache_hit`、`cache_miss` 和 `fallback_pyav`。

## 17. 后续建议

- 每次改训练配置时，同步记录：
  - config 文件路径
  - output_dir
  - cache 路径
  - SwanLab run name
  - benchmark 结果文件
- SFT、RL、Eval 的视觉分辨率和帧数尽量对齐。
- 大规模评测前先 `MAX_SAMPLES=1` smoke test。
- 如果 benchmark 很慢，优先检查 CPU/视频解码，而不是只看 GPU 利用率。
- bad case 统一记录到 `docs/bad_case.md` 的模板中。
