# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Dict, Optional, Tuple

from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16

    def post_init(self):
        if self.image_dir is not None:
            if os.path.exists(self.image_dir):  # ray job uses absolute path
                self.image_dir = os.path.abspath(self.image_dir)
            else:
                print(f"Image directory {self.image_dir} not found.")
                self.image_dir = None

        if self.format_prompt is not None:
            if os.path.exists(self.format_prompt):  # ray job uses absolute path
                self.format_prompt = os.path.abspath(self.format_prompt)
            else:
                print(f"Format prompt file {self.format_prompt} not found.")
                self.format_prompt = None


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator, support `gae`, `grpo`, `reinforce_plus_plus`, `remax`, `rloo`"""
    disable_kl: bool = False
    """disable reference model"""
    use_kl_loss: bool = False
    """use kl loss instead of kl in reward"""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_coef: float = 1e-3
    """kl coefficient"""
    kl_type: str = "fixed"
    """kl controller type, support `fixed`, `adaptive`"""
    kl_horizon: float = 10000.0
    """kl horizon for adaptive kl controller"""
    kl_target: float = 0.1
    """target kl for adaptive kl controller"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""
    filter_key_min_std: float = 0.0
    """minimum group std of filter_key; rejects groups with no answer-quality ranking signal"""
    filter_min_std: float = 0.0
    """minimum group std of the final outcome reward; 0 keeps backward-compatible filtering"""
    filter_type_min_std: Dict[str, float] = field(default_factory=dict)
    """optional problem-type overrides for minimum filter-key std"""
    filter_type_min_range: Dict[str, float] = field(default_factory=dict)
    """optional minimum filter-key range by problem type"""
    filter_type_max_ratio: Dict[str, float] = field(default_factory=dict)
    """optional caps on each problem type's share of accepted prompt groups"""
    filter_adaptive_refill: bool = False
    """adapt refill generation size to the remaining groups and per-type keep-rate EMA"""
    filter_keep_rate_ema_alpha: float = 0.1
    """EMA update weight assigned to the latest per-type keep rate"""
    filter_keep_rate_default: float = 0.3
    """initial keep-rate estimate for problem types without an explicit value"""
    filter_type_initial_keep_rate: Dict[str, float] = field(default_factory=dict)
    """optional initial filtering keep-rate estimate by normalized problem type"""
    filter_refill_oversample: float = 1.2
    """expected accepted-group safety factor when sizing a refill rollout"""
    filter_refill_min_batch_size: int = 2
    """minimum number of prompt groups in an adaptive refill rollout"""
    temporal: bool = False
    """enable Video-R1 style T-GRPO temporal reward shaping"""
    shuffled_rollout_ratio: float = 0.5
    """number of shuffled-frame rollouts as a ratio of rollout.n"""
    temporal_reward: float = 0.3
    """reward bonus added to correct ordered video responses when they beat shuffled frames"""
    temporal_compare_ratio: float = 0.8
    """ordered accuracy must be >= this ratio * shuffled accuracy"""
    temporal_correct_threshold: float = 0.1
    """accuracy threshold for responses eligible for temporal reward"""
    len_control: bool = False
    """enable Video-R1 length-control reward"""
    len_reward: float = 0.2
    """reward bonus for correct responses in the target length range"""
    len_min: int = 320
    """minimum response length for length-control reward"""
    len_max: int = 512
    """maximum response length for length-control reward"""

    def post_init(self):
        if self.filter_key_min_std < 0 or self.filter_min_std < 0:
            raise ValueError(
                "filter_key_min_std and filter_min_std must be non-negative, got "
                f"{self.filter_key_min_std} and {self.filter_min_std}."
            )

        for name, values in (
            ("filter_type_min_std", self.filter_type_min_std),
            ("filter_type_min_range", self.filter_type_min_range),
        ):
            invalid = {key: value for key, value in values.items() if value < 0}
            if invalid:
                raise ValueError(f"{name} values must be non-negative, got {invalid}.")

        invalid_ratios = {
            key: value for key, value in self.filter_type_max_ratio.items() if not 0 < value <= 1
        }
        if invalid_ratios:
            raise ValueError(f"filter_type_max_ratio values must be in (0, 1], got {invalid_ratios}.")

        if not 0 < self.filter_keep_rate_ema_alpha <= 1:
            raise ValueError("filter_keep_rate_ema_alpha must be in (0, 1].")
        if not 0 < self.filter_keep_rate_default <= 1:
            raise ValueError("filter_keep_rate_default must be in (0, 1].")
        invalid_keep_rates = {
            key: value for key, value in self.filter_type_initial_keep_rate.items() if not 0 < value <= 1
        }
        if invalid_keep_rates:
            raise ValueError(f"filter_type_initial_keep_rate values must be in (0, 1], got {invalid_keep_rates}.")
        if self.filter_refill_oversample < 1:
            raise ValueError("filter_refill_oversample must be at least 1.")
        if self.filter_refill_min_batch_size < 1:
            raise ValueError("filter_refill_min_batch_size must be at least 1.")

        if self.online_filtering and self.filter_low >= self.filter_high:
            raise ValueError(
                f"filter_low must be smaller than filter_high, got {self.filter_low} >= {self.filter_high}."
            )


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "easy_r1"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    ray_timeline: Optional[str] = None
    """file to save ray timeline"""
    find_last_checkpoint: bool = True
    """automatically find the last checkpoint in the save checkpoint path to resume training"""

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # ray job uses absolute path
        if self.load_checkpoint_path is not None:
            if os.path.exists(self.load_checkpoint_path):  # ray job uses absolute path
                self.load_checkpoint_path = os.path.abspath(self.load_checkpoint_path)
            else:
                print(f"Model checkpoint {self.load_checkpoint_path} not found.")
                self.load_checkpoint_path = None


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
