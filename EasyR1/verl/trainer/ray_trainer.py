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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.dataset import process_video
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_group_score_stats,
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    estimate_adaptive_refill_size,
    normalize_problem_type,
    reduce_metrics,
    summarize_group_score_stats,
    summarize_reward_metrics_by_problem_type,
)


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        # if gpus_available < gpus_required:
            # raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    """Compute advantage estimates for policy optimization."""
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
        "data_type": data.non_tensor_batch["data_type"],
        "problem_type": data.non_tensor_batch["problem_type"],
    }

    # print(adv_inputs)

    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None
        self._pending_prompt_batch = None
        self.filter_keep_rate_ema = {
            normalize_problem_type(problem_type): float(keep_rate)
            for problem_type, keep_rate in config.algorithm.filter_type_initial_keep_rate.items()
        }

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor, rollout and ref
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        val_problem_types = []
        length_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)
            val_problem_types.extend(test_batch.non_tensor_batch["problem_type"].tolist())

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_type_metrics = summarize_reward_metrics_by_problem_type(
            val_problem_types,
            reward_metrics_lst,
            prefix="val/by_problem_type",
            mean_suffix=False,
        )
        val_macro_metrics = {}
        for metric_name in ("accuracy", "overall", "format"):
            typed_values = [
                value for key, value in val_type_metrics.items() if key.endswith(f"/{metric_name}")
            ]
            if typed_values:
                val_macro_metrics[f"val/macro/{metric_name}"] = float(np.mean(typed_values))

        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        print("Finish validation.")
        return {
            "val/reward_score": self.val_reward_score,
            **val_reward_metrics,
            **val_type_metrics,
            **val_macro_metrics,
            **val_length_metrics,
        }

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _get_video_indices(self, gen_batch: DataProto) -> list[int]:
        multi_modal_data = gen_batch.non_tensor_batch.get("multi_modal_data")
        if multi_modal_data is None:
            return []

        video_indices = []
        for idx, mm_data in enumerate(multi_modal_data):
            if isinstance(mm_data, dict) and len(mm_data.get("videos") or []) > 0:
                video_indices.append(idx)

        return video_indices

    def _process_video_for_shuffle(self, video: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        if isinstance(video, tuple) and len(video) == 2 and isinstance(video[1], dict):
            return video[0], deepcopy(video[1])

        processed_video, _ = process_video(
            video,
            min_pixels=self.config.data.min_pixels,
            max_pixels=self.config.data.max_pixels,
            video_fps=self.config.data.video_fps,
            return_fps=True,
        )

        if (
            isinstance(processed_video, tuple)
            and len(processed_video) == 2
            and isinstance(processed_video[0], torch.Tensor)
            and isinstance(processed_video[1], dict)
        ):
            return processed_video[0], deepcopy(processed_video[1])

        raise TypeError(f"Unsupported processed video type for T-GRPO shuffle: {type(processed_video)}")

    def _build_shuffled_video_mm_data(self, mm_data: dict[str, Any]) -> Optional[dict[str, Any]]:
        shuffled_mm_data = deepcopy(mm_data)
        shuffled_videos = []
        for video in mm_data.get("videos") or []:
            video_tensor, metadata = self._process_video_for_shuffle(video)
            if len(video_tensor) == 0:
                return None

            indices = np.random.permutation(len(video_tensor))
            shuffled_tensor = video_tensor[torch.as_tensor(indices, device=video_tensor.device)]
            if isinstance(metadata.get("frames_indices"), list) and len(metadata["frames_indices"]) == len(indices):
                metadata["frames_indices"] = [metadata["frames_indices"][i] for i in indices]
            shuffled_videos.append((shuffled_tensor, metadata))

        shuffled_mm_data["videos"] = shuffled_videos
        return shuffled_mm_data

    def _build_shuffled_gen_batch(self, gen_batch: DataProto, video_indices: list[int]) -> Optional[DataProto]:
        if len(video_indices) == 0:
            return None

        shuffled_gen_batch = gen_batch[video_indices]
        shuffled_gen_batch.meta_info = deepcopy(gen_batch.meta_info)
        shuffled_n = max(1, int(self.config.worker.rollout.n * self.config.algorithm.shuffled_rollout_ratio))
        shuffled_gen_batch.meta_info["n"] = shuffled_n

        shuffled_mm_data = []
        for mm_data in shuffled_gen_batch.non_tensor_batch["multi_modal_data"]:
            try:
                shuffled = self._build_shuffled_video_mm_data(mm_data)
            except Exception as exc:
                print(f"[T-GRPO] skip temporal shuffle for one sample: {exc}")
                return None
            if shuffled is None:
                return None
            shuffled_mm_data.append(shuffled)

        shuffled_gen_batch.non_tensor_batch["multi_modal_data"] = np.array(shuffled_mm_data, dtype=object)
        return shuffled_gen_batch

    @staticmethod
    def _add_last_token_bonus(reward_tensor: torch.Tensor, response_mask: torch.Tensor, bonus: np.ndarray) -> None:
        if len(bonus) == 0:
            return

        bonus_tensor = torch.as_tensor(bonus, dtype=reward_tensor.dtype, device=reward_tensor.device)
        response_lengths = response_mask.sum(dim=-1).long().to(reward_tensor.device)
        valid = response_lengths > 0
        if not torch.any(valid):
            return

        row_idx = torch.arange(reward_tensor.shape[0], device=reward_tensor.device)[valid]
        col_idx = response_lengths[valid] - 1
        reward_tensor[row_idx, col_idx] += bonus_tensor[valid]

    def _apply_tgrpo_rewards(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_metrics: dict[str, list[float]],
        shuffled_batch: Optional[DataProto],
    ) -> None:
        accuracy = np.asarray(reward_metrics.get("accuracy", [0.0] * len(batch)), dtype=np.float32)
        temporal_bonus = np.zeros(len(batch), dtype=np.float32)
        temporal_applied = np.zeros(len(batch), dtype=np.float32)
        shuffled_group_accuracy = []
        temporal_group_counts = {
            "ordered_correct_shuffled_wrong": 0,
            "ordered_correct_shuffled_correct": 0,
            "ordered_wrong_shuffled_correct": 0,
            "ordered_wrong_shuffled_wrong": 0,
            "ordered_gt_shuffled": 0,
            "compare_pass": 0,
            "total": 0,
        }

        if self.config.algorithm.temporal and shuffled_batch is not None:
            shuffled_reward_tensor, shuffled_reward_metrics = ray.get(self.reward_fn.compute_reward.remote(shuffled_batch))
            del shuffled_reward_tensor

            shuffled_accuracy = np.asarray(
                shuffled_reward_metrics.get("accuracy", [0.0] * len(shuffled_batch)), dtype=np.float32
            )
            ordered_uids = np.asarray(batch.non_tensor_batch["uid"], dtype=object)
            shuffled_uids = np.asarray(shuffled_batch.non_tensor_batch["uid"], dtype=object)
            video_uids = set(shuffled_uids.tolist())

            for uid in video_uids:
                ordered_idx = np.flatnonzero(ordered_uids == uid)
                shuffled_idx = np.flatnonzero(shuffled_uids == uid)
                if len(ordered_idx) == 0 or len(shuffled_idx) == 0:
                    continue

                ordered_acc = float(np.mean(accuracy[ordered_idx]))
                shuffled_acc = float(np.mean(shuffled_accuracy[shuffled_idx]))
                shuffled_group_accuracy.extend([shuffled_acc] * len(ordered_idx))
                ordered_correct = ordered_acc > self.config.algorithm.temporal_correct_threshold
                shuffled_correct = shuffled_acc > self.config.algorithm.temporal_correct_threshold
                temporal_group_counts["total"] += 1
                temporal_group_counts["ordered_gt_shuffled"] += int(ordered_acc > shuffled_acc)

                compare_pass = ordered_acc >= self.config.algorithm.temporal_compare_ratio * shuffled_acc
                temporal_group_counts["compare_pass"] += int(compare_pass)

                if ordered_correct and shuffled_correct:
                    temporal_group_counts["ordered_correct_shuffled_correct"] += 1
                elif ordered_correct and not shuffled_correct:
                    temporal_group_counts["ordered_correct_shuffled_wrong"] += 1
                elif not ordered_correct and shuffled_correct:
                    temporal_group_counts["ordered_wrong_shuffled_correct"] += 1
                else:
                    temporal_group_counts["ordered_wrong_shuffled_wrong"] += 1

                if compare_pass:
                    correct_idx = ordered_idx[accuracy[ordered_idx] > self.config.algorithm.temporal_correct_threshold]
                    temporal_bonus[correct_idx] = self.config.algorithm.temporal_reward
                    temporal_applied[correct_idx] = 1.0

        length_bonus = np.zeros(len(batch), dtype=np.float32)
        if self.config.algorithm.len_control:
            response_lengths = batch.batch["response_mask"].sum(dim=-1).detach().cpu().numpy()
            correct = accuracy > self.config.algorithm.temporal_correct_threshold
            in_range = (response_lengths >= self.config.algorithm.len_min) & (
                response_lengths <= self.config.algorithm.len_max
            )
            length_bonus[correct & in_range] = self.config.algorithm.len_reward

        total_bonus = temporal_bonus + length_bonus
        self._add_last_token_bonus(reward_tensor, batch.batch["response_mask"], total_bonus)

        final_overall = reward_tensor.sum(dim=-1).detach().cpu().tolist()
        reward_metrics["final_overall"] = final_overall
        reward_metrics["temporal_bonus"] = temporal_bonus.tolist()
        reward_metrics["temporal_applied"] = temporal_applied.tolist()
        if len(shuffled_group_accuracy) > 0:
            reward_metrics["shuffled_accuracy"] = shuffled_group_accuracy
        if temporal_group_counts["total"] > 0:
            total = float(temporal_group_counts["total"])
            reward_metrics["temporal_video_groups"] = [float(temporal_group_counts["total"])]
            reward_metrics["temporal_ordered_correct_shuffled_wrong_ratio"] = [
                temporal_group_counts["ordered_correct_shuffled_wrong"] / total
            ]
            reward_metrics["temporal_ordered_correct_shuffled_correct_ratio"] = [
                temporal_group_counts["ordered_correct_shuffled_correct"] / total
            ]
            reward_metrics["temporal_ordered_wrong_shuffled_correct_ratio"] = [
                temporal_group_counts["ordered_wrong_shuffled_correct"] / total
            ]
            reward_metrics["temporal_ordered_wrong_shuffled_wrong_ratio"] = [
                temporal_group_counts["ordered_wrong_shuffled_wrong"] / total
            ]
            reward_metrics["temporal_ordered_gt_shuffled_ratio"] = [
                temporal_group_counts["ordered_gt_shuffled"] / total
            ]
            reward_metrics["temporal_compare_pass_ratio"] = [temporal_group_counts["compare_pass"] / total]
        reward_metrics["length_bonus"] = length_bonus.tolist()

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        pre_filter_problem_types = []
        pre_filter_reward_metrics = defaultdict(list)
        accepted_problem_types = []
        accepted_reward_metrics = defaultdict(list)
        filter_group_stats = {}
        outcome_group_stats = {}
        accepted_group_counts = defaultdict(int)
        has_reward_metrics = False
        num_try_make_batch = 0
        generated_group_count = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }

            prompt_pool = self._pending_prompt_batch
            target_pool_size = self.config.data.rollout_batch_size
            while prompt_pool is None or len(prompt_pool) < target_pool_size:
                try:
                    batch_dict = next(self.data_iterator)
                except StopIteration:
                    self.data_iterator = iter(self.train_dataloader)
                    batch_dict = next(self.data_iterator)
                fetched_batch = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
                prompt_pool = (
                    DataProto.concat([prompt_pool, fetched_batch]) if prompt_pool is not None else fetched_batch
                )

            current_batch_size = 0 if batch is None else len(batch) // self.config.worker.rollout.n
            remaining_groups = self.config.data.rollout_batch_size - current_batch_size
            if self.config.algorithm.filter_adaptive_refill and batch is not None:
                requested_group_count = estimate_adaptive_refill_size(
                    prompt_pool.non_tensor_batch["problem_type"].tolist(),
                    remaining_groups=remaining_groups,
                    keep_rate_ema=self.filter_keep_rate_ema,
                    default_keep_rate=self.config.algorithm.filter_keep_rate_default,
                    oversample=self.config.algorithm.filter_refill_oversample,
                    min_batch_size=self.config.algorithm.filter_refill_min_batch_size,
                    batch_size_multiple=self.actor_rollout_ref_wg.world_size,
                    max_batch_size=target_pool_size,
                )
            else:
                requested_group_count = min(self.config.data.rollout_batch_size, len(prompt_pool))

            new_batch = prompt_pool[:requested_group_count]
            self._pending_prompt_batch = (
                prompt_pool[requested_group_count:] if requested_group_count < len(prompt_pool) else None
            )
            generated_group_count += requested_group_count
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )

            # pop those keys for generation
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            video_indices = self._get_video_indices(gen_batch)
            shuffled_gen_batch = None
            if self.config.algorithm.temporal:
                shuffled_gen_batch = self._build_shuffled_gen_batch(gen_batch, video_indices)

            # generate a batch
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
            shuffled_gen_batch_output = None
            if shuffled_gen_batch is not None:
                shuffled_n = shuffled_gen_batch.meta_info["n"]
                shuffled_gen_batch, shuffled_pad_size = pad_dataproto_to_divisor(
                    shuffled_gen_batch, self.actor_rollout_ref_wg.world_size
                )
                shuffled_gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(shuffled_gen_batch)
                shuffled_gen_batch_output = unpad_dataproto(
                    shuffled_gen_batch_output, pad_size=shuffled_pad_size * shuffled_n
                )

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            # repeat to align with repeated responses in rollout
            prompt_only_batch = new_batch
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            reward_metrics = None
            if self.config.algorithm.temporal or self.config.algorithm.len_control:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                if shuffled_gen_batch_output is not None:
                    shuffled_n = shuffled_gen_batch.meta_info["n"]
                    base_video_batch = prompt_only_batch[video_indices]
                    shuffled_batch = base_video_batch.repeat(repeat_times=shuffled_n, interleave=True)
                    shuffled_batch = shuffled_batch.union(shuffled_gen_batch_output)
                else:
                    shuffled_batch = None

                self._apply_tgrpo_rewards(new_batch, reward_tensor, reward_metrics, shuffled_batch)
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)
                has_reward_metrics = True

            # filter group
            if self.config.algorithm.online_filtering:
                if "token_level_scores" not in new_batch.batch:
                    reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                    new_batch.batch["token_level_scores"] = reward_tensor
                    for k, v in reward_metrics.items():
                        all_metrics[k].extend(v)
                    has_reward_metrics = True

            selected_sample_idxs = list(range(len(new_batch)))
            if reward_metrics is not None:
                problem_types = new_batch.non_tensor_batch["problem_type"].tolist()
                tracked_reward_metrics = {
                    key: value
                    for key, value in reward_metrics.items()
                    if key in {"accuracy", "overall", "format", "final_overall", "length_bonus"}
                    and len(value) == len(new_batch)
                }
                pre_filter_problem_types.extend(problem_types)
                for key, value in tracked_reward_metrics.items():
                    pre_filter_reward_metrics[key].extend(value)

                if self.config.algorithm.online_filtering:
                    filter_scores = reward_metrics[self.config.algorithm.filter_key]
                    outcome_scores = new_batch.batch["token_level_scores"].sum(dim=-1).detach().cpu().tolist()
                    uids = new_batch.non_tensor_batch["uid"].tolist()
                    current_filter_stats = compute_group_score_stats(uids, problem_types, filter_scores)
                    current_outcome_stats = compute_group_score_stats(uids, problem_types, outcome_scores)
                    kept_uids = set()
                    for uid, group_stats in current_filter_stats.items():
                        problem_type = normalize_problem_type(group_stats["problem_type"])
                        type_min_std = self.config.algorithm.filter_type_min_std.get(
                            problem_type, self.config.algorithm.filter_key_min_std
                        )
                        type_min_range = self.config.algorithm.filter_type_min_range.get(problem_type, 0.0)
                        mean_pass = (
                            group_stats["mean"] > self.config.algorithm.filter_low
                            and group_stats["mean"] < self.config.algorithm.filter_high
                        )
                        filter_variance_pass = (
                            group_stats["std"] >= type_min_std
                            and group_stats["range"] >= type_min_range
                        )
                        outcome_variance_pass = (
                            current_outcome_stats[uid]["std"] >= self.config.algorithm.filter_min_std
                        )
                        variance_pass = filter_variance_pass and outcome_variance_pass
                        signal_pass = mean_pass and variance_pass
                        max_ratio = self.config.algorithm.filter_type_max_ratio.get(problem_type)
                        max_groups = (
                            max(1, int(self.config.data.rollout_batch_size * max_ratio))
                            if max_ratio is not None
                            else None
                        )
                        quota_pass = max_groups is None or accepted_group_counts[problem_type] < max_groups
                        kept = signal_pass and quota_pass
                        group_stats.update(
                            mean_pass=mean_pass,
                            filter_variance_pass=filter_variance_pass,
                            outcome_variance_pass=outcome_variance_pass,
                            variance_pass=variance_pass,
                            signal_pass=signal_pass,
                            quota_pass=quota_pass,
                            kept=kept,
                        )
                        current_outcome_stats[uid].update(
                            mean_pass=mean_pass,
                            filter_variance_pass=filter_variance_pass,
                            outcome_variance_pass=outcome_variance_pass,
                            variance_pass=variance_pass,
                            signal_pass=signal_pass,
                            quota_pass=quota_pass,
                            kept=kept,
                        )
                        if kept:
                            kept_uids.add(uid)
                            accepted_group_counts[problem_type] += 1

                    filter_group_stats.update(current_filter_stats)
                    outcome_group_stats.update(current_outcome_stats)
                    selected_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]

                    type_signal_counts = defaultdict(lambda: [0, 0])
                    for group_stats in current_filter_stats.values():
                        problem_type = normalize_problem_type(group_stats["problem_type"])
                        type_signal_counts[problem_type][1] += 1
                        type_signal_counts[problem_type][0] += int(group_stats["signal_pass"])
                    alpha = self.config.algorithm.filter_keep_rate_ema_alpha
                    for problem_type, (passed, total) in type_signal_counts.items():
                        observed_rate = passed / total
                        old_rate = self.filter_keep_rate_ema.get(
                            problem_type, self.config.algorithm.filter_keep_rate_default
                        )
                        self.filter_keep_rate_ema[problem_type] = (
                            (1.0 - alpha) * old_rate + alpha * observed_rate
                        )

                if selected_sample_idxs:
                    accepted_problem_types.extend([problem_types[idx] for idx in selected_sample_idxs])
                    for key, value in tracked_reward_metrics.items():
                        accepted_reward_metrics[key].extend([value[idx] for idx in selected_sample_idxs])

            if selected_sample_idxs:
                new_batch = new_batch[selected_sample_idxs]
                batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch

            current_batch_size = 0 if batch is None else len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if has_reward_metrics:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})
                target_sample_count = self.config.data.rollout_batch_size * self.config.worker.rollout.n
                final_batch = batch[:target_sample_count]
                metrics["train/filter/attempt_count"] = num_try_make_batch
                metrics["train/filter/generated_group_count"] = generated_group_count
                metrics["train/filter/generated_response_count"] = (
                    generated_group_count * self.config.worker.rollout.n
                )
                for problem_type, keep_rate in sorted(self.filter_keep_rate_ema.items()):
                    metrics[f"train/filter/{problem_type}/keep_rate_ema"] = keep_rate
                if pre_filter_problem_types:
                    metrics.update(
                        summarize_reward_metrics_by_problem_type(
                            pre_filter_problem_types,
                            pre_filter_reward_metrics,
                            prefix="train/reward/pre_filter",
                            metric_names=tuple(pre_filter_reward_metrics.keys()),
                        )
                    )

                if accepted_problem_types:
                    final_problem_types = accepted_problem_types[:target_sample_count]
                    final_reward_metrics = {
                        key: value[:target_sample_count] for key, value in accepted_reward_metrics.items()
                    }
                    metrics.update(
                        summarize_reward_metrics_by_problem_type(
                            final_problem_types,
                            final_reward_metrics,
                            prefix="train/reward/post_filter",
                            metric_names=tuple(final_reward_metrics.keys()),
                        )
                    )

                if filter_group_stats:
                    metrics.update(
                        summarize_group_score_stats(
                            filter_group_stats,
                            prefix="train/filter",
                            score_name="filter_score",
                        )
                    )
                    metrics.update(
                        summarize_group_score_stats(
                            outcome_group_stats,
                            prefix="train/filter",
                            score_name="outcome_reward",
                        )
                    )

                return final_batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch, metrics=metrics)

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # compute reward
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)

                # recompute old_log_probs
                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)

                # compute ref_log_probs
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                # compute values
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                    # apply kl penalty if available
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        # apply kl penalty to reward
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    update_group_stats = compute_group_score_stats(
                        batch.non_tensor_batch["uid"].tolist(),
                        batch.non_tensor_batch["problem_type"].tolist(),
                        batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu().tolist(),
                    )
                    metrics.update(
                        summarize_group_score_stats(
                            update_group_stats,
                            prefix="train/update",
                            score_name="outcome_reward",
                        )
                    )

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                    )

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
