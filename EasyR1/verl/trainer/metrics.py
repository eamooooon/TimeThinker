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

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from ..protocol import DataProto


def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    return {key: np.mean(value) for key, value in metrics.items()}


def normalize_problem_type(problem_type: Any) -> str:
    """Normalize problem types for stable metric keys."""
    normalized = str(problem_type or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"[^a-z0-9_]+", "_", normalized).strip("_")
    if normalized == "free_form":
        return "open_ended"

    return normalized or "unknown"


def estimate_adaptive_refill_size(
    candidate_problem_types: Sequence[Any],
    remaining_groups: int,
    keep_rate_ema: Mapping[str, float],
    default_keep_rate: float,
    oversample: float = 1.2,
    min_batch_size: int = 2,
    batch_size_multiple: int = 1,
    max_batch_size: int | None = None,
) -> int:
    """Size a refill so its expected accepted groups cover the remaining target.

    Candidates stay in their dataloader order. Their type-specific EMA probabilities
    are accumulated until the expected yield reaches ``remaining * oversample``.
    """
    candidates = list(candidate_problem_types)
    if max_batch_size is not None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        candidates = candidates[:max_batch_size]

    if remaining_groups <= 0 or not candidates:
        return 0

    if batch_size_multiple < 1:
        raise ValueError("batch_size_multiple must be at least 1")

    def align_size(size: int) -> int:
        max_aligned_size = (len(candidates) // batch_size_multiple) * batch_size_multiple
        if max_aligned_size == 0:
            raise ValueError(
                f"Need at least {batch_size_multiple} candidates for an aligned refill, "
                f"got {len(candidates)}"
            )
        aligned = ((size + batch_size_multiple - 1) // batch_size_multiple) * batch_size_multiple
        return min(aligned, max_aligned_size)

    min_batch_size = min(max(1, min_batch_size), len(candidates))
    expected_kept = 0.0
    target_expected_kept = remaining_groups * oversample
    for index, raw_problem_type in enumerate(candidates, start=1):
        problem_type = normalize_problem_type(raw_problem_type)
        keep_rate = float(keep_rate_ema.get(problem_type, default_keep_rate))
        expected_kept += min(1.0, max(1e-6, keep_rate))
        if index >= min_batch_size and expected_kept >= target_expected_kept:
            return align_size(index)

    return align_size(len(candidates))


def summarize_reward_metrics_by_problem_type(
    problem_types: Sequence[Any],
    reward_metrics: Mapping[str, Sequence[Any]],
    prefix: str,
    metric_names: Sequence[str] = ("accuracy", "overall", "format"),
    mean_suffix: bool = True,
) -> dict[str, Any]:
    """Compute per-problem-type sample means and counts from aligned reward arrays."""
    normalized_types = [normalize_problem_type(problem_type) for problem_type in problem_types]
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for idx, problem_type in enumerate(normalized_types):
        grouped_indices[problem_type].append(idx)

    result: dict[str, Any] = {}
    for problem_type, indices in sorted(grouped_indices.items()):
        result[f"{prefix}/{problem_type}/sample_count"] = len(indices)
        for metric_name in metric_names:
            if metric_name not in reward_metrics:
                continue

            values = reward_metrics[metric_name]
            if len(values) != len(normalized_types):
                raise ValueError(
                    f"Reward metric {metric_name!r} has {len(values)} values, "
                    f"but there are {len(normalized_types)} problem types."
                )

            metric_key = f"{metric_name}_mean" if mean_suffix else metric_name
            result[f"{prefix}/{problem_type}/{metric_key}"] = float(
                np.mean([float(values[idx]) for idx in indices])
            )

    return result


def compute_group_score_stats(
    uids: Sequence[Any], problem_types: Sequence[Any], scores: Sequence[Any]
) -> dict[Any, dict[str, Any]]:
    """Group aligned scalar scores by prompt UID and compute outcome statistics."""
    if not (len(uids) == len(problem_types) == len(scores)):
        raise ValueError(
            "uids, problem_types, and scores must have the same length: "
            f"{len(uids)}, {len(problem_types)}, {len(scores)}"
        )

    grouped_scores: dict[Any, list[float]] = defaultdict(list)
    grouped_types: dict[Any, str] = {}
    for uid, raw_problem_type, score in zip(uids, problem_types, scores):
        problem_type = normalize_problem_type(raw_problem_type)
        previous_type = grouped_types.setdefault(uid, problem_type)
        if previous_type != problem_type:
            raise ValueError(f"UID {uid!r} contains multiple problem types: {previous_type!r}, {problem_type!r}")

        grouped_scores[uid].append(float(score))

    result: dict[Any, dict[str, Any]] = {}
    for uid, values in grouped_scores.items():
        score_array = np.asarray(values, dtype=np.float64)
        result[uid] = {
            "problem_type": grouped_types[uid],
            "sample_count": len(values),
            "mean": float(np.mean(score_array)),
            "std": float(np.std(score_array)),
            "range": float(np.max(score_array) - np.min(score_array)),
        }

    return result


def summarize_group_score_stats(
    group_stats: Mapping[Any, Mapping[str, Any]],
    prefix: str,
    score_name: str,
    zero_std_threshold: float = 1e-6,
) -> dict[str, Any]:
    """Reduce precomputed group statistics by problem type for experiment logging."""
    stats_by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for stats in group_stats.values():
        stats_by_type[str(stats["problem_type"])].append(stats)

    result: dict[str, Any] = {}
    for problem_type, typed_stats in sorted(stats_by_type.items()):
        means = [float(stats["mean"]) for stats in typed_stats]
        stds = [float(stats["std"]) for stats in typed_stats]
        ranges = [float(stats["range"]) for stats in typed_stats]
        base_key = f"{prefix}/{problem_type}"
        result[f"{base_key}/group_count"] = len(typed_stats)
        result[f"{base_key}/{score_name}_mean"] = float(np.mean(means))
        result[f"{base_key}/{score_name}_group_std_mean"] = float(np.mean(stds))
        result[f"{base_key}/{score_name}_group_std_min"] = float(np.min(stds))
        result[f"{base_key}/{score_name}_group_range_mean"] = float(np.mean(ranges))
        result[f"{base_key}/{score_name}_zero_std_ratio"] = float(
            np.mean([std <= zero_std_threshold for std in stds])
        )

        for flag_name in (
            "mean_pass",
            "filter_variance_pass",
            "outcome_variance_pass",
            "variance_pass",
            "signal_pass",
            "quota_pass",
            "kept",
        ):
            if all(flag_name in stats for stats in typed_stats):
                flag_values = [bool(stats[flag_name]) for stats in typed_stats]
                result[f"{base_key}/{flag_name}_group_count"] = int(sum(flag_values))
                result[f"{base_key}/{flag_name}_ratio"] = float(np.mean(flag_values))

    return result


def compute_length_metrics(batch: DataProto) -> dict[str, Any]:
    max_response_length = batch.batch["responses"].size(-1)
    max_prompt_length = batch.batch["attention_mask"].size(-1) - max_response_length

    prompt_length = batch.batch["attention_mask"][:, :-max_response_length].sum(-1).float()
    response_length = batch.batch["attention_mask"][:, -max_response_length:].sum(-1).float()

    return {
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.eq(response_length, max_response_length).float().mean().detach().item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.eq(prompt_length, max_prompt_length).float().mean().detach().item(),
    }


def compute_data_metrics(batch: DataProto, use_critic: bool = False) -> dict[str, Any]:
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].size(-1)
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    return {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        **compute_length_metrics(batch),
    }


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    num_response_tokens = torch.sum(batch.batch["response_mask"]).item()
    num_overall_tokens = sum(batch.meta_info["global_token_num"])
    num_tokens_of_section = {
        **dict.fromkeys(["gen", "reward"], num_response_tokens),
        **dict.fromkeys(["ref", "old", "values", "adv", "update_critic", "update_actor"], num_overall_tokens),
    }
    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], num_gpus: int) -> dict[str, Any]:
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * num_gpus),
    }
