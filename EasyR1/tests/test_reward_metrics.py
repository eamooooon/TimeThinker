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

import unittest

from scripts.prompting.timethinker import build_prompt
from verl.trainer.metrics import (
    compute_group_score_stats,
    estimate_adaptive_refill_size,
    normalize_problem_type,
    summarize_group_score_stats,
    summarize_reward_metrics_by_problem_type,
)


class RewardMetricsTest(unittest.TestCase):
    def test_adaptive_refill_uses_type_specific_keep_rates(self):
        candidates = ["multiple choice", "OCR", "regression", "OCR", "regression", "OCR"]
        size = estimate_adaptive_refill_size(
            candidates,
            remaining_groups=2,
            keep_rate_ema={"multiple_choice": 0.2, "ocr": 0.5, "regression": 0.8},
            default_keep_rate=0.3,
            oversample=1.0,
            min_batch_size=2,
        )
        # Expected kept count by prefix: .2, .7, 1.5, 2.0.
        self.assertEqual(size, 4)

    def test_adaptive_refill_respects_minimum_and_available_candidates(self):
        self.assertEqual(
            estimate_adaptive_refill_size(
                ["OCR"] * 8,
                remaining_groups=1,
                keep_rate_ema={"ocr": 1.0},
                default_keep_rate=0.3,
                oversample=1.0,
                min_batch_size=2,
            ),
            2,
        )
        self.assertEqual(
            estimate_adaptive_refill_size(
                ["multiple choice"] * 3,
                remaining_groups=5,
                keep_rate_ema={"multiple_choice": 0.2},
                default_keep_rate=0.3,
            ),
            3,
        )

    def test_adaptive_refill_aligns_to_rollout_world_size(self):
        size = estimate_adaptive_refill_size(
            ["OCR"] * 16,
            remaining_groups=1,
            keep_rate_ema={"ocr": 0.5},
            default_keep_rate=0.3,
            oversample=1.0,
            min_batch_size=2,
            batch_size_multiple=4,
        )
        self.assertEqual(size, 4)
        self.assertEqual(size % 4, 0)

    def test_adaptive_refill_never_exceeds_generation_cap(self):
        size = estimate_adaptive_refill_size(
            ["multiple choice"] * 24,
            remaining_groups=16,
            keep_rate_ema={"multiple_choice": 0.01},
            default_keep_rate=0.3,
            batch_size_multiple=4,
            max_batch_size=16,
        )
        self.assertEqual(size, 16)

    def test_problem_type_normalization_and_ocr_prompt(self):
        self.assertEqual(normalize_problem_type("OCR"), "ocr")
        self.assertEqual(normalize_problem_type(" open-ended "), "open_ended")
        self.assertEqual(normalize_problem_type("free-form"), "open_ended")
        self.assertIn("must be only the transcribed text", build_prompt("Read this image.", "OCR"))

    def test_summarize_reward_metrics_by_problem_type(self):
        metrics = summarize_reward_metrics_by_problem_type(
            ["OCR", "OCR", "multiple choice", "multiple choice"],
            {
                "accuracy": [0.2, 0.6, 0.0, 1.0],
                "overall": [0.24, 0.62, 0.05, 1.0],
                "format": [1.0, 1.0, 1.0, 1.0],
            },
            prefix="val/by_problem_type",
            mean_suffix=False,
        )

        self.assertEqual(metrics["val/by_problem_type/ocr/sample_count"], 2)
        self.assertAlmostEqual(metrics["val/by_problem_type/ocr/accuracy"], 0.4)
        self.assertAlmostEqual(metrics["val/by_problem_type/multiple_choice/accuracy"], 0.5)

    def test_group_score_stats_distinguish_mean_from_variance(self):
        uids = ["useful"] * 4 + ["constant"] * 4
        problem_types = ["multiple choice"] * 4 + ["open-ended"] * 4
        scores = [0.0, 0.0, 1.0, 1.0] + [0.5, 0.5, 0.5, 0.5]
        group_stats = compute_group_score_stats(uids, problem_types, scores)
        group_stats["useful"].update(
            mean_pass=True,
            filter_variance_pass=True,
            outcome_variance_pass=True,
            variance_pass=True,
            kept=True,
        )
        group_stats["constant"].update(
            mean_pass=True,
            filter_variance_pass=False,
            outcome_variance_pass=False,
            variance_pass=False,
            kept=False,
        )

        self.assertAlmostEqual(group_stats["useful"]["mean"], 0.5)
        self.assertGreater(group_stats["useful"]["std"], 0)
        self.assertAlmostEqual(group_stats["constant"]["mean"], 0.5)
        self.assertEqual(group_stats["constant"]["std"], 0)

        metrics = summarize_group_score_stats(
            group_stats,
            prefix="train/filter",
            score_name="outcome_reward",
        )
        self.assertEqual(metrics["train/filter/open_ended/outcome_reward_zero_std_ratio"], 1.0)
        self.assertEqual(metrics["train/filter/open_ended/filter_variance_pass_ratio"], 0.0)
        self.assertEqual(metrics["train/filter/open_ended/outcome_variance_pass_ratio"], 0.0)
        self.assertEqual(metrics["train/filter/open_ended/kept_ratio"], 0.0)
        self.assertEqual(metrics["train/filter/multiple_choice/kept_ratio"], 1.0)

    def test_group_score_stats_reject_misaligned_inputs(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            compute_group_score_stats(["uid"], ["OCR", "OCR"], [0.5])


if __name__ == "__main__":
    unittest.main()
