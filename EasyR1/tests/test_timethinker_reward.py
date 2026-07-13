import unittest

from verl.reward_function.timethinker_reward import (
    accuracy_reward,
    compute_score,
    open_ended_accuracy_v3,
    regression_accuracy_v3,
)


class TimeThinkerRewardV3Test(unittest.TestCase):
    def test_ocr_handles_cjk_and_latex_without_space_tokenization(self):
        cjk_score = accuracy_reward("<answer>今天天气很好</answer>", "今天天气真好", "image", "OCR", "v3")
        latex_score = accuracy_reward(
            "<answer>12\\times\\frac{2}{3}=8</answer>",
            "12 \\times \\frac { 2 } { 3 } = 8",
            "image",
            "OCR",
            "v3",
        )
        self.assertGreater(cjk_score, 0.7)
        self.assertGreater(latex_score, 0.8)

    def test_open_ended_penalizes_unsupported_extra_text(self):
        reference = "The person opens the door and sits on the bed."
        concise = open_ended_accuracy_v3(reference, "The person opens the door and sits on the bed.")
        verbose = open_ended_accuracy_v3(
            reference,
            "The person opens the door and sits on the bed, then flies to the moon and wins a race.",
        )
        self.assertEqual(concise, 1.0)
        self.assertLess(verbose, concise)

    def test_regression_reward_is_smooth_and_monotonic(self):
        exact = regression_accuracy_v3(10.0, 10.0)
        close = regression_accuracy_v3(11.0, 10.0)
        far = regression_accuracy_v3(20.0, 10.0)
        self.assertEqual(exact, 1.0)
        self.assertGreater(close, far)
        self.assertGreater(far, 0.0)

    def test_batch_api_selects_v3_explicitly(self):
        inputs = [{
            "response": "<think>estimate</think><answer>11</answer>",
            "ground_truth": "<answer>10</answer>",
            "data_type": "image",
            "problem_type": "regression",
        }]
        legacy = compute_score(inputs, format_weight=0.0, formula_version="legacy")[0]["accuracy"]
        v3 = compute_score(inputs, format_weight=0.0, formula_version="v3")[0]["accuracy"]
        self.assertNotEqual(legacy, v3)


if __name__ == "__main__":
    unittest.main()
