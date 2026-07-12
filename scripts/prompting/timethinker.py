"""Canonical QA prompt shared by SFT, RL, and evaluation."""

from __future__ import annotations


QUESTION_TEMPLATE = (
    "{Question}\n\n"
    "Please answer based on the visual content.\n"
    "Respond exactly in this format:\n"
    "<think>\n"
    "Your reasoning here.\n"
    "</think>\n"
    "<answer>\n"
    "Your final answer here.\n"
    "</answer>\n"
    "\n"
)

TYPE_TEMPLATE = {
    "multiple choice": (
        "The final answer inside <answer> must be only one option letter from the given options.\n"
    ),
    "numerical": "The final answer inside <answer> must be only the numerical value.\n",
    "ocr": "The final answer inside <answer> must be only the transcribed text.\n",
    "open-ended": "The final answer inside <answer> must be only the concise text answer.\n",
    "free-form": "The final answer inside <answer> must be only the concise text answer.\n",
    "regression": "The final answer inside <answer> must be only the numerical estimate.\n",
    "math": "The final answer inside <answer> must be only the final result (a number or LaTeX formula).\n",
}


def build_prompt(question: str, problem_type: str) -> str:
    """Build the canonical prompt for a QA example."""
    type_key = (problem_type or "").strip().lower()
    return QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE.get(type_key, "")
