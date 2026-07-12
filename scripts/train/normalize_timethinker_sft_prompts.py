#!/usr/bin/env python3
"""Normalize TimeThinker SFT user prompts to the shared canonical QA prompt."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prompting.timethinker import build_prompt


DEFAULT_DATASETS = (
    REPO_ROOT / "LLaMA-Factory/data/timethinker_sft_video.json",
    REPO_ROOT / "LLaMA-Factory/data/timethinker_sft_image.json",
)
PROMPT_STARTS = (
    "\nPlease answer this question based on the visual content",
    "\n\nPlease answer based on the visual content",
)


def infer_problem_type(prompt_tail: str) -> str:
    normalized = prompt_tail.lower()
    if "option letter" in normalized:
        return "multiple choice"
    if "transcribed text" in normalized:
        return "ocr"
    if "text answer" in normalized:
        return "open-ended"
    if "numerical estimate" in normalized or "<answer>42.7</answer>" in normalized:
        return "regression"
    if "numerical value" in normalized:
        return "numerical"
    if "final result" in normalized:
        return "math"
    raise ValueError(f"Cannot infer problem type from prompt tail: {prompt_tail[:160]!r}")


def split_question_and_prompt_tail(content: str) -> tuple[str, str]:
    for prompt_start in PROMPT_STARTS:
        if prompt_start in content:
            return content.split(prompt_start, 1)
    raise ValueError(f"Prompt start not found: {content[:160]!r}")


def normalize_record(record: dict[str, Any]) -> bool:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Missing messages")
    user_message = messages[0]
    if user_message.get("role") != "user" or not isinstance(user_message.get("content"), str):
        raise ValueError("First message must be a text user message")

    content = user_message["content"]
    question, prompt_tail = split_question_and_prompt_tail(content)
    problem_type = infer_problem_type(prompt_tail)
    normalized = build_prompt(question, problem_type)
    if content == normalized:
        return False
    user_message["content"] = normalized
    return True


def rewrite_dataset(path: Path, write: bool) -> tuple[int, int]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}")
    changed = sum(normalize_record(record) for record in records)
    if not write:
        return len(records), changed

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        json.dump(records, temp_file, ensure_ascii=False, indent=2)
        temp_file.write("\n")
        temp_path = Path(temp_file.name)
    os.replace(temp_path, path)
    return len(records), changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Rewrite datasets in place.")
    parser.add_argument("datasets", nargs="*", type=Path, default=DEFAULT_DATASETS)
    args = parser.parse_args()

    action = "updated" if args.write else "would update"
    for dataset_path in args.datasets:
        total, changed = rewrite_dataset(dataset_path, write=args.write)
        print(f"{dataset_path}: {action} {changed}/{total} records")


if __name__ == "__main__":
    main()
