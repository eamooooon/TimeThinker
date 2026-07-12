#!/usr/bin/env python3
"""Convert local Video-R1 JSON files into TimeThinker SFT/RL training files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


QUESTION_TEMPLATE = (
    "{question}\n"
    "Please answer this question based on the visual content. "
    "Provide your thinking process between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags. "
    "At the end, you must output the final answer in the format:\n"
    "<answer><your_answer_here></answer>\n"
)

TYPE_TEMPLATE = {
    "multiple choice": (
        "Please provide only the single option letter (e.g., A, B, C, D, etc.) "
        "within the <answer>...</answer> tags.\n"
        "Example:\n<answer>A</answer>"
    ),
    "numerical": (
        "Please provide only the numerical value within the <answer>...</answer> tags.\n"
        "Example:\n<answer>3.14</answer>"
    ),
    "OCR": (
        "Please provide only the transcribed text within the <answer>...</answer> tags.\n"
        "Example:\n<answer>Hello World</answer>"
    ),
    "open-ended": (
        "Please provide only your text answer within the <answer>...</answer> tags.\n"
        "Example:\n<answer>The capital of France is Paris.</answer>"
    ),
    "regression": (
        "Please provide only the numerical value within the <answer>...</answer> tags.\n"
        "Example:\n<answer>42.7</answer>"
    ),
}


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return data


def dump_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def normalize_problem_type(problem_type: str) -> str:
    ptype = (problem_type or "").strip()
    if ptype.lower() == "free-form":
        return "open-ended"
    if ptype.lower() == "ocr":
        return "OCR"
    return ptype


def normalize_media_path(raw_path: str) -> str:
    path = (raw_path or "").strip()
    if path.startswith("./"):
        path = path[2:]
    if path.startswith("data/"):
        return path
    return f"data/{path}"


def build_question(example: dict[str, Any], problem_type: str) -> str:
    question = str(example.get("problem") or "").strip()
    options = example.get("options")
    if problem_type == "multiple choice" and isinstance(options, list) and options:
        question = f"{question}\nOptions:\n" + "\n".join(str(option) for option in options)

    return QUESTION_TEMPLATE.format(question=question) + TYPE_TEMPLATE.get(problem_type, "")


def build_sft_example(example: dict[str, Any], repo_root: Path) -> tuple[dict[str, Any] | None, str | None]:
    data_type = (example.get("data_type") or "").strip().lower()
    if data_type not in {"image", "video"}:
        return None, f"unsupported_data_type:{data_type}"

    process = str(example.get("process") or "").strip()
    solution = str(example.get("solution") or "").strip()
    if not process or not solution:
        return None, "missing_process_or_solution"

    media_path = normalize_media_path(str(example.get("path") or ""))
    if not (repo_root / media_path).is_file():
        return None, "missing_media"

    problem_type = normalize_problem_type(str(example.get("problem_type") or ""))
    placeholder = "<image>" if data_type == "image" else "<video>"
    user_content = f"{placeholder}\n{build_question(example, problem_type)}"
    assistant_content = f"{process}\n{solution}"

    converted = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }
    if data_type == "image":
        converted["images"] = [media_path]
    else:
        converted["videos"] = [media_path]

    return converted, None


def build_rl_example(example: dict[str, Any], repo_root: Path) -> tuple[dict[str, Any] | None, str | None]:
    data_type = (example.get("data_type") or "").strip().lower()
    if data_type not in {"image", "video"}:
        return None, f"unsupported_data_type:{data_type}"

    solution = str(example.get("solution") or "").strip()
    if not solution:
        return None, "missing_solution"

    media_path = normalize_media_path(str(example.get("path") or ""))
    if not (repo_root / media_path).is_file():
        return None, "missing_media"

    problem_type = normalize_problem_type(str(example.get("problem_type") or ""))
    converted: dict[str, Any] = {
        "problem_id": example.get("problem_id"),
        "problem": example.get("problem") or "",
        "answer": solution,
        "data_type": data_type,
        "problem_type": problem_type,
        "options": example.get("options") or [],
        "data_source": example.get("data_source") or "",
    }
    if data_type == "image":
        converted["images"] = [media_path]
    else:
        converted["videos"] = [media_path]

    return converted, None


def convert_sft(data: list[dict[str, Any]], repo_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    image_data: list[dict[str, Any]] = []
    video_data: list[dict[str, Any]] = []
    skipped: Counter = Counter()

    for example in data:
        converted, reason = build_sft_example(example, repo_root)
        if converted is None:
            skipped[reason or "unknown"] += 1
            continue

        if "images" in converted:
            image_data.append(converted)
        else:
            video_data.append(converted)

    return image_data, video_data, skipped


def convert_rl(data: list[dict[str, Any]], repo_root: Path) -> tuple[list[dict[str, Any]], Counter]:
    rl_data: list[dict[str, Any]] = []
    skipped: Counter = Counter()

    for example in data:
        converted, reason = build_rl_example(example, repo_root)
        if converted is None:
            skipped[reason or "unknown"] += 1
            continue
        rl_data.append(converted)

    return rl_data, skipped


def summarize(name: str, data: list[dict[str, Any]]) -> None:
    modality = Counter("image" if "images" in x else "video" if "videos" in x else "text" for x in data)
    problem_types = Counter(str(x.get("problem_type", "")) for x in data)
    print(f"{name}: {len(data)} examples")
    print(f"  modality: {dict(modality)}")
    print(f"  problem_type: {dict(problem_types.most_common())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--sft-input", type=Path, default=Path("data/Video-R1-COT-165k.json"))
    parser.add_argument("--rl-input", type=Path, default=Path("data/Video-R1-260k.json"))
    parser.add_argument("--sft-image-output", type=Path, default=Path("LLaMA-Factory/data/timethinker_sft_image.json"))
    parser.add_argument("--sft-video-output", type=Path, default=Path("LLaMA-Factory/data/timethinker_sft_video.json"))
    parser.add_argument("--rl-output", type=Path, default=Path("EasyR1/data/timethinker_rl_train.json"))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    sft_input = (repo_root / args.sft_input).resolve()
    rl_input = (repo_root / args.rl_input).resolve()

    print(f"Loading SFT data: {sft_input}")
    sft_raw = load_json(sft_input)
    sft_image, sft_video, sft_skipped = convert_sft(sft_raw, repo_root)

    print(f"Loading RL data: {rl_input}")
    rl_raw = load_json(rl_input)
    rl_data, rl_skipped = convert_rl(rl_raw, repo_root)

    dump_json(repo_root / args.sft_image_output, sft_image)
    dump_json(repo_root / args.sft_video_output, sft_video)
    dump_json(repo_root / args.rl_output, rl_data)

    summarize("SFT image", sft_image)
    summarize("SFT video", sft_video)
    summarize("RL", rl_data)
    print(f"SFT skipped: {dict(sft_skipped)}")
    print(f"RL skipped: {dict(rl_skipped)}")
    print("Wrote:")
    print(f"  {args.sft_image_output}")
    print(f"  {args.sft_video_output}")
    print(f"  {args.rl_output}")


if __name__ == "__main__":
    main()
