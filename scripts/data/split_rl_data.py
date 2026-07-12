#!/usr/bin/env python3
"""Create the deterministic train/validation JSON files expected by RL configs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("EasyR1/data/timethinker_rl_train.json"))
    parser.add_argument("--train-output", type=Path, default=Path("EasyR1/data/timethinker_rl_train_split.json"))
    parser.add_argument("--val-output", type=Path, default=Path("EasyR1/data/timethinker_rl_val_512.json"))
    parser.add_argument("--val-size", type=int, default=512, help="Number of held-out validation examples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    args = parse_args()
    for path in (args.train_output, args.val_output):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite {path}; pass --overwrite to replace it.")

    with args.input.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise SystemExit(f"Expected a JSON list in {args.input}")
    if not 0 < args.val_size < len(rows):
        raise SystemExit(f"--val-size must be in [1, {len(rows) - 1}], got {args.val_size}")

    val_indices = set(random.Random(args.seed).sample(range(len(rows)), args.val_size))
    train_rows = [row for index, row in enumerate(rows) if index not in val_indices]
    val_rows = [row for index, row in enumerate(rows) if index in val_indices]
    write_json(args.train_output, train_rows)
    write_json(args.val_output, val_rows)
    print(f"[DONE] train={len(train_rows)} -> {args.train_output}")
    print(f"[DONE] validation={len(val_rows)} -> {args.val_output}")


if __name__ == "__main__":
    main()
