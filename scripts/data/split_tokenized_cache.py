#!/usr/bin/env python
"""Create train/validation caches from an existing tokenized dataset.

This avoids re-running image/video preprocessing when you only need a smaller
debug cache or a different validation split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="LLaMA-Factory/cache/timethinker_sft_tokenized")
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples-per-modality",
        type=int,
        default=-1,
        help="Use -1 for all cached samples. Otherwise select up to N image-only and N video-only samples.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validation-overlaps-train",
        action="store_true",
        help=(
            "Keep the full source train split and add a validation split sampled from it. "
            "This is fast but validation samples are still present in train."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)

    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output}. Pass --overwrite to replace it.")
    if output.exists():
        import shutil

        shutil.rmtree(output)

    cached = load_from_disk(str(source))
    dataset = cached["train"] if isinstance(cached, DatasetDict) else cached

    if args.max_samples_per_modality > 0:
        image_idx: list[int] = []
        video_idx: list[int] = []
        meta = dataset.select_columns(["images", "videos"])

        for idx, example in enumerate(meta):
            has_image = bool(example["images"])
            has_video = bool(example["videos"])
            if has_image and not has_video and len(image_idx) < args.max_samples_per_modality:
                image_idx.append(idx)
            elif has_video and not has_image and len(video_idx) < args.max_samples_per_modality:
                video_idx.append(idx)

            if len(image_idx) >= args.max_samples_per_modality and len(video_idx) >= args.max_samples_per_modality:
                break

        pieces = []
        if image_idx:
            pieces.append(dataset.select(image_idx))
        if video_idx:
            pieces.append(dataset.select(video_idx))
        if not pieces:
            raise SystemExit("No image/video samples were selected from the cache.")

        dataset = concatenate_datasets(pieces) if len(pieces) > 1 else pieces[0]
        print(f"Selected image={len(image_idx)}, video={len(video_idx)}, total={len(dataset)}")
    else:
        print(f"Using all cached samples: total={len(dataset)}")

    split = dataset.train_test_split(test_size=args.val_size, seed=args.seed)
    train_split = dataset if args.validation_overlaps_train else split["train"]
    out = DatasetDict({"train": train_split, "validation": split["test"]})
    out.save_to_disk(str(output))
    print(f"Saved: {output}")
    print(f"train={len(out['train'])}, validation={len(out['validation'])}")


if __name__ == "__main__":
    main()
