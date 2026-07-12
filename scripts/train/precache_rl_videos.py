#!/usr/bin/env python3
"""Pre-decode RL videos into the on-disk frame cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "EasyR1"))

from verl.utils.dataset import (  # noqa: E402
    _frame_cache_key,
    _load_frame_cache,
    _rl_frame_cache_root,
    _sample_video_frames_for_cache,
    _save_frame_cache,
)


def iter_json_records(path: Path):
    try:
        import ijson

        with path.open("rb") as f:
            for record in ijson.items(f, "item"):
                if isinstance(record, dict):
                    yield record
        return
    except Exception:
        pass

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("train", "data", "instances"):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")

    for record in data:
        if isinstance(record, dict):
            yield record


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required when using --config") from exc

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_data_files(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise TypeError(f"Unsupported data file value: {value!r}")


def resolve_video_path(video: str, image_dir: str | None) -> str:
    if video.startswith("file://"):
        raw = video[len("file://") :]
        if image_dir and not os.path.isabs(raw):
            raw = os.path.join(image_dir, raw)
        return f"file://{raw}"

    if image_dir and not os.path.isabs(video):
        return os.path.join(image_dir, video)
    return video


def collect_videos(
    data_files: list[str],
    video_key: str,
    image_dir: str | None,
    limit: int | None,
) -> list[str]:
    videos: list[str] = []
    seen: set[str] = set()

    for data_file in data_files:
        path = Path(data_file)
        if "@" in data_file:
            path = Path(data_file.split("@", 1)[0])
        if not path.is_absolute():
            path = REPO_ROOT / path

        for record in iter_json_records(path):
            raw_videos = record.get(video_key)
            if not isinstance(raw_videos, list):
                continue
            for raw_video in raw_videos:
                if not isinstance(raw_video, str):
                    continue
                video = resolve_video_path(raw_video, image_dir)
                if video not in seen:
                    seen.add(video)
                    videos.append(video)
                    if limit is not None and len(videos) >= limit:
                        return videos

    return videos


def warm_one(
    video: str,
    min_pixels: int | None,
    max_pixels: int | None,
    max_frames: int,
    video_fps: float,
) -> tuple[str, str]:
    cache_root = _rl_frame_cache_root()
    if cache_root is None:
        raise RuntimeError("RL frame cache is disabled")

    vision_info = {
        "video": video,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "max_frames": max_frames,
        "fps": video_fps,
    }
    key, key_data = _frame_cache_key(video, min_pixels, max_pixels, max_frames, video_fps)
    if _load_frame_cache(cache_root, key) is not None:
        return video, "hit"

    frames, raw_fps, sample_fps = _sample_video_frames_for_cache(vision_info)
    if _save_frame_cache(cache_root, key, key_data, frames, raw_fps=raw_fps, sample_fps=sample_fps):
        return video, "write"
    return video, "skip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/rl/qwen3_rl_bs16.yaml", help="RL yaml config.")
    parser.add_argument("--data-file", action="append", default=None, help="JSON data file. Can be passed multiple times.")
    parser.add_argument("--include-val", action="store_true", help="Also warm config.data.val_files.")
    parser.add_argument("--cache-dir", default="data/.cache/rl_frames", help="RL frame cache directory.")
    parser.add_argument("--video-key", default=None, help="Dataset video key. Defaults to config.data.video_key.")
    parser.add_argument("--image-dir", default=None, help="Base directory for relative media paths. Defaults to config.data.image_dir.")
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Only warm the first N unique videos.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("RL_FRAME_CACHE_DIR", args.cache_dir)

    config = load_config(REPO_ROOT / args.config) if args.config else {}
    data_config = config.get("data", {}) if isinstance(config, dict) else {}

    data_files = args.data_file or normalize_data_files(data_config.get("train_files"))
    if args.include_val:
        data_files.extend(normalize_data_files(data_config.get("val_files")))
    if not data_files:
        raise ValueError("No data files found. Pass --data-file or --config.")

    video_key = args.video_key or data_config.get("video_key", "videos")
    image_dir = args.image_dir if args.image_dir is not None else data_config.get("image_dir")
    if image_dir is not None and not os.path.isabs(image_dir):
        image_dir = str(REPO_ROOT / image_dir)

    min_pixels = args.min_pixels if args.min_pixels is not None else data_config.get("min_pixels")
    max_pixels = args.max_pixels if args.max_pixels is not None else data_config.get("max_pixels")
    video_fps = args.video_fps if args.video_fps is not None else float(data_config.get("video_fps", 2.0))

    videos = collect_videos(data_files, video_key=video_key, image_dir=image_dir, limit=args.limit)
    cache_root = _rl_frame_cache_root()
    if cache_root is None:
        raise RuntimeError("RL frame cache is disabled")
    cache_root.mkdir(parents=True, exist_ok=True)

    print(f"[RLFrameCache] dir={cache_root}")
    print(f"[RLFrameCache] videos={len(videos)} workers={args.workers} max_frames={args.max_frames} fps={video_fps}")

    stats = {"hit": 0, "write": 0, "skip": 0, "error": 0}
    errors: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(warm_one, video, min_pixels, max_pixels, args.max_frames, video_fps): video
            for video in videos
        }
        for idx, future in enumerate(as_completed(futures), 1):
            video = futures[future]
            try:
                _, status = future.result()
                stats[status] = stats.get(status, 0) + 1
            except Exception as exc:
                stats["error"] += 1
                if len(errors) < 20:
                    errors.append((video, str(exc)))

            if idx % 100 == 0 or idx == len(videos):
                print(
                    "[RLFrameCache] "
                    f"{idx}/{len(videos)} hit={stats['hit']} write={stats['write']} "
                    f"skip={stats['skip']} error={stats['error']}",
                    flush=True,
                )

    if errors:
        print("[RLFrameCache] first errors:")
        for video, error in errors:
            print(f"  {video}: {error}")

    print(f"[RLFrameCache] done {stats}")


if __name__ == "__main__":
    main()
