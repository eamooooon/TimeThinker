#!/usr/bin/env python
"""Download publicly available media for the Video-R1 evaluation JSON files.

The Video-R1 eval repository only contains JSON indexes. This helper downloads
media from the original benchmark repositories and lays them out under:

    Evaluation/data/Evaluation/<benchmark>/...

"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


ROOT = Path("Evaluation/data/Evaluation")
DATA_ROOT = Path("Evaluation/data")


def unzip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / f".{zip_path.stem}.done"
    if marker.exists():
        print(f"[OK] already extracted {zip_path.name} -> {out_dir}")
        return
    print(f"[UNZIP] {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    marker.touch()


def assert_safe_archive_member(out_dir: Path, member_name: str) -> None:
    target = (out_dir / member_name).resolve()
    root = out_dir.resolve()
    if root != target and root not in target.parents:
        raise RuntimeError(f"Archive member escapes output directory: {member_name}")


def untar(tar_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / f".{tar_path.name}.done"
    if marker.exists():
        print(f"[OK] already extracted {tar_path.name} -> {out_dir}")
        return
    print(f"[UNTAR] {tar_path} -> {out_dir}")
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            assert_safe_archive_member(out_dir, member.name)
        tf.extractall(out_dir)
    marker.touch()


def download_zip(repo: str, filename: str, out_dir: Path, cache_dir: Path) -> Path:
    print(f"[HF] {repo}:{filename}")
    path = hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        filename=filename,
        cache_dir=str(cache_dir),
    )
    zip_path = Path(path)
    unzip(zip_path, out_dir)
    return zip_path


def download_tar_gz(repo: str, filename: str, out_dir: Path, cache_dir: Path) -> Path:
    print(f"[HF] {repo}:{filename}")
    path = hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        filename=filename,
        cache_dir=str(cache_dir),
    )
    tar_path = Path(path)
    untar(tar_path, out_dir)
    return tar_path


def symlink_to_expected_path(source: Path, target: Path) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return False
    rel_source = os.path.relpath(source.resolve(), start=target.parent.resolve())
    target.symlink_to(rel_source)
    return True


def materialize_expected_media_paths(json_path: Path, media_root: Path) -> None:
    if not json_path.exists():
        print(f"[WARN] missing index json: {json_path}")
        return
    if not media_root.exists():
        print(f"[WARN] missing media directory: {media_root}")
        return

    data = json.loads(json_path.read_text(encoding="utf-8"))
    media_files = [
        path
        for path in media_root.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    ]
    by_name = {}
    by_stem = {}
    duplicate_stems = set()
    for path in media_files:
        by_name.setdefault(path.name, path)
        stem = path.stem
        if stem in by_stem:
            duplicate_stems.add(stem)
        else:
            by_stem[stem] = path

    created = 0
    existing = 0
    missing = []
    for item in data:
        rel_path = (item.get("path") or "").lstrip("./").lstrip("/")
        if not rel_path:
            continue
        expected = DATA_ROOT / rel_path
        if expected.exists() or expected.is_symlink():
            existing += 1
            continue

        expected_name = expected.name
        source = by_name.get(expected_name)
        if source is None and expected.stem not in duplicate_stems:
            source = by_stem.get(expected.stem)
        if source is None:
            missing.append(rel_path)
            continue
        if symlink_to_expected_path(source, expected):
            created += 1
        else:
            existing += 1

    print(
        f"[LINK] {json_path.name}: created={created} existing={existing} missing={len(missing)}"
    )
    if missing:
        print(f"[WARN] first missing paths: {missing[:5]}")


def download_tempcompass(cache_dir: Path) -> None:
    download_zip("lmms-lab/TempCompass", "tempcompass_videos.zip", ROOT / "TempCompass", cache_dir)


def download_videomme(cache_dir: Path) -> None:
    out_dir = ROOT / "VideoMME"
    download_zip("lmms-lab/Video-MME", "subtitle.zip", out_dir, cache_dir)
    for idx in range(1, 21):
        download_zip("lmms-lab/Video-MME", f"videos_chunked_{idx:02d}.zip", out_dir, cache_dir)


def download_videommmu(cache_dir: Path) -> None:
    out_dir = ROOT / "VideoMMMU"
    for filename in [
        "Art.zip",
        "Business.zip",
        "Engineering.zip",
        "Humanities.zip",
        "Medicine.zip",
        "Science.zip",
        "question_only_videos.zip",
    ]:
        download_zip("lmms-lab/VideoMMMU", filename, out_dir, cache_dir)


def download_mvbench(cache_dir: Path) -> None:
    out_dir = ROOT / "MVBench"
    for filename in [
        "video/FunQA_test.zip",
        "video/Moments_in_Time_Raw.zip",
        "video/clevrer.zip",
        "video/data0613.zip",
        "video/perception.zip",
        "video/scene_qa.zip",
        "video/ssv2_video.zip",
        "video/sta.zip",
        "video/star.zip",
        "video/tvqa.zip",
        "video/vlnqa.zip",
    ]:
        download_zip("OpenGVLab/MVBench", filename, out_dir, cache_dir)


def download_vsibench(cache_dir: Path) -> None:
    out_dir = ROOT / "VSIBench"
    for filename in ["arkitscenes.zip", "scannet.zip", "scannetpp.zip"]:
        download_zip("nyu-visionx/VSI-Bench", filename, out_dir, cache_dir)


def download_mmvu(cache_dir: Path) -> None:
    target = ROOT / "MMVU"
    target.mkdir(parents=True, exist_ok=True)
    print("[HF] yale-nlp/MMVU snapshot videos/**")
    snap = Path(
        snapshot_download(
            repo_id="yale-nlp/MMVU",
            repo_type="dataset",
            allow_patterns=["videos/**", "validation.json"],
            cache_dir=str(cache_dir),
        )
    )
    if (target / "videos").exists():
        print(f"[OK] already exists {target / 'videos'}")
    else:
        shutil.copytree(snap / "videos", target / "videos")
    shutil.copy2(snap / "validation.json", target / "validation.json")


def download_onethinker_jsons(cache_dir: Path) -> None:
    for filename in ["eval_longvideoreason.json", "eval_videomathqa.json"]:
        print(f"[HF] OneThink/OneThinker-eval:{filename}")
        path = Path(
            hf_hub_download(
                repo_id="OneThink/OneThinker-eval",
                repo_type="dataset",
                filename=filename,
                cache_dir=str(cache_dir),
            )
        )
        target = Path("Evaluation/data") / filename
        target.write_bytes(path.read_bytes())
        print(f"[OK] {target}")


def download_longvideoreason(cache_dir: Path) -> None:
    out_dir = ROOT / "LongVideoReason"
    for idx in range(10):
        download_tar_gz(
            "LongVideo-Reason/longvideo_eval_videos",
            f"longvideo_eval_subset{idx}.tar.gz",
            out_dir,
            cache_dir,
        )

    # The upstream LongVideo-Reason eval archive is documented as extracting
    # into longvila_videos/, while eval_longvideoreason.json references
    # Evaluation/LongVideoReason/*.mp4. Flatten it into that expected layout.
    nested = out_dir / "longvila_videos"
    if nested.exists():
        for child in nested.iterdir():
            target = out_dir / child.name
            if target.exists():
                continue
            shutil.move(str(child), str(target))
        try:
            nested.rmdir()
        except OSError:
            pass
    materialize_expected_media_paths(DATA_ROOT / "eval_longvideoreason.json", out_dir)


def download_videomathqa(cache_dir: Path) -> None:
    out_dir = ROOT
    for idx in range(1, 3):
        download_zip(
            "OneThink/OneThinker-eval",
            f"VideoMathQA/VideoMathQA_part{idx}.zip",
            out_dir,
            cache_dir,
        )
    materialize_expected_media_paths(DATA_ROOT / "eval_videomathqa.json", ROOT / "VideoMathQA")


DOWNLOADERS = {
    "tempcompass": download_tempcompass,
    "videomme": download_videomme,
    "videommmu": download_videommmu,
    "mvbench": download_mvbench,
    "vsibench": download_vsibench,
    "mmvu": download_mmvu,
    "onethinker-jsons": download_onethinker_jsons,
    "longvideoreason": download_longvideoreason,
    "videomathqa": download_videomathqa,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datasets",
        nargs="*",
        default=["tempcompass", "mmvu", "vsibench", "mvbench"],
        choices=sorted(DOWNLOADERS.keys()) + ["public-small", "public-all", "onethinker-compatible"],
    )
    parser.add_argument("--cache-dir", default="Evaluation/data/.hf_cache")
    args = parser.parse_args()

    datasets = list(args.datasets)
    if "public-small" in datasets:
        datasets = ["tempcompass", "mmvu", "vsibench", "mvbench"]
    elif "public-all" in datasets:
        datasets = ["tempcompass", "mmvu", "vsibench", "mvbench", "videomme"]
    elif "onethinker-compatible" in datasets:
        datasets = ["onethinker-jsons", "videomathqa", "longvideoreason"]

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in datasets:
        print(f"\n=== {name} ===")
        DOWNLOADERS[name](cache_dir)


if __name__ == "__main__":
    main()
