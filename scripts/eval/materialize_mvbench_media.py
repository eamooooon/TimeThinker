#!/usr/bin/env python
"""Materialize MVBench media paths expected by the eval JSON.

The MVBench archives contain a few assets in layouts that differ from the
Video-R1 eval JSON. This helper creates lightweight links where possible and
converts TVQA frame directories into mp4 files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import av
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_url
from remotezip import RemoteZip


FPS_TVQA = 3
NTURGBD_REPO = "Wangxc1000/nturgbd"
NTURGBD_ZIP = "nturgbd.zip"


def link_relative(src: Path, dst: Path) -> bool:
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.relpath(src, dst.parent), dst)
    return True


def load_mvbench_paths(data_root: Path) -> list[Path]:
    with open(data_root / "eval_mvbench.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return [data_root / item["path"].lstrip("./") for item in data if item.get("path")]


def repair_links(data_root: Path) -> None:
    root = data_root / "Evaluation/MVBench"
    paths = load_mvbench_paths(data_root)

    ssv2_created = 0
    relocated_created = 0
    for dst in paths:
        rel = dst.relative_to(data_root)
        rel_s = rel.as_posix()
        if dst.exists():
            continue

        if "/ssv2_video/" in rel_s:
            src = dst.with_suffix(".webm")
            if src.exists():
                ssv2_created += int(link_relative(src, dst))
            continue

        if "/MVBench/star/" in rel_s or "/MVBench/clevrer/" in rel_s:
            inner = rel.relative_to("Evaluation/MVBench")
            src = root / "data0613" / inner
            if src.exists():
                relocated_created += int(link_relative(src, dst))

    print(f"[LINK] ssv2_created={ssv2_created} relocated_created={relocated_created}")


def encode_frame_dir(frame_dir: Path, out_path: Path, fps: int = FPS_TVQA) -> bool:
    frames = sorted(frame_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No jpg frames found in {frame_dir}")

    first = Image.open(frames[0]).convert("RGB")
    width, height = first.size
    width -= width % 2
    height -= height % 2

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    container = av.open(str(tmp_path), mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": "veryfast", "crf": "23"}

    try:
        for fp in frames:
            image = Image.open(fp).convert("RGB")
            if image.size != (width, height):
                image = image.crop((0, 0, width, height))
            frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()

    tmp_path.replace(out_path)
    return True


def materialize_tvqa(data_root: Path, limit: int | None = None) -> None:
    paths = load_mvbench_paths(data_root)
    tvqa_paths = [p for p in paths if "/MVBench/tvqa/" in p.as_posix()]
    created = 0
    skipped = 0
    missing_dirs = 0

    for out_path in tvqa_paths:
        if out_path.exists():
            skipped += 1
            continue
        frame_dir = out_path.with_suffix("")
        if not frame_dir.is_dir():
            missing_dirs += 1
            print(f"[MISS] {frame_dir}")
            continue
        print(f"[ENCODE] {frame_dir} -> {out_path}", flush=True)
        encode_frame_dir(frame_dir, out_path)
        created += 1
        if limit is not None and created >= limit:
            break

    print(f"[TVQA] created={created} skipped={skipped} missing_dirs={missing_dirs}")


def materialize_nturgbd(data_root: Path) -> None:
    paths = load_mvbench_paths(data_root)
    ntu_paths = [p for p in paths if "/MVBench/nturgbd/" in p.as_posix()]
    targets = [p for p in ntu_paths if not p.exists()]
    if not targets:
        print("[NTURGBD] nothing to do")
        return

    url = hf_hub_url(NTURGBD_REPO, NTURGBD_ZIP, repo_type="dataset")
    created = 0
    extracted = 0
    missing = 0
    with RemoteZip(url) as zf:
        names = set(zf.namelist())
        for dst in targets:
            avi_path = dst.with_suffix(".avi")
            if not avi_path.exists():
                member = f"nturgbd/{dst.with_suffix('.avi').name}"
                if member not in names:
                    missing += 1
                    print(f"[MISS] {member}")
                    continue
                print(f"[EXTRACT] {member} -> {avi_path}", flush=True)
                avi_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(avi_path, "wb") as out:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                extracted += 1
            created += int(link_relative(avi_path, dst))

    print(f"[NTURGBD] extracted={extracted} linked={created} missing={missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="Evaluation/data")
    parser.add_argument("--skip-tvqa", action="store_true")
    parser.add_argument("--skip-nturgbd", action="store_true")
    parser.add_argument("--tvqa-limit", type=int)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    repair_links(data_root)
    if not args.skip_nturgbd:
        materialize_nturgbd(data_root)
    if not args.skip_tvqa:
        materialize_tvqa(data_root, args.tvqa_limit)


if __name__ == "__main__":
    main()
