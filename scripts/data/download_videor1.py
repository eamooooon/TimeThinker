#!/usr/bin/env python3
"""Download and safely extract the public Video-R1 training data.

The upstream dataset stores the two JSON indexes and media as split ZIP files.
This helper keeps the project layout expected by ``convert_data.py``::

    data/Video-R1-COT-165k.json
    data/Video-R1-260k.json
    data/<media bucket>/...

Examples:

    # Metadata only (small and safe as a first step).
    python scripts/data/download_videor1.py --metadata-only

    # Download only the chart and math media buckets.
    python scripts/data/download_videor1.py --components Chart Math

    # Download every public media bucket. This is very large.
    python scripts/data/download_videor1.py --all
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


DATASET_REPO = "Video-R1/Video-R1-data"
METADATA_FILES = ("README.md", "Video-R1-COT-165k.json", "Video-R1-260k.json")
COMPONENTS = (
    "CLEVRER",
    "Chart",
    "General",
    "Knowledge",
    "LLaVA-Video-178K",
    "Math",
    "NeXT-QA",
    "OCR",
    "PerceptionTest",
    "STAR",
    "Spatial",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download only the two JSON indexes and upstream README (the default).",
    )
    selection.add_argument(
        "--components",
        nargs="+",
        choices=COMPONENTS,
        help="Download and extract one or more media buckets.",
    )
    selection.add_argument("--all", action="store_true", help="Download and extract every media bucket.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Destination data directory.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/.hf_cache/Video-R1"),
        help="Hugging Face cache directory. It can be removed after a successful extraction.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded ZIP files in --cache-dir after extraction.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print selected upstream files without downloading.")
    return parser.parse_args()


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract a ZIP only when every member remains below ``destination``."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe archive member in {archive}: {member.filename}")
        zf.extractall(destination)


def selected_files(args: argparse.Namespace) -> list[str]:
    api = HfApi()
    all_files = api.list_repo_files(DATASET_REPO, repo_type="dataset")
    wanted = list(METADATA_FILES)
    components = COMPONENTS if args.all else (args.components or ())
    for component in components:
        prefix = f"{component}/"
        wanted.extend(path for path in all_files if path.startswith(prefix) and path.endswith(".zip"))
    return wanted


def main() -> None:
    args = parse_args()
    files = selected_files(args)
    components = "all" if args.all else ", ".join(args.components or []) or "metadata only"
    print(f"[Video-R1] repo={DATASET_REPO} selection={components} files={len(files)}")
    for name in files:
        print(f"  {name}")
    if args.dry_run:
        return

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for filename in files:
        downloaded = Path(
            hf_hub_download(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                filename=filename,
                cache_dir=str(args.cache_dir),
                local_dir=str(args.cache_dir / "downloads"),
            )
        )
        if downloaded.suffix.lower() != ".zip":
            target = args.data_dir / Path(filename).name
            shutil.copy2(downloaded, target)
            print(f"[OK] {target}")
            continue

        print(f"[EXTRACT] {downloaded.name} -> {args.data_dir}")
        safe_extract(downloaded, args.data_dir)
        if not args.keep_archives:
            local_copy = args.cache_dir / "downloads" / filename
            if local_copy.exists():
                local_copy.unlink()

    print("[DONE] Run scripts/data/convert_data.py after the required media is present.")
    print(f"[NOTE] Hugging Face cache remains in {args.cache_dir}; remove it manually if disk space is needed.")


if __name__ == "__main__":
    main()
