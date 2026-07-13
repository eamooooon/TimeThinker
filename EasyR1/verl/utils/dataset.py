# 

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import json
import hashlib
import shutil
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prompting.timethinker import build_prompt

from . import torch_functional as VF


RL_FRAME_CACHE_VERSION = 1
RL_FRAME_CACHE_IMAGE_EXT = "jpg"
RL_FRAME_CACHE_JPEG_QUALITY = 95

def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    # print(max_pixels)

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def _fetch_video_with_pyav(
    vision_info: dict[str, Any], image_patch_size: int = 16, return_fps: bool = False
):
    import av
    from qwen_vl_utils.vision_process import smart_nframes

    video_path = vision_info["video"]
    if isinstance(video_path, str) and video_path.startswith("file://"):
        video_path = video_path[len("file://") :]

    frames = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        raw_fps = float(stream.average_rate) if stream.average_rate else 24.0
        for frame in container.decode(stream):
            frames.append(frame.to_image().convert("RGB"))

    if len(frames) == 0:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    nframes = smart_nframes(vision_info, total_frames=len(frames), video_fps=raw_fps)
    indices = np.linspace(0, len(frames) - 1, nframes).round().astype(int).tolist()
    sampled_frames = [frames[i] for i in indices]
    sample_fps = nframes / max(len(frames), 1e-6) * raw_fps

    fallback_info = dict(vision_info)
    fallback_info["video"] = sampled_frames
    fallback_info["sample_fps"] = sample_fps
    fallback_info["raw_fps"] = raw_fps
    return fetch_video(
        fallback_info,
        image_patch_size=image_patch_size,
        return_video_sample_fps=return_fps,
        return_video_metadata=return_fps,
    )


def _clean_video_path(video_path: str) -> str:
    if isinstance(video_path, str) and video_path.startswith("file://"):
        return video_path[len("file://") :]
    return video_path


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _rl_frame_cache_root() -> Optional[Path]:
    if _truthy_env("RL_FRAME_CACHE_DISABLE"):
        return None

    cache_dir = os.environ.get("RL_FRAME_CACHE_DIR")
    if not cache_dir:
        return None
    return Path(cache_dir)


def _frame_cache_key(
    video_path: str,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    max_frames: int,
    video_fps: float,
) -> tuple[str, dict[str, Any]]:
    clean_path = _clean_video_path(video_path)
    abs_path = os.path.abspath(clean_path)
    try:
        stat = os.stat(clean_path)
        source = {
            "path": abs_path,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    except Exception:
        source = {
            "path": abs_path,
            "size": None,
            "mtime_ns": None,
        }

    key_data = {
        "version": RL_FRAME_CACHE_VERSION,
        "source": source,
        "min_pixels": None if min_pixels is None else int(min_pixels),
        "max_pixels": None if max_pixels is None else int(max_pixels),
        "max_frames": int(max_frames),
        "fps": float(video_fps),
        "image_ext": RL_FRAME_CACHE_IMAGE_EXT,
        "jpeg_quality": RL_FRAME_CACHE_JPEG_QUALITY,
    }
    digest = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode("utf-8")).hexdigest()
    return digest, key_data


def _frame_cache_dir_for(cache_root: Path, key: str) -> Path:
    return cache_root / key[:2] / key


def _load_frame_cache(cache_root: Optional[Path], key: str) -> Optional[tuple[list[ImageObject], dict[str, Any]]]:
    if cache_root is None:
        return None

    cache_dir = _frame_cache_dir_for(cache_root, key)
    meta_path = cache_dir / "metadata.json"
    if not meta_path.exists():
        return None

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        frame_names = meta.get("frames", [])
        if not isinstance(frame_names, list) or not frame_names:
            return None

        frames = []
        for name in frame_names:
            frame_path = cache_dir / str(name)
            if not frame_path.exists():
                return None
            with Image.open(frame_path) as image:
                frames.append(image.convert("RGB").copy())

        if len(frames) == 1:
            frames.append(frames[0].copy())
        return frames, meta
    except Exception:
        return None


def _save_frame_cache(
    cache_root: Optional[Path],
    key: str,
    key_data: dict[str, Any],
    frames: list[ImageObject],
    raw_fps: float,
    sample_fps: float,
) -> bool:
    if cache_root is None or not frames:
        return False

    cache_dir = _frame_cache_dir_for(cache_root, key)
    meta_path = cache_dir / "metadata.json"
    if meta_path.exists():
        return False

    parent = cache_dir.parent
    tmp_dir = parent / f".{key}.{os.getpid()}.tmp"
    try:
        parent.mkdir(parents=True, exist_ok=True)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        frame_names = []
        for idx, frame in enumerate(frames):
            name = f"{idx:04d}.{RL_FRAME_CACHE_IMAGE_EXT}"
            frame_names.append(name)
            frame.convert("RGB").save(
                tmp_dir / name,
                format="JPEG",
                quality=RL_FRAME_CACHE_JPEG_QUALITY,
                optimize=False,
            )

        with (tmp_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "key": key,
                    "key_data": key_data,
                    "frames": frame_names,
                    "num_frames": len(frame_names),
                    "raw_fps": float(raw_fps),
                    "sample_fps": float(sample_fps),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        if cache_dir.exists() and not meta_path.exists():
            shutil.rmtree(cache_dir)
        tmp_dir.rename(cache_dir)
        return True
    except FileExistsError:
        return False
    except Exception as exc:
        print(f"[RLFrameCache] failed to save {cache_dir}: {exc}")
        return False
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _sample_video_frames_decord(vision_info: dict[str, Any]) -> tuple[list[ImageObject], float, float]:
    from decord import VideoReader, cpu
    from qwen_vl_utils.vision_process import smart_nframes

    video_path = _clean_video_path(vision_info["video"])
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    if total_frames == 0:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    raw_fps = float(vr.get_avg_fps() or 24.0)
    nframes = smart_nframes(vision_info, total_frames=total_frames, video_fps=raw_fps)
    indices = np.linspace(0, total_frames - 1, nframes).round().astype(int).tolist()
    batch = vr.get_batch(indices).asnumpy()
    frames = [Image.fromarray(frame).convert("RGB") for frame in batch]
    if len(frames) == 1:
        frames.append(frames[0].copy())

    sample_fps = nframes / max(total_frames, 1e-6) * raw_fps
    return frames, raw_fps, sample_fps


def _sample_video_frames_pyav(vision_info: dict[str, Any]) -> tuple[list[ImageObject], float, float]:
    import av
    from qwen_vl_utils.vision_process import smart_nframes

    video_path = _clean_video_path(vision_info["video"])
    decoded_frames = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        raw_fps = float(stream.average_rate) if stream.average_rate else 24.0
        for frame in container.decode(stream):
            decoded_frames.append(frame.to_image().convert("RGB"))

    if len(decoded_frames) == 0:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    nframes = smart_nframes(vision_info, total_frames=len(decoded_frames), video_fps=raw_fps)
    indices = np.linspace(0, len(decoded_frames) - 1, nframes).round().astype(int).tolist()
    frames = [decoded_frames[i] for i in indices]
    if len(frames) == 1:
        frames.append(frames[0].copy())

    sample_fps = nframes / max(len(decoded_frames), 1e-6) * raw_fps
    return frames, raw_fps, sample_fps


def _sample_video_frames_for_cache(vision_info: dict[str, Any]) -> tuple[list[ImageObject], float, float]:
    backend = os.environ.get("FORCE_QWENVL_VIDEO_READER", "decord").strip().lower()
    if backend != "pyav":
        try:
            return _sample_video_frames_decord(vision_info)
        except Exception as exc:
            if _truthy_env("RL_FRAME_CACHE_VERBOSE"):
                print(f"[RLFrameCache] decord failed for {vision_info['video']}, falling back to pyav: {exc}")

    return _sample_video_frames_pyav(vision_info)


def _fetch_video_from_frames(
    frames: list[ImageObject],
    vision_info: dict[str, Any],
    raw_fps: float,
    sample_fps: float,
    image_patch_size: int = 16,
    return_fps: bool = False,
):
    cached_info = dict(vision_info)
    cached_info["video"] = frames
    cached_info["sample_fps"] = sample_fps
    cached_info["raw_fps"] = raw_fps
    return fetch_video(
        cached_info,
        image_patch_size=image_patch_size,
        return_video_sample_fps=return_fps,
        return_video_metadata=return_fps,
    )


def _fetch_video_with_frame_cache(
    vision_info: dict[str, Any], image_patch_size: int = 16, return_fps: bool = False
):
    cache_root = _rl_frame_cache_root()
    key, key_data = _frame_cache_key(
        vision_info["video"],
        vision_info.get("min_pixels"),
        vision_info.get("max_pixels"),
        vision_info.get("max_frames", 128),
        vision_info.get("fps", 2.0),
    )

    cached = _load_frame_cache(cache_root, key)
    if cached is not None:
        frames, meta = cached
        if _truthy_env("RL_FRAME_CACHE_VERBOSE"):
            print(f"[RLFrameCache] hit {vision_info['video']}")
        return _fetch_video_from_frames(
            frames,
            vision_info,
            raw_fps=float(meta.get("raw_fps") or vision_info.get("fps", 2.0)),
            sample_fps=float(meta.get("sample_fps") or vision_info.get("fps", 2.0)),
            image_patch_size=image_patch_size,
            return_fps=return_fps,
        )

    frames, raw_fps, sample_fps = _sample_video_frames_for_cache(vision_info)
    _save_frame_cache(cache_root, key, key_data, frames, raw_fps=raw_fps, sample_fps=sample_fps)
    if _truthy_env("RL_FRAME_CACHE_VERBOSE"):
        print(f"[RLFrameCache] miss {vision_info['video']}")
    return _fetch_video_from_frames(
        frames,
        vision_info,
        raw_fps=raw_fps,
        sample_fps=sample_fps,
        image_patch_size=image_patch_size,
        return_fps=return_fps,
    )


def process_video(
    video: str, min_pixels: int = 4*32*32, max_pixels: int = 64*32*32, max_frames: int = 128, video_fps: float = 2, return_fps: bool = False
):
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "max_frames": max_frames, "fps": video_fps}
    is_video_path = isinstance(video, (str, os.PathLike))
    try:
        if is_video_path and _rl_frame_cache_root() is not None:
            return _fetch_video_with_frame_cache(
                vision_info,
                image_patch_size=16,
                return_fps=return_fps,
            )

        return fetch_video(
            vision_info,
            image_patch_size=16,
            return_video_sample_fps=return_fps,
            return_video_metadata=return_fps,
        )
    except Exception as exc:
        if not is_video_path:
            raise

        print(f"fetch_video failed for {video}, falling back to pyav: {exc}")
        return _fetch_video_with_pyav(vision_info, image_patch_size=16, return_fps=return_fps)


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            self.dataset = self._load_local_file(data_path)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    @staticmethod
    def _load_local_file(data_path: str) -> HFDataset:
        ext = os.path.splitext(data_path)[-1].lower()
        with open(data_path, encoding="utf-8") as f:
            if ext == ".jsonl":
                records = [json.loads(line) for line in f if line.strip()]
            elif ext == ".json":
                records = json.load(f)
            else:
                raise ValueError(f"Unsupported local dataset file extension: {ext}")

        if isinstance(records, dict):
            for key in ("train", "data", "instances"):
                if isinstance(records.get(key), list):
                    records = records[key]
                    break

        if not isinstance(records, list):
            raise ValueError(f"Expected a list of records in local dataset file: {data_path}")

        all_keys = set()
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"Expected every record to be a dict in local dataset file: {data_path}")
            all_keys.update(record.keys())

        for record in records:
            for key in all_keys:
                record.setdefault(key, None)

        return HFDataset.from_list(records)


    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        pt = (example.get("problem_type") or "").strip().lower()
        question = prompt_str

        if (pt == "multiple choice") and isinstance(example.get("options"), list) and example["options"]:
            opts = "\n".join(example["options"])
            question = f"{question}\nOptions:\n{opts}"

        prompt_str = build_prompt(question, pt)

        if self.image_key in example and isinstance(example.get(self.image_key), list) and len(example.get(self.image_key)) > 0:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            image_count = len(example.get(self.image_key))
            if "<image>" not in prompt_str:
                content_list.extend({"type": "image"} for _ in range(image_count))
                content_list.append({"type": "text", "text": prompt_str})
                return [{"role": "user", "content": content_list}]

            inserted_images = 0
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0 and inserted_images < image_count:
                    content_list.append({"type": "image"})
                    inserted_images += 1

                if content:
                    content_list.append({"type": "text", "text": content})

            while inserted_images < image_count:
                content_list.append({"type": "image"})
                inserted_images += 1

            # print(content_list)

            return [{"role": "user", "content": content_list}]
        elif self.video_key in example and isinstance(example.get(self.video_key), list) and len(example.get(self.video_key)) > 0:
            content_list = []
            video_count = len(example.get(self.video_key))
            if "<video>" not in prompt_str:
                content_list.extend({"type": "video"} for _ in range(video_count))
                content_list.append({"type": "text", "text": prompt_str})
                return [{"role": "user", "content": content_list}]

            inserted_videos = 0
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0 and inserted_videos < video_count:
                    content_list.append({"type": "video"})
                    inserted_videos += 1

                if content:
                    content_list.append({"type": "text", "text": content})

            while inserted_videos < video_count:
                content_list.append({"type": "video"})
                inserted_videos += 1

            # print(content_list)

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]


    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example and isinstance(example.get(self.image_key), list) and len(example.get(self.image_key)) > 0:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            try:
                if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                    images = [os.path.join(self.image_dir, image) for image in images]

            except Exception as e:
                print(f"images type: {type(images)} | value: {images}")
                print("full example:", example)



            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            
            print(images, model_inputs["input_ids"].size(-1))
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example and isinstance(example.get(self.video_key), list) and len(example.get(self.video_key)) > 0:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(
                    process_video(
                        video,
                        min_pixels=self.min_pixels,
                        max_pixels=self.max_pixels,
                        video_fps=self.video_fps,
                    )
                )

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            # print(videos, model_inputs["input_ids"].size(-1))
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)
        example["problem_reserved_text"] = example.get(self.prompt_key, "")
        example["multi_modal_data"] = None
        example.pop(self.prompt_key, None)

        if self.image_key in example and isinstance(example.get(self.image_key), list) and len(example.get(self.image_key)) > 0:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif self.video_key in example and isinstance(example.get(self.video_key), list) and len(example.get(self.video_key)) > 0:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                    video_fps=self.video_fps,
                    return_fps=True,
                )
                video_kwargs = {"do_sample_frames": False}
                processed_videos.append(processed_video)


                video_fps_list.append(video_fps)

            # print([prompt])
            if processed_video is not None:
                # print(processed_video)
                # print(video_kwargs)
                # print(processed_video[0].shape)
                processed_video, video_metadatas = processed_video
                processed_video, video_metadatas = [processed_video], [video_metadatas]
            else:
                video_metadatas = None
            model_inputs= self.processor(text=[prompt], videos=processed_video, add_special_tokens=False, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)

            # print(videos, model_inputs["input_ids"].size(-1))


            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]


        if "images" in example: 
            example.pop("images")
        elif "videos" in example:
            example.pop("videos")

        # print(example)

        # print(example.keys())


        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index



            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)

        # print(example)
        # print(input_ids.shape)


        # print(example)
        return example
