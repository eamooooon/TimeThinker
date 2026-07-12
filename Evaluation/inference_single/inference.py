# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# Import SAM2 visualization (used only for segmentation)
from simple_sam2_vis import visualize_segmentation

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prompting.timethinker import QUESTION_TEMPLATE, TYPE_TEMPLATE as QA_TYPE_TEMPLATE

# vLLM multiprocessing mode
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# ================== MANUAL CONFIGURATION ==================
# Model checkpoint
CHECKPOINT_PATH = "models/TimeThinker-4B"

# Media path (image or video)
MEDIA_PATH = ""
# Whether the media is a video
IS_VIDEO = False
# Question text
QUESTION_TEXT = ""

# Problem type (must be a key in TYPE_TEMPLATE)
# Examples:
# "open-ended", "multiple choice", "math",
# "temporal grounding", "spatial grounding",
# "spatial-temporal grounding", "tracking",
# "segmentation_image", "segmentation_video"
PROBLEM_TYPE = "segmentation_video"

# ==========================================================

TYPE_TEMPLATE = {
    **QA_TYPE_TEMPLATE,
    "temporal grounding": (
        "Provide only the time span in seconds as JSON with key \"time\" inside <answer>.\n"
        "The value of \"time\" must be a two-number list [start, end]."
    ),
    "spatial grounding": (
        "Provide only one bounding box as JSON with key \"boxes\" inside <answer>.\n"
        "The value of \"boxes\" must be a four-number list [x1, y1, x2, y2]."
    ),
    "spatial-temporal grounding": (
        "Provide the time span and bounding boxes as JSON inside <answer>.\n"
        "Use a two-number \"time\" list and a \"boxes\" object. Each \"boxes\" key must be an integer second "
        "within the span, and each value must be [x1, y1, x2, y2]."
    ),
    "tracking": (
        "Track the target object and provide one bounding box per second, up to 32 seconds, as JSON inside <answer>.\n"
        "Use a \"boxes\" object whose keys are seconds and whose values are [x1, y1, x2, y2]."
    ),
    "segmentation_image": (
        "This task prepares inputs for image object segmentation with a specialized model (e.g., SAM2).\n"
        "Provide JSON inside <answer> with \"boxes\", \"positive_points\", and \"negative_points\".\n"
        "Use one [x1, y1, x2, y2] box, three positive points inside the object, and three negative points outside it."
    ),
    "segmentation_video": (
        "This task prepares inputs for video object segmentation with a specialized model (e.g., SAM2).\n"
        "Provide JSON inside <answer> with \"time\", \"boxes\", \"positive_points\", and \"negative_points\".\n"
        "Use one representative time in seconds, one [x1, y1, x2, y2] box, three positive points inside the "
        "object, and three negative points outside it."
    )
}


def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    print(f"video_kwargs: {video_kwargs}")

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


if __name__ == "__main__":
    # Build final prompt text
    type_hint = TYPE_TEMPLATE.get(PROBLEM_TYPE, "")
    full_text = QUESTION_TEMPLATE.format(Question=QUESTION_TEXT) + type_hint

    # Build message in plain form
    if IS_VIDEO:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": MEDIA_PATH,
                        "max_pixels": 256 * 32 * 32,
                        "max_frames": 128,
                        "fps": 2,
                    },
                    {"type": "text", "text": full_text},
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": MEDIA_PATH,
                        "max_pixels": 1024 * 32 * 32,
                    },
                    {"type": "text", "text": full_text},
                ],
            }
        ]

    # vLLM inference
    processor = AutoProcessor.from_pretrained(CHECKPOINT_PATH)
    inputs = [prepare_inputs_for_vllm(messages, processor)]

    llm = LLM(
        model=CHECKPOINT_PATH,
        mm_encoder_tp_mode="data",
        tensor_parallel_size=1,
        max_model_len = 81920,
        gpu_memory_utilization=0.7,
        seed=0,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        top_k=-1,
        stop_token_ids=[],
    )

    print("\n========== PROMPT ==========")
    print(inputs[0]["prompt"])
    print("============================\n")

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        print("\n========== MODEL OUTPUT ==========")
        print(generated_text)

        # Automatically visualize if it is a segmentation task
        if PROBLEM_TYPE in ["segmentation_image", "segmentation_video"]:
            print("\n[Segmentation] Running SAM2 visualization...")
            vis_path = visualize_segmentation(
                media_path=MEDIA_PATH,
                answer_text=generated_text,
                is_video=IS_VIDEO,
            )
            print(f"[Segmentation] Visualization saved to: {vis_path}")
