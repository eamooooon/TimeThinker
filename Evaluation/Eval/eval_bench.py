#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import argparse
import random
import hashlib
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prompting.timethinker import QUESTION_TEMPLATE, TYPE_TEMPLATE

from tqdm import tqdm
try:
    import torch
except Exception:
    torch = None
try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None
try:
    import av
except Exception:
    av = None
try:
    if av is not None:
        av.logging.set_level(av.logging.PANIC)
except Exception:
    pass
from PIL import Image
try:
    import cv2
except Exception:
    cv2 = None
try:
    from decord import VideoReader, cpu
except Exception:
    VideoReader = cpu = None

try:
    from transformers import AutoProcessor
except Exception:
    AutoProcessor = None
try:
    from vllm import LLM, SamplingParams
except Exception:
    LLM = SamplingParams = None
try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None

if torch is not None:
    torch.set_num_threads(int(os.environ.get("EVAL_TORCH_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "8"))))
    torch.set_num_interop_threads(int(os.environ.get("EVAL_TORCH_INTEROP_THREADS", "1")))
if cv2 is not None:
    cv2.setNumThreads(int(os.environ.get("OPENCV_NUM_THREADS", "1")))

# ====== Optional: math equivalence dependencies (automatically degrade if unavailable)======
try:
    from math_verify import parse as math_parse, verify as math_verify
except Exception:
    math_parse = math_verify = None

try:
    from mathruler.grader import grade_answer as math_grade_answer
except Exception:
    math_grade_answer = None
# ====================================================

# =========================
# Default parameters (overridable by command line)
# =========================
DEFAULT_BASE_PREFIX = ""
DEFAULT_BSZ = 64
DEFAULT_SEED = 0
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.001
DEFAULT_TOP_K = -1

# Video/Image preprocessing parameters
DEFAULT_MAX_PIXELS_VIDEO = 256 * 32 * 32
DEFAULT_MAX_FRAMES = 128
DEFAULT_FPS = 2
DEFAULT_MAX_PIXELS_IMAGE = 1024 * 32 * 32
FRAME_CACHE_VERSION = 1
FRAME_CACHE_IMAGE_EXT = "jpg"
FRAME_CACHE_JPEG_QUALITY = 95

# =========================
# Utility functions (parsing and metrics)
# =========================
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")

STRICT_THINK_ANSWER_RE = re.compile(
    r"\A\s*<think>.*?</think>\s*<answer>.*?</answer>\s*\Z",
    re.DOTALL,
)

def extract_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else text.strip()

def valid_option_letters(options: Any) -> set:
    if isinstance(options, list) and options:
        return {chr(ord("A") + i) for i in range(len(options))}
    return set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def extract_multiple_choice_answer(text: str, options: Any = None) -> str:
    if not isinstance(text, str):
        return ""
    valid_letters = valid_option_letters(options)

    tagged = ANSWER_RE.search(text)
    if tagged:
        tagged_text = tagged.group(1).strip()
        exact = re.fullmatch(r"\(?([A-Za-z])\)?[.)]?", tagged_text)
        if exact and exact.group(1).upper() in valid_letters:
            return exact.group(1).upper()
        m = re.search(r"\b([A-Za-z])\b", tagged_text)
        if m and m.group(1).upper() in valid_letters:
            return m.group(1).upper()

    stripped = text.strip()

    # Common malformed strict-format output: "... A\n</answer>" without the opening tag.
    m = re.search(r"(?:^|\n|\s)([A-Za-z])\s*</answer>\s*$", stripped, re.IGNORECASE)
    if m and m.group(1).upper() in valid_letters:
        return m.group(1).upper()

    phrase_patterns = [
        r"(?:the\s+)?(?:correct\s+)?(?:answer|option|choice)(?:\s+is|\s*:)?\s*\(?([A-Za-z])\)?(?:\.|\b)",
        r"(?:therefore|thus|so|hence)[^\n.]{0,120}?\b(?:answer|option|choice)\s*(?:is|:)\s*\(?([A-Za-z])\)?",
        r"\boption\s+([A-Za-z])\b",
    ]
    hits: List[Tuple[int, str]] = []
    for pattern in phrase_patterns:
        for match in re.finditer(pattern, stripped, re.IGNORECASE):
            letter = match.group(1).upper()
            if letter in valid_letters:
                hits.append((match.start(), letter))
    if hits:
        return sorted(hits)[-1][1]

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    for line in reversed(lines[-8:]):
        m = re.fullmatch(r"\(?([A-Za-z])\)?[.)]?", line)
        if m and m.group(1).upper() in valid_letters:
            return m.group(1).upper()
        m = re.match(r"^([A-Za-z])\s*[.)]\s+", line)
        if m and m.group(1).upper() in valid_letters:
            return m.group(1).upper()

    tail = stripped[-500:]
    tail_hits = [
        match.group(1).upper()
        for match in re.finditer(r"\b([A-Za-z])\b", tail)
        if match.group(1).upper() in valid_letters
    ]
    return tail_hits[-1] if tail_hits else ""

def extract_number_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""
    tagged = ANSWER_RE.search(text)
    search_text = tagged.group(1) if tagged else text

    phrase_match = re.search(
        r"(?:answer|value|result)(?:\s+is|\s*:)?\s*(" + NUMBER_RE.pattern + r")",
        search_text,
        re.IGNORECASE,
    )
    if phrase_match:
        return phrase_match.group(1).replace(",", "")

    matches = NUMBER_RE.findall(search_text)
    return matches[-1].replace(",", "") if matches else ""

def extract_prediction_answer(text: str, problem_type: str = "", options: Any = None) -> str:
    ptype = (problem_type or "").strip().lower()
    if ptype == "multiple choice":
        return extract_multiple_choice_answer(text, options)
    if ptype in {"numerical", "regression"}:
        return extract_number_answer(text)
    return extract_answer(text)

def has_answer_tag(text: str) -> bool:
    return bool(isinstance(text, str) and ANSWER_RE.search(text))

def has_think_tag(text: str) -> bool:
    return bool(isinstance(text, str) and "<think>" in text and "</think>" in text)

def has_strict_think_answer(text: str) -> bool:
    return bool(isinstance(text, str) and STRICT_THINK_ANSWER_RE.fullmatch(text))

def normalize_number(num_str: str) -> Optional[float]:
    try:
        return float((num_str or "").replace(",", ""))
    except Exception:
        return None

def _is_list_of_numbers(x, n=None) -> bool:
    if not isinstance(x, list):
        return False
    if n is not None and len(x) != n:
        return False
    try:
        for v in x:
            float(v)
        return True
    except Exception:
        return False

def iou_1d(pred: List[float], gt: List[float]) -> float:
    if not _is_list_of_numbers(pred, 2) or not _is_list_of_numbers(gt, 2):
        return 0.0
    s1, e1 = float(pred[0]), float(pred[1])
    s2, e2 = float(gt[0]), float(gt[1])
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 1e-12 else 0.0

def iou_2d(box1: List[float], box2: List[float]) -> float:
    if not _is_list_of_numbers(box1, 4) or not _is_list_of_numbers(box2, 4):
        return 0.0
    x1, y1, x2, y2 = map(float, box1)
    X1, Y1, X2, Y2 = map(float, box2)
    inter_x1, inter_y1 = max(x1, X1), max(y1, Y1)
    inter_x2, inter_y2 = min(x2, X2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area2 = max(0.0, X2 - X1) * max(0.0, Y2 - Y1)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 1e-12 else 0.0

def mean_iou_over_gt_frames(pred_boxes: Dict[str, List[float]], gt_boxes: Dict[str, List[float]]) -> float:
    if not isinstance(gt_boxes, dict) or not gt_boxes:
        return 0.0
    total, n = 0.0, 0
    for k, gbox in gt_boxes.items():
        total += iou_2d(pred_boxes.get(k, []), gbox)
        n += 1
    return total / n if n > 0 else 0.0

def mean_iou_over_intersection(pred_boxes: Dict[str, List[float]], gt_boxes: Dict[str, List[float]]) -> float:
    if not isinstance(pred_boxes, dict) or not isinstance(gt_boxes, dict):
        return 0.0
    common = [k for k in pred_boxes.keys() if k in gt_boxes]
    if not common:
        return 0.0
    vals = [iou_2d(pred_boxes[k], gt_boxes[k]) for k in common]
    return sum(vals) / len(vals) if vals else 0.0

def wer(reference: str, hypothesis: str) -> float:
    ref_words, hyp_words = (reference or "").split(), (hypothesis or "").split()
    m, n = len(ref_words), len(hyp_words)
    d = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): d[i][0] = i
    for j in range(n+1): d[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[m][n] / max(1, m)

def compute_rouge_score(reference: str, hypothesis: str) -> float:
    if rouge_scorer is None:
        ref_tokens = set((reference or "").lower().split())
        hyp_tokens = set((hypothesis or "").lower().split())
        if not ref_tokens and not hyp_tokens:
            return 1.0
        if not ref_tokens or not hyp_tokens:
            return 0.0
        return len(ref_tokens & hyp_tokens) / len(ref_tokens | hyp_tokens)
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    scores = scorer.score(reference or "", hypothesis or "")
    return (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3.0

# =============== Math equivalence (math) ===============
def _math_equivalent(gt: str, pred: str) -> bool:
    # Prefer math_verify; if it fails, fall back to mathruler.grader; if that also fails, use exact string match
    try:
        if math_parse and math_verify:
            return bool(math_verify(math_parse(gt), math_parse(pred)))
    except Exception:
        pass
    try:
        if math_grade_answer:
            return bool(math_grade_answer(pred, gt))
    except Exception:
        pass
    return gt.strip() == pred.strip()

# =============== Strict MRA (Regression) ===============
def mean_relative_accuracy_strict(pred: float, target: float) -> float:
    """
    MRA = (1/10) * sum_{θ in {0.5,0.55,...,0.95}}  1[ |y_hat - y| / |y| < 1 - θ ]
    If target≈0 then return 0 (avoid division by zero; this metric assumes y!=0).
    """
    try:
        p = float(pred); t = float(target)
        if abs(t) < 1e-12:
            return 0.0
    except Exception:
        return 0.0

    rel = abs(p - t) / abs(t)
    count = 0
    for k in range(10):  # 0.5, 0.55, ..., 0.95
        theta = 0.5 + 0.05 * k
        if rel < (1.0 - theta):
            count += 1
    return count / 10.0

# =============== accuracy (return value + component details) ===============
def accuracy_only(
    response: str,
    ground_truth: str,
    data_type: str,
    problem_type: str,
    options: Any = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Returns:
      - accuracy ∈ [0,1]
      - components: raw score of each sub-metric (for composite tasks only sub-metrics are recorded, not the weighted sum)
        e.g.:
          temporal grounding: {"tiou": x}
          spatial grounding : {"iou": x}
          spatial-temporal  : {"tiou": a, "miou_inter": b}
          tracking          : {"miou_gt": x}
          seg_image         : {"iou": i, "pos_sim": p, "neg_sim": n}
          seg_video         : {"iou": i, "time_sim": t, "pos_sim": p, "neg_sim": n}
    """
    ans = extract_prediction_answer(response, problem_type, options) or response.strip()
    gt  = extract_answer(ground_truth) or ground_truth or ""
    ptype = (problem_type or "").strip()
    ptype_l = ptype.lower()
    dtype = (data_type or "").lower()

    # multiple choice
    if ptype_l == "multiple choice":
        a = (ans.strip()[:1] if ans else "").upper()
        g = (gt.strip()[:1]  if gt  else "").upper()
        return (1.0 if a == g and a in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" else 0.0, {})

    # numerical (strict to two decimals)
    if ptype_l == "numerical":
        a_num, g_num = normalize_number(ans), normalize_number(gt)
        ok = (a_num is not None and g_num is not None and round(a_num,2) == round(g_num,2))
        return (1.0 if ok else 0.0, {})

    # regression (strict MRA)
    if ptype_l == "regression":
        a_num, g_num = normalize_number(ans), normalize_number(gt)
        if a_num is None or g_num is None:
            return (0.0, {})
        mra = mean_relative_accuracy_strict(a_num, g_num)
        return (float(mra), {})   # single metric, not split into components

    # ocr
    if ptype_l == "ocr" or ptype == "OCR":
        return (max(0.0, min(1.0, 1.0 - wer(gt, ans))), {})

    # open-ended (ROUGE)
    if ptype_l in {"open-ended", "free-form"}:
        return (max(0.0, min(1.0, compute_rouge_score(gt, ans))), {})

    # math (symbolic equivalence)
    if ptype_l == "math":
        return (1.0 if _math_equivalent(gt, ans) else 0.0, {})

    # JSON types
    def _json(s: str):
        try:
            return json.loads(s)
        except Exception:
            return None

    # temporal grounding: tIoU
    if ptype_l == "temporal grounding":
        pred = _json(ans); gtj = _json(gt)
        tiou = iou_1d(pred.get("time") if isinstance(pred, dict) else None,
                      gtj.get("time")  if isinstance(gtj, dict)  else None)
        # accuracy = tiou; components only record tiou
        return (float(tiou), {"tiou": float(tiou)})

    # spatial grounding: IoU
    if ptype_l == "spatial grounding":
        pred = _json(ans); gtj = _json(gt)
        iou = iou_2d(pred.get("boxes") if isinstance(pred, dict) else None,
                     gtj.get("boxes")  if isinstance(gtj, dict)  else None)
        return (float(iou), {"iou": float(iou)})

    # spatial-temporal grounding: record components separately, do not put the combined value into components
    if ptype_l == "spatial-temporal grounding":
        pred = _json(ans); gtj = _json(gt)
        if not isinstance(pred, dict) or not isinstance(gtj, dict):
            return (0.0, {"tiou": 0.0, "miou_inter": 0.0})
        tiou = iou_1d(pred.get("time"), gtj.get("time"))
        pboxes, gboxes = pred.get("boxes"), gtj.get("boxes")
        miou = mean_iou_over_intersection(pboxes if isinstance(pboxes, dict) else {},
                                          gboxes if isinstance(gboxes, dict) else {})
        # accuracy is still 0.5*tiou + 0.5*miou for overall; components only record tiou and miou
        acc = 0.5 * tiou + 0.5 * miou
        return (float(acc), {"tiou": float(tiou), "miou_inter": float(miou)})

    # tracking: mean mIoU over GT frames (missing=0)
    if ptype_l == "tracking":
        pred = _json(ans); gtj = _json(gt)
        if not isinstance(pred, dict) or not isinstance(gtj, dict):
            return (0.0, {"miou_gt": 0.0})
        pboxes, gboxes = pred.get("boxes"), gtj.get("boxes")
        miou_gt = mean_iou_over_gt_frames(pboxes if isinstance(pboxes, dict) else {},
                                          gboxes if isinstance(gboxes, dict) else {})
        return (float(miou_gt), {"miou_gt": float(miou_gt)})

    # segmentation —— do not compute metrics, directly return 0.0 and empty components (beyond the added logic, keep unchanged)
    if ptype_l == "segmentation":
        return (0.0, {})

    # unknown type
    return (0.0, {})

# =============== Component-level R@t statistics ===============
RECALL_THRESHOLDS = [0.3, 0.5, 0.7]
BOOTSTRAP_SAMPLES = int(os.environ.get("EVAL_BOOTSTRAP_SAMPLES", "1000"))
BOOTSTRAP_SEED = int(os.environ.get("EVAL_BOOTSTRAP_SEED", "0"))

def sanitize_metric_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    text = re.sub(r"\s+", "_", text)
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)

def category_for_example(example: Dict[str, Any]) -> str:
    # Keep categories broad enough to be useful; data_source/path are too high-cardinality.
    for key in ("dim", "task_type", "original_question_type", "sub_category", "domain"):
        value = example.get(key)
        if value is not None and str(value).strip() and str(value).strip().lower() != "none":
            return str(value).strip()
    return ""

def valid_answer_for_problem(pred_ans: str, example: Dict[str, Any]) -> bool:
    ptype = (example.get("problem_type") or "").strip().lower()
    ans = (pred_ans or "").strip()

    if ptype == "multiple choice":
        valid_letters = valid_option_letters(example.get("options"))
        return len(ans) == 1 and ans.upper() in valid_letters

    if ptype in {"numerical", "regression"}:
        return normalize_number(ans) is not None

    if ptype in {"ocr", "open-ended", "free-form", "math"}:
        return bool(ans)

    return bool(ans)

def bootstrap_ci(values: List[float], samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1 or samples <= 0:
        mean_value = sum(values) / len(values)
        return (float(mean_value), float(mean_value))

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    low_idx = int(0.025 * (samples - 1))
    high_idx = int(0.975 * (samples - 1))
    return (float(means[low_idx]), float(means[high_idx]))

def init_aggregator() -> Dict[str, Any]:
    return {
        "overall_acc_sum": 0.0,
        "overall_count": 0,
        "acc_values": [],
        "per_type_acc_sum": defaultdict(float),
        "per_type_count": defaultdict(int),
        "category_acc_sum": defaultdict(float),
        "category_count": defaultdict(int),
        "answer_extract_count": 0,
        "invalid_answer_count": 0,
        "output_token_sum": 0,
        "output_token_count": 0,
        "truncated_count": 0,
        "has_think_count": 0,
        "strict_format_count": 0,

        # Component-level (by problem_type, then by component name)
        "comp_sum": defaultdict(lambda: defaultdict(float)),     # comp_sum[ptype][comp]
        "comp_count": defaultdict(lambda: defaultdict(int)),     # comp_count[ptype][comp]
        "comp_recall_hits": defaultdict(lambda: defaultdict(lambda: defaultdict(int))),  # hits[ptype][comp][t]
    }

def accumulate_metrics(
    agg: Dict[str, Any],
    ptype: str,
    acc: float,
    comp: Dict[str, float],
    answer_extracted: bool = False,
    invalid_answer: bool = False,
    output_tokens: Optional[int] = None,
    truncated: bool = False,
    category: str = "",
    has_think: bool = False,
    strict_format: bool = False,
):
    pkey = (ptype or "").strip()

    # overall + per_type accuracy
    agg["overall_acc_sum"] += float(acc)
    agg["overall_count"]   += 1
    agg["acc_values"].append(float(acc))
    agg["per_type_acc_sum"][pkey] += float(acc)
    agg["per_type_count"][pkey]   += 1

    if category:
        agg["category_acc_sum"][category] += float(acc)
        agg["category_count"][category] += 1

    if answer_extracted:
        agg["answer_extract_count"] += 1
    if invalid_answer:
        agg["invalid_answer_count"] += 1
    if output_tokens is not None:
        agg["output_token_sum"] += int(output_tokens)
        agg["output_token_count"] += 1
    if truncated:
        agg["truncated_count"] += 1
    if has_think:
        agg["has_think_count"] += 1
    if strict_format:
        agg["strict_format_count"] += 1

    # components
    for cname, cval in comp.items():
        val = float(cval)
        agg["comp_sum"][pkey][cname]   += val
        agg["comp_count"][pkey][cname] += 1
        for t in RECALL_THRESHOLDS:
            if val >= t:
                agg["comp_recall_hits"][pkey][cname][t] += 1

def finalize_metrics(agg: Dict[str, Any], include_bootstrap: bool = False) -> Dict[str, Any]:
    out = {}
    count = max(1, agg["overall_count"])

    # overall
    out["overall/acc"] = agg["overall_acc_sum"] / count
    out["answer/acc"] = out["overall/acc"]
    out["answer/extract_rate"] = agg["answer_extract_count"] / count
    out["answer/invalid_rate"] = agg["invalid_answer_count"] / count
    out["output/avg_tokens"] = agg["output_token_sum"] / max(1, agg["output_token_count"])
    out["output/truncation_rate"] = agg["truncated_count"] / count
    out["format/has_think_rate"] = agg["has_think_count"] / count
    out["format/strict_rate"] = agg["strict_format_count"] / count
    out["answer_acc"] = out["answer/acc"]
    out["answer_extract_rate"] = out["answer/extract_rate"]
    out["invalid_answer_rate"] = out["answer/invalid_rate"]
    out["avg_output_tokens"] = out["output/avg_tokens"]
    out["truncation_rate"] = out["output/truncation_rate"]

    if include_bootstrap:
        low, high = bootstrap_ci(agg["acc_values"])
        out["answer/bootstrap_ci_low"] = low
        out["answer/bootstrap_ci_high"] = high
        out["bootstrap_ci"] = {"low": low, "high": high}

    # per_type accuracy
    for pkey, cnt in agg["per_type_count"].items():
        out[f"{pkey}/acc"] = agg["per_type_acc_sum"][pkey] / max(1, cnt)

    if agg["category_count"]:
        category_accs = []
        per_category_acc = {}
        per_category_count = {}
        for category, cnt in sorted(agg["category_count"].items()):
            acc = agg["category_acc_sum"][category] / max(1, cnt)
            category_accs.append(acc)
            per_category_acc[category] = acc
            per_category_count[category] = cnt
            slug = sanitize_metric_key(category)
            out[f"category/{slug}/acc"] = acc
            out[f"category/{slug}/count"] = cnt
        out["per_category/macro_acc"] = sum(category_accs) / max(1, len(category_accs))
        out["per_category_acc"] = per_category_acc
        out["per_category_count"] = per_category_count

    # per_type component means + recalls
    for pkey, comp_sums in agg["comp_sum"].items():
        for cname, s in comp_sums.items():
            c = agg["comp_count"][pkey][cname]
            if c > 0:
                out[f"{pkey}/{cname}/mean"] = s / c
                for t in RECALL_THRESHOLDS:
                    hits = agg["comp_recall_hits"][pkey][cname][t]
                    out[f"{pkey}/{cname}/R@{t}"] = hits / c
    return out

# =========================
# vLLM input packing
# =========================
def build_user_content_item(
    data_type: str,
    full_path: Any,
    add_image_path: Optional[str],
    max_pixels_video: int,
    max_frames: int,
    fps: int,
    max_pixels_image: int
) -> List[Dict[str, Any]]:
    content = []
    if data_type == "video":
        content.append({
            "type": "video",
            "video": full_path,
            "max_pixels": max_pixels_video,
            "max_frames": max_frames,
            "fps": fps,
            "sample_fps": fps,
        })
    elif data_type == "image":
        content.append({
            "type": "image",
            "image": full_path,
            "max_pixels": max_pixels_image
        })
    elif data_type == "video-image":
        content.append({
            "type": "video",
            "video": full_path,
            "max_pixels": max_pixels_video,
            "max_frames": max_frames,
            "fps": fps,
            "sample_fps": fps,
        })
        if not add_image_path:
            raise ValueError("data_type=video-image requires additional_path (image path)")
        content.append({
            "type": "image",
            "image": add_image_path,
            "max_pixels": max_pixels_image
        })
    return content


def read_video_frames_pyav(
    video_path: str,
    max_frames: int,
    fps: int,
    video_start: Optional[float] = None,
    video_end: Optional[float] = None,
) -> List[Any]:
    """Decode a video into PIL frames before qwen-vl-utils sees it.

    Path-based video decoding in qwen-vl-utils depends on decord/torchvision.
    In this eval environment decord fails on some MVBench files, then falls
    back to torchvision.io.read_video, which no longer exists in torchvision
    0.26. Passing a frame list uses qwen-vl-utils' stable image-list branch.
    """
    if video_path.startswith("file://"):
        video_path = video_path[7:]

    start_time = float(video_start or 0.0)
    end_time = float(video_end) if video_end is not None else None
    target_fps = max(float(fps or DEFAULT_FPS), 1e-6)
    step = 1.0 / target_fps
    max_frames = max(1, int(max_frames))

    frames = []
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        video_fps = float(stream.average_rate) if stream.average_rate else target_fps
        next_time = start_time

        for idx, frame in enumerate(container.decode(stream)):
            frame_time = frame.time if frame.time is not None else idx / max(video_fps, 1e-6)
            if frame_time < start_time:
                continue
            if end_time is not None and frame_time > end_time:
                break
            if frame_time + 1e-6 < next_time:
                continue

            frames.append(frame.to_image().convert("RGB"))
            next_time += step
            if len(frames) >= max_frames:
                break
    finally:
        container.close()

    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    if len(frames) == 1:
        frames.append(frames[0].copy())
    return frames


def read_video_frames_decord(
    video_path: str,
    max_frames: int,
    fps: int,
    video_start: Optional[float] = None,
    video_end: Optional[float] = None,
) -> List[Any]:
    if VideoReader is None or cpu is None:
        raise RuntimeError("decord is not available")

    if video_path.startswith("file://"):
        video_path = video_path[7:]

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    if total_frames <= 0:
        raise ValueError(f"No frames decoded from {video_path}")

    source_fps = float(vr.get_avg_fps() or fps or DEFAULT_FPS)
    target_fps = max(float(fps or DEFAULT_FPS), 1e-6)
    max_frames = max(1, int(max_frames))

    start_frame = 0
    if video_start is not None:
        start_frame = max(0, int(float(video_start) * source_fps))

    end_frame = total_frames - 1
    if video_end is not None:
        end_frame = min(total_frames - 1, int(float(video_end) * source_fps))

    if end_frame < start_frame:
        raise ValueError(f"Invalid video time range for {video_path}")

    step = max(1, int(round(source_fps / target_fps)))
    indices = list(range(start_frame, end_frame + 1, step))
    if not indices:
        indices = [start_frame]

    if len(indices) > max_frames:
        stride = len(indices) / max_frames
        indices = [indices[min(int(i * stride), len(indices) - 1)] for i in range(max_frames)]

    batch = vr.get_batch(indices).asnumpy()
    frames = [Image.fromarray(frame).convert("RGB") for frame in batch]
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    if len(frames) == 1:
        frames.append(frames[0].copy())
    return frames


def _clean_video_path(video_path: str) -> str:
    if video_path.startswith("file://"):
        video_path = video_path[7:]
    return video_path


def frame_cache_key(
    video_path: str,
    max_frames: int,
    fps: int,
    video_start: Optional[float] = None,
    video_end: Optional[float] = None,
) -> Tuple[str, Dict[str, Any]]:
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
        "version": FRAME_CACHE_VERSION,
        "source": source,
        "max_frames": int(max_frames),
        "fps": float(fps),
        "video_start": None if video_start is None else float(video_start),
        "video_end": None if video_end is None else float(video_end),
        "image_ext": FRAME_CACHE_IMAGE_EXT,
        "jpeg_quality": FRAME_CACHE_JPEG_QUALITY,
    }
    digest = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode("utf-8")).hexdigest()
    return digest, key_data


def frame_cache_dir_for(cache_root: Path, key: str) -> Path:
    return cache_root / key[:2] / key


def load_frame_cache(cache_root: Optional[Path], key: str) -> Optional[List[Any]]:
    if cache_root is None:
        return None

    cache_dir = frame_cache_dir_for(cache_root, key)
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
        return frames
    except Exception:
        return None


def save_frame_cache(
    cache_root: Optional[Path],
    key: str,
    key_data: Dict[str, Any],
    frames: List[Any],
) -> bool:
    if cache_root is None or not frames:
        return False

    cache_dir = frame_cache_dir_for(cache_root, key)
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
            name = f"{idx:04d}.{FRAME_CACHE_IMAGE_EXT}"
            frame_names.append(name)
            frame.convert("RGB").save(
                tmp_dir / name,
                format="JPEG",
                quality=FRAME_CACHE_JPEG_QUALITY,
                optimize=False,
            )

        with (tmp_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "key": key,
                    "key_data": key_data,
                    "frames": frame_names,
                    "num_frames": len(frame_names),
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
        print(f"[Warn] Failed to save frame cache {cache_dir}: {exc}")
        return False
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def prepare_inputs_for_vllm_single(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs
    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs
    }

# =========================
# Main pipeline
# =========================
def main():
    eval_start_time = time.perf_counter()
    parser = argparse.ArgumentParser(description="Multimodal Evaluation (accuracy + component-wise recalls)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_json", type=str, required=True)

    # Output name: out_dir / (basename(input_json) + suffix + .json)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--suffix", type=str, default="_eval")

    parser.add_argument("--base_prefix", type=str, default=DEFAULT_BASE_PREFIX)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BSZ)
    parser.add_argument("--max_samples", type=int, default=-1, help="Run only the first N samples when > 0.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)

    parser.add_argument("--max_pixels_video", type=int, default=DEFAULT_MAX_PIXELS_VIDEO)
    parser.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--max_pixels_image", type=int, default=DEFAULT_MAX_PIXELS_IMAGE)
    parser.add_argument(
        "--video_reader",
        choices=["auto", "decord", "pyav", "qwen"],
        default=os.environ.get("EVAL_VIDEO_READER", "auto"),
        help="auto decodes with decord first and falls back to pyav, skipping torchvision.",
    )
    parser.add_argument(
        "--frame_cache_dir",
        type=str,
        default=os.environ.get("EVAL_FRAME_CACHE_DIR", ""),
        help="Directory for runtime decoded-frame cache. Defaults to <base_prefix>/.cache/eval_frames.",
    )
    parser.add_argument(
        "--disable_frame_cache",
        action="store_true",
        default=os.environ.get("EVAL_DISABLE_FRAME_CACHE", "0") == "1",
        help="Disable runtime decoded-frame cache.",
    )
    parser.add_argument(
        "--rescore_existing",
        action="store_true",
        help="Only rescore an existing output JSON with the current answer extraction logic; do not load the model.",
    )
    args = parser.parse_args()

    in_base = Path(args.input_json).stem
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = out_dir / f"{in_base}{args.suffix}.json"

    default_cache_base = Path(args.base_prefix) if args.base_prefix else Path(args.input_json).parent
    frame_cache_root = None
    if not args.disable_frame_cache:
        frame_cache_root = Path(args.frame_cache_dir) if args.frame_cache_dir else default_cache_base / ".cache" / "eval_frames"
        frame_cache_root.mkdir(parents=True, exist_ok=True)
        print(f"[FrameCache] dir={frame_cache_root}")
    else:
        print("[FrameCache] disabled")
    frame_cache_stats = defaultdict(int)

    # Read data
    if args.input_json.endswith(".jsonl"):
        data = []
        with open(args.input_json, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    else:
        with open(args.input_json, "r", encoding="utf-8") as f:
            data = json.load(f)

    if args.max_samples > 0:
        data = data[: args.max_samples]
        print(f"[Smoke] Running first {len(data)} samples from {args.input_json}.")

    # Resume from checkpoint
    final_output: List[Dict[str, Any]] = []
    agg = init_aggregator()
    start_idx = 0
    existing_meta: Dict[str, Any] = {}
    if output_json_path.exists():
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            final_output = existing.get("results", [])
            existing_meta = existing.get("meta", {}) if isinstance(existing.get("meta"), dict) else {}
            # Replay once to restore aggregation (ensure consistency)
            agg = init_aggregator()
            for sample in final_output:
                ptype = sample.get("problem_type", "")
                out_text = sample.get("output", "")
                pred_ans = extract_prediction_answer(out_text, ptype, sample.get("options"))
                acc, comps = accuracy_only(
                    response=out_text,
                    ground_truth=sample.get("solution", ""),
                    data_type=(sample.get("data_type") or "").strip().lower(),
                    problem_type=ptype,
                    options=sample.get("options"),
                )
                sample["prediction"] = pred_ans
                sample["accuracy"] = float(acc)
                sample["answer_extracted"] = bool(pred_ans)
                sample["invalid_answer"] = not valid_answer_for_problem(pred_ans, sample)
                if comps:
                    sample["components"] = {k: float(v) for k, v in comps.items()}
                elif "components" in sample:
                    sample.pop("components", None)
                output_tokens = sample.get("output_tokens")
                if output_tokens is not None:
                    try:
                        output_tokens = int(output_tokens)
                    except Exception:
                        output_tokens = None
                accumulate_metrics(
                    agg,
                    ptype,
                    acc,
                    comps,
                    answer_extracted=bool(pred_ans),
                    invalid_answer=not valid_answer_for_problem(pred_ans, sample),
                    output_tokens=output_tokens,
                    truncated=bool(sample.get("truncated", False)),
                    category=sample.get("category", category_for_example(sample)),
                    has_think=bool(sample.get("has_think", has_think_tag(out_text))),
                    strict_format=bool(sample.get("strict_format", has_strict_think_answer(out_text))),
                )
            start_idx = len(final_output)
            print(f"[Resume] Found {start_idx} processed samples, resume from {start_idx}.")
        except Exception as e:
            print(f"[Warn] Failed to read existing output: {e}")

    if args.rescore_existing:
        if not output_json_path.exists():
            raise FileNotFoundError(f"--rescore_existing requires an existing output file: {output_json_path}")
        metrics_dict = finalize_metrics(agg, include_bootstrap=True)
        meta = dict(existing_meta)
        meta["rescored"] = True
        meta["rescore_elapsed_seconds"] = time.perf_counter() - eval_start_time
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "results": final_output,
                    "metrics": metrics_dict,
                    "meta": meta,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[Rescore] Saved {len(final_output)} samples to {output_json_path}")
        print("[Metrics]")
        for k, v in metrics_dict.items():
            print(f"{k}: {v}")
        return

    # Initialize model
    missing_runtime = [
        name
        for name, obj in (
            ("torch", torch),
            ("transformers", AutoProcessor),
            ("vllm", LLM),
            ("SamplingParams", SamplingParams),
            ("qwen_vl_utils", process_vision_info),
        )
        if obj is None
    ]
    if missing_runtime:
        raise RuntimeError(
            "Full evaluation requires missing runtime dependencies: "
            + ", ".join(missing_runtime)
            + ". Use --rescore_existing to update existing outputs without generation."
        )
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    torch.manual_seed(args.seed)

    processor = AutoProcessor.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        max_model_len = 81920,
        gpu_memory_utilization=0.8,
        mm_encoder_tp_mode="data",
        tensor_parallel_size=torch.cuda.device_count(),
        seed=args.seed
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        stop_token_ids=[],
    )

    # Build prompt
    def build_prompt_text(example: Dict[str, Any]) -> str:
        prompt_str = example.get("problem") or ""
        data_type = (example.get("data_type") or "").strip().lower()
        pt = example.get("problem_type") or ""

        # multiple choice: append options
        question = prompt_str
        if (pt == "multiple choice") and isinstance(example.get("options"), list) and example["options"]:
            opts = "\n".join(example["options"])
            question = f"{question}\nOptions:\n{opts}"

        pt_lower = pt.strip().lower()
        type_key = pt if pt in TYPE_TEMPLATE else pt_lower

        tail = TYPE_TEMPLATE.get(type_key, "")
        return QUESTION_TEMPLATE.format(Question=question) + tail

    def frame_cache_meta() -> Dict[str, Any]:
        return {
            "enabled": frame_cache_root is not None,
            "dir": str(frame_cache_root) if frame_cache_root is not None else None,
            "hit": int(frame_cache_stats["hit"]),
            "miss": int(frame_cache_stats["miss"]),
            "write": int(frame_cache_stats["write"]),
            "write_skip": int(frame_cache_stats["write_skip"]),
            "fallback_to_pyav": int(frame_cache_stats["fallback_to_pyav"]),
        }

    def output_meta() -> Dict[str, Any]:
        return {
            "batch_size": BSZ,
            "model_path": args.model_path,
            "input_json": args.input_json,
            "elapsed_seconds": time.perf_counter() - eval_start_time,
            "frame_cache": frame_cache_meta(),
        }

    def read_video_frames_with_cache(
        video_path: str,
        video_reader: str,
        video_start: Optional[float] = None,
        video_end: Optional[float] = None,
    ) -> List[Any]:
        cache_key, key_data = frame_cache_key(
            video_path,
            max_frames=args.max_frames,
            fps=args.fps,
            video_start=video_start,
            video_end=video_end,
        )

        cached_frames = load_frame_cache(frame_cache_root, cache_key)
        if cached_frames is not None:
            frame_cache_stats["hit"] += 1
            return cached_frames

        if frame_cache_root is not None:
            frame_cache_stats["miss"] += 1

        if video_reader == "decord":
            frames = read_video_frames_decord(
                video_path,
                max_frames=args.max_frames,
                fps=args.fps,
                video_start=video_start,
                video_end=video_end,
            )
        elif video_reader == "pyav":
            frames = read_video_frames_pyav(
                video_path,
                max_frames=args.max_frames,
                fps=args.fps,
                video_start=video_start,
                video_end=video_end,
            )
        else:
            try:
                frames = read_video_frames_decord(
                    video_path,
                    max_frames=args.max_frames,
                    fps=args.fps,
                    video_start=video_start,
                    video_end=video_end,
                )
            except Exception as e:
                frame_cache_stats["fallback_to_pyav"] += 1
                print(f"[Warn] decord video reader failed, fallback to pyav: {e}")
                frames = read_video_frames_pyav(
                    video_path,
                    max_frames=args.max_frames,
                    fps=args.fps,
                    video_start=video_start,
                    video_end=video_end,
                )

        if save_frame_cache(frame_cache_root, cache_key, key_data, frames):
            frame_cache_stats["write"] += 1
        elif frame_cache_root is not None:
            frame_cache_stats["write_skip"] += 1
        return frames

    def prepare_example_input(example: Dict[str, Any], video_reader: str = "qwen") -> Dict[str, Any]:
        raw_path = example.get("path") or ""
        full_path = os.path.join(args.base_prefix, raw_path.lstrip("./").lstrip("/"))
        data_type = (example.get("data_type") or "").strip().lower()

        add_path = None
        if data_type == "video-image":
            add_raw = example.get("additional_path") or ""
            add_path = os.path.join(args.base_prefix, add_raw.lstrip("./").lstrip("/"))

        video_payload = full_path
        if data_type in {"video", "video-image"}:
            video_start = example.get("video_start", example.get("start"))
            video_end = example.get("video_end", example.get("end"))
            if video_reader in {"auto", "decord", "pyav"}:
                video_payload = read_video_frames_with_cache(
                    full_path,
                    video_reader=video_reader,
                    video_start=video_start,
                    video_end=video_end,
                )

        content = build_user_content_item(
            data_type=data_type,
            full_path=video_payload,
            add_image_path=add_path,
            max_pixels_video=args.max_pixels_video,
            max_frames=args.max_frames,
            fps=args.fps,
            max_pixels_image=args.max_pixels_image
        )
        content.append({"type": "text", "text": build_prompt_text(example)})
        return prepare_inputs_for_vllm_single([{"role": "user", "content": content}], processor)

    # Main loop
    BSZ = args.batch_size
    progress_desc = f"{in_base} batches"
    for i in tqdm(range(start_idx, len(data), BSZ), desc=progress_desc):
        batch = data[i:i+BSZ]

        inputs_for_vllm = []
        for example in batch:
            inputs_for_vllm.append(prepare_example_input(example, video_reader=args.video_reader))

        # Generation
        try:
            outputs = llm.generate(inputs_for_vllm, sampling_params=sampling_params, use_tqdm=False)
            texts = []
            completion_infos = []
            for output in outputs:
                completion = output.outputs[0] if getattr(output, "outputs", None) else None
                if completion is None:
                    texts.append("<answer>ERROR</answer>")
                    completion_infos.append({
                        "finish_reason": "missing_completion",
                        "stop_reason": None,
                        "output_tokens": 0,
                        "truncated": False,
                    })
                    continue

                text = completion.text
                token_ids = getattr(completion, "token_ids", None)
                output_tokens = len(token_ids) if token_ids is not None else None
                finish_reason = getattr(completion, "finish_reason", None)
                stop_reason = getattr(completion, "stop_reason", None)
                finish_reason_l = str(finish_reason or "").lower()
                truncated = (
                    finish_reason_l == "length"
                    or (
                        output_tokens is not None
                        and output_tokens >= args.max_tokens
                        and finish_reason_l not in {"stop", "eos_token"}
                    )
                )

                texts.append(text)
                completion_infos.append({
                    "finish_reason": finish_reason,
                    "stop_reason": stop_reason,
                    "output_tokens": output_tokens,
                    "truncated": truncated,
                })
        except Exception as e:
            print(f"[Error] vLLM generate failed at batch start_idx={i}: {e}")
            texts = ["<answer>ERROR</answer>"] * len(inputs_for_vllm)
            completion_infos = [
                {
                    "finish_reason": "generation_error",
                    "stop_reason": None,
                    "output_tokens": 0,
                    "truncated": False,
                }
                for _ in inputs_for_vllm
            ]

        # Evaluation + accumulation + write results
        for example, out_text, completion_info in zip(batch, texts, completion_infos):
            pred_ans = extract_prediction_answer(out_text, example.get("problem_type", ""), example.get("options"))
            gt_ans   = example.get("solution", "")

            acc, components = accuracy_only(
                response=out_text,
                ground_truth=gt_ans,
                data_type=(example.get("data_type") or "").strip().lower(),
                problem_type=example.get("problem_type",""),
                options=example.get("options"),
            )

            answer_extracted = bool(pred_ans)
            invalid_answer = not valid_answer_for_problem(pred_ans, example)
            output_tokens = completion_info.get("output_tokens")
            truncated = bool(completion_info.get("truncated", False))
            has_think = has_think_tag(out_text)
            strict_format = has_strict_think_answer(out_text)
            category = category_for_example(example)

            sample_out = dict(example)
            sample_out["output"] = out_text
            sample_out["prediction"] = pred_ans
            sample_out["answer_extracted"] = answer_extracted
            sample_out["invalid_answer"] = invalid_answer
            sample_out["output_tokens"] = output_tokens
            sample_out["finish_reason"] = completion_info.get("finish_reason")
            sample_out["stop_reason"] = completion_info.get("stop_reason")
            sample_out["truncated"] = truncated
            sample_out["has_think"] = has_think
            sample_out["strict_format"] = strict_format
            sample_out["category"] = category
            # —— New: for Segmentation, only extract <answer> content into predicted_answer_norm (no metric calculation)
            if (example.get("problem_type","").strip().lower() == "segmentation"):
                sample_out["predicted_answer_norm"] = pred_ans
            # —— Others remain unchanged
            sample_out["accuracy"] = float(acc)
            if components:
                sample_out["components"] = {k: float(v) for k, v in components.items()}

            final_output.append(sample_out)
            accumulate_metrics(
                agg,
                example.get("problem_type",""),
                float(acc),
                components,
                answer_extracted=answer_extracted,
                invalid_answer=invalid_answer,
                output_tokens=output_tokens,
                truncated=truncated,
                category=category,
                has_think=has_think,
                strict_format=strict_format,
            )

        # Write to disk per batch (including accumulated metrics)
        metrics_dict = finalize_metrics(agg, include_bootstrap=False)
        try:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "results": final_output,
                        "metrics": metrics_dict,
                        "meta": output_meta()
                    },
                    f, indent=2, ensure_ascii=False
                )
        except Exception as e:
            print(f"[Warn] Failed to write output json at batch end (i={i}): {e}")

    # Final write to disk
    metrics_dict = finalize_metrics(agg, include_bootstrap=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": final_output,
                "metrics": metrics_dict,
                "meta": output_meta()
            },
            f, indent=2, ensure_ascii=False
        )
    print(f"[Done] Saved {len(final_output)} samples to {output_json_path}")
    print(f"[Time] elapsed_seconds: {time.perf_counter() - eval_start_time:.2f}")
    print("[FrameCache]")
    for k, v in frame_cache_meta().items():
        print(f"  {k}: {v}")
    print("[Metrics]")
    for k, v in metrics_dict.items():
        try:
            print(f"  {k}: {v:.4f}")
        except Exception:
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
