# -*- coding: utf-8 -*-
# Rewards for multimodal tasks with <think>...</think><answer>...</answer> outputs.
import math
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import torch
from rouge_score import rouge_scorer
from math_verify import parse as math_parse, verify as math_verify
from mathruler.grader import grade_answer

# ===================== Model-based reward configuration =====================
# Whether to use external Reward Model to compute accuracy for open-ended type
USE_MODEL_FOR_OPEN_ENDED: bool = False

# External RM model and service address (kept consistent with example)
RM_MODEL_PATH = "internlm/POLAR-7B"
RM_SERVER_ADDRESS = "xx.xx.xx.xx:xxxx"
# ==========================================================

# ===================== External RM evaluation dependencies =====================
from verl.workers.reward.model_reward import RewardModelClient
import numpy as np
# =========================================================


# -------------------------
# Patterns for format check
# -------------------------
THINK_ANSWER_PATTERN = re.compile(
    r"\A\s*<think>.*?</think>\s*<answer>.*?</answer>\s*\Z",
    re.DOTALL
)

ANSWER_CAPTURE_PATTERN = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.DOTALL
)

ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
ROUGE_L_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


# -------------------------
# Utilities
# -------------------------
def extract_answer(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = ANSWER_CAPTURE_PATTERN.search(text)
    return m.group(1).strip() if m else None


def normalize_number(num_str: str) -> Optional[float]:
    try:
        return float((num_str or "").replace(",", ""))
    except Exception:
        return None


def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05) -> float:
    pred_t = torch.tensor(pred, dtype=torch.float32)
    tgt_t  = torch.tensor(target, dtype=torch.float32)
    rel_error = torch.abs(pred_t - tgt_t) / (torch.abs(tgt_t) + 1e-8)
    thresholds = torch.arange(start, end + interval/2, interval, dtype=torch.float32)
    return (rel_error < (1 - thresholds)).float().mean().item()


def wer(reference: str, hypothesis: str) -> float:
    ref_words, hyp_words = (reference or "").split(), (hypothesis or "").split()
    m, n = len(ref_words), len(hyp_words)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): d[i][0] = i
    for j in range(n + 1): d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            d[i][j] = d[i - 1][j - 1] if ref_words[i - 1] == hyp_words[j - 1] else 1 + min(
                d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]
            )
    return d[m][n] / max(1, m)


def compute_rouge_score(reference: str, hypothesis: str) -> float:
    scores = ROUGE_SCORER.score(reference or "", hypothesis or "")
    return (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3.0


def _normalize_text(text: str, *, compact: bool = False, casefold: bool = False) -> str:
    """Apply conservative Unicode/whitespace normalization for text rewards."""
    value = unicodedata.normalize("NFKC", text or "")
    value = value.replace("\u200b", "").replace("\ufeff", "")
    value = re.sub(r"\s+", " ", value).strip()
    if compact:
        value = re.sub(r"\s+", "", value)
    if casefold:
        value = value.casefold()
    return value


def _text_tokens(text: str) -> List[str]:
    """Tokenize Latin text, LaTeX commands, numbers and CJK characters without extra models."""
    normalized = _normalize_text(text, casefold=True)
    return re.findall(
        r"\\[A-Za-z]+|[A-Za-z0-9]+(?:[._/:-][A-Za-z0-9]+)*|[\u3400-\u9fff]|[^\s]",
        normalized,
    )


def _token_f1(reference: str, hypothesis: str) -> float:
    ref_tokens, hyp_tokens = _text_tokens(reference), _text_tokens(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return float(ref_tokens == hyp_tokens)
    overlap = sum((Counter(ref_tokens) & Counter(hyp_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(hyp_tokens)
    recall = overlap / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _sequence_similarity(reference: str, hypothesis: str, *, compact: bool = False, casefold: bool = False) -> float:
    ref = _normalize_text(reference, compact=compact, casefold=casefold)
    hyp = _normalize_text(hypothesis, compact=compact, casefold=casefold)
    if not ref or not hyp:
        return float(ref == hyp)
    # Disabling autojunk is more faithful for short OCR strings, but can be
    # quadratic on long repetitive open-ended answers.
    autojunk = max(len(ref), len(hyp)) >= 200
    return SequenceMatcher(None, ref, hyp, autojunk=autojunk).ratio()


def ocr_accuracy_v3(reference: str, hypothesis: str) -> float:
    """Character-aware OCR score that also handles CJK and whitespace-heavy LaTeX."""
    ref = _normalize_text(reference)
    hyp = _normalize_text(hypothesis)
    if ref == hyp:
        return 1.0

    spaced = _sequence_similarity(ref, hyp)
    compact = _sequence_similarity(ref, hyp, compact=True)
    case_insensitive = _sequence_similarity(ref, hyp, compact=True, casefold=True)
    return max(0.0, min(1.0, 0.25 * spaced + 0.60 * compact + 0.15 * case_insensitive))


def open_ended_accuracy_v3(reference: str, hypothesis: str) -> float:
    """Hybrid lexical score with precision pressure to reduce reference-copy reward hacking.

    This remains a deterministic single-reference proxy, not a semantic judge.  Token F1
    penalizes unsupported verbosity, ROUGE-L rewards ordered content, and character
    similarity keeps short/CJK answers useful.
    """
    ref = _normalize_text(reference)
    hyp = _normalize_text(hypothesis)
    if ref.casefold() == hyp.casefold():
        return 1.0
    rouge_l = ROUGE_L_SCORER.score(ref, hyp)["rougeL"].fmeasure
    token_f1 = _token_f1(ref, hyp)
    char_similarity = _sequence_similarity(ref, hyp, casefold=True)
    return max(0.0, min(1.0, 0.50 * token_f1 + 0.35 * rouge_l + 0.15 * char_similarity))


def regression_accuracy_v3(prediction: float, target: float) -> float:
    """Smooth, bounded regression reward based on a stabilized symmetric relative error."""
    scale = max((abs(prediction) + abs(target)) / 2.0, 1.0)
    normalized_error = abs(prediction - target) / scale
    return float(math.exp(-2.0 * normalized_error))


# -------------------------
# Format reward
# -------------------------
def tag_format_reward(response: str) -> float:
    """
    Format requirement (format reward):
      Must strictly be: <think>...</think><answer>...</answer>
      Arbitrary newlines/whitespaces are allowed in the middle, but tag order and closures must be correct.
      Returns 1.0 if satisfied; otherwise 0.0.
    """
    return 1.0 if THINK_ANSWER_PATTERN.fullmatch(response or "") else 0.0


# -------------------------
# Math equivalence helper
# -------------------------
def _math_equivalent(gt: str, pred: str) -> bool:
    """
    Use math_verify to perform symbolic equivalence checking; if it fails (exceptions, etc.),
    fall back to grade_answer.
    """
    try:
        return bool(math_verify(math_parse(gt), math_parse(pred)))
    except Exception:
        return grade_answer(pred, gt)


# -------------------------
# Accuracy reward (normalized to [0,1])
# -------------------------
def accuracy_reward(response: str,
                    ground_truth: str,
                    data_type: str,
                    problem_type: str,
                    formula_version: str = "legacy") -> float:
    """
    Normalized accuracy ∈ [0,1]. Strict format requirement: if the format is invalid, always return 0.
    Wrapped with try/except: any exception → 0.0.
    """
    try:
        ans = extract_answer(response) or response.strip()
        ptype = (problem_type or "").lower()
        gt = ground_truth or ""

        # ------ Pure QA type ------
        if ptype == "multiple choice":
            return 1.0 if grade_answer(ans.strip(), gt.strip()) else 0.0

        if ptype == "numerical":
            gt_num, pr_num = normalize_number(gt), normalize_number(ans)
            return 1.0 if (gt_num is not None and pr_num is not None and round(gt_num, 2) == round(pr_num, 2)) else 0.0

        if ptype == "regression":
            gt_num, pr_num = normalize_number(gt), normalize_number(ans)
            if gt_num is None or pr_num is None:
                return 0.0
            if formula_version == "v3":
                return regression_accuracy_v3(pr_num, gt_num)
            return mean_relative_accuracy(pr_num, gt_num)

        if ptype == "ocr":
            if formula_version == "v3":
                return ocr_accuracy_v3(gt, ans)
            return max(0.0, min(1.0, 1.0 - wer(gt, ans)))

        if ptype == "open-ended":
            if formula_version == "v3":
                return open_ended_accuracy_v3(gt, ans)
            return max(0.0, min(1.0, compute_rouge_score(gt, ans)))

        if ptype == "math":
            return 1.0 if _math_equivalent(gt, ans) else 0.0

        # Unknown type
        return 0.0
    except Exception:
        # Outer fallback: any exception will be scored as 0
        return 0.0


# ===================== Wrapper: batch call external model for open-ended =====================
def evaluate_open_ended_with_rm(
    open_ended_queue: List[Dict[str, Any]],
    results: List[Dict[str, float]],
    format_weight: float,
    rm_server_type: str,
    rm_batch_size: int,
    normalize_model_reward_by_problem_id: bool
) -> None:
    """
    Take open-ended samples in open_ended_queue, and call external RM in batches to evaluate accuracy.
    Failed batches fall back to ROUGE. Optionally apply mean-std → min-max normalization within
    each problem_id group.
    After evaluation, this function will fill results[idx]['accuracy'] in-place and recompute
    results[idx]['overall'].
    """
    if not USE_MODEL_FOR_OPEN_ENDED or not open_ended_queue:
        return

    client = RewardModelClient(
        RM_MODEL_PATH,
        server_type=rm_server_type,
        server_address=RM_SERVER_ADDRESS
    )

    def _chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    model_scores: List[float] = [0.0] * len(open_ended_queue)

    for batch_id, batch in enumerate(_chunks(open_ended_queue, rm_batch_size)):
        data = [{"prompt": b["prompt"], "reference": b["reference"], "output": b["output"]} for b in batch]
        try:
            rewards = client(data)  # expected to return list[float]
            for j, sc in enumerate(rewards):
                model_scores[(batch_id * rm_batch_size) + j] = float(sc)
        except Exception:
            # Fallback: use ROUGE to compute scores for this batch
            for j, b in enumerate(batch):
                ref = b["reference"]
                out = b["output"]
                model_scores[(batch_id * rm_batch_size) + j] = float(max(0.0, min(1.0, compute_rouge_score(ref, out))))

    if normalize_model_reward_by_problem_id:
        groups: Dict[Any, List[int]] = {}
        for k, b in enumerate(open_ended_queue):
            gid = b.get("problem_id", None)
            groups.setdefault(gid, []).append(k)

        for gid, indices in groups.items():
            vals = np.array([model_scores[k] for k in indices], dtype=np.float32)
            mean, std = vals.mean(), vals.std()
            if std == 0:
                norm_vals = np.ones_like(vals)
            else:
                z = (vals - mean) / (std + 1e-6)
                norm_vals = (z - z.min()) / (z.max() - z.min() + 1e-12)
            for t, k in enumerate(indices):
                model_scores[k] = float(norm_vals[t])

    # Fill back accuracy, and recompute overall
    for k, b in enumerate(open_ended_queue):
        idx = b["idx"]
        results[idx]["accuracy"] = float(max(0.0, min(1.0, model_scores[k])))
        results[idx]["overall"] = (
            (1.0 - format_weight) * results[idx]["accuracy"]
            + format_weight * results[idx]["format"]
        )
# ==================================================================


# -------------------------
# Public API
# -------------------------
def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.2,
    formula_version: str = "legacy",
    # ===== Still kept as configurable parameters =====
    rm_server_type: str = "vllm",
    rm_batch_size: int = 64,
    normalize_model_reward_by_problem_id: bool = True,
) -> List[Dict[str, float]]:
    """
    Batch interface.
    Each item:
        {
            "response": str,
            "response_length": int,
            "ground_truth": str,   # may also contain <answer>...</answer>, here we extract it first
            "data_type": str,      # "image" | "video" | ...
            "problem_type": str    # see branches above
            # Optional additional fields:
            # "problem": str        # used as prompt for external RM in open-ended tasks
            # "problem_id": Any     # grouping key for normalization
        }
    Returns: list of dict with keys {overall, format, accuracy}
    overall = (1 - format_weight) * accuracy + format_weight * format
      - format: 1.0 if <think>...</think><answer>...</answer>, otherwise 0.0
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for this reward function.")
    if formula_version not in {"legacy", "v3"}:
        raise ValueError(f"Unsupported formula_version: {formula_version!r}")

    results: List[Dict[str, float]] = []
    # ===================== Collect open-ended samples to be evaluated =====================
    open_ended_queue = []  # Each item: {idx, prompt, reference, output, problem_id}
    # ================================================================

    for idx, item in enumerate(reward_inputs):
        try:
            # Normalize tag whitespaces, e.g. < / think > → </think>
            raw_response = item.get("response", "") or ""
            response = re.sub(r"\s*(<|>|/)\s*", r"\1", raw_response)

            # print(response)

            data_type = item.get("data_type", "") or ""
            problem_type = item.get("problem_type", "") or ""

            # ground_truth may also be wrapped in <answer>...</answer>; extract it first here
            raw_gt = item.get("ground_truth", "") or ""
            gt_extracted = extract_answer(raw_gt) or raw_gt

            # 1) format reward —— requires strict tag structure: <think>...</think><answer>...</answer>
            f_score = tag_format_reward(response)

            # 2) answer accuracy (all normalized to [0,1])
            ans = extract_answer(response) or ""

            if USE_MODEL_FOR_OPEN_ENDED and (problem_type or "").lower() == "open-ended":
                # First set to 0, and finally compute with external model and fill back
                a_score = 0.0
                open_ended_queue.append({
                    "idx": idx,
                    "prompt": item.get("problem", "") or "",
                    "reference": gt_extracted or "",
                    "output": ans or "",
                    "problem_id": item.get("problem_id", None),
                })
            else:
                a_score = accuracy_reward(
                    response, gt_extracted, data_type, problem_type, formula_version=formula_version
                )

            overall = (1.0 - format_weight) * a_score + format_weight * f_score

            results.append({
                "overall": float(overall),
                "format": float(f_score),
                "accuracy": float(a_score),
            })
        except Exception:
            # Fallback for the entire sample: any exception, all fields are set to 0
            results.append({
                "overall": 0.0,
                "format": 0.0,
                "accuracy": 0.0,
            })



    # ===================== Call wrapper for batch external evaluation and fill back =====================
    evaluate_open_ended_with_rm(
        open_ended_queue=open_ended_queue,
        results=results,
        format_weight=format_weight,
        rm_server_type=rm_server_type,
        rm_batch_size=rm_batch_size,
        normalize_model_reward_by_problem_id=normalize_model_reward_by_problem_id
    )
    # ======================================================================

    return results
