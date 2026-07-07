#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def metric(metrics: Dict[str, Any], *names: str) -> Optional[float]:
    for name in names:
        value = metrics.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def benchmark_name(path: Path) -> str:
    stem = path.stem
    if stem.startswith("eval_"):
        stem = stem[len("eval_"):]
    return stem


def summarize_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    results = payload.get("results", [])
    if not isinstance(results, list):
        results = []
    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    frame_cache = meta.get("frame_cache", {})
    if not isinstance(frame_cache, dict):
        frame_cache = {}
    elapsed_seconds = meta.get("elapsed_seconds")
    if not isinstance(elapsed_seconds, (int, float)):
        elapsed_seconds = None

    row = {
        "path": str(path),
        "samples": len(results),
        "elapsed_seconds": float(elapsed_seconds) if elapsed_seconds is not None else None,
        "elapsed_min": float(elapsed_seconds) / 60.0 if elapsed_seconds is not None else None,
        "frame_cache_hit": frame_cache.get("hit"),
        "frame_cache_miss": frame_cache.get("miss"),
        "frame_cache_write": frame_cache.get("write"),
        "frame_cache_fallback_to_pyav": frame_cache.get("fallback_to_pyav"),
        "answer_acc": metric(metrics, "answer_acc", "answer/acc", "overall/acc"),
        "answer_extract_rate": metric(metrics, "answer_extract_rate", "answer/extract_rate"),
        "invalid_answer_rate": metric(metrics, "invalid_answer_rate", "answer/invalid_rate"),
        "avg_output_tokens": metric(metrics, "avg_output_tokens", "output/avg_tokens"),
        "truncation_rate": metric(metrics, "truncation_rate", "output/truncation_rate"),
        "per_category_macro_acc": metric(metrics, "per_category/macro_acc"),
        "bootstrap_ci": metrics.get("bootstrap_ci"),
    }
    return row


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def summarize_dir(result_dir: Path, pattern: str) -> Dict[str, Any]:
    files = sorted(p for p in result_dir.glob(pattern) if p.is_file())
    benchmarks: Dict[str, Dict[str, Any]] = {}

    for path in files:
        try:
            benchmarks[benchmark_name(path)] = summarize_file(path)
        except Exception as exc:
            benchmarks[benchmark_name(path)] = {
                "path": str(path),
                "error": str(exc),
            }

    accs = [
        row["answer_acc"]
        for row in benchmarks.values()
        if isinstance(row.get("answer_acc"), (int, float))
    ]
    macro_acc = sum(accs) / len(accs) if accs else None

    return {
        "result_dir": str(result_dir),
        "metrics": {
            "macro_avg/by_benchmark": macro_acc,
            "num_benchmarks": len(accs),
        },
        "benchmarks": benchmarks,
    }


def to_markdown(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# Eval Summary: {summary['result_dir']}")
    lines.append("")
    lines.append(f"- macro_avg/by_benchmark: {fmt(summary['metrics']['macro_avg/by_benchmark'])}")
    lines.append(f"- num_benchmarks: {summary['metrics']['num_benchmarks']}")
    lines.append("")
    lines.append(
        "| benchmark | samples | elapsed_min | cache_hit | cache_miss | cache_write | fallback_pyav | answer_acc | extract_rate | invalid_rate | trunc_rate | avg_tokens | category_macro | bootstrap_ci |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    for name, row in sorted(summary["benchmarks"].items()):
        if "error" in row:
            lines.append(f"| {name} | - | - | - | - | - | - | - | - | - | - | - | - | ERROR: {row['error']} |")
            continue
        ci = row.get("bootstrap_ci")
        if isinstance(ci, dict):
            ci_text = f"[{fmt(ci.get('low'))}, {fmt(ci.get('high'))}]"
        else:
            ci_text = "-"
        lines.append(
            "| {name} | {samples} | {elapsed} | {cache_hit} | {cache_miss} | {cache_write} | {fallback} | {answer_acc} | {extract} | {invalid} | {trunc} | {tokens} | {cat} | {ci} |".format(
                name=name,
                samples=row.get("samples", "-"),
                elapsed=fmt(row.get("elapsed_min")),
                cache_hit=fmt(row.get("frame_cache_hit")),
                cache_miss=fmt(row.get("frame_cache_miss")),
                cache_write=fmt(row.get("frame_cache_write")),
                fallback=fmt(row.get("frame_cache_fallback_to_pyav")),
                answer_acc=fmt(row.get("answer_acc")),
                extract=fmt(row.get("answer_extract_rate")),
                invalid=fmt(row.get("invalid_answer_rate")),
                trunc=fmt(row.get("truncation_rate")),
                tokens=fmt(row.get("avg_output_tokens")),
                cat=fmt(row.get("per_category_macro_acc")),
                ci=ci_text,
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize eval_*.json files under model result dirs.")
    parser.add_argument("result_dirs", nargs="+", help="Result dirs such as Evaluation/results/MODEL/frames16")
    parser.add_argument("--pattern", default="eval_*.json", help="Glob pattern inside each result dir.")
    parser.add_argument("--no_write", action="store_true", help="Only print markdown; do not write summary files.")
    args = parser.parse_args()

    for raw_dir in args.result_dirs:
        result_dir = Path(raw_dir)
        summary = summarize_dir(result_dir, args.pattern)
        markdown = to_markdown(summary)
        print(markdown)

        if args.no_write:
            continue
        result_dir.mkdir(parents=True, exist_ok=True)
        with (result_dir / "_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        with (result_dir / "_summary.md").open("w", encoding="utf-8") as f:
            f.write(markdown)


if __name__ == "__main__":
    main()
