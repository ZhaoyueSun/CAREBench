#!/usr/bin/env python3
"""
Usage:
  /home/ubuntu/anaconda3/bin/python scripts/evaluate_counterfactual_emotion.py \
    --model Qwen-3.5-9B \
    --first_person_root data/first_person \
    --counterfactual_gold_root data/counterfactual \
    --baseline_root output/first_person/baseline_with_cog_story \
    --counterfactual_pred_root output/first_person/counterfactual_emotion \
    --prompt_path scripts/prompts/baseline_prompt.toml \
    --output_file results.json

Notes:
- Correlations are computed per appraisal dimension.
- For level tasks:
    delta_model = mean(third_person_model_levels) - mean(first_person_model_levels)
    delta_human = mean(third_person_human_levels) - first_person_human_level
- For each emotion category label:
    delta_model = third_person_model_probability - first_person_model_probability
    delta_human = third_person_human_probability - first_person_human_probability
- For samples with multiple third-person entries, third-person level and category occurrence
  are averaged first, then used in sample-level correlation.
- Output is written to:
    output/first_person/counterfactual_emotion/<model>/<output_file>
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # for python < 3.11


LEVEL_TASKS = ["positive-level", "negative-level"]
LABEL_TASKS = ["positive-labels", "negative-labels"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        payload = tomllib.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Top-level TOML must be object: {path}")
    return payload


def safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / float(len(values))


def numeric_score(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def parse_level_value(text: Any) -> Tuple[Optional[int], Optional[str]]:
    if not isinstance(text, str):
        return None, None
    cleaned = text.strip()
    if not cleaned:
        return None, None

    match = re.match(r"^\s*(\d+)\s*-\s*(.+?)\s*$", cleaned)
    if match:
        return int(match.group(1)), match.group(2).strip()

    digits = re.findall(r"\d+", cleaned)
    score = int(digits[0]) if digits else None
    return score, cleaned


def rankdata(values: List[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (float(i + 1) + float(j + 1)) / 2.0
        for k in range(i, j + 1):
            original_idx = indexed[k][0]
            ranks[original_idx] = avg_rank
        i = j + 1
    return ranks


def pearson_corr(x: List[float], y: List[float]) -> Tuple[Optional[float], Optional[str]]:
    n = len(x)
    if n != len(y):
        return None, "Length mismatch"
    if n < 2:
        return None, "Need at least 2 samples"

    mean_x = safe_mean(x)
    mean_y = safe_mean(y)
    if mean_x is None or mean_y is None:
        return None, "Empty input"

    centered_x = [v - mean_x for v in x]
    centered_y = [v - mean_y for v in y]
    var_x = sum(v * v for v in centered_x)
    var_y = sum(v * v for v in centered_y)
    if var_x == 0.0 or var_y == 0.0:
        return None, "Zero variance"

    cov = sum(a * b for a, b in zip(centered_x, centered_y))
    denom = math.sqrt(var_x * var_y)
    if denom == 0.0:
        return None, "Zero denominator"
    return cov / denom, None


def spearman_corr(x: List[float], y: List[float]) -> Tuple[Optional[float], Optional[str]]:
    if len(x) != len(y):
        return None, "Length mismatch"
    if len(x) < 2:
        return None, "Need at least 2 samples"
    rx = rankdata(x)
    ry = rankdata(y)
    return pearson_corr(rx, ry)


def run_sort_key(path: Path) -> int:
    match = re.search(r"run_(\d+)$", path.name)
    return int(match.group(1)) if match else 10**9


def resolve_baseline_run_roots(model_root: Path) -> List[Path]:
    direct_task = model_root / "positive-level"
    if direct_task.exists() and direct_task.is_dir():
        return [model_root]

    run_roots = sorted(
        [p for p in model_root.glob("run_*") if p.is_dir()],
        key=run_sort_key,
    )
    if run_roots:
        return run_roots

    raise FileNotFoundError(
        f"Cannot find baseline prediction folders under: {model_root}. "
        "Expected either direct task folders or run_*/task folders."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate counterfactual emotion deltas")
    parser.add_argument("--model", type=str, required=True, help="Model directory name")
    parser.add_argument(
        "--first_person_root",
        type=str,
        default="data/first_person",
        help="First-person gold root",
    )
    parser.add_argument(
        "--counterfactual_gold_root",
        type=str,
        default="data/counterfactual",
        help="Counterfactual human gold root",
    )
    parser.add_argument(
        "--baseline_root",
        type=str,
        default="output/first_person/baseline_with_cog_story",
        help="Baseline prediction root",
    )
    parser.add_argument(
        "--counterfactual_pred_root",
        type=str,
        default="output/first_person/counterfactual_emotion",
        help="Counterfactual emotion prediction root",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="scripts/prompts/baseline_prompt.toml",
        help="Prompt TOML for emotion label options",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results.json",
        help="Output filename under counterfactual emotion model folder",
    )
    return parser


def _load_first_person_human_level(first_person_payload: Dict[str, Any], task: str) -> Optional[float]:
    key = "positive_level" if task == "positive-level" else "negative_level"
    emotion_labels = first_person_payload.get("emotion_labels", {})
    if not isinstance(emotion_labels, dict):
        return None
    value, _ = parse_level_value(emotion_labels.get(key))
    return float(value) if isinstance(value, int) else None


def _load_first_person_human_label_prob(
    first_person_payload: Dict[str, Any],
    task: str,
    label: str,
) -> Optional[float]:
    key = "positive_emotion_labels" if task == "positive-labels" else "negative_emotion_labels"
    emotion_labels = first_person_payload.get("emotion_labels", {})
    if not isinstance(emotion_labels, dict):
        return None
    labels_raw = emotion_labels.get(key)
    if not isinstance(labels_raw, list):
        return None
    labels_set = {v for v in labels_raw if isinstance(v, str)}
    return 1.0 if label in labels_set else 0.0


def _load_first_person_model_level_from_runs(run_roots: List[Path], task: str, sample_name: str) -> Optional[float]:
    values: List[float] = []
    for run_root in run_roots:
        path = run_root / task / sample_name
        if not path.exists() or not path.is_file():
            continue
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        score = numeric_score(payload.get("score"))
        if score is not None:
            values.append(score)
    return safe_mean(values)


def _load_first_person_model_label_prob_from_runs(
    run_roots: List[Path],
    task: str,
    sample_name: str,
    label: str,
) -> Optional[float]:
    probs: List[float] = []
    for run_root in run_roots:
        path = run_root / task / sample_name
        if not path.exists() or not path.is_file():
            continue
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        labels_raw = payload.get("labels")
        if not isinstance(labels_raw, list):
            continue
        labels_set = {v for v in labels_raw if isinstance(v, str)}
        probs.append(1.0 if label in labels_set else 0.0)
    return safe_mean(probs)


def _load_third_person_human_level(gold_counterfactual_payload: Any, task: str) -> Optional[float]:
    key = "positive_level" if task == "positive-level" else "negative_level"
    if not isinstance(gold_counterfactual_payload, list) or not gold_counterfactual_payload:
        return None

    values: List[float] = []
    for item in gold_counterfactual_payload:
        if not isinstance(item, dict):
            continue
        emotion_labels = item.get("emotion_labels", {})
        if not isinstance(emotion_labels, dict):
            continue
        score, _ = parse_level_value(emotion_labels.get(key))
        if isinstance(score, int):
            values.append(float(score))
    return safe_mean(values)


def _load_third_person_model_level(pred_counterfactual_payload: Any) -> Optional[float]:
    if not isinstance(pred_counterfactual_payload, list) or not pred_counterfactual_payload:
        return None

    values: List[float] = []
    for item in pred_counterfactual_payload:
        if not isinstance(item, dict):
            continue
        score = numeric_score(item.get("score"))
        if score is not None:
            values.append(score)
    return safe_mean(values)


def _load_third_person_human_label_prob(
    gold_counterfactual_payload: Any,
    task: str,
    label: str,
) -> Optional[float]:
    key = "positive_emotion_labels" if task == "positive-labels" else "negative_emotion_labels"
    if not isinstance(gold_counterfactual_payload, list) or not gold_counterfactual_payload:
        return None

    probs: List[float] = []
    for item in gold_counterfactual_payload:
        if not isinstance(item, dict):
            continue
        emotion_labels = item.get("emotion_labels", {})
        if not isinstance(emotion_labels, dict):
            continue
        labels_raw = emotion_labels.get(key)
        if not isinstance(labels_raw, list):
            continue
        labels_set = {v for v in labels_raw if isinstance(v, str)}
        probs.append(1.0 if label in labels_set else 0.0)
    return safe_mean(probs)


def _load_third_person_model_label_prob(pred_counterfactual_payload: Any, label: str) -> Optional[float]:
    if not isinstance(pred_counterfactual_payload, list) or not pred_counterfactual_payload:
        return None

    probs: List[float] = []
    for item in pred_counterfactual_payload:
        if not isinstance(item, dict):
            continue
        labels_raw = item.get("labels")
        if not isinstance(labels_raw, list):
            continue
        labels_set = {v for v in labels_raw if isinstance(v, str)}
        probs.append(1.0 if label in labels_set else 0.0)
    return safe_mean(probs)


def _build_base_stats(gold_files: int, pred_files: int) -> Dict[str, int]:
    return {
        "gold_files": gold_files,
        "pred_files": pred_files,
        "evaluated_samples": 0,
        "skipped_missing_first_person": 0,
        "skipped_missing_baseline": 0,
        "skipped_missing_counterfactual_pred_file": 0,
        "skipped_invalid_first_person_human": 0,
        "skipped_invalid_baseline": 0,
        "skipped_invalid_human_counterfactual": 0,
        "skipped_invalid_model_counterfactual": 0,
    }


def _evaluate_level_task_for_dimension(
    task: str,
    gold_dim_dir: Path,
    pred_dim_dir: Path,
    first_person_root: Path,
    baseline_run_roots: List[Path],
) -> Dict[str, Any]:
    gold_files = sorted([p for p in gold_dim_dir.glob("*.json") if p.is_file()])
    pred_files = len([p for p in pred_dim_dir.glob("*.json") if p.is_file()]) if pred_dim_dir.exists() else 0
    stats = _build_base_stats(gold_files=len(gold_files), pred_files=pred_files)

    model_deltas: List[float] = []
    human_deltas: List[float] = []

    for gold_file in gold_files:
        sample_name = gold_file.name
        first_person_file = first_person_root / sample_name
        pred_file = pred_dim_dir / sample_name

        if not first_person_file.exists() or not first_person_file.is_file():
            stats["skipped_missing_first_person"] += 1
            continue
        if not pred_file.exists() or not pred_file.is_file():
            stats["skipped_missing_counterfactual_pred_file"] += 1
            continue

        first_person_payload = load_json(first_person_file)
        gold_counterfactual_payload = load_json(gold_file)
        pred_counterfactual_payload = load_json(pred_file)

        if not isinstance(first_person_payload, dict):
            stats["skipped_invalid_first_person_human"] += 1
            continue

        first_person_human_level = _load_first_person_human_level(first_person_payload, task)
        if first_person_human_level is None:
            stats["skipped_invalid_first_person_human"] += 1
            continue

        first_person_model_level = _load_first_person_model_level_from_runs(baseline_run_roots, task, sample_name)
        if first_person_model_level is None:
            stats["skipped_missing_baseline"] += 1
            continue

        third_person_human_level = _load_third_person_human_level(gold_counterfactual_payload, task)
        if third_person_human_level is None:
            stats["skipped_invalid_human_counterfactual"] += 1
            continue

        third_person_model_level = _load_third_person_model_level(pred_counterfactual_payload)
        if third_person_model_level is None:
            stats["skipped_invalid_model_counterfactual"] += 1
            continue

        human_deltas.append(third_person_human_level - first_person_human_level)
        model_deltas.append(third_person_model_level - first_person_model_level)
        stats["evaluated_samples"] += 1

    pearson_value, pearson_reason = pearson_corr(model_deltas, human_deltas)
    spearman_value, spearman_reason = spearman_corr(model_deltas, human_deltas)

    warnings: List[str] = []
    if pearson_reason is not None:
        warnings.append(f"Pearson unavailable: {pearson_reason}")
    if spearman_reason is not None:
        warnings.append(f"Spearman unavailable: {spearman_reason}")

    return {
        "pearson": pearson_value,
        "spearman": spearman_value,
        "stats": stats,
        "warnings": warnings,
    }


def _evaluate_label_task_for_dimension(
    task: str,
    allowed_labels: List[str],
    gold_dim_dir: Path,
    pred_dim_dir: Path,
    first_person_root: Path,
    baseline_run_roots: List[Path],
) -> Dict[str, Any]:
    gold_files = sorted([p for p in gold_dim_dir.glob("*.json") if p.is_file()])
    pred_files = len([p for p in pred_dim_dir.glob("*.json") if p.is_file()]) if pred_dim_dir.exists() else 0
    stats = _build_base_stats(gold_files=len(gold_files), pred_files=pred_files)

    model_deltas_by_label: Dict[str, List[float]] = {label: [] for label in allowed_labels}
    human_deltas_by_label: Dict[str, List[float]] = {label: [] for label in allowed_labels}

    for gold_file in gold_files:
        sample_name = gold_file.name
        first_person_file = first_person_root / sample_name
        pred_file = pred_dim_dir / sample_name

        if not first_person_file.exists() or not first_person_file.is_file():
            stats["skipped_missing_first_person"] += 1
            continue
        if not pred_file.exists() or not pred_file.is_file():
            stats["skipped_missing_counterfactual_pred_file"] += 1
            continue

        first_person_payload = load_json(first_person_file)
        gold_counterfactual_payload = load_json(gold_file)
        pred_counterfactual_payload = load_json(pred_file)

        if not isinstance(first_person_payload, dict):
            stats["skipped_invalid_first_person_human"] += 1
            continue
        if not isinstance(gold_counterfactual_payload, list) or not gold_counterfactual_payload:
            stats["skipped_invalid_human_counterfactual"] += 1
            continue
        if not isinstance(pred_counterfactual_payload, list) or not pred_counterfactual_payload:
            stats["skipped_invalid_model_counterfactual"] += 1
            continue

        sample_model_deltas: Dict[str, float] = {}
        sample_human_deltas: Dict[str, float] = {}

        for label in allowed_labels:
            first_person_human_prob = _load_first_person_human_label_prob(first_person_payload, task, label)
            if first_person_human_prob is None:
                sample_model_deltas = {}
                sample_human_deltas = {}
                break

            first_person_model_prob = _load_first_person_model_label_prob_from_runs(
                baseline_run_roots,
                task,
                sample_name,
                label,
            )
            if first_person_model_prob is None:
                sample_model_deltas = {}
                sample_human_deltas = {}
                break

            third_person_human_prob = _load_third_person_human_label_prob(gold_counterfactual_payload, task, label)
            if third_person_human_prob is None:
                sample_model_deltas = {}
                sample_human_deltas = {}
                break

            third_person_model_prob = _load_third_person_model_label_prob(pred_counterfactual_payload, label)
            if third_person_model_prob is None:
                sample_model_deltas = {}
                sample_human_deltas = {}
                break

            sample_human_deltas[label] = third_person_human_prob - first_person_human_prob
            sample_model_deltas[label] = third_person_model_prob - first_person_model_prob

        if not sample_model_deltas or not sample_human_deltas:
            # Align skip reasons to dominant failure type by checking baseline/human/model components once.
            # This keeps behavior deterministic while preserving focused counters.
            probe_label = allowed_labels[0] if allowed_labels else None
            if probe_label is None:
                stats["skipped_invalid_first_person_human"] += 1
                continue
            if _load_first_person_human_label_prob(first_person_payload, task, probe_label) is None:
                stats["skipped_invalid_first_person_human"] += 1
                continue
            if _load_first_person_model_label_prob_from_runs(
                baseline_run_roots, task, sample_name, probe_label
            ) is None:
                stats["skipped_missing_baseline"] += 1
                continue
            if _load_third_person_human_label_prob(gold_counterfactual_payload, task, probe_label) is None:
                stats["skipped_invalid_human_counterfactual"] += 1
                continue
            stats["skipped_invalid_model_counterfactual"] += 1
            continue

        for label in allowed_labels:
            human_deltas_by_label[label].append(sample_human_deltas[label])
            model_deltas_by_label[label].append(sample_model_deltas[label])

        stats["evaluated_samples"] += 1

    categories: Dict[str, Any] = {}
    for label in allowed_labels:
        pearson_value, pearson_reason = pearson_corr(model_deltas_by_label[label], human_deltas_by_label[label])
        spearman_value, spearman_reason = spearman_corr(model_deltas_by_label[label], human_deltas_by_label[label])

        warnings: List[str] = []
        if pearson_reason is not None:
            warnings.append(f"Pearson unavailable: {pearson_reason}")
        if spearman_reason is not None:
            warnings.append(f"Spearman unavailable: {spearman_reason}")

        categories[label] = {
            "pearson": pearson_value,
            "spearman": spearman_value,
            "stats": dict(stats),
            "warnings": warnings,
        }

    return {
        "stats": stats,
        "categories": categories,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    first_person_root = Path(args.first_person_root)
    counterfactual_gold_root = Path(args.counterfactual_gold_root)
    baseline_model_root = Path(args.baseline_root) / args.model
    counterfactual_model_root = Path(args.counterfactual_pred_root) / args.model
    prompt_path = Path(args.prompt_path)

    if not first_person_root.exists() or not first_person_root.is_dir():
        raise FileNotFoundError(f"first_person_root not found: {first_person_root}")
    if not counterfactual_gold_root.exists() or not counterfactual_gold_root.is_dir():
        raise FileNotFoundError(f"counterfactual_gold_root not found: {counterfactual_gold_root}")
    if not baseline_model_root.exists() or not baseline_model_root.is_dir():
        raise FileNotFoundError(f"baseline model folder not found: {baseline_model_root}")
    if not counterfactual_model_root.exists() or not counterfactual_model_root.is_dir():
        raise FileNotFoundError(f"counterfactual model folder not found: {counterfactual_model_root}")
    if not prompt_path.exists() or not prompt_path.is_file():
        raise FileNotFoundError(f"prompt_path not found: {prompt_path}")

    baseline_run_roots = resolve_baseline_run_roots(baseline_model_root)
    prompt_cfg = load_toml(prompt_path)

    dim_to_statement = prompt_cfg.get("appraisals", {}).get("dimension_to_statement", {})
    if not isinstance(dim_to_statement, dict) or not dim_to_statement:
        raise ValueError("Missing [appraisals.dimension_to_statement] in prompt TOML")

    pos_label_options = prompt_cfg.get("label_options", {}).get("positive-labels", {}).get("values")
    neg_label_options = prompt_cfg.get("label_options", {}).get("negative-labels", {}).get("values")
    if not isinstance(pos_label_options, list) or not pos_label_options:
        raise ValueError("Missing [label_options.positive-labels.values] in prompt TOML")
    if not isinstance(neg_label_options, list) or not neg_label_options:
        raise ValueError("Missing [label_options.negative-labels.values] in prompt TOML")

    positive_labels = [str(v) for v in pos_label_options if isinstance(v, str)]
    negative_labels = [str(v) for v in neg_label_options if isinstance(v, str)]
    if not positive_labels or not negative_labels:
        raise ValueError("Emotion label options are empty after filtering string values")

    results: Dict[str, Any] = {
        "meta": {
            "model": args.model,
            "first_person_root": str(first_person_root),
            "counterfactual_gold_root": str(counterfactual_gold_root),
            "baseline_model_root": str(baseline_model_root),
            "counterfactual_model_root": str(counterfactual_model_root),
            "baseline_run_roots": [str(p) for p in baseline_run_roots],
            "prompt_path": str(prompt_path),
            "tasks": LEVEL_TASKS + LABEL_TASKS,
        },
        "dimension": {},
    }

    for dimension in dim_to_statement.keys():
        gold_dim_dir = counterfactual_gold_root / dimension
        if not gold_dim_dir.exists() or not gold_dim_dir.is_dir():
            results["dimension"][dimension] = {
                "positive-level": {
                    "pearson": None,
                    "spearman": None,
                    "stats": _build_base_stats(gold_files=0, pred_files=0),
                    "warnings": [f"Missing human counterfactual folder: {gold_dim_dir}"],
                },
                "negative-level": {
                    "pearson": None,
                    "spearman": None,
                    "stats": _build_base_stats(gold_files=0, pred_files=0),
                    "warnings": [f"Missing human counterfactual folder: {gold_dim_dir}"],
                },
                "positive-labels": {
                    "stats": _build_base_stats(gold_files=0, pred_files=0),
                    "categories": {
                        label: {
                            "pearson": None,
                            "spearman": None,
                            "stats": _build_base_stats(gold_files=0, pred_files=0),
                            "warnings": [f"Missing human counterfactual folder: {gold_dim_dir}"],
                        }
                        for label in positive_labels
                    },
                },
                "negative-labels": {
                    "stats": _build_base_stats(gold_files=0, pred_files=0),
                    "categories": {
                        label: {
                            "pearson": None,
                            "spearman": None,
                            "stats": _build_base_stats(gold_files=0, pred_files=0),
                            "warnings": [f"Missing human counterfactual folder: {gold_dim_dir}"],
                        }
                        for label in negative_labels
                    },
                },
            }
            continue

        dim_obj: Dict[str, Any] = {}

        for level_task in LEVEL_TASKS:
            pred_dim_dir = counterfactual_model_root / level_task / dimension
            dim_obj[level_task] = _evaluate_level_task_for_dimension(
                level_task,
                gold_dim_dir,
                pred_dim_dir,
                first_person_root,
                baseline_run_roots,
            )

        dim_obj["positive-labels"] = _evaluate_label_task_for_dimension(
            "positive-labels",
            positive_labels,
            gold_dim_dir,
            counterfactual_model_root / "positive-labels" / dimension,
            first_person_root,
            baseline_run_roots,
        )
        dim_obj["negative-labels"] = _evaluate_label_task_for_dimension(
            "negative-labels",
            negative_labels,
            gold_dim_dir,
            counterfactual_model_root / "negative-labels" / dimension,
            first_person_root,
            baseline_run_roots,
        )

        results["dimension"][dimension] = dim_obj

    output_file = args.output_file.strip()
    if not output_file:
        raise ValueError("output_file cannot be empty")

    output_path = counterfactual_model_root / output_file
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[done] wrote results to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
