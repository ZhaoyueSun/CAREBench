#!/usr/bin/env python3
"""
Usage:
  python scripts/evaluate_per_sample.py \
    --gold_folder data/first_person \
    --pred_folder output/first_person/baseline/gpt-5.2 \
    --prompt_path scripts/prompts/baseline_prompt.toml \
    --output_file results_samples.json

Notes:
- --gold_folder and --pred_folder are required.
- The script evaluates first_person outputs per sample and writes:
    <pred_folder>/<output_file> (default: results_samples.json)
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
    import tomli as tomllib  # for python version < 3.11


TASKS = [
    # "appraisals",
    "positive-level",
    "negative-level",
    "positive-labels",
    "negative-labels",
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Top-level JSON must be object: {path}")
    return payload


def load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        payload = tomllib.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Top-level TOML must be object: {path}")
    return payload


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


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


def normalized_rmse(diffs: List[float], value_range: float) -> Optional[float]:
    if not diffs:
        return None
    mse = sum(((delta / value_range) ** 2 for delta in diffs)) / len(diffs)
    return math.sqrt(mse)


def list_gold_ids(gold_folder: Path) -> Set[str]:
    return {p.stem for p in gold_folder.glob("*.json") if p.is_file()}


def build_pred_index(pred_folder: Path, task: str) -> Dict[str, Path]:
    task_folder = pred_folder / task
    if not task_folder.exists() or not task_folder.is_dir():
        return {}
    return {p.stem: p for p in task_folder.glob("*.json") if p.is_file()}


def compute_example_f1(pred_set: Set[str], gold_set: Set[str]) -> float:
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    if not pred_set and not gold_set:
        return 1.0

    p_i = safe_div(tp, tp + fp)
    r_i = safe_div(tp, tp + fn)
    return safe_div(2 * p_i * r_i, p_i + r_i) if (p_i + r_i) > 0 else 0.0


def evaluate_sample_appraisals_rmse(
    sample_id: str,
    gold_payload: Dict[str, Any],
    pred_by_id: Dict[str, Path],
    dim_to_statement: Dict[str, str],
    label_map: Dict[str, Any],
) -> Optional[float]:
    default_score = 3
    appraisal_ratings = gold_payload.get("appraisal_ratings", {})
    if not isinstance(appraisal_ratings, dict):
        return None

    pred_payload: Optional[Dict[str, Any]] = None
    pred_file = pred_by_id.get(sample_id)
    if pred_file is not None:
        pred_payload = load_json(pred_file)

    diffs: List[float] = []
    for dimension, statement in dim_to_statement.items():
        gold_label_raw = appraisal_ratings.get(statement)
        if not isinstance(gold_label_raw, str) or gold_label_raw not in label_map:
            continue

        gold_score = int(label_map[gold_label_raw])
        pred_score = default_score
        if pred_payload is not None:
            pred_dim = pred_payload.get(dimension)
            if isinstance(pred_dim, dict) and isinstance(pred_dim.get("score"), (int, float)):
                pred_score = int(pred_dim["score"])

        diffs.append(float(pred_score - gold_score))

    return normalized_rmse(diffs, value_range=4.0)


def evaluate_sample_level_rmse(
    sample_id: str,
    gold_payload: Dict[str, Any],
    pred_by_id: Dict[str, Path],
    task: str,
) -> Optional[float]:
    default_score = 0
    gold_key = "positive_level" if task == "positive-level" else "negative_level"

    emotion_labels = gold_payload.get("emotion_labels", {})
    if not isinstance(emotion_labels, dict):
        return None
    gold_level_raw = emotion_labels.get(gold_key)
    gold_score, _ = parse_level_value(gold_level_raw)
    if gold_score is None:
        return None

    pred_score = default_score
    pred_file = pred_by_id.get(sample_id)
    if pred_file is not None:
        pred_payload = load_json(pred_file)
        pred_score_raw = pred_payload.get("score")
        if isinstance(pred_score_raw, (int, float)):
            pred_score = int(pred_score_raw)

    return normalized_rmse([float(pred_score - gold_score)], value_range=6.0)


def evaluate_sample_labels_example_f1(
    sample_id: str,
    gold_payload: Dict[str, Any],
    pred_by_id: Dict[str, Path],
    task: str,
    allowed_labels: Set[str],
) -> Optional[float]:
    gold_key = "positive_emotion_labels" if task == "positive-labels" else "negative_emotion_labels"

    emotion_labels = gold_payload.get("emotion_labels", {})
    if not isinstance(emotion_labels, dict):
        return None
    gold_labels_raw = emotion_labels.get(gold_key, [])
    if not isinstance(gold_labels_raw, list):
        return None

    gold_set = {str(label) for label in gold_labels_raw if isinstance(label, str)}
    gold_set = {label for label in gold_set if label in allowed_labels}

    pred_set: Set[str] = set()
    pred_file = pred_by_id.get(sample_id)
    if pred_file is not None:
        pred_payload = load_json(pred_file)
        pred_labels_raw = pred_payload.get("labels", [])
        if isinstance(pred_labels_raw, list):
            pred_set = {str(label) for label in pred_labels_raw if isinstance(label, str)}
            pred_set = {label for label in pred_set if label in allowed_labels}

    return compute_example_f1(pred_set, gold_set)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate first_person prediction outputs per sample")
    parser.add_argument(
        "--gold_folder",
        type=str,
        required=True,
        help="Gold annotation folder, e.g., data/first_person",
    )
    parser.add_argument(
        "--pred_folder",
        type=str,
        required=True,
        help="Prediction folder, e.g., output/first_person/baseline/gpt-5.2",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="scripts/prompts/baseline_prompt.toml",
        help="Prompt TOML path for label maps and dimensions",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results_samples.json",
        help="Output filename under pred_folder, default: results_samples.json",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    gold_folder = Path(args.gold_folder)
    pred_folder = Path(args.pred_folder)
    prompt_path = Path(args.prompt_path)

    if not gold_folder.exists() or not gold_folder.is_dir():
        raise FileNotFoundError(f"gold_folder not found or not a folder: {gold_folder}")
    if not pred_folder.exists() or not pred_folder.is_dir():
        raise FileNotFoundError(f"pred_folder not found or not a folder: {pred_folder}")
    if not prompt_path.exists() or not prompt_path.is_file():
        raise FileNotFoundError(f"prompt_path not found: {prompt_path}")

    output_file = args.output_file.strip()
    if not output_file:
        raise ValueError("output_file cannot be empty")

    selected_tasks = TASKS

    prompt_cfg = load_toml(prompt_path)

    dim_to_statement: Dict[str, str] = {}
    appraisal_label_map: Dict[str, Any] = {}
    if "appraisals" in selected_tasks:
        dim_to_statement = prompt_cfg.get("appraisals", {}).get("dimension_to_statement", {})
        if not isinstance(dim_to_statement, dict) or not dim_to_statement:
            raise ValueError("Missing [appraisals.dimension_to_statement] in prompt TOML")

        appraisal_label_map = prompt_cfg.get("label_maps", {}).get("appraisals", {})
        if not isinstance(appraisal_label_map, dict) or not appraisal_label_map:
            raise ValueError("Missing [label_maps.appraisals] in prompt TOML")

    positive_allowed_labels: Set[str] = set()
    negative_allowed_labels: Set[str] = set()
    if "positive-labels" in selected_tasks:
        positive_label_options = prompt_cfg.get("label_options", {}).get("positive-labels", {})
        positive_values = positive_label_options.get("values") if isinstance(positive_label_options, dict) else None
        if not isinstance(positive_values, list) or not positive_values:
            raise ValueError("Missing [label_options.positive-labels.values] in prompt TOML")
        positive_allowed_labels = {str(label) for label in positive_values}

    if "negative-labels" in selected_tasks:
        negative_label_options = prompt_cfg.get("label_options", {}).get("negative-labels", {})
        negative_values = negative_label_options.get("values") if isinstance(negative_label_options, dict) else None
        if not isinstance(negative_values, list) or not negative_values:
            raise ValueError("Missing [label_options.negative-labels.values] in prompt TOML")
        negative_allowed_labels = {str(label) for label in negative_values}

    pred_by_task = {task: build_pred_index(pred_folder, task) for task in selected_tasks}

    gold_ids = sorted(list_gold_ids(gold_folder))
    results: Dict[str, Dict[str, Optional[float]]] = {}

    for sample_id in gold_ids:
        gold_payload = load_json(gold_folder / f"{sample_id}.json")
        sample_result: Dict[str, Optional[float]] = {}

        for task in selected_tasks:
            pred_by_id = pred_by_task[task]
            if task == "appraisals":
                sample_result["appraisals_rmse"] = evaluate_sample_appraisals_rmse(
                    sample_id,
                    gold_payload,
                    pred_by_id,
                    dim_to_statement,
                    appraisal_label_map,
                )
            elif task == "positive-level":
                sample_result["positive-level_rmse"] = evaluate_sample_level_rmse(
                    sample_id,
                    gold_payload,
                    pred_by_id,
                    task="positive-level",
                )
            elif task == "negative-level":
                sample_result["negative-level_rmse"] = evaluate_sample_level_rmse(
                    sample_id,
                    gold_payload,
                    pred_by_id,
                    task="negative-level",
                )
            elif task == "positive-labels":
                # example_F1 is computed from labels task and mapped to user-requested key name.
                sample_result["positive-level_example_F1"] = evaluate_sample_labels_example_f1(
                    sample_id,
                    gold_payload,
                    pred_by_id,
                    task="positive-labels",
                    allowed_labels=positive_allowed_labels,
                )
            elif task == "negative-labels":
                # example_F1 is computed from labels task and mapped to user-requested key name.
                sample_result["negative-level_example_F1"] = evaluate_sample_labels_example_f1(
                    sample_id,
                    gold_payload,
                    pred_by_id,
                    task="negative-labels",
                    allowed_labels=negative_allowed_labels,
                )
            else:
                raise ValueError(f"Unknown task: {task}")

        results[sample_id] = sample_result

    output_path = pred_folder / output_file
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[done] wrote per-sample results to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())