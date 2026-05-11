#!/usr/bin/env python3
"""
Usage:
  python scripts/evaluate_counterfactual.py \
    --model claude-sonnet-4.6 \
    --first_person_root data/first_person \
    --counterfactual_gold_root data/counterfactual \
    --baseline_root output/first_person/baseline_with_cog_story \
    --counterfactual_pred_root output/first_person/counterfactual \
    --prompt_path scripts/prompts/counterfactual_prompt.toml \
    --output_file results.json

Notes:
- For each appraisal dimension, this script computes correlation between:
    model_delta = mean(third_person_model_scores) - first_person_model_score
    human_delta = mean(third_person_human_scores) - first_person_human_score
- Pearson and Spearman are both reported per dimension.
- Output is written to:
    output/first_person/counterfactual/<model>/results.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # for python < 3.11


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


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def numeric_score(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


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


def resolve_baseline_appraisals_dir(model_root: Path) -> Path:
    direct = model_root / "appraisals"
    if direct.exists() and direct.is_dir():
        return direct

    nested_candidates: List[Path] = []
    if model_root.exists() and model_root.is_dir():
        for child in sorted(model_root.iterdir()):
            if child.is_dir() and (child / "appraisals").is_dir():
                nested_candidates.append(child / "appraisals")

    if not nested_candidates:
        raise FileNotFoundError(f"Cannot find baseline appraisals dir under: {model_root}")
    return nested_candidates[0]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate counterfactual appraisal deltas")
    parser.add_argument("--model", type=str, required=True, help="Model directory name")
    parser.add_argument("--first_person_root", type=str, default="data/first_person", help="First-person gold root")
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
        default="output/first_person/counterfactual",
        help="Counterfactual prediction root",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="scripts/prompts/counterfactual_prompt.toml",
        help="Prompt TOML for dimension/label mapping",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results.json",
        help="Output filename under counterfactual model folder",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
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
    if not counterfactual_model_root.exists() or not counterfactual_model_root.is_dir():
        raise FileNotFoundError(f"counterfactual model folder not found: {counterfactual_model_root}")
    if not prompt_path.exists() or not prompt_path.is_file():
        raise FileNotFoundError(f"prompt_path not found: {prompt_path}")

    baseline_appraisals_dir = resolve_baseline_appraisals_dir(baseline_model_root)
    prompt_cfg = load_toml(prompt_path)

    dim_to_statement = prompt_cfg.get("appraisals", {}).get("dimension_to_statement", {})
    if not isinstance(dim_to_statement, dict) or not dim_to_statement:
        raise ValueError("Missing [appraisals.dimension_to_statement] in prompt TOML")

    label_map = prompt_cfg.get("label_maps", {}).get("appraisals", {})
    if not isinstance(label_map, dict) or not label_map:
        raise ValueError("Missing [label_maps.appraisals] in prompt TOML")

    results: Dict[str, Any] = {
        "meta": {
            "model": args.model,
            "first_person_root": str(first_person_root),
            "counterfactual_gold_root": str(counterfactual_gold_root),
            "baseline_appraisals_dir": str(baseline_appraisals_dir),
            "counterfactual_model_root": str(counterfactual_model_root),
            "prompt_path": str(prompt_path),
        },
        "dimension": {},
    }

    for dimension, statement in dim_to_statement.items():
        if not isinstance(statement, str) or not statement.strip():
            continue

        gold_dim_dir = counterfactual_gold_root / dimension
        pred_dim_dir = counterfactual_model_root / dimension

        dim_stats: Dict[str, Any] = {
            "gold_files": 0,
            "pred_files": 0,
            "evaluated_samples": 0,
            "prediction_count": 0,
            "human_counterfactual_count": 0,
            "skipped_missing_first_person": 0,
            "skipped_missing_baseline": 0,
            "skipped_missing_counterfactual_pred_file": 0,
            "skipped_invalid_first_person_human": 0,
            "skipped_invalid_baseline": 0,
            "skipped_invalid_human_counterfactual": 0,
            "skipped_invalid_model_counterfactual": 0,
        }

        model_deltas: List[float] = []
        human_deltas: List[float] = []

        if not gold_dim_dir.exists() or not gold_dim_dir.is_dir():
            results["dimension"][dimension] = {
                "pearson": None,
                "spearman": None,
                "stats": dim_stats,
                "warnings": [f"Missing human counterfactual folder: {gold_dim_dir}"],
            }
            continue

        gold_files = sorted([p for p in gold_dim_dir.glob("*.json") if p.is_file()])
        dim_stats["gold_files"] = len(gold_files)

        if pred_dim_dir.exists() and pred_dim_dir.is_dir():
            dim_stats["pred_files"] = len([p for p in pred_dim_dir.glob("*.json") if p.is_file()])

        for gold_file in gold_files:
            sample_name = gold_file.name
            first_person_file = first_person_root / sample_name
            baseline_file = baseline_appraisals_dir / sample_name
            pred_file = pred_dim_dir / sample_name

            if not first_person_file.exists() or not first_person_file.is_file():
                dim_stats["skipped_missing_first_person"] += 1
                continue
            if not baseline_file.exists() or not baseline_file.is_file():
                dim_stats["skipped_missing_baseline"] += 1
                continue
            if not pred_file.exists() or not pred_file.is_file():
                dim_stats["skipped_missing_counterfactual_pred_file"] += 1
                continue

            first_person_payload = load_json(first_person_file)
            baseline_payload = load_json(baseline_file)
            gold_counterfactual_payload = load_json(gold_file)
            pred_counterfactual_payload = load_json(pred_file)

            if not isinstance(first_person_payload, dict):
                dim_stats["skipped_invalid_first_person_human"] += 1
                continue
            if not isinstance(baseline_payload, dict):
                dim_stats["skipped_invalid_baseline"] += 1
                continue
            if not isinstance(gold_counterfactual_payload, list) or not gold_counterfactual_payload:
                dim_stats["skipped_invalid_human_counterfactual"] += 1
                continue
            if not isinstance(pred_counterfactual_payload, list) or not pred_counterfactual_payload:
                dim_stats["skipped_invalid_model_counterfactual"] += 1
                continue

            appraisal_ratings = first_person_payload.get("appraisal_ratings", {})
            if not isinstance(appraisal_ratings, dict):
                dim_stats["skipped_invalid_first_person_human"] += 1
                continue

            first_person_label = appraisal_ratings.get(statement)
            if not isinstance(first_person_label, str) or first_person_label not in label_map:
                dim_stats["skipped_invalid_first_person_human"] += 1
                continue
            first_person_human_score = float(label_map[first_person_label])

            baseline_dim = baseline_payload.get(dimension)
            if not isinstance(baseline_dim, dict):
                dim_stats["skipped_invalid_baseline"] += 1
                continue
            first_person_model_score = numeric_score(baseline_dim.get("score"))
            if first_person_model_score is None:
                dim_stats["skipped_invalid_baseline"] += 1
                continue

            human_third_scores: List[float] = []
            for item in gold_counterfactual_payload:
                if not isinstance(item, dict):
                    continue
                ratings = item.get("appraisal_ratings", {})
                if not isinstance(ratings, dict):
                    continue
                label = ratings.get(statement)
                if isinstance(label, str) and label in label_map:
                    human_third_scores.append(float(label_map[label]))
            if not human_third_scores:
                dim_stats["skipped_invalid_human_counterfactual"] += 1
                continue

            model_third_scores: List[float] = []
            for item in pred_counterfactual_payload:
                if not isinstance(item, dict):
                    continue
                score = numeric_score(item.get("score"))
                if score is not None:
                    model_third_scores.append(score)
            if not model_third_scores:
                dim_stats["skipped_invalid_model_counterfactual"] += 1
                continue

            dim_stats["human_counterfactual_count"] += len(human_third_scores)
            dim_stats["prediction_count"] += len(model_third_scores)

            human_third_mean = safe_mean(human_third_scores)
            model_third_mean = safe_mean(model_third_scores)
            if human_third_mean is None or model_third_mean is None:
                continue

            human_delta = human_third_mean - first_person_human_score
            model_delta = model_third_mean - first_person_model_score

            human_deltas.append(human_delta)
            model_deltas.append(model_delta)
            dim_stats["evaluated_samples"] += 1

        pearson_value, pearson_reason = pearson_corr(model_deltas, human_deltas)
        spearman_value, spearman_reason = spearman_corr(model_deltas, human_deltas)

        warnings: List[str] = []
        if pearson_reason is not None:
            warnings.append(f"Pearson unavailable: {pearson_reason}")
        if spearman_reason is not None:
            warnings.append(f"Spearman unavailable: {spearman_reason}")

        results["dimension"][dimension] = {
            "pearson": pearson_value,
            "spearman": spearman_value,
            "stats": dim_stats,
            "warnings": warnings,
        }

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