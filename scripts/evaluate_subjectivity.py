#!/usr/bin/env python3
"""Evaluate the subjectivity-modeling experiments.

The script reports the five metric families used in the paper: mean MAE,
variance MAE, Wasserstein distance, sample-level Spearman correlation between
human and model dimension variances, and Precision@k.

Example:
  python scripts/evaluate_subjectivity.py \
    --model_folder output/subjectivity/Qwen-3.5-9B \
    --gold_folder data/third_person
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr, wasserstein_distance

try:
    import tomllib
except ImportError:
    import tomli as tomllib


TASK_ORDER = (
    "appraisals",
    "appraisals-persona",
    "appraisals-third-person-scenario",
    "appraisals-persona-third-person-scenario",
)
LABEL_SYNONYMS = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Somewhat disagree": 2,
    "Neither agree nor disagree": 3,
    "Somewhat agree": 4,
    "Agree": 4,
    "Strongly agree": 5,
}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        payload = tomllib.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a TOML object in {path}")
    return payload


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def standard_deviation(values: list[float], ddof: int = 0) -> float | None:
    if len(values) <= ddof:
        return None
    return float(np.std(values, ddof=ddof))


def format_metric(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def population_variance(values: list[float]) -> float:
    return float(np.var(values, ddof=0))


def score_from_label(value: object, label_map: dict[str, int]) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    label = value.strip()
    score = label_map.get(label, LABEL_SYNONYMS.get(label))
    return float(score) if score is not None else None


def appraisal_config(
    prompt_path: Path,
) -> tuple[dict[str, str], dict[str, int]]:
    config = load_toml(prompt_path)
    statements = config.get("appraisals", {}).get("dimension_to_statement", {})
    labels = config.get("label_maps", {}).get("appraisals", {})
    if not isinstance(statements, dict) or not statements:
        raise ValueError(f"Missing appraisal statements in {prompt_path}")
    if not isinstance(labels, dict) or not labels:
        raise ValueError(f"Missing appraisal label map in {prompt_path}")
    return (
        {str(dimension): str(statement) for dimension, statement in statements.items()},
        {str(label): int(score) for label, score in labels.items()},
    )


def records(payload: Any, source: str) -> list[dict[str, Any]]:
    values = payload if isinstance(payload, list) else [payload]
    if not values or not all(isinstance(value, dict) for value in values):
        raise ValueError(f"{source} must be a JSON object or a list of objects")
    return values


def predicted_scores(
    payload: Any,
    dimensions: list[str],
) -> dict[str, list[float]]:
    result = {dimension: [] for dimension in dimensions}
    for record in records(payload, "Prediction file"):
        for dimension in dimensions:
            value = record.get(dimension)
            if not isinstance(value, dict):
                continue
            score = value.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                result[dimension].append(float(score))
    return result


def human_scores(
    payload: Any,
    statements: dict[str, str],
    label_map: dict[str, int],
) -> dict[str, list[float]]:
    result = {dimension: [] for dimension in statements}
    for record in records(payload, "Gold file"):
        ratings = record.get("appraisal_ratings", {})
        if not isinstance(ratings, dict):
            continue
        for dimension, statement in statements.items():
            score = score_from_label(ratings.get(statement), label_map)
            if score is not None:
                result[dimension].append(score)
    return result


def top_k_dimensions(scores: dict[str, list[float]], k: int) -> set[str]:
    ranked = sorted(
        (
            (dimension, population_variance(values))
            for dimension, values in scores.items()
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return {dimension for dimension, _ in ranked[:k]}


def spearman_variance_correlation(
    gold: dict[str, list[float]],
    prediction: dict[str, list[float]],
    dimensions: list[str],
) -> tuple[float, float] | None:
    valid = [
        dimension
        for dimension in dimensions
        if gold.get(dimension) and prediction.get(dimension)
    ]
    if len(valid) < 2:
        return None
    gold_variances = [population_variance(gold[dimension]) for dimension in valid]
    prediction_variances = [
        population_variance(prediction[dimension]) for dimension in valid
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = spearmanr(gold_variances, prediction_variances)
    return float(result.statistic), float(result.pvalue)


def empty_dimension_metrics() -> dict[str, list[float]]:
    return {
        "mean_mae": [],
        "variance_mae": [],
        "wasserstein": [],
        "prediction_mean": [],
        "prediction_variance": [],
        "gold_mean": [],
        "gold_variance": [],
    }


def summarize_dimension(metrics: dict[str, list[float]]) -> dict[str, Any]:
    return {
        name: mean(values)
        for name, values in metrics.items()
    } | {"count": len(metrics["mean_mae"])}


def summarize_top_k(
    values: dict[int, dict[str, list[float]]],
) -> dict[str, dict[str, float | int | None]]:
    summary = {}
    for k, metrics in values.items():
        summary[str(k)] = {
            "mean_precision": mean(metrics["precision"]),
            "std_precision": standard_deviation(metrics["precision"], ddof=1),
            "mean_recall": mean(metrics["recall"]),
            "std_recall": standard_deviation(metrics["recall"], ddof=1),
            "mean_f1": mean(metrics["f1"]),
            "std_f1": standard_deviation(metrics["f1"], ddof=1),
            "mean_jaccard": mean(metrics["jaccard"]),
            "std_jaccard": standard_deviation(metrics["jaccard"], ddof=1),
            "n": len(metrics["precision"]),
        }
    return summary


def evaluate_task(
    task_folder: Path,
    gold_index: dict[str, Path],
    statements: dict[str, str],
    label_map: dict[str, int],
    k_values: list[int],
    include_per_sample: bool,
) -> dict[str, Any]:
    dimensions = list(statements)
    dimension_metrics = {
        dimension: empty_dimension_metrics() for dimension in dimensions
    }
    top_k_values = {
        k: {name: [] for name in ("precision", "recall", "f1", "jaccard")}
        for k in k_values
    }
    rhos: list[float] = []
    pvalues: list[float] = []
    per_sample: list[dict[str, Any]] = []
    missing_gold = 0
    invalid_predictions = 0
    nan_correlations = 0

    prediction_files = sorted(task_folder.glob("*.json"))
    for prediction_path in prediction_files:
        sample_id = prediction_path.stem
        gold_path = gold_index.get(sample_id)
        if gold_path is None:
            missing_gold += 1
            continue

        prediction = predicted_scores(load_json(prediction_path), dimensions)
        gold = human_scores(load_json(gold_path), statements, label_map)
        valid = [
            dimension
            for dimension in dimensions
            if prediction[dimension] and gold[dimension]
        ]
        if not valid:
            invalid_predictions += 1
            continue

        sample_result: dict[str, Any] = {"sample_id": sample_id}
        for dimension in valid:
            prediction_values = prediction[dimension]
            gold_values = gold[dimension]
            metrics = dimension_metrics[dimension]
            metrics["mean_mae"].append(
                abs(float(np.mean(prediction_values)) - float(np.mean(gold_values)))
            )
            metrics["variance_mae"].append(
                abs(
                    population_variance(prediction_values)
                    - population_variance(gold_values)
                )
            )
            metrics["wasserstein"].append(
                float(wasserstein_distance(prediction_values, gold_values))
            )
            metrics["prediction_mean"].append(float(np.mean(prediction_values)))
            metrics["prediction_variance"].append(
                population_variance(prediction_values)
            )
            metrics["gold_mean"].append(float(np.mean(gold_values)))
            metrics["gold_variance"].append(population_variance(gold_values))

        correlation = spearman_variance_correlation(gold, prediction, dimensions)
        if correlation is not None:
            rho, pvalue = correlation
            if math.isfinite(rho):
                rhos.append(rho)
                pvalues.append(pvalue)
                sample_result["spearman_rho"] = rho
                sample_result["spearman_pvalue"] = pvalue
            else:
                nan_correlations += 1

        gold_valid = {dimension: gold[dimension] for dimension in valid}
        prediction_valid = {
            dimension: prediction[dimension] for dimension in valid
        }
        sample_top_k = {}
        for k in k_values:
            effective_k = min(k, len(valid))
            gold_top = top_k_dimensions(gold_valid, effective_k)
            prediction_top = top_k_dimensions(prediction_valid, effective_k)
            overlap = len(gold_top & prediction_top)
            union = len(gold_top | prediction_top)
            precision = overlap / len(prediction_top) if prediction_top else 0.0
            recall = overlap / len(gold_top) if gold_top else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            jaccard = overlap / union if union else 0.0
            top_k_values[k]["precision"].append(precision)
            top_k_values[k]["recall"].append(recall)
            top_k_values[k]["f1"].append(f1)
            top_k_values[k]["jaccard"].append(jaccard)
            if include_per_sample:
                sample_top_k[str(k)] = {
                    "precision": precision,
                    "gold_dimensions": sorted(gold_top),
                    "prediction_dimensions": sorted(prediction_top),
                }

        if include_per_sample:
            sample_result["precision_at_k"] = sample_top_k
            per_sample.append(sample_result)

    all_metrics = {
        name: [
            value
            for dimension in dimension_metrics.values()
            for value in dimension[name]
        ]
        for name in empty_dimension_metrics()
    }
    result: dict[str, Any] = {
        "distribution": {
            "overall": summarize_dimension(all_metrics),
            "by_dimension": {
                dimension: summarize_dimension(metrics)
                for dimension, metrics in dimension_metrics.items()
            },
        },
        "disagreement": {
            "spearman": {
                "mean_rho": mean(rhos),
                "std_rho": standard_deviation(rhos),
                "significant_fraction": (
                    float(np.mean([pvalue < 0.05 for pvalue in pvalues]))
                    if pvalues
                    else None
                ),
                "n": len(rhos),
                "n_nan": nan_correlations,
            },
            "precision_at_k": summarize_top_k(top_k_values),
        },
        "stats": {
            "prediction_files": len(prediction_files),
            "gold_files": len(gold_index),
            "evaluated": len(prediction_files) - missing_gold - invalid_predictions,
            "missing_gold": missing_gold,
            "invalid_predictions": invalid_predictions,
        },
    }
    if include_per_sample:
        result["per_sample"] = per_sample
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_folder",
        type=Path,
        required=True,
        help="Model output folder containing the four subjectivity task directories",
    )
    parser.add_argument(
        "--gold_folder",
        type=Path,
        default=Path("data/third_person"),
        help="Folder containing five observer records per narrative",
    )
    parser.add_argument(
        "--prompt_path",
        type=Path,
        default=Path("scripts/prompts/subjectivity_prompt.toml"),
    )
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument(
        "--output_file",
        type=Path,
        default=None,
        help="Output path; defaults to <model_folder>/results.json",
    )
    parser.add_argument("--include_per_sample", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model_folder.is_dir():
        raise FileNotFoundError(f"Model folder not found: {args.model_folder}")
    if not args.gold_folder.is_dir():
        raise FileNotFoundError(f"Gold folder not found: {args.gold_folder}")
    if not args.prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {args.prompt_path}")
    if not args.k_values or any(k <= 0 for k in args.k_values):
        raise ValueError("--k_values must contain positive integers")

    statements, label_map = appraisal_config(args.prompt_path)
    gold_index = {
        path.stem: path
        for path in args.gold_folder.glob("*.json")
        if path.is_file()
    }
    task_folders = [
        (task, args.model_folder / task)
        for task in TASK_ORDER
        if (args.model_folder / task).is_dir()
    ]
    if not task_folders:
        raise ValueError(
            f"No subjectivity-modeling task directories found under {args.model_folder}"
        )

    results = {
        task: evaluate_task(
            folder,
            gold_index,
            statements,
            label_map,
            sorted(set(args.k_values)),
            args.include_per_sample,
        )
        for task, folder in task_folders
    }
    output = {
        "model_folder": str(args.model_folder),
        "gold_folder": str(args.gold_folder),
        "tasks": results,
    }
    output_path = args.output_file or args.model_folder / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2, allow_nan=False)

    print("Task,Mean MAE,Variance MAE,Wasserstein,Spearman rho")
    for task, task_result in results.items():
        distribution = task_result["distribution"]["overall"]
        correlation = task_result["disagreement"]["spearman"]
        print(
            f"{task},{format_metric(distribution['mean_mae'])},"
            f"{format_metric(distribution['variance_mae'])},"
            f"{format_metric(distribution['wasserstein'])},"
            f"{format_metric(correlation['mean_rho'])}"
        )
    print(f"[saved] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
