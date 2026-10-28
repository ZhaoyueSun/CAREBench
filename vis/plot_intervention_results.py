#!/usr/bin/env python3
"""Generate the appraisal-intervention figures used in the CAREBench paper.

Example:
  python vis/plot_intervention_results.py \
    --results-root output/first_person \
    --output-dir figures

Benjamini-Hochberg correction is applied separately within each model and task.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


MODELS = (
    ("Claude", "claude-sonnet-4.6", "claude"),
    ("GPT", "gpt-5.2", "gpt"),
    ("Qwen", "Qwen-3.5-9B", "qwen"),
    ("Llama", "Llama-3.1-8B-Instruct", "llama"),
    ("PsyLLM", "PsyLLM-8B", "psyllm"),
    ("EmoLlama", "Emollama-chat-13b", "emollama"),
)
APPRAISALS = (
    "relevance.general",
    "relevance.urgency",
    "relevance.goals",
    "relevance.bodily motives",
    "relevance.social motives",
    "relevance.identity motives",
    "certainty.construal",
    "certainty.outlook",
    "certainty.predictability",
    "certainty.novelty",
    "congruence.general",
    "congruence.outlook",
    "congruence.positive prediction error",
    "congruence.negative prediction error",
    "control.general",
    "control.select",
    "control.vicarious",
    "control.effortful",
    "accountability.self",
    "accountability.other",
    "accountability.intentionality",
    "accountability.fairness",
)
POSITIVE_LABELS = (
    "Hopeful, optimistic, encouraged",
    "Grateful, appreciative, thankful",
    "Glad, happy, joyful",
    "Love, closeness, trust",
    "Amused, entertained",
    "Sympathy, concern, compassion",
    "Inspiration, admiration, appreciation",
    "Calm, peaceful, relaxed",
    "Proud, confident, determined",
    "Excited, enthusiastic, elated",
)
NEGATIVE_LABELS = (
    "Angry, frustrated, annoyed",
    "Worried, nervous, fearful",
    "Sad, downhearted, unhappy",
    "Disgust, distaste, revulsion",
    "Despair, hopelessness, sorrow",
    "Ashamed, humiliated, embarrassed",
    "Lonely, isolated, disconnected from others",
    "Panicked, alarmed, freaked out",
    "Guilty, blameworthy, repentant",
    "Confused, disoriented, surprised",
)
GROUP_SPANS = (
    (0, 6, "relevance"),
    (6, 10, "certainty"),
    (10, 14, "congruence"),
    (14, 18, "control"),
    (18, 22, "accountability"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CAREBench appraisal-intervention figures."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("output/first_person"),
        help="Root containing counterfactual and counterfactual_emotion.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--figure-group",
        choices=("all", "appraisal", "emotion"),
        default="all",
    )
    return parser.parse_args()


def optional_float(value: object) -> float:
    if value is None or value == "":
        return math.nan
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def pearson_p_value(correlation: float, sample_count: float) -> float:
    if not math.isfinite(correlation) or not math.isfinite(sample_count):
        return math.nan
    count = int(sample_count)
    if count < 3 or abs(correlation) > 1:
        return math.nan
    if abs(correlation) == 1:
        return 0.0
    statistic = abs(correlation) * math.sqrt(
        (count - 2) / (1 - correlation * correlation)
    )
    return float(2 * student_t.sf(statistic, df=count - 2))


def append_record(
    records: list[dict[str, Any]],
    model: str,
    task: str,
    dimension: str,
    output: str,
    item: object,
) -> None:
    if not isinstance(item, dict):
        return
    correlation = optional_float(item.get("pearson"))
    stats = item.get("stats", {})
    sample_count = optional_float(
        stats.get("evaluated_samples") if isinstance(stats, dict) else None
    )
    records.append(
        {
            "model": model,
            "task": task,
            "appraisal_dimension": dimension,
            "output": output,
            "pearson_r": correlation,
            "n": sample_count,
            "p_raw": pearson_p_value(correlation, sample_count),
        }
    )


def load_records(results_root: Path) -> list[dict[str, Any]]:
    appraisal_root = results_root / "counterfactual"
    emotion_root = results_root / "counterfactual_emotion"
    records: list[dict[str, Any]] = []
    missing = []

    for display_name, model_id, _ in MODELS:
        appraisal_path = appraisal_root / model_id / "results.json"
        if appraisal_path.is_file():
            dimensions = load_json(appraisal_path).get("dimension", {})
            if isinstance(dimensions, dict):
                for dimension, item in dimensions.items():
                    append_record(
                        records,
                        display_name,
                        "appraisals",
                        str(dimension),
                        str(dimension),
                        item,
                    )
        else:
            missing.append(appraisal_path)

        emotion_path = emotion_root / model_id / "results.json"
        if emotion_path.is_file():
            dimensions = load_json(emotion_path).get("dimension", {})
            if isinstance(dimensions, dict):
                for dimension, item in dimensions.items():
                    if not isinstance(item, dict):
                        continue
                    append_record(
                        records,
                        display_name,
                        "positive-level",
                        str(dimension),
                        "positive_intensity",
                        item.get("positive-level"),
                    )
                    append_record(
                        records,
                        display_name,
                        "negative-level",
                        str(dimension),
                        "negative_intensity",
                        item.get("negative-level"),
                    )
                    for task in ("positive-labels", "negative-labels"):
                        task_item = item.get(task, {})
                        categories = (
                            task_item.get("categories", {})
                            if isinstance(task_item, dict)
                            else {}
                        )
                        if not isinstance(categories, dict):
                            continue
                        for label, category in categories.items():
                            append_record(
                                records,
                                display_name,
                                task,
                                str(dimension),
                                str(label),
                                category,
                            )
        else:
            missing.append(emotion_path)

    if not records:
        expected = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "No RQ3 results.json files were found. Expected paths include:\n"
            f"{expected}"
        )
    if missing:
        print("Missing RQ3 result files:")
        for path in missing:
            print(f"  - {path}")
    return records


def benjamini_hochberg(indexed_values: list[tuple[int, float]]) -> dict[int, float]:
    ordered = sorted(indexed_values, key=lambda item: item[1])
    adjusted: dict[int, float] = {}
    running_minimum = 1.0
    total = len(ordered)
    for reverse_index in range(total - 1, -1, -1):
        original_index, value = ordered[reverse_index]
        rank = reverse_index + 1
        running_minimum = min(running_minimum, value * total / rank)
        adjusted[original_index] = min(running_minimum, 1.0)
    return adjusted


def add_adjusted_p_values(records: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for index, record in enumerate(records):
        p_value = record["p_raw"]
        if math.isfinite(p_value):
            groups[(record["model"], record["task"])].append((index, p_value))
    adjusted = {}
    for values in groups.values():
        adjusted.update(benjamini_hochberg(values))
    for index, record in enumerate(records):
        record["p_fdr_bh"] = adjusted.get(index, math.nan)


def significance_marker(value: float) -> str:
    if not math.isfinite(value):
        return ""
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return ""


def build_lookup(records: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    return {
        (
            record["model"],
            record["task"],
            record["appraisal_dimension"],
            record["output"],
        ): record
        for record in records
    }


def absolute_limit(*arrays: np.ndarray) -> float:
    values = np.concatenate([array.ravel() for array in arrays])
    values = values[np.isfinite(values)]
    if not values.size:
        return 1.0
    maximum = float(np.max(np.abs(values)))
    return maximum or 1.0


def annotate_markers(
    ax: plt.Axes,
    markers: list[list[str]],
    fontsize: float,
) -> None:
    for row_index, row in enumerate(markers):
        for column_index, marker in enumerate(row):
            if marker:
                ax.text(
                    column_index,
                    row_index,
                    marker,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color="black",
                )


def configure_appraisal_groups(ax: plt.Axes, top: bool) -> None:
    for boundary in (5.5, 9.5, 13.5, 17.5):
        ax.axvline(boundary, color="black", linewidth=1.2, alpha=0.9)
    if top:
        secondary = ax.secondary_xaxis("top")
        secondary.set_xticks(
            [(start + end - 1) / 2 for start, end, _ in GROUP_SPANS]
        )
        secondary.set_xticklabels([label for _, _, label in GROUP_SPANS])
        secondary.tick_params(axis="x", length=0, pad=2, labelsize=16)


def plot_appraisal_results(
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    output_path: Path,
) -> None:
    correlations = np.full((len(MODELS), len(APPRAISALS)), np.nan)
    markers = [["" for _ in APPRAISALS] for _ in MODELS]
    for model_index, (display_name, _, _) in enumerate(MODELS):
        for dimension_index, dimension in enumerate(APPRAISALS):
            record = lookup.get((display_name, "appraisals", dimension, dimension))
            if record:
                correlations[model_index, dimension_index] = record["pearson_r"]
                markers[model_index][dimension_index] = significance_marker(
                    record["p_fdr_bh"]
                )

    limit = absolute_limit(correlations)
    fig, ax = plt.subplots(figsize=(14, 2.8), constrained_layout=True)
    image = ax.imshow(
        correlations,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
        interpolation="none",
    )
    annotate_markers(ax, markers, fontsize=7)
    ax.set_xticks(np.arange(len(APPRAISALS)))
    ax.set_xticklabels([dimension.split(".", 1)[1] for dimension in APPRAISALS])
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_yticklabels([model[0] for model in MODELS])
    ax.tick_params(axis="x", rotation=20, labelsize=12)
    ax.tick_params(axis="y", rotation=0, labelsize=16)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.set_xticks(np.arange(-0.5, len(APPRAISALS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    configure_appraisal_groups(ax, top=True)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85, pad=0.01)
    colorbar.ax.tick_params(labelsize=16)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def first_word(label: str) -> str:
    return label.split(",", 1)[0].strip()


def emotion_panel(
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    model: str,
    level_task: str,
    label_task: str,
    labels: tuple[str, ...],
) -> tuple[np.ndarray, list[list[str]], list[str]]:
    row_labels = ["Intensity", "", *(first_word(label) for label in labels)]
    values = np.full((len(row_labels), len(APPRAISALS)), np.nan)
    markers = [["" for _ in APPRAISALS] for _ in row_labels]
    intensity_output = (
        "positive_intensity" if level_task == "positive-level" else "negative_intensity"
    )
    for dimension_index, dimension in enumerate(APPRAISALS):
        intensity = lookup.get((model, level_task, dimension, intensity_output))
        if intensity:
            values[0, dimension_index] = intensity["pearson_r"]
            markers[0][dimension_index] = significance_marker(intensity["p_fdr_bh"])
        for label_index, label in enumerate(labels, start=2):
            record = lookup.get((model, label_task, dimension, label))
            if record:
                values[label_index, dimension_index] = record["pearson_r"]
                markers[label_index][dimension_index] = significance_marker(
                    record["p_fdr_bh"]
                )
    return values, markers, row_labels


def draw_emotion_panel(
    ax: plt.Axes,
    values: np.ndarray,
    markers: list[list[str]],
    row_labels: list[str],
    limit: float,
    panel_label: str,
    show_top_groups: bool,
    show_bottom_labels: bool,
) -> Any:
    image = ax.imshow(
        np.ma.masked_invalid(values),
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
        interpolation="none",
    )
    annotate_markers(ax, markers, fontsize=11)
    ax.axhline(0.5, color="black", linewidth=1.8)
    ax.axhline(1.5, color="black", linewidth=1.8)
    configure_appraisal_groups(ax, top=show_top_groups)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_ylabel(panel_label, fontsize=14)
    ax.tick_params(axis="y", rotation=0, labelsize=12)
    ax.set_xticks(np.arange(len(APPRAISALS)))
    if show_bottom_labels:
        ax.set_xticklabels(
            [dimension.split(".", 1)[1] for dimension in APPRAISALS]
        )
        ax.tick_params(axis="x", rotation=35, labelsize=11)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", bottom=False)
    ax.set_xticks(np.arange(-0.5, len(APPRAISALS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)
    return image


def plot_emotion_results(
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    model: str,
    output_path: Path,
) -> None:
    positive = emotion_panel(
        lookup,
        model,
        "positive-level",
        "positive-labels",
        POSITIVE_LABELS,
    )
    negative = emotion_panel(
        lookup,
        model,
        "negative-level",
        "negative-labels",
        NEGATIVE_LABELS,
    )
    limit = absolute_limit(positive[0], negative[0])
    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(18, 10.5),
        gridspec_kw={"hspace": 0.10},
    )
    image = draw_emotion_panel(
        top,
        *positive,
        limit,
        panel_label="Positive Emotion",
        show_top_groups=True,
        show_bottom_labels=False,
    )
    draw_emotion_panel(
        bottom,
        *negative,
        limit,
        panel_label="Negative Emotion",
        show_top_groups=False,
        show_bottom_labels=True,
    )
    fig.subplots_adjust(left=0.10, right=0.90, top=0.92, bottom=0.15)
    colorbar_axis = fig.add_axes((0.92, 0.18, 0.016, 0.66))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.ax.tick_params(labelsize=13)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    records = load_records(args.results_root)
    add_adjusted_p_values(records)
    lookup = build_lookup(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    if args.figure_group in {"all", "appraisal"}:
        plot_appraisal_results(
            lookup,
            args.output_dir / "couterfactual_heatmap.pdf",
        )
        generated.append("couterfactual_heatmap.pdf")

    if args.figure_group in {"all", "emotion"}:
        for display_name, _, short_name in MODELS:
            plot_emotion_results(
                lookup,
                display_name,
                args.output_dir / f"counterfactual_emotion_{short_name}.pdf",
            )
            generated.append(f"counterfactual_emotion_{short_name}.pdf")

    print(f"Loaded {len(records):,} intervention correlations")
    print(f"Generated {len(generated)} intervention figures in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
