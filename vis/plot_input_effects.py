#!/usr/bin/env python3
"""Generate the RQ2 input-effect figures used in the CAREBench paper.

Example:
  python vis/plot_input_effects.py \
    --results-root output/first_person \
    --output-dir figures
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np


APPRAISAL_ORDER = (
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
CORE_ORDER = ("accountability", "control", "congruence", "certainty", "relevance")
HEATMAP_MODELS = (
    ("qwen", "Qwen-3.5-9B"),
    ("llama", "Llama-3.1-8B-Instruct"),
    ("psyllm", "PsyLLM-8B"),
    ("emollama", "Emollama-chat-13b"),
)
SETTING_COLORS = (
    ("baseline (scenario only)", "tab:blue"),
    ("+ 1st-person reasoning", "tab:orange"),
    ("+ 1st-person appraisals", "tab:green"),
    ("+ 1st-person reasoning & appraisals", "tab:red"),
    ("+ predicted reasoning", "tab:purple"),
)
PANEL_TITLES = (
    "Positive Micro-F1",
    "Positive Macro-F1",
    "Negative Micro-F1",
    "Negative Macro-F1",
)
MODEL_ORDER = (
    ("Claude", "claude-sonnet-4.6"),
    ("GPT", "gpt-5.2"),
    ("Qwen", "Qwen-3.5-9B"),
    ("Llama", "Llama-3.1-8B-Instruct"),
    ("PsyLLM", "PsyLLM-8B"),
    ("EmoLlama", "Emollama-chat-13b"),
)
OPEN_SOURCE_MODELS = {
    "Qwen-3.5-9B",
    "Llama-3.1-8B-Instruct",
    "PsyLLM-8B",
    "Emollama-chat-13b",
}
SETTING_DIRECTORIES = (
    "baseline",
    "baseline_with_cog_story",
    "baseline_with_appraisal",
    "baseline_with_cog_story_and_appraisals",
    "baseline_with_pred_cog_story",
)
PANEL_METRICS = (
    ("Positive Micro-F1", ("positive-labels", "micro-F1", "f1")),
    ("Positive Macro-F1", ("positive-labels", "macro-F1", "overall", "f1")),
    ("Negative Micro-F1", ("negative-labels", "micro-F1", "f1")),
    ("Negative Macro-F1", ("negative-labels", "macro-F1", "overall", "f1")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CAREBench RQ2 figures.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("output/first_person"),
        help="Root containing the five RQ2 setting directories.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--figure-group",
        choices=("all", "f1", "heatmaps"),
        default="all",
    )
    return parser.parse_args()


def first_word(value: str) -> str:
    token = value.strip().split()[0] if value.strip() else "-"
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", token) or "-"


def result_number(payload: dict[str, Any], path: tuple[str, ...]) -> float:
    value = nested_value(payload, path)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return np.nan


def baseline_result(results_root: Path, model_id: str) -> dict[str, Any]:
    model_root = results_root / "baseline" / model_id
    run_one = model_root / "run_1" / "results.json"
    single = model_root / "results.json"
    if run_one.is_file():
        return load_json(run_one)
    if single.is_file():
        return load_json(single)
    raise FileNotFoundError(f"Baseline results not found under {model_root}")


def single_core_result(
    results_root: Path,
    model_id: str,
    core_dimension: str,
) -> dict[str, Any]:
    path = (
        results_root
        / "baseline_with_pred_single_core_appraisal"
        / model_id
        / core_dimension
        / "results.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Single-core appraisal results not found: {path}")
    return load_json(path)


def label_keys(payload: dict[str, Any], label_task: str) -> list[str]:
    groups = nested_value(payload, (label_task, "macro-F1", "label_group"))
    if not isinstance(groups, dict):
        return []
    return [str(label) for label in groups]


def build_heatmap_matrices(
    results_root: Path,
    model_id: str,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, list[str]]:
    baseline = baseline_result(results_root, model_id)
    interventions = {
        core: single_core_result(results_root, model_id, core)
        for core in CORE_ORDER
    }

    appraisal = np.full((len(CORE_ORDER), len(APPRAISAL_ORDER)), np.nan)
    for row_index, core in enumerate(CORE_ORDER):
        for column_index, dimension in enumerate(APPRAISAL_ORDER):
            metric_path = ("appraisals", "dimension", dimension, "rmse")
            base_value = result_number(baseline, metric_path)
            intervention_value = result_number(interventions[core], metric_path)
            if np.isfinite(base_value) and np.isfinite(intervention_value):
                appraisal[row_index, column_index] = -(
                    intervention_value - base_value
                )

    label_outputs = []
    for label_task in ("positive-labels", "negative-labels"):
        labels = label_keys(baseline, label_task)
        values = np.full((len(CORE_ORDER), len(labels)), np.nan)
        for row_index, core in enumerate(CORE_ORDER):
            for column_index, label in enumerate(labels):
                metric_path = (
                    label_task,
                    "macro-F1",
                    "label_group",
                    label,
                    "f1",
                )
                base_value = result_number(baseline, metric_path)
                intervention_value = result_number(
                    interventions[core], metric_path
                )
                if np.isfinite(base_value) and np.isfinite(intervention_value):
                    values[row_index, column_index] = (
                        intervention_value - base_value
                    )
        label_outputs.append((values, [first_word(label) for label in labels]))

    positive, negative = label_outputs
    return appraisal, positive[0], positive[1], negative[0], negative[1]


def draw_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    column_labels: list[str],
    limit: float,
    show_y_labels: bool,
) -> Any:
    image = ax.imshow(
        values,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
        interpolation="none",
    )
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels)
    ax.set_yticks(np.arange(len(CORE_ORDER)))
    ax.set_yticklabels(CORE_ORDER if show_y_labels else [])
    ax.set_xticks(np.arange(-0.5, len(column_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CORE_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    return image


def plot_model_heatmaps(
    results_root: Path,
    model_id: str,
    output_path: Path,
) -> None:
    appraisal, positive, positive_labels, negative, negative_labels = (
        build_heatmap_matrices(results_root, model_id)
    )
    appraisal_labels = [f"A{index}" for index in range(1, 23)]

    combined = np.concatenate(
        [appraisal.ravel(), negative.ravel(), positive.ravel()]
    )
    finite = combined[np.isfinite(combined)]
    limit = float(np.max(np.abs(finite))) if finite.size else 1.0
    limit = limit or 1.0

    plt.rcParams.update(
        {
            "font.size": 19,
            "axes.titlesize": 19,
            "axes.labelsize": 19,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(25, 5.2),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (2.6, 1.6, 1.6)},
    )
    first_heatmap = draw_heatmap(
        axes[0],
        appraisal,
        appraisal_labels,
        limit,
        show_y_labels=True,
    )
    for boundary in (6, 10, 14, 18):
        axes[0].axvline(boundary, color="black", linewidth=2.0, alpha=0.9)
    axes[0].set_title("Δ Appraisal RMSE", pad=12)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    axes[0].tick_params(axis="y", labelrotation=30, labelsize=20)
    fig.canvas.draw()
    offset = transforms.ScaledTranslation(0, -8 / 72.0, fig.dpi_scale_trans)
    for label in axes[0].get_yticklabels():
        label.set_ha("right")
        label.set_transform(label.get_transform() + offset)

    for ax, frame, labels, title in (
        (axes[1], negative, negative_labels, "Δ Negative Emotion Label F1"),
        (axes[2], positive, positive_labels, "Δ Positive Emotion Label F1"),
    ):
        draw_heatmap(
            ax,
            frame,
            labels,
            limit,
            show_y_labels=False,
        )
        ax.set_title(title, pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False)

    for ax in axes:
        ax.tick_params(axis="x", labelrotation=38)
        for label in ax.get_xticklabels():
            label.set_ha("right")

    spans = (
        (0, 6, "relevance"),
        (6, 10, "certainty"),
        (10, 14, "congruence"),
        (14, 18, "control"),
        (18, 22, "accountability"),
    )
    secondary = axes[0].secondary_xaxis("bottom")
    secondary.set_xticks([(start + end) / 2 for start, end, _ in spans])
    secondary.set_xticklabels([label for _, _, label in spans])
    secondary.spines["bottom"].set_visible(False)
    secondary.spines["bottom"].set_position(("outward", 50))
    secondary.tick_params(axis="x", length=0, pad=12, labelsize=19)

    colorbar = fig.colorbar(first_heatmap, ax=axes, fraction=0.02, pad=0.02)
    colorbar.set_label("Delta Value", fontsize=19)
    colorbar.ax.tick_params(labelsize=19)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def nested_value(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def run_sort_key(path: Path) -> int:
    match = re.search(r"run_(\d+)$", path.name)
    return int(match.group(1)) if match else 10**9


def load_model_runs(setting_root: Path, model_id: str) -> list[dict[str, Any]]:
    model_dir = setting_root / model_id
    if not model_dir.is_dir():
        return []

    if model_id in OPEN_SOURCE_MODELS:
        expected_runs = [
            model_dir / f"run_{index}" / "results.json"
            for index in range(1, 6)
        ]
        runs = [load_json(path) for path in expected_runs if path.is_file()]
        if runs:
            return runs

    single_result = model_dir / "results.json"
    if single_result.is_file():
        return [load_json(single_result)]

    run_paths = sorted(
        model_dir.glob("run_*/results.json"),
        key=lambda path: run_sort_key(path.parent),
    )
    return [load_json(path) for path in run_paths]


def metric_values(
    setting_root: Path,
    model_id: str,
    metric_path: tuple[str, ...],
) -> np.ndarray:
    values = []
    for result in load_model_runs(setting_root, model_id):
        value = nested_value(result, metric_path)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return np.asarray(values, dtype=float)


def mean_and_sem(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, 0.0
    mean = float(np.mean(values))
    sem = (
        float(np.std(values, ddof=1) / np.sqrt(values.size))
        if values.size > 1
        else 0.0
    )
    return mean, sem


def collect_f1_results(results_root: Path) -> dict[str, Any]:
    panels: dict[str, Any] = {}
    for panel_title, metric_path in PANEL_METRICS:
        panels[panel_title] = {}
        for (setting_label, _), setting_directory in zip(
            SETTING_COLORS,
            SETTING_DIRECTORIES,
        ):
            means = []
            sems = []
            setting_root = results_root / setting_directory
            for _, model_id in MODEL_ORDER:
                mean, sem = mean_and_sem(
                    metric_values(setting_root, model_id, metric_path)
                )
                means.append(mean)
                sems.append(sem)
            panels[panel_title][setting_label] = {"mean": means, "sem": sems}
    return {
        "models": [display_name for display_name, _ in MODEL_ORDER],
        "panels": panels,
    }


def plot_f1_summary(summary: dict[str, Any], output_path: Path) -> None:
    models = summary["models"]
    panels = summary["panels"]
    fig, axes = plt.subplots(2, 2, figsize=(19.5, 9.2), constrained_layout=False)
    x = np.arange(len(models))
    width = 0.15
    offsets = (np.arange(len(SETTING_COLORS)) - (len(SETTING_COLORS) - 1) / 2) * width

    for ax, title in zip(axes.ravel(), PANEL_TITLES):
        for index, (setting, color) in enumerate(SETTING_COLORS):
            values = panels[title][setting]
            means = np.asarray(values["mean"], dtype=float)
            sems = np.asarray(values.get("sem", np.zeros(len(models))), dtype=float)
            ax.bar(
                x + offsets[index],
                means,
                width=width,
                yerr=sems,
                capsize=2,
                label=setting,
                color=color,
                edgecolor="none",
            )
        ax.set_title(title, fontsize=20, pad=8)
        ax.set_ylabel("F1", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylim(0.3, 0.8)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.80, 0.55),
        frameon=False,
        fontsize=16,
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.76,
        top=0.93,
        bottom=0.09,
        wspace=0.26,
        hspace=0.32,
    )
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    if args.figure_group in {"all", "f1"}:
        plot_f1_summary(
            collect_f1_results(args.results_root),
            args.output_dir / "emotion_f1_bar_chat.pdf",
        )
        generated.append("emotion_f1_bar_chat.pdf")

    if args.figure_group in {"all", "heatmaps"}:
        for short_name, model_id in HEATMAP_MODELS:
            plot_model_heatmaps(
                args.results_root,
                model_id,
                args.output_dir / f"appraisal_heatmap_{short_name}.pdf",
            )
            generated.append(f"appraisal_heatmap_{short_name}.pdf")

    print(f"Generated {len(generated)} RQ2 figures in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
