#!/usr/bin/env python3
"""Generate the human-evaluation figures used in the CAREBench paper.

Example:
  python vis/plot_human_evaluation.py \
    --human-eval-dir data/human_eval \
    --output-dir figures
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


DIMENSIONS = ("relevance", "congruence", "accountability", "control", "certainty")
PREFERRED_SOURCES = (
    "third-person",
    "claude-sonnet-4.6",
    "gpt-5.2",
    "Qwen-3.5-9B",
    "Llama-3.1-8B-Instruct",
    "PsyLLM-8B",
    "Emollama-chat-13b",
)
QUESTION_TO_DIMENSION = {
    "When the event happened, how much did it matter to you? Why?": "relevance",
    "In what ways did this situation help or hurt you?": "congruence",
    "Who or what did you feel was most responsible for all of this?": "accountability",
    "Did you feel like you were in control? Why?": "control",
    "Did things go as you expected? If not, what was unexpected?": "certainty",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CAREBench human-evaluation figures."
    )
    parser.add_argument(
        "--human-eval-dir",
        type=Path,
        default=Path("data/human_eval"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def pair_key(scenario: object, question: object) -> tuple[str, str]:
    return normalize_text(scenario), normalize_text(question)


def score_key(source: str) -> str:
    return "third_person_score" if source == "third-person" else f"{source}_score"


def empty_dimension_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "best_model": {"annotator_1": None, "annotator_2": None}
    }
    for source in PREFERRED_SOURCES:
        payload[score_key(source)] = {"annotator_1": None, "annotator_2": None}
    return payload


def consistency_response_index(task_question: object) -> int | None:
    match = re.fullmatch(
        r"Is response\s+(\d+)\s+consistent with the reference answer\?",
        normalize_text(task_question),
    )
    if not match:
        return None
    index = int(match.group(1))
    return index if 1 <= index <= 7 else None


def best_response_index(answer: object) -> int | None:
    match = re.fullmatch(r"response\s+(\d+)", normalize_text(answer).lower())
    if not match:
        return None
    index = int(match.group(1))
    return index if 1 <= index <= 7 else None


def load_batch_rows(batch_dir: Path, batch_number: int) -> list[dict[str, Any]]:
    results_path = batch_dir / f"results_{batch_number}.csv"
    mapping_path = batch_dir / f"mapping_{batch_number}.csv"
    evaluate_path = batch_dir / f"evaluate_{batch_number}.csv"
    for path in (results_path, mapping_path, evaluate_path):
        if not path.is_file():
            raise FileNotFoundError(f"Human-evaluation file not found: {path}")

    mappings: dict[tuple[str, str], dict[str, str]] = {}
    scenarios: dict[str, dict[str, Any]] = {}
    scenario_order: list[str] = []

    with mapping_path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            question = normalize_text(row.get("question"))
            dimension = QUESTION_TO_DIMENSION.get(question)
            if dimension is None:
                continue
            scenario = normalize_text(row.get("scenario"))
            mappings[pair_key(scenario, question)] = {
                f"response {index}": normalize_text(row.get(f"response {index}"))
                for index in range(1, 8)
            }
            if scenario not in scenarios:
                scenarios[scenario] = {
                    "scenario": scenario,
                    "dimension": {
                        name: empty_dimension_payload() for name in DIMENSIONS
                    },
                }
                scenario_order.append(scenario)

    evaluate_pairs: set[tuple[str, str]] = set()
    with evaluate_path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            evaluate_pairs.add(pair_key(row.get("scenario"), row.get("question")))

    with results_path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            question = normalize_text(row.get("question"))
            dimension = QUESTION_TO_DIMENSION.get(question)
            if dimension is None:
                continue
            scenario = normalize_text(row.get("scenario"))
            key = pair_key(scenario, question)
            slot_sources = mappings.get(key)
            if slot_sources is None or key not in evaluate_pairs:
                continue
            if scenario not in scenarios:
                scenarios[scenario] = {
                    "scenario": scenario,
                    "dimension": {
                        name: empty_dimension_payload() for name in DIMENSIONS
                    },
                }
                scenario_order.append(scenario)

            payload = scenarios[scenario]["dimension"][dimension]
            first = normalize_text(row.get("Annotator1_Response"))
            second = normalize_text(row.get("Annotator2_Response"))
            response_index = consistency_response_index(row.get("Task_Question"))
            if response_index is not None:
                source = slot_sources.get(f"response {response_index}", "")
                key_name = score_key(source)
                if key_name in payload:
                    payload[key_name]["annotator_1"] = first
                    payload[key_name]["annotator_2"] = second
                continue

            if normalize_text(row.get("Task_Question")) != (
                "Which response is most consistent with the reference?"
            ):
                continue

            def resolve_best(answer: str) -> str:
                if answer.lower() == "none of them":
                    return "None of them"
                index = best_response_index(answer)
                return (
                    slot_sources.get(f"response {index}", answer)
                    if index is not None
                    else answer
                )

            payload["best_model"]["annotator_1"] = resolve_best(first)
            payload["best_model"]["annotator_2"] = resolve_best(second)

    return [scenarios[scenario] for scenario in scenario_order]


def load_detail_rows(human_eval_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for batch_number in (1, 3):
        rows.extend(
            load_batch_rows(
                human_eval_dir / f"batch_{batch_number}",
                batch_number,
            )
        )
    if not rows:
        raise ValueError("No human-evaluation rows were loaded")
    return rows


def agreement(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    return sum(first == second for first, second in pairs) / len(pairs)


def collect_annotator_pairs(
    rows: list[dict[str, Any]],
) -> tuple[
    dict[str, list[tuple[str, str]]],
    dict[str, list[tuple[str, str]]],
]:
    consistency = {dimension: [] for dimension in DIMENSIONS}
    best_response = {dimension: [] for dimension in DIMENSIONS}

    for row in rows:
        dimensions = row.get("dimension", {})
        if not isinstance(dimensions, dict):
            continue
        for dimension in DIMENSIONS:
            values = dimensions.get(dimension, {})
            if not isinstance(values, dict):
                continue
            for score_key, score in values.items():
                if not str(score_key).endswith("_score") or not isinstance(score, dict):
                    continue
                first = normalize_text(score.get("annotator_1")).lower()
                second = normalize_text(score.get("annotator_2")).lower()
                if first in {"yes", "no"} and second in {"yes", "no"}:
                    consistency[dimension].append((first, second))

            best = values.get("best_model", {})
            if isinstance(best, dict):
                first = normalize_text(best.get("annotator_1"))
                second = normalize_text(best.get("annotator_2"))
                if first and second:
                    best_response[dimension].append((first, second))

    return consistency, best_response


def agreement_series(
    pairs_by_dimension: dict[str, list[tuple[str, str]]],
) -> list[float | None]:
    overall = [
        pair
        for dimension in DIMENSIONS
        for pair in pairs_by_dimension.get(dimension, [])
    ]
    return [
        agreement(overall),
        *(agreement(pairs_by_dimension.get(dimension, [])) for dimension in DIMENSIONS),
    ]


def plot_annotator_agreement(rows: list[dict[str, Any]], output_path: Path) -> None:
    consistency, best_response = collect_annotator_pairs(rows)
    specifications = (
        (
            agreement_series(consistency),
            "(a) Consistency Task",
            "#4C78A8",
            0.5,
        ),
        (
            agreement_series(best_response),
            "(b) Best-Response Task",
            "#F58518",
            1.0 / 8.0,
        ),
    )
    labels = ("overall", *DIMENSIONS)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, (values, title, color, chance) in zip(axes, specifications):
        safe_values = [value if value is not None else 0.0 for value in values]
        ax.bar(labels, safe_values, color=color, alpha=0.9)
        ax.axhline(chance, color="black", linestyle="--", linewidth=1.5)
        for index, value in enumerate(values):
            text = "NA" if value is None else f"{value:.2f}"
            height = safe_values[index] + 0.02 if value is not None else 0.02
            ax.text(index, height, text, ha="center", va="bottom", fontsize=13)
        ax.set_title(title, fontsize=18)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.tick_params(axis="both", labelsize=14)
        for tick in ax.get_xticklabels():
            tick.set_rotation(28)
            tick.set_horizontalalignment("right")
    axes[0].set_ylabel("Agreement Percent", fontsize=16)
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Chance level (a: 0.50, b: 0.125)",
            )
        ],
        loc="upper center",
        frameon=False,
        fontsize=14,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def source_from_score_key(score_key: str) -> str:
    if score_key == "third_person_score":
        return "third-person"
    return score_key[: -len("_score")]


def empty_bucket() -> dict[str, int]:
    return {"yes": 0, "consistency_total": 0, "best": 0, "best_total": 0}


def collect_source_scores(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float | None]]]:
    counts: dict[str, dict[str, dict[str, int]]] = {
        dimension: {} for dimension in DIMENSIONS
    }
    for row in rows:
        dimensions = row.get("dimension", {})
        if not isinstance(dimensions, dict):
            continue
        for dimension in DIMENSIONS:
            values = dimensions.get(dimension, {})
            if not isinstance(values, dict):
                continue
            best = values.get("best_model", {})
            best_first = normalize_text(best.get("annotator_1")) if isinstance(best, dict) else ""
            best_second = normalize_text(best.get("annotator_2")) if isinstance(best, dict) else ""

            for score_key, score in values.items():
                if not str(score_key).endswith("_score") or not isinstance(score, dict):
                    continue
                source = source_from_score_key(str(score_key))
                bucket = counts[dimension].setdefault(source, empty_bucket())
                for annotator in ("annotator_1", "annotator_2"):
                    answer = normalize_text(score.get(annotator)).lower()
                    if answer in {"yes", "no"}:
                        bucket["consistency_total"] += 1
                        bucket["yes"] += answer == "yes"
                for answer in (best_first, best_second):
                    if answer:
                        bucket["best_total"] += 1
                        bucket["best"] += answer == source

    ratios: dict[str, dict[str, dict[str, float | None]]] = {}
    for dimension, sources in counts.items():
        ratios[dimension] = {}
        for source, bucket in sources.items():
            ratios[dimension][source] = {
                "consistency": (
                    bucket["yes"] / bucket["consistency_total"]
                    if bucket["consistency_total"]
                    else None
                ),
                "best": (
                    bucket["best"] / bucket["best_total"]
                    if bucket["best_total"]
                    else None
                ),
            }
    return ratios


def ordered_sources(scores: dict[str, dict[str, dict[str, float | None]]]) -> list[str]:
    sources = {
        source
        for dimension_scores in scores.values()
        for source in dimension_scores
    }
    ranks = {source: index for index, source in enumerate(PREFERRED_SOURCES)}
    return sorted(sources, key=lambda source: (ranks.get(source, len(ranks)), source))


def source_colors(sources: list[str]) -> dict[str, Any]:
    palette = plt.get_cmap("tab20")
    base = {source: palette(index) for index, source in enumerate(PREFERRED_SOURCES)}
    shifted = {
        "third-person": base["Emollama-chat-13b"],
        "claude-sonnet-4.6": base["third-person"],
        "gpt-5.2": base["claude-sonnet-4.6"],
        "Qwen-3.5-9B": base["gpt-5.2"],
        "Llama-3.1-8B-Instruct": base["Qwen-3.5-9B"],
        "PsyLLM-8B": base["Llama-3.1-8B-Instruct"],
        "Emollama-chat-13b": base["PsyLLM-8B"],
    }
    for index, source in enumerate(sources):
        shifted.setdefault(source, palette((index + 7) % 20))
    return shifted


def plot_source_scores(rows: list[dict[str, Any]], output_path: Path) -> None:
    scores = collect_source_scores(rows)
    sources = ordered_sources(scores)
    colors = source_colors(sources)
    x = np.arange(len(DIMENSIONS))
    width = 0.85 / max(1, len(sources))
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
    specifications = (
        ("consistency", "(a) Consistency Proportion by Dimension"),
        ("best", "(b) Best Proportion by Dimension"),
    )
    for ax, (metric, title) in zip(axes, specifications):
        for index, source in enumerate(sources):
            values = [
                scores.get(dimension, {}).get(source, {}).get(metric) or 0.0
                for dimension in DIMENSIONS
            ]
            offset = (index - (len(sources) - 1) / 2) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                label=source,
                color=colors[source],
                alpha=0.9,
            )
        ax.set_title(title, fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels(DIMENSIONS, fontsize=16)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.tick_params(axis="y", labelsize=16)
    axes[0].set_ylabel("Proportion", fontsize=16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=16,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = load_detail_rows(args.human_eval_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_annotator_agreement(
        rows,
        args.output_dir / "human_agreement_two_tasks.pdf",
    )
    plot_source_scores(
        rows,
        args.output_dir / "model_human_by_dimension_consistency_best.pdf",
    )
    print(f"Loaded {len(rows):,} human-evaluation rows")
    print(f"Wrote figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
