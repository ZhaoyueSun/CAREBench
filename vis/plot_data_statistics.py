#!/usr/bin/env python3
"""Generate the dataset statistics figures used in the CAREBench paper.

Example:
  python vis/plot_data_statistics.py \
    --data-root data \
    --output-dir figures
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


LIKERT_LEVELS = (
    "Strongly disagree",
    "Somewhat disagree",
    "Neither agree nor disagree",
    "Somewhat agree",
    "Strongly agree",
)
LIKERT_COLORS = ("#b2182b", "#ef8a62", "#d9d9d9", "#67a9cf", "#2166ac")

APPRAISALS = (
    ("Relevance", "General", "This situation matters to me."),
    (
        "Relevance",
        "Urgency",
        "I need to do something about this situation right away.",
    ),
    ("Relevance", "Goals", "This situation relates to my goals and plans."),
    (
        "Relevance",
        "Bodily motives",
        "This situation relates to my physical well-being.",
    ),
    (
        "Relevance",
        "Social motives",
        "This situation involves people who matter to me.",
    ),
    (
        "Relevance",
        "Identity motives",
        "This situation concerns who I am and what I stand for.",
    ),
    (
        "Certainty",
        "Construal",
        "It is clear to me what is going on in this situation.",
    ),
    ("Certainty", "Outlook", "I know what will come next in this situation."),
    ("Certainty", "Predictability", "I saw this situation coming."),
    ("Certainty", "Novelty", "This is a new kind of situation for me."),
    ("Congruence", "General", "This is a good situation."),
    ("Congruence", "Outlook", "This situation will get better with time."),
    (
        "Congruence",
        "Positive prediction error",
        "This situation is better than I expected.",
    ),
    (
        "Congruence",
        "Negative prediction error",
        "This situation is worse than I expected.",
    ),
    ("Control", "General", "This situation is under my control."),
    (
        "Control",
        "Selectability",
        "I can decide whether to stay in this situation or leave it.",
    ),
    ("Control", "Vicarious", "Someone can handle this situation for me."),
    ("Control", "Effortful", "I have to exert effort in this situation."),
    ("Accountability", "Self", "I am responsible for this situation."),
    ("Accountability", "Other", "Someone else is responsible for this situation."),
    (
        "Accountability",
        "Intentionality",
        "This situation was caused intentionally.",
    ),
    ("Accountability", "Fairness", "This situation is fair and deserved."),
)
FAMILIES = tuple(dict.fromkeys(family for family, _, _ in APPRAISALS))

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
POSITIVE_COLORS = {"first": "#1f4e8c", "third": "#6fa8ff"}
NEGATIVE_COLORS = {"first": "#8b1e1e", "third": "#f28b82"}
GROUP_COLORS = {"first": "#1f4e8c", "third": "#cc3d3d"}
INTENSITY_COLORS = (
    "#deebf7",
    "#c6dbef",
    "#9ecae1",
    "#6baed6",
    "#4292c6",
    "#2171b5",
    "#08519c",
)
TRAITS = (
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CAREBench dataset statistics figures."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--figure-group",
        choices=("all", "appraisal", "emotion", "demographics"),
        default="all",
    )
    return parser.parse_args()


def load_records(folder: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(folder.glob("*.json")):
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, dict):
            records.append(payload)
        elif isinstance(payload, list):
            records.extend(item for item in payload if isinstance(item, dict))
    if not records:
        raise RuntimeError(f"No annotation records found in {folder}")
    return records


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def collect_ratings(records: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    ratings = []
    for record in records:
        values = record.get("appraisal_ratings")
        if isinstance(values, dict):
            ratings.append(values)
    if not ratings:
        raise RuntimeError("No appraisal ratings found in the annotation records")
    return ratings


def proportions(values: Iterable[str]) -> np.ndarray:
    counts = Counter(value for value in values if value in LIKERT_LEVELS)
    total = sum(counts.values())
    if total == 0:
        raise RuntimeError("Cannot compute a distribution from zero valid ratings")
    return np.array([counts[level] / total for level in LIKERT_LEVELS]) * 100.0


def family_distributions(ratings: list[dict[str, str]]) -> dict[str, np.ndarray]:
    distributions = {}
    for family in FAMILIES:
        statements = [
            statement
            for item_family, _, statement in APPRAISALS
            if item_family == family
        ]
        values = [
            record[statement]
            for record in ratings
            for statement in statements
            if statement in record
        ]
        distributions[family] = proportions(values)
    return distributions


def dimension_distributions(
    ratings: list[dict[str, str]],
) -> dict[str, np.ndarray]:
    return {
        statement: proportions(
            record[statement] for record in ratings if statement in record
        )
        for _, _, statement in APPRAISALS
    }


def likert_handles() -> list[Patch]:
    labels = (
        "Strongly disagree",
        "Somewhat disagree",
        "Neutral",
        "Somewhat agree",
        "Strongly agree",
    )
    return [
        Patch(facecolor=color, edgecolor="none", label=label)
        for color, label in zip(LIKERT_COLORS, labels)
    ]


def perspective_handles() -> list[Patch]:
    return [
        Patch(
            facecolor="white",
            edgecolor="#333333",
            linewidth=0.6,
            label="First-person",
        ),
        Patch(
            facecolor="white",
            edgecolor="#333333",
            linewidth=0.6,
            hatch="////",
            label="Third-person",
        ),
    ]


def add_vertical_stack(
    ax: plt.Axes,
    x: np.ndarray,
    values: np.ndarray,
    width: float,
    hatch: str | None,
) -> None:
    bottoms = np.zeros(len(x))
    for index, color in enumerate(LIKERT_COLORS):
        heights = values[:, index]
        ax.bar(
            x,
            heights,
            width,
            bottom=bottoms,
            color=color,
            edgecolor="#333333" if hatch else "white",
            linewidth=0.25,
            hatch=hatch,
        )
        bottoms += heights


def plot_appraisal_family(
    first_person: dict[str, np.ndarray],
    third_person: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    x = np.arange(len(FAMILIES))
    width = 0.34
    first_values = np.vstack([first_person[family] for family in FAMILIES])
    third_values = np.vstack([third_person[family] for family in FAMILIES])
    add_vertical_stack(ax, x - width / 2, first_values, width, None)
    add_vertical_stack(ax, x + width / 2, third_values, width, "////")

    ax.set_ylim(0, 100)
    ax.set_ylabel("Responses (%)", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(FAMILIES, fontsize=6.4)
    ax.set_yticks((0, 25, 50, 75, 100))
    ax.tick_params(axis="y", labelsize=6.4, length=2)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.grid(axis="y", color="#cccccc", linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    likert_legend = ax.legend(
        handles=likert_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=5.4,
        handlelength=1.15,
        handletextpad=0.35,
        columnspacing=0.65,
        borderaxespad=0,
    )
    ax.add_artist(likert_legend)
    ax.legend(
        handles=perspective_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=5.8,
        handlelength=1.2,
        columnspacing=1.4,
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.15, right=0.995, top=0.78, bottom=0.24)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def add_horizontal_stack(
    ax: plt.Axes,
    y: float,
    values: np.ndarray,
    height: float,
    hatch: str | None,
) -> None:
    left = 0.0
    for value, color in zip(values, LIKERT_COLORS):
        ax.barh(
            y,
            value,
            height,
            left=left,
            color=color,
            edgecolor="#333333" if hatch else "white",
            linewidth=0.25,
            hatch=hatch,
        )
        left += value


def plot_appraisal_dimensions(
    first_person: dict[str, np.ndarray],
    third_person: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    family_panels = (("Relevance", "Certainty"), ("Congruence", "Control", "Accountability"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.45))

    for ax, families in zip(axes, family_panels):
        rows: list[tuple[str, str | None]] = []
        for family in families:
            rows.append((family.upper(), None))
            rows.extend(
                (label, statement)
                for item_family, label, statement in APPRAISALS
                if item_family == family
            )

        positions = []
        labels = []
        for y, (label, statement) in enumerate(rows):
            if statement is None:
                ax.axhspan(y - 0.42, y + 0.42, color="#eeeeee", zorder=-2)
                ax.text(
                    0.02,
                    y,
                    label,
                    transform=ax.get_yaxis_transform(),
                    ha="left",
                    va="center",
                    fontsize=9.2,
                    fontweight="bold",
                )
                continue
            positions.append(y)
            labels.append(label)
            add_horizontal_stack(ax, y - 0.16, first_person[statement], 0.27, None)
            add_horizontal_stack(ax, y + 0.16, third_person[statement], 0.27, "////")

        ax.set_xlim(0, 100)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xlabel("Responses (%)", fontsize=9.5)
        ax.set_xticks((0, 25, 50, 75, 100))
        ax.tick_params(axis="x", labelsize=8.8, length=2)
        ax.grid(axis="x", color="#cccccc", linewidth=0.45, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=8.8)

    likert_legend = fig.legend(
        handles=likert_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=False,
        fontsize=8.2,
        handlelength=1.5,
        handletextpad=0.45,
        columnspacing=1.15,
        borderaxespad=0,
    )
    fig.add_artist(likert_legend)
    fig.legend(
        handles=perspective_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
        fontsize=8.2,
        handlelength=1.4,
        columnspacing=1.8,
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.16, right=0.995, top=0.84, bottom=0.11, wspace=0.65)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def normalize_labels(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() == "none":
            return []
        return [item.strip() for item in value.split(";") if item.strip()]
    return []


def parse_intensity(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text and text[0].isdigit():
        try:
            return int(text.split("-", 1)[0].strip())
        except ValueError:
            pass
    names = {
        "Not at all": 0,
        "Very slightly": 1,
        "Slightly": 2,
        "Moderately": 3,
        "Quite a bit": 4,
        "Very": 5,
        "Extremely": 6,
    }
    for prefix, level in names.items():
        if text.startswith(prefix):
            return level
    return None


def collect_emotion_statistics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    positive_levels: list[int] = []
    negative_levels: list[int] = []
    positive_counts: list[int] = []
    negative_counts: list[int] = []
    positive_presence: Counter[str] = Counter()
    negative_presence: Counter[str] = Counter()
    count = 0

    for record in records:
        count += 1
        emotions = record.get("emotion_labels", {})
        if not isinstance(emotions, dict):
            emotions = {}
        positive_level = parse_intensity(emotions.get("positive_level"))
        negative_level = parse_intensity(emotions.get("negative_level"))
        if positive_level is not None:
            positive_levels.append(positive_level)
        if negative_level is not None:
            negative_levels.append(negative_level)
        positive_labels = normalize_labels(emotions.get("positive_emotion_labels"))
        negative_labels = normalize_labels(emotions.get("negative_emotion_labels"))
        positive_counts.append(len(positive_labels))
        negative_counts.append(len(negative_labels))
        positive_presence.update(set(positive_labels))
        negative_presence.update(set(negative_labels))

    return {
        "positive_levels": positive_levels,
        "negative_levels": negative_levels,
        "positive_counts": positive_counts,
        "negative_counts": negative_counts,
        "positive_presence": positive_presence,
        "negative_presence": negative_presence,
        "n": count,
    }


def percentage_distribution(values: list[int], length: int) -> np.ndarray:
    counts = np.bincount(values, minlength=length) if values else np.zeros(length)
    return counts / counts.sum() * 100 if counts.sum() else np.zeros(length)


def plot_emotion_statistics(
    first_records: list[dict[str, Any]],
    third_records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    first = collect_emotion_statistics(first_records)
    third = collect_emotion_statistics(third_records)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=220)
    title_size, axis_size, tick_size, legend_size = 20, 20, 18, 15

    ax = axes[0, 0]
    series = (
        first["positive_levels"],
        third["positive_levels"],
        first["negative_levels"],
        third["negative_levels"],
    )
    x = np.arange(4)
    bottom = np.zeros(4)
    distributions = [percentage_distribution(values, 7) for values in series]
    for level in range(7):
        values = np.array([distribution[level] for distribution in distributions])
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=INTENSITY_COLORS[level],
            edgecolor="white",
            linewidth=0.4,
            label=str(level),
        )
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(
        (
            "First-person positive",
            "Third-person positive",
            "First-person negative",
            "Third-person negative",
        ),
        rotation=15,
        ha="right",
        fontsize=tick_size,
    )
    ax.tick_params(axis="y", labelsize=tick_size)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percent", fontsize=axis_size)
    ax.set_title("(a) Emotion Intensity Distribution", fontsize=title_size)
    ax.legend(title="Level", ncol=3, fontsize=legend_size, loc="lower left")

    ax = axes[0, 1]
    count_series = (
        first["positive_counts"],
        third["positive_counts"],
        first["negative_counts"],
        third["negative_counts"],
    )
    maximum = max([value for values in count_series for value in values] + [0])
    plotted_maximum = min(maximum, 10)
    k_values = np.arange(plotted_maximum + 1)
    distributions = [percentage_distribution(values, maximum + 1) for values in count_series]
    width = 0.2
    colors = (
        POSITIVE_COLORS["first"],
        POSITIVE_COLORS["third"],
        NEGATIVE_COLORS["first"],
        NEGATIVE_COLORS["third"],
    )
    labels = (
        "First-person positive",
        "Third-person positive",
        "First-person negative",
        "Third-person negative",
    )
    for index, (values, color, label) in enumerate(zip(distributions, colors, labels)):
        ax.bar(
            k_values + (index - 1.5) * width,
            values[: plotted_maximum + 1],
            width=width,
            color=color,
            label=label,
        )
    ax.set_xticks(k_values)
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.set_xlabel("# labels per sample", fontsize=axis_size)
    ax.set_ylabel("Percent", fontsize=axis_size)
    ax.set_title("(b) Label Number Distribution", fontsize=title_size)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=legend_size)

    for ax, label_names, key, title, colors in (
        (
            axes[1, 0],
            POSITIVE_LABELS,
            "positive_presence",
            "(c) Positive Label Distribution",
            POSITIVE_COLORS,
        ),
        (
            axes[1, 1],
            NEGATIVE_LABELS,
            "negative_presence",
            "(d) Negative Label Distribution",
            NEGATIVE_COLORS,
        ),
    ):
        positions = np.arange(len(label_names))
        width = 0.38
        first_values = [first[key][label] / first["n"] * 100 for label in label_names]
        third_values = [third[key][label] / third["n"] * 100 for label in label_names]
        ax.bar(
            positions - width / 2,
            first_values,
            width=width,
            label="first_person",
            color=colors["first"],
        )
        ax.bar(
            positions + width / 2,
            third_values,
            width=width,
            label="third_person",
            color=colors["third"],
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [label.split(",", 1)[0] for label in label_names],
            rotation=30,
            ha="right",
            fontsize=tick_size,
        )
        ax.tick_params(axis="y", labelsize=tick_size)
        ax.set_ylabel("Prevalence (%)", fontsize=axis_size)
        ax.set_title(title, fontsize=title_size)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=legend_size)

    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def demographic_value(record: dict[str, Any], key: str) -> Any:
    demographics = record.get("demographics", {})
    return demographics.get(key) if isinstance(demographics, dict) else None


def normalize_country(value: Any) -> str | None:
    text = str(value).strip().lower() if value is not None else ""
    if text in {
        "united states",
        "united states of america",
        "usa",
        "us",
        "u.s.",
        "u.s.a.",
        "america",
    }:
        return "US"
    if text in {
        "united kingdom",
        "uk",
        "u.k.",
        "great britain",
        "britain",
        "england",
        "scotland",
        "wales",
        "northern ireland",
    }:
        return "UK"
    return None


def plot_country_distribution(
    first_records: list[dict[str, Any]],
    third_records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    order = ("US", "UK")
    group_percentages = []
    for records in (first_records, third_records):
        countries = [
            country
            for record in records
            if (country := normalize_country(demographic_value(record, "Country of residence")))
        ]
        counts = Counter(countries)
        group_percentages.append(
            np.array([counts[label] for label in order]) / max(len(countries), 1) * 100
        )

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=220)
    x = np.arange(len(order))
    width = 0.36
    bars = (
        ax.bar(
            x - width / 2,
            group_percentages[0],
            width=width,
            color=GROUP_COLORS["first"],
            label="first_person",
        ),
        ax.bar(
            x + width / 2,
            group_percentages[1],
            width=width,
            color=GROUP_COLORS["third"],
            label="third_person",
        ),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Percent (%)")
    ax.set_title("Country of Residence")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    for group in bars:
        for bar in group:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.5,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_ylim(0, max(value.max() for value in group_percentages) + 6)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def parse_age(value: Any) -> float | None:
    try:
        age = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return age if 18 <= age < 90 else None


def plot_age_distribution(
    first_records: list[dict[str, Any]],
    third_records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    groups = [
        [
            age
            for record in records
            if (age := parse_age(demographic_value(record, "Age"))) is not None
        ]
        for records in (first_records, third_records)
    ]
    maximum = int(max(max(values) for values in groups))
    last_decade = max(20, maximum // 10 * 10)
    labels = ["18-19", *(f"{start}-{start + 9}" for start in range(20, last_decade + 1, 10))]
    bins = [18, 20, *(start + 10 for start in range(20, last_decade + 1, 10))]
    percentages = []
    for values in groups:
        counts, _ = np.histogram(values, bins=bins)
        percentages.append(counts / counts.sum() * 100)

    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=220)
    x = np.arange(len(labels))
    width = 0.36
    bars = (
        ax.bar(
            x - width / 2,
            percentages[0],
            width=width,
            color=GROUP_COLORS["first"],
            label="first_person",
        ),
        ax.bar(
            x + width / 2,
            percentages[1],
            width=width,
            color=GROUP_COLORS["third"],
            label="third_person",
        ),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Percent (%)")
    ax.set_xlabel("Age")
    ax.set_title("Age Distribution")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    for group in bars:
        for bar in group:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.25,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    ax.set_ylim(0, max(value.max() for value in percentages) + 6)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def collect_trait_values(
    records: Iterable[dict[str, Any]],
) -> dict[str, list[float]]:
    result = {trait: [] for trait in TRAITS}
    for record in records:
        scores = record.get("BigFive_scores", {})
        if not isinstance(scores, dict):
            continue
        for trait in TRAITS:
            value = scores.get(trait)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[trait].append(float(value))
    return result


def plot_bfi_distribution(
    first_records: list[dict[str, Any]],
    third_records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    values = (
        collect_trait_values(first_records),
        collect_trait_values(third_records),
    )
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=220)
    base_y = np.arange(len(TRAITS))
    for group_index, (name, offset) in enumerate((("first", -0.18), ("third", 0.18))):
        datasets = [values[group_index][trait] for trait in TRAITS]
        positions = base_y + offset
        parts = ax.violinplot(
            datasets,
            positions=positions,
            vert=False,
            widths=0.32,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(GROUP_COLORS[name])
            body.set_edgecolor("white")
            body.set_alpha(0.35)
            body.set_linewidth(0.6)
        parts["cmedians"].set_color(GROUP_COLORS[name])
        parts["cmedians"].set_linewidth(1.4)
        ax.scatter(
            [np.mean(dataset) for dataset in datasets],
            positions,
            s=24,
            color=GROUP_COLORS[name],
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    ax.set_yticks(base_y)
    ax.set_yticklabels(TRAITS)
    ax.invert_yaxis()
    ax.set_xlabel("Big Five score")
    ax.set_title("BFI Distribution")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Patch(
                facecolor=GROUP_COLORS["first"],
                edgecolor="none",
                alpha=0.45,
                label="first_person",
            ),
            Patch(
                facecolor=GROUP_COLORS["third"],
                edgecolor="none",
                alpha=0.45,
                label="third_person",
            ),
        ],
        frameon=False,
        fontsize=9,
        loc="upper left",
    )
    all_values = [value for group in values for trait in TRAITS for value in group[trait]]
    maximum = max(all_values)
    padding = max((maximum - min(all_values)) * 0.08, 0.5)
    ax.set_xlim(2.0, maximum + padding)
    ax.set_xticks(np.arange(2, int(np.floor(maximum + padding)) + 1, 1))
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    first_records = load_records(args.data_root / "first_person")
    third_records = load_records(args.data_root / "third_person")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    if args.figure_group in {"all", "appraisal"}:
        first_ratings = collect_ratings(first_records)
        third_ratings = collect_ratings(third_records)
        plot_appraisal_family(
            family_distributions(first_ratings),
            family_distributions(third_ratings),
            args.output_dir / "appraisal_family_likert_distribution.pdf",
        )
        plot_appraisal_dimensions(
            dimension_distributions(first_ratings),
            dimension_distributions(third_ratings),
            args.output_dir / "appraisal_dimension_likert_distribution.pdf",
        )
    if args.figure_group in {"all", "emotion"}:
        plot_emotion_statistics(
            first_records,
            third_records,
            args.output_dir / "emotion_statistics.pdf",
        )
    if args.figure_group in {"all", "demographics"}:
        plot_age_distribution(
            first_records,
            third_records,
            args.output_dir / "age_distribution.pdf",
        )
        plot_country_distribution(
            first_records,
            third_records,
            args.output_dir / "country_of_residence.pdf",
        )
        plot_bfi_distribution(
            first_records,
            third_records,
            args.output_dir / "bfi_distribution.pdf",
        )

    print(f"Loaded {len(first_records):,} first-person records")
    print(f"Loaded {len(third_records):,} third-person records")
    print(f"Wrote figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
