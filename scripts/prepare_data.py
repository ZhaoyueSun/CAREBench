#!/usr/bin/env python3
"""Convert the released CAREBench JSONL files into experiment-ready folders.

Example:
  python scripts/prepare_data.py \
    --first-person-file data/raw/first_person.jsonl \
    --third-person-file data/raw/third_person.jsonl \
    --output-root data

The script creates:
  data/first_person/<sample_id>.json
  data/first_person_500/<sample_id>.json
  data/third_person/<sample_id>.json
  data/counterfactual/<appraisal_dimension>/<sample_id>.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


AGREE_LABELS = {"strongly agree", "somewhat agree", "agree"}
DISAGREE_LABELS = {"strongly disagree", "somewhat disagree", "disagree"}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} on line {line_number}: {error.msg}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"Every row in {path} must be a JSON object; "
                    f"line {line_number} contains {type(row).__name__}."
                )
            rows.append(row)

    if not rows:
        raise ValueError(f"JSONL file contains no records: {path}")
    return rows


def load_appraisal_mapping(path: Path) -> Dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Prompt configuration not found: {path}")
    with path.open("rb") as file:
        config = tomllib.load(file)

    mapping = config.get("appraisals", {}).get("dimension_to_statement", {})
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(
            "Prompt configuration must define appraisals.dimension_to_statement."
        )

    result: Dict[str, str] = {}
    for dimension, statement in mapping.items():
        if not isinstance(dimension, str) or not isinstance(statement, str):
            raise ValueError("Appraisal dimensions and statements must be strings.")
        validate_path_component(dimension, "appraisal dimension")
        result[dimension] = statement
    return result


def validate_path_component(value: str, label: str) -> None:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"Invalid {label}: {value!r}")


def validate_first_person(rows: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("The first-person file must contain JSONL records.")

    validated: Dict[str, Dict[str, Any]] = {}
    for line_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(
                f"First-person record on line {line_number} must be a JSON object."
            )
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str):
            raise ValueError(
                f"First-person record on line {line_number} must have a string sample_id."
            )
        validate_path_component(sample_id, "sample ID")
        if sample_id in validated:
            raise ValueError(
                f"Duplicate first-person sample_id {sample_id!r} on line {line_number}."
            )
        annotation = dict(row)
        del annotation["sample_id"]
        if not isinstance(annotation.get("appraisal_ratings"), dict):
            raise ValueError(
                f"First-person annotation for {sample_id!r} has no appraisal_ratings object."
            )
        validated[sample_id] = annotation
    return validated


def validate_third_person(rows: Any) -> Dict[str, List[Dict[str, Any]]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("The third-person file must contain JSONL records.")

    grouped: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for line_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(
                f"Third-person record on line {line_number} must be a JSON object."
            )
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str):
            raise ValueError(
                f"Third-person record on line {line_number} must have a string sample_id."
            )
        validate_path_component(sample_id, "sample ID")
        observer_index = row.get("observer_index")
        if (
            isinstance(observer_index, bool)
            or not isinstance(observer_index, int)
            or observer_index not in range(1, 6)
        ):
            raise ValueError(
                f"Third-person record on line {line_number} must have an "
                "observer_index from 1 to 5."
            )
        annotations = grouped.setdefault(sample_id, {})
        if observer_index in annotations:
            raise ValueError(
                f"Duplicate observer_index {observer_index} for sample_id {sample_id!r}."
            )

        annotation = dict(row)
        del annotation["sample_id"]
        del annotation["observer_index"]
        if not isinstance(annotation.get("appraisal_ratings"), dict):
            raise ValueError(
                f"Third-person annotation for {sample_id!r}, observer "
                f"{observer_index}, has no appraisal_ratings object."
            )
        annotations[observer_index] = annotation

    validated: Dict[str, List[Dict[str, Any]]] = {}
    expected_indices = set(range(1, 6))
    for sample_id, annotations in grouped.items():
        indices = set(annotations)
        if indices != expected_indices:
            missing = sorted(expected_indices - indices)
            raise ValueError(
                f"Third-person sample {sample_id!r} must contain observer indices "
                f"1 to 5; missing indices: {missing}."
            )
        validated[sample_id] = [annotations[index] for index in range(1, 6)]
    return validated


def normalize_label(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().lower().split())


def label_polarity(value: Any) -> str:
    label = normalize_label(value)
    if label in AGREE_LABELS:
        return "agree"
    if label in DISAGREE_LABELS:
        return "disagree"
    return "neutral"


def has_opposite_polarity(first_value: Any, third_value: Any) -> bool:
    polarities = {label_polarity(first_value), label_polarity(third_value)}
    return polarities == {"agree", "disagree"}


def build_counterfactual_samples(
    first_person: Mapping[str, Dict[str, Any]],
    third_person: Mapping[str, List[Dict[str, Any]]],
    appraisal_mapping: Mapping[str, str],
) -> Tuple[Dict[str, Dict[str, List[Dict[str, Any]]]], Dict[str, int]]:
    missing_first_person = sorted(set(third_person) - set(first_person))
    if missing_first_person:
        preview = ", ".join(missing_first_person[:5])
        raise ValueError(
            "Third-person samples are missing from the first-person data: "
            f"{preview}"
        )

    samples_by_dimension: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        dimension: {} for dimension in appraisal_mapping
    }

    for sample_id, observer_annotations in third_person.items():
        first_ratings = first_person[sample_id]["appraisal_ratings"]
        for dimension, statement in appraisal_mapping.items():
            first_value = first_ratings.get(statement)
            selected_annotations = []
            for annotation in observer_annotations:
                observer_ratings = annotation.get("appraisal_ratings", {})
                if not isinstance(observer_ratings, dict):
                    continue
                if has_opposite_polarity(
                    first_value,
                    observer_ratings.get(statement),
                ):
                    selected_annotations.append(annotation)

            if selected_annotations:
                samples_by_dimension[dimension][sample_id] = selected_annotations

    counts = {
        dimension: len(samples)
        for dimension, samples in samples_by_dimension.items()
    }
    return samples_by_dimension, counts


def ensure_writable_targets(targets: List[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in targets if path.exists()]
    if existing:
        preview = ", ".join(str(path) for path in existing[:5])
        suffix = "" if len(existing) <= 5 else f" (and {len(existing) - 5} more)"
        raise FileExistsError(
            f"Output files already exist: {preview}{suffix}. Use --overwrite to replace them."
        )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def prepare_data(
    first_person_file: Path,
    third_person_file: Path,
    prompt_config: Path,
    output_root: Path,
    overwrite: bool,
) -> Dict[str, Any]:
    first_person = validate_first_person(load_jsonl(first_person_file))
    third_person = validate_third_person(load_jsonl(third_person_file))
    appraisal_mapping = load_appraisal_mapping(prompt_config)
    counterfactual, counterfactual_counts = build_counterfactual_samples(
        first_person,
        third_person,
        appraisal_mapping,
    )

    first_targets = {
        sample_id: output_root / "first_person" / f"{sample_id}.json"
        for sample_id in first_person
    }
    first_subset_targets = {
        sample_id: output_root / "first_person_500" / f"{sample_id}.json"
        for sample_id in third_person
    }
    third_targets = {
        sample_id: output_root / "third_person" / f"{sample_id}.json"
        for sample_id in third_person
    }
    counterfactual_targets = {
        (dimension, sample_id): output_root
        / "counterfactual"
        / dimension
        / f"{sample_id}.json"
        for dimension, samples in counterfactual.items()
        for sample_id in samples
    }
    all_targets = [
        *first_targets.values(),
        *first_subset_targets.values(),
        *third_targets.values(),
        *counterfactual_targets.values(),
    ]
    ensure_writable_targets(all_targets, overwrite)

    for sample_id, annotation in first_person.items():
        write_json(first_targets[sample_id], annotation)
    for sample_id in third_person:
        write_json(first_subset_targets[sample_id], first_person[sample_id])
    for sample_id, annotations in third_person.items():
        write_json(third_targets[sample_id], annotations)
    for dimension, samples in counterfactual.items():
        for sample_id, annotations in samples.items():
            write_json(counterfactual_targets[(dimension, sample_id)], annotations)

    return {
        "first_person_samples": len(first_person),
        "first_person_subset_samples": len(third_person),
        "third_person_samples": len(third_person),
        "appraisal_dimensions": len(appraisal_mapping),
        "counterfactual_files": sum(counterfactual_counts.values()),
        "counterfactual_sample_counts": counterfactual_counts,
        "output_root": str(output_root),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the released CAREBench JSONL files into the directory layout "
            "used by the experiment scripts."
        )
    )
    parser.add_argument(
        "--first-person-file",
        type=Path,
        required=True,
        help="Released first_person.jsonl file.",
    )
    parser.add_argument(
        "--third-person-file",
        type=Path,
        required=True,
        help="Released third_person.jsonl file.",
    )
    parser.add_argument(
        "--prompt-config",
        type=Path,
        default=Path("scripts/prompts/baseline_prompt.toml"),
        help="Prompt TOML containing the appraisal dimension mapping.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Root directory for the prepared data folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output files that already exist.",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    summary = prepare_data(
        first_person_file=args.first_person_file,
        third_person_file=args.third_person_file,
        prompt_config=args.prompt_config,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
