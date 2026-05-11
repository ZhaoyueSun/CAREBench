#!/usr/bin/env python3
"""
Usage:
  python scripts/evaluate.py \
    --gold_folder data/first_person \
    --pred_folder output/first_person/baseline_with_cog_story_and_appraisals/claude-sonnet-4.6 \
    --prompt_path scripts/prompts/baseline_prompt.toml \
    --output_file results.json

Notes:
- --gold_folder and --pred_folder are required.
- The script evaluates all first_person tasks and writes:
    <pred_folder>/<output_file> (default: results.json)
"""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import tomllib
except ImportError:
    import tomli as tomllib # for python version < 3.11

try:
    from bert_score import BERTScorer
except ImportError:
    BERTScorer = None

try:
    import torch
except ImportError:
    torch = None

try:
    from bleurt import score as bleurt_score
except ImportError:
    bleurt_score = None


TASKS = [
    # "appraisals",
    "positive-level",
    "negative-level",
    "positive-labels",
    "negative-labels",
    # "core-appraisals",
]

CORE_APPRAISAL_DIMENSIONS = [
    "relevance",
    "congruence",
    "accountability",
    "control",
    "certainty",
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


def normalized_rmse(diffs: List[float], value_range: float) -> float:
    if not diffs:
        return 0.0
    mse = sum(((delta / value_range) ** 2 for delta in diffs)) / len(diffs)
    return math.sqrt(mse)


def list_gold_ids(gold_folder: Path) -> Set[str]:
    return {p.stem for p in gold_folder.glob("*.json") if p.is_file()}


def get_pred_files(pred_folder: Path, task: str) -> List[Path]:
    task_folder = pred_folder / task
    if not task_folder.exists() or not task_folder.is_dir():
        return []
    return sorted([p for p in task_folder.glob("*.json") if p.is_file()])


def tokenize_text(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def make_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_bleu_score(reference: str, prediction: str, max_n: int = 4) -> float:
    ref_tokens = tokenize_text(reference)
    pred_tokens = tokenize_text(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        pred_ngrams = make_ngrams(pred_tokens, n)
        ref_ngrams = make_ngrams(ref_tokens, n)
        if not pred_ngrams:
            return 0.0

        pred_counts = Counter(pred_ngrams)
        ref_counts = Counter(ref_ngrams)
        overlap = sum(min(count, ref_counts[gram]) for gram, count in pred_counts.items())
        precision_n = safe_div(float(overlap), float(sum(pred_counts.values())))
        if precision_n <= 0.0:
            return 0.0
        precisions.append(precision_n)

    log_precision = sum(math.log(p) for p in precisions) / max_n
    pred_len = len(pred_tokens)
    ref_len = len(ref_tokens)
    bp = 1.0 if pred_len > ref_len else math.exp(1.0 - (float(ref_len) / float(pred_len)))
    return bp * math.exp(log_precision)


def rouge_n_f1(reference: str, prediction: str, n: int) -> float:
    ref_tokens = tokenize_text(reference)
    pred_tokens = tokenize_text(prediction)
    ref_ngrams = make_ngrams(ref_tokens, n)
    pred_ngrams = make_ngrams(pred_tokens, n)

    if not ref_ngrams or not pred_ngrams:
        return 0.0

    ref_counts = Counter(ref_ngrams)
    pred_counts = Counter(pred_ngrams)
    overlap = sum(min(count, pred_counts[gram]) for gram, count in ref_counts.items())

    precision = safe_div(float(overlap), float(sum(pred_counts.values())))
    recall = safe_div(float(overlap), float(sum(ref_counts.values())))
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[j - 1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref_tokens = tokenize_text(reference)
    pred_tokens = tokenize_text(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0

    lcs = lcs_length(ref_tokens, pred_tokens)
    precision = safe_div(float(lcs), float(len(pred_tokens)))
    recall = safe_div(float(lcs), float(len(ref_tokens)))
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def get_gold_core_answer(payload: Dict[str, Any], dimension: str) -> Optional[str]:
    summary = payload.get("cognitive_questions", {}).get("summary_answers", {})
    if not isinstance(summary, dict):
        return None
    value = summary.get(dimension)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def get_pred_core_answer(payload: Dict[str, Any], dimension: str) -> Optional[str]:
    value = payload.get(dimension)
    if isinstance(value, dict):
        answer = value.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
    elif isinstance(value, str) and value.strip():
        return value.strip()
    return None


def evaluate_core_appraisals(
    gold_folder: Path,
    pred_folder: Path,
    gold_ids: Set[str],
    bleurt_checkpoint: Optional[str],
    bertscore_batch_size: int,
) -> Dict[str, Any]:
    pred_files = get_pred_files(pred_folder, "core-appraisals")
    pred_by_id = {p.stem: p for p in pred_files}

    metric_keys = ["bleu", "rouge-1", "rouge-2", "rouge-l", "bertscore", "bleurt"]
    metric_sums: Dict[str, float] = {k: 0.0 for k in metric_keys}
    metric_counts: Dict[str, int] = {k: 0 for k in metric_keys}

    dim_metric_sums: Dict[str, Dict[str, float]] = {
        dim: {k: 0.0 for k in metric_keys}
        for dim in CORE_APPRAISAL_DIMENSIONS
    }
    dim_metric_counts: Dict[str, Dict[str, int]] = {
        dim: {k: 0 for k in metric_keys}
        for dim in CORE_APPRAISAL_DIMENSIONS
    }

    warnings: List[str] = []
    bert_scorer = None
    if BERTScorer is None:
        warnings.append("BERTScore unavailable: install bert-score")
    else:
        try:
            bertscore_device = "cpu"
            if torch is not None and torch.cuda.is_available():
                bertscore_device = "cuda"
            bert_scorer = BERTScorer(lang="en", device=bertscore_device)
        except Exception as exc:
            warnings.append(f"BERTScorer init failed: {exc}")
            bert_scorer = None

    bleurt_scorer = None
    if bleurt_score is None:
        warnings.append("BLEURT unavailable: install bleurt and provide checkpoint")
    elif bleurt_checkpoint and bleurt_checkpoint.strip():
        try:
            bleurt_scorer = bleurt_score.BleurtScorer(bleurt_checkpoint)
        except Exception as exc:
            warnings.append(f"BLEURT init failed: {exc}")
            bleurt_scorer = None
    else:
        warnings.append("BLEURT checkpoint not provided; set --bleurt_checkpoint")

    missing_pred_sample_count = 0
    missing_pred_answer_count = 0
    missing_gold_answer_count = 0
    evaluated_sample_ids: Set[str] = set()
    evaluated_pairs = 0
    bertscore_pending: List[Tuple[str, str, str]] = []

    for sample_id in sorted(gold_ids):
        gold_payload = load_json(gold_folder / f"{sample_id}.json")

        pred_payload: Optional[Dict[str, Any]] = None
        pred_file = pred_by_id.get(sample_id)
        if pred_file is not None:
            pred_payload = load_json(pred_file)
        else:
            missing_pred_sample_count += 1

        matched_any_dimension = False
        for dimension in CORE_APPRAISAL_DIMENSIONS:
            gold_text = get_gold_core_answer(gold_payload, dimension)
            pred_text = get_pred_core_answer(pred_payload, dimension) if pred_payload else None

            if gold_text is None:
                missing_gold_answer_count += 1
                if pred_text is None:
                    missing_pred_answer_count += 1
                continue

            if pred_text is None:
                missing_pred_answer_count += 1
                values: Dict[str, Optional[float]] = {
                    "bleu": 0.0,
                    "rouge-1": 0.0,
                    "rouge-2": 0.0,
                    "rouge-l": 0.0,
                    "bertscore": 0.0 if bert_scorer is not None else None,
                    "bleurt": 0.0 if bleurt_scorer is not None else None,
                }
            else:
                values = {
                    "bleu": compute_bleu_score(gold_text, pred_text),
                    "rouge-1": rouge_n_f1(gold_text, pred_text, n=1),
                    "rouge-2": rouge_n_f1(gold_text, pred_text, n=2),
                    "rouge-l": rouge_l_f1(gold_text, pred_text),
                    "bertscore": None,
                    "bleurt": None,
                }
                if bert_scorer is not None:
                    bertscore_pending.append((dimension, gold_text, pred_text))
                if bleurt_scorer is not None:
                    try:
                        bleurt_values = bleurt_scorer.score(references=[gold_text], candidates=[pred_text])
                        if isinstance(bleurt_values, list) and bleurt_values:
                            values["bleurt"] = float(bleurt_values[0])
                    except Exception as exc:
                        if f"BLEURT scoring failed: {exc}" not in warnings:
                            warnings.append(f"BLEURT scoring failed: {exc}")

            for metric_name, metric_value in values.items():
                if metric_value is None:
                    continue
                metric_sums[metric_name] += float(metric_value)
                metric_counts[metric_name] += 1
                dim_metric_sums[dimension][metric_name] += float(metric_value)
                dim_metric_counts[dimension][metric_name] += 1

            matched_any_dimension = True
            evaluated_pairs += 1

        if matched_any_dimension:
            evaluated_sample_ids.add(sample_id)

    if bert_scorer is not None and bertscore_pending:
        batch_size = max(1, int(bertscore_batch_size))
        for idx in range(0, len(bertscore_pending), batch_size):
            batch = bertscore_pending[idx : idx + batch_size]
            cands = [pred_text for _, _, pred_text in batch]
            refs = [gold_text for _, gold_text, _ in batch]
            try:
                _, _, f1 = bert_scorer.score(cands, refs, verbose=False, batch_size=batch_size)
            except Exception as exc:
                message = f"BERTScore failed: {exc}"
                if message not in warnings:
                    warnings.append(message)
                continue

            for (dimension, _, _), score in zip(batch, f1):
                metric_value = float(score)
                metric_sums["bertscore"] += metric_value
                metric_counts["bertscore"] += 1
                dim_metric_sums[dimension]["bertscore"] += metric_value
                dim_metric_counts[dimension]["bertscore"] += 1

    overall = {
        metric_name: (safe_div(metric_sums[metric_name], metric_counts[metric_name]) if metric_counts[metric_name] > 0 else None)
        for metric_name in metric_keys
    }

    dimension_result: Dict[str, Dict[str, Optional[float]]] = {}
    for dimension in CORE_APPRAISAL_DIMENSIONS:
        dimension_result[dimension] = {
            metric_name: (
                safe_div(dim_metric_sums[dimension][metric_name], dim_metric_counts[dimension][metric_name])
                if dim_metric_counts[dimension][metric_name] > 0
                else None
            )
            for metric_name in metric_keys
        }

    return {
        "overall": overall,
        "dimension": dimension_result,
        "stats": {
            "gold_total": len(gold_ids),
            "pred_total": len(pred_files),
            "missing_pred_samples": missing_pred_sample_count,
            "missing_pred_answers": missing_pred_answer_count,
            "missing_gold_answers": missing_gold_answer_count,
            "evaluated": len(evaluated_sample_ids),
            "evaluated_pairs": evaluated_pairs,
            "skipped_in_gold": max(0, len(gold_ids) - len(evaluated_sample_ids)),
        },
        "warnings": warnings,
    }


def evaluate_appraisals(
    gold_folder: Path,
    pred_folder: Path,
    prompt_cfg: Dict[str, Any],
    gold_ids: Set[str],
) -> Dict[str, Any]:
    dim_to_statement = prompt_cfg.get("appraisals", {}).get("dimension_to_statement", {})
    if not isinstance(dim_to_statement, dict) or not dim_to_statement:
        raise ValueError("Missing [appraisals.dimension_to_statement] in prompt TOML")

    label_map = prompt_cfg.get("label_maps", {}).get("appraisals", {})
    if not isinstance(label_map, dict) or not label_map:
        raise ValueError("Missing [label_maps.appraisals] in prompt TOML")

    score_to_label = {int(v): str(k) for k, v in label_map.items()}
    pred_files = get_pred_files(pred_folder, "appraisals")
    pred_by_id = {p.stem: p for p in pred_files}
    default_score = 3
    default_label = "Neither agree nor disagree"

    overall_diffs: List[float] = []
    overall_exact = 0
    overall_count = 0

    dimension_stats: Dict[str, Dict[str, Any]] = {
        dim: {
            "diffs": [],
            "exact": 0,
            "count": 0,
        }
        for dim in dim_to_statement.keys()
    }

    evaluated_sample_ids: Set[str] = set()
    missing_pred_count = 0

    for sample_id in sorted(gold_ids):
        gold_path = gold_folder / f"{sample_id}.json"
        pred_file = pred_by_id.get(sample_id)
        if pred_file is None:
            missing_pred_count += 1
            pred_payload = {
                dimension: {
                    "score": default_score,
                    "label": default_label,
                }
                for dimension in dim_to_statement.keys()
            }
        else:
            pred_payload = load_json(pred_file)
        gold_payload = load_json(gold_path)

        appraisal_ratings = gold_payload.get("appraisal_ratings", {})
        if not isinstance(appraisal_ratings, dict):
            continue

        matched_any_dimension = False
        for dimension, statement in dim_to_statement.items():
            pred_dim = pred_payload.get(dimension)
            if not isinstance(pred_dim, dict):
                continue

            pred_score_raw = pred_dim.get("score")
            pred_label_raw = pred_dim.get("label")
            gold_label_raw = appraisal_ratings.get(statement)

            if not isinstance(pred_score_raw, (int, float)):
                continue
            pred_score = int(pred_score_raw)

            if not isinstance(gold_label_raw, str):
                continue
            if gold_label_raw not in label_map:
                continue
            gold_score = int(label_map[gold_label_raw])

            pred_label: Optional[str] = None
            if isinstance(pred_label_raw, str) and pred_label_raw.strip():
                pred_label = pred_label_raw.strip()
            elif pred_score in score_to_label:
                pred_label = score_to_label[pred_score]

            diff = float(pred_score - gold_score)
            overall_diffs.append(diff)
            overall_count += 1
            dimension_stats[dimension]["diffs"].append(diff)
            dimension_stats[dimension]["count"] += 1

            if pred_label is not None and pred_label == gold_label_raw:
                overall_exact += 1
                dimension_stats[dimension]["exact"] += 1

            matched_any_dimension = True

        if matched_any_dimension:
            evaluated_sample_ids.add(sample_id)

    dimension_result: Dict[str, Dict[str, float]] = {}
    for dimension, stats in dimension_stats.items():
        count = int(stats["count"])
        exact = int(stats["exact"])
        diffs = stats["diffs"]
        dimension_result[dimension] = {
            "rmse": normalized_rmse(diffs, value_range=4.0),
            "accuracy": safe_div(exact, count),
        }

    return {
        "overall": {
            "rmse": normalized_rmse(overall_diffs, value_range=4.0),
            "accuracy": safe_div(overall_exact, overall_count),
        },
        "dimension": dimension_result,
        "stats": {
            "gold_total": len(gold_ids),
            "pred_total": len(pred_files),
            "missing_pred": missing_pred_count,
            "evaluated": len(evaluated_sample_ids),
            "skipped_in_gold": max(0, len(gold_ids) - len(evaluated_sample_ids)),
        },
    }


def evaluate_level_task(
    task: str,
    gold_folder: Path,
    pred_folder: Path,
    prompt_cfg: Dict[str, Any],
    gold_ids: Set[str],
) -> Dict[str, Any]:
    label_map = prompt_cfg.get("label_maps", {}).get(task, {})
    if not isinstance(label_map, dict) or not label_map:
        raise ValueError(f"Missing [label_maps.{task}] in prompt TOML")

    pred_files = get_pred_files(pred_folder, task)
    pred_by_id = {p.stem: p for p in pred_files}
    if task == "positive-level":
        default_score = 0
        default_label = "Not at all positive"
    else:
        default_score = 0
        default_label = "Not at all negative"
    diffs: List[float] = []
    exact = 0
    count = 0
    evaluated_sample_ids: Set[str] = set()
    missing_pred_count = 0

    gold_key = "positive_level" if task == "positive-level" else "negative_level"

    for sample_id in sorted(gold_ids):
        gold_path = gold_folder / f"{sample_id}.json"
        pred_file = pred_by_id.get(sample_id)
        if pred_file is None:
            missing_pred_count += 1
            pred_score_raw: Any = default_score
            pred_label_raw: Any = default_label
        else:
            pred_payload = load_json(pred_file)
            pred_score_raw = pred_payload.get("score")
            pred_label_raw = pred_payload.get("label")
        gold_payload = load_json(gold_path)
        if not isinstance(pred_score_raw, (int, float)):
            continue
        pred_score = int(pred_score_raw)

        emotion_labels = gold_payload.get("emotion_labels", {})
        if not isinstance(emotion_labels, dict):
            continue
        gold_level_raw = emotion_labels.get(gold_key)
        gold_score, gold_label = parse_level_value(gold_level_raw)
        if gold_score is None or gold_label is None:
            continue

        diffs.append(float(pred_score - gold_score))
        count += 1
        if isinstance(pred_label_raw, str) and pred_label_raw.strip() == gold_label:
            exact += 1
        evaluated_sample_ids.add(sample_id)

    return {
        "rmse": normalized_rmse(diffs, value_range=6.0),
        "accuracy": safe_div(exact, count),
        "stats": {
            "gold_total": len(gold_ids),
            "pred_total": len(pred_files),
            "missing_pred": missing_pred_count,
            "evaluated": len(evaluated_sample_ids),
            "skipped_in_gold": max(0, len(gold_ids) - len(evaluated_sample_ids)),
        },
    }


def evaluate_labels_task(
    task: str,
    gold_folder: Path,
    pred_folder: Path,
    prompt_cfg: Dict[str, Any],
    gold_ids: Set[str],
) -> Dict[str, Any]:
    label_options = prompt_cfg.get("label_options", {}).get(task, {})
    label_groups = label_options.get("values") if isinstance(label_options, dict) else None
    if not isinstance(label_groups, list) or not label_groups:
        raise ValueError(f"Missing [label_options.{task}.values] in prompt TOML")
    label_groups = [str(label) for label in label_groups]

    gold_key = "positive_emotion_labels" if task == "positive-labels" else "negative_emotion_labels"

    pred_files = get_pred_files(pred_folder, task)
    pred_by_id = {p.stem: p for p in pred_files}

    exact_match = 0
    sample_count = 0
    example_p_sum = 0.0
    example_r_sum = 0.0
    example_f1_sum = 0.0

    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    label_tp: Dict[str, int] = {label: 0 for label in label_groups}
    label_fp: Dict[str, int] = {label: 0 for label in label_groups}
    label_fn: Dict[str, int] = {label: 0 for label in label_groups}

    evaluated_sample_ids: Set[str] = set()
    missing_pred_count = 0

    for sample_id in sorted(gold_ids):
        gold_path = gold_folder / f"{sample_id}.json"
        pred_file = pred_by_id.get(sample_id)
        if pred_file is None:
            missing_pred_count += 1
            pred_labels_raw: Any = []
        else:
            pred_payload = load_json(pred_file)
            pred_labels_raw = pred_payload.get("labels", [])
        gold_payload = load_json(gold_path)
        if not isinstance(pred_labels_raw, list):
            continue
        pred_set = {str(label) for label in pred_labels_raw if isinstance(label, str)}
        pred_set = {label for label in pred_set if label in label_tp}

        emotion_labels = gold_payload.get("emotion_labels", {})
        if not isinstance(emotion_labels, dict):
            continue
        gold_labels_raw = emotion_labels.get(gold_key, [])
        if not isinstance(gold_labels_raw, list):
            continue
        gold_set = {str(label) for label in gold_labels_raw if isinstance(label, str)}
        gold_set = {label for label in gold_set if label in label_tp}

        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        if not pred_set and not gold_set:
            p_i, r_i, f1_i = 1.0, 1.0, 1.0
        else:
            p_i = safe_div(tp, tp + fp)
            r_i = safe_div(tp, tp + fn)
            f1_i = safe_div(2 * p_i * r_i, p_i + r_i) if (p_i + r_i) > 0 else 0.0

        example_p_sum += p_i
        example_r_sum += r_i
        example_f1_sum += f1_i

        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

        for label in label_groups:
            in_pred = label in pred_set
            in_gold = label in gold_set
            if in_pred and in_gold:
                label_tp[label] += 1
            elif in_pred and not in_gold:
                label_fp[label] += 1
            elif (not in_pred) and in_gold:
                label_fn[label] += 1

        if pred_set == gold_set:
            exact_match += 1

        sample_count += 1
        evaluated_sample_ids.add(sample_id)

    micro_p = safe_div(micro_tp, micro_tp + micro_fp)
    micro_r = safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = safe_div(2 * micro_p * micro_r, micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    label_group_metrics: Dict[str, Dict[str, float]] = {}
    macro_p_sum = 0.0
    macro_r_sum = 0.0
    macro_f1_sum = 0.0

    for label in label_groups:
        p_l = safe_div(label_tp[label], label_tp[label] + label_fp[label])
        r_l = safe_div(label_tp[label], label_tp[label] + label_fn[label])
        f1_l = safe_div(2 * p_l * r_l, p_l + r_l) if (p_l + r_l) > 0 else 0.0
        label_group_metrics[label] = {
            "p": p_l,
            "r": r_l,
            "f1": f1_l,
        }
        macro_p_sum += p_l
        macro_r_sum += r_l
        macro_f1_sum += f1_l

    num_labels = len(label_groups)
    macro_overall = {
        "p": safe_div(macro_p_sum, num_labels),
        "r": safe_div(macro_r_sum, num_labels),
        "f1": safe_div(macro_f1_sum, num_labels),
    }

    return {
        "accuracy": safe_div(exact_match, sample_count),
        "example-F1": {
            "p": safe_div(example_p_sum, sample_count),
            "r": safe_div(example_r_sum, sample_count),
            "f1": safe_div(example_f1_sum, sample_count),
        },
        "micro-F1": {
            "p": micro_p,
            "r": micro_r,
            "f1": micro_f1,
        },
        "macro-F1": {
            "overall": macro_overall,
            "label_group": label_group_metrics,
        },
        "stats": {
            "gold_total": len(gold_ids),
            "pred_total": len(pred_files),
            "missing_pred": missing_pred_count,
            "evaluated": len(evaluated_sample_ids),
            "skipped_in_gold": max(0, len(gold_ids) - len(evaluated_sample_ids)),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate first_person prediction outputs")
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
        "--bleurt_checkpoint",
        type=str,
        default="",
        help="Optional BLEURT checkpoint path for core-appraisals",
    )
    parser.add_argument(
        "--bertscore_batch_size",
        type=int,
        default=64,
        help="Batch size for BERTScorer when evaluating core-appraisals",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results.json",
        help="Output filename under pred_folder, default: results.json",
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

    prompt_cfg = load_toml(prompt_path)
    gold_ids = list_gold_ids(gold_folder)

    results: Dict[str, Any] = {}
    for task in TASKS:
        if task == "appraisals":
            results[task] = evaluate_appraisals(gold_folder, pred_folder, prompt_cfg, gold_ids)
        elif task in {"positive-level", "negative-level"}:
            results[task] = evaluate_level_task(task, gold_folder, pred_folder, prompt_cfg, gold_ids)
        elif task in {"positive-labels", "negative-labels"}:
            results[task] = evaluate_labels_task(task, gold_folder, pred_folder, prompt_cfg, gold_ids)
        elif task == "core-appraisals":
            results[task] = evaluate_core_appraisals(
                gold_folder,
                pred_folder,
                gold_ids,
                bleurt_checkpoint=args.bleurt_checkpoint,
                bertscore_batch_size=args.bertscore_batch_size,
            )
        else:
            raise ValueError(f"Unknown task in TASKS: {task}")

    output_path = pred_folder / output_file
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[done] wrote results to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
