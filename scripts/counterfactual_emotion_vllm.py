#!/usr/bin/env python3
"""
Usage:
  python scripts/counterfactual_emotion_vllm.py \
    --model lzw1008/Emollama-chat-13b \
    --task all \
    --source_root data/counterfactual \
    --target_root output/first_person/counterfactual_emotion \
    --base_url http://localhost:8002 \
    --num_threads 4 \
    --max_tokens 512 \
    --verbose false

Notes:
- This script runs only emotion tasks on counterfactual samples:
    positive-level, negative-level, positive-labels, negative-labels.
- Input layout: data/counterfactual/<appraisal_dimension>/*.json
- Output layout: output/first_person/counterfactual_emotion/<model>/<task>/<appraisal_dimension>/*.json
- Existing output files are skipped automatically.
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tomllib

try:
    import requests
except ImportError:
    requests = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TASKS = [
    "positive-level",
    "negative-level",
    "positive-labels",
    "negative-labels",
]


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value_lower = value.strip().lower()
    if value_lower in {"1", "true", "yes", "y", "on"}:
        return True
    if value_lower in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def load_toml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"TOML file not found: {path}")
    with path.open("rb") as f:
        return tomllib.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\\n?", "", text)
        text = re.sub(r"\\n?```$", "", text)
    return text.strip()


def normalize_match_text(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def extract_answer_text(text: str) -> str:
    cleaned = strip_code_fence(text).strip()
    if not cleaned:
        return ""

    marker_match = re.search(r"my\s*answer\s*:\s*(.*)$", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if marker_match:
        candidate = marker_match.group(1).strip()
    else:
        candidate = cleaned

    for line in candidate.splitlines():
        stripped = line.strip()
        if stripped:
            answer = stripped.strip().strip("`\"'").strip()
            return answer.rstrip("。.!?;,")

    return ""


def resolve_score(
    raw_text: str,
    label_map: Dict[str, int],
    min_score: int,
    max_score: int,
) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    answer = extract_answer_text(raw_text)
    if not answer:
        return None, None, "Empty answer"

    normalized_to_label: Dict[str, str] = {}
    for label_text in label_map:
        normalized = normalize_match_text(label_text)
        if normalized:
            normalized_to_label[normalized] = label_text

    answer_norm = normalize_match_text(answer)
    if answer_norm in normalized_to_label:
        canonical_label = normalized_to_label[answer_norm]
        score = label_map[canonical_label]
        if min_score <= score <= max_score:
            return score, canonical_label, None

    matches: List[str] = []
    wrapped_answer = f" {answer_norm} "
    for normalized_label, canonical_label in normalized_to_label.items():
        if normalized_label and f" {normalized_label} " in wrapped_answer:
            matches.append(canonical_label)

    if len(matches) == 1:
        canonical_label = matches[0]
        score = label_map[canonical_label]
        if min_score <= score <= max_score:
            return score, canonical_label, None

    if len(matches) > 1:
        return None, None, f"Ambiguous label in answer: {answer}"

    return None, answer, f"Unknown label: {answer}"


def get_task_label_options(prompt_cfg: Dict[str, Any], task: str) -> List[str]:
    label_options = prompt_cfg.get("label_options", {}).get(task)
    if isinstance(label_options, dict):
        values = label_options.get("values")
        if isinstance(values, list):
            return [str(item).strip() for item in values if str(item).strip()]
    if isinstance(label_options, list):
        return [str(item).strip() for item in label_options if str(item).strip()]
    return []


def build_token_to_label_map(allowed_labels: List[str]) -> Dict[str, str]:
    token_to_label: Dict[str, str] = {}
    for label in allowed_labels:
        for token in label.split(","):
            normalized_token = normalize_match_text(token)
            if normalized_token and normalized_token not in token_to_label:
                token_to_label[normalized_token] = label
    return token_to_label


def resolve_label_list(raw_text: str, allowed_labels: List[str]) -> Tuple[List[str], bool]:
    answer = extract_answer_text(raw_text)
    if not answer:
        return [], True

    if normalize_match_text(answer) == "none":
        return [], False

    normalized_to_label = {normalize_match_text(label): label for label in allowed_labels}
    token_to_label = build_token_to_label_map(allowed_labels)

    raw_parts = [part.strip().strip("`\"'").rstrip("。.!?,") for part in answer.split(";")]
    parts = [part for part in raw_parts if part]

    resolved: List[str] = []
    seen = set()
    used_inexact_mapping = False

    for part in parts:
        normalized = normalize_match_text(part)
        if not normalized or normalized == "none":
            continue

        canonical = normalized_to_label.get(normalized)
        if canonical:
            if canonical not in seen:
                seen.add(canonical)
                resolved.append(canonical)
            continue

        candidate_labels: List[str] = []
        candidate_seen = set()
        for token in part.split(","):
            token_norm = normalize_match_text(token)
            if not token_norm or token_norm == "none":
                continue
            mapped = token_to_label.get(token_norm)
            if mapped and mapped not in candidate_seen:
                candidate_seen.add(mapped)
                candidate_labels.append(mapped)

        if candidate_labels:
            used_inexact_mapping = True
            for candidate in candidate_labels:
                if candidate not in seen:
                    seen.add(candidate)
                    resolved.append(candidate)

    is_ambiguous = used_inexact_mapping or (not resolved and normalize_match_text(answer) != "none")
    return resolved, is_ambiguous


@dataclass
class ChatResponse:
    content: str
    raw: Dict[str, Any]


class VLLMClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        timeout_seconds: int,
        max_tokens: Optional[int],
        temperature: Optional[float],
    ) -> None:
        self.model = model.strip()
        self.base_url = base_url.strip().rstrip("/")
        self.api_key = api_key.strip()
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.temperature = temperature

        if not self.model:
            raise ValueError("Missing model name")
        if not self.base_url:
            raise ValueError("Missing base_url")
        if requests is None:
            raise RuntimeError("Missing dependency 'requests'. Install with: pip install requests")

    def chat(self, system_prompt: str, user_prompt: str) -> ChatResponse:
        endpoint = f"{self.base_url}/v1/chat/completions"
        messages: List[Dict[str, str]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            raise RuntimeError(f"vLLM request error: {exc}") from exc

        if not resp.ok:
            raise RuntimeError(f"vLLM API error: status={resp.status_code}, body={resp.text}")

        try:
            data = resp.json()
        except ValueError as exc:
            raise RuntimeError(f"vLLM response is not valid JSON: {exc}; body={resp.text}") from exc

        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected vLLM response type: {type(data)}")

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"Missing choices in vLLM response: {data}")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError(f"Unexpected choice type in vLLM response: {type(first_choice)}")

        message = first_choice.get("message", {})
        content = ""
        if isinstance(message, dict):
            raw_content = message.get("content")
            if isinstance(raw_content, str):
                content = raw_content
            elif isinstance(raw_content, list):
                parts: List[str] = []
                for item in raw_content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str) and text.strip():
                            parts.append(text)
                content = "\n".join(parts)

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"Empty content in vLLM response: {data}")

        return ChatResponse(content=content, raw=data)


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-") or "model"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Counterfactual emotion evaluator via vLLM")
    parser.add_argument("--model", type=str, required=True, help="Model name served by vLLM")
    parser.add_argument("--task", type=str, default="all", help="Task name or all")
    parser.add_argument("--source_root", type=str, default="data/counterfactual", help="Counterfactual input root")
    parser.add_argument(
        "--target_root",
        type=str,
        default="output/first_person/counterfactual_emotion",
        help="Counterfactual emotion output root",
    )
    parser.add_argument(
        "--dimension",
        type=str,
        default="all",
        help="Single appraisal dimension folder name, or all",
    )
    parser.add_argument("--base_url", type=str, default="http://localhost:8001", help="vLLM base URL")
    parser.add_argument("--api_key", type=str, default="", help="Optional API key for vLLM endpoint")
    parser.add_argument("--timeout_seconds", type=int, default=120, help="HTTP timeout in seconds")
    parser.add_argument("--max_tokens", type=int, default=512, help="Optional max tokens")
    parser.add_argument("--temperature", type=float, default=0.2, help="Optional temperature")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--verbose", type=str2bool, default=True, help="Print last sample info")
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="scripts/prompts/baseline_prompt.toml",
        help="Path to prompt TOML config",
    )
    return parser


def build_prompt(prompt_cfg: Dict[str, Any], task: str, scenario: str) -> Tuple[str, str]:
    templates = prompt_cfg.get("templates", {})
    task_tpl = templates.get(task, {})
    user_tpl = str(task_tpl.get("user", "")).strip()
    if not user_tpl:
        raise ValueError(f"Prompt content missing for task: {task}")
    system_prompt = str(task_tpl.get("system", "")).strip()
    user_prompt = user_tpl.format(scenario=scenario, question="", statement="")
    return system_prompt, user_prompt


def parse_counterfactual_list(input_file: Path) -> List[Dict[str, Any]]:
    with input_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list) or len(payload) == 0:
        raise ValueError("Input JSON must be a non-empty list")
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {idx} must be an object")
    return payload


def extract_scenario(item: Dict[str, Any]) -> str:
    scenario = item.get("cognitive_questions", {}).get("final_scenario", "")
    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError("Missing cognitive_questions.final_scenario")
    return scenario.strip()


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.num_threads <= 0:
        raise ValueError("--num_threads must be >= 1")
    if tqdm is None:
        raise RuntimeError("Missing dependency 'tqdm'. Install with: pip install tqdm")

    task_value = args.task.strip().lower()
    if task_value == "all":
        selected_tasks = TASKS
    else:
        if task_value not in TASKS:
            raise ValueError(f"Invalid --task: {args.task}. Valid: all or {', '.join(TASKS)}")
        selected_tasks = [task_value]

    prompt_cfg = load_toml(Path(args.prompt_path))
    templates = prompt_cfg.get("templates", {})
    for task in selected_tasks:
        tpl = templates.get(task, {})
        if not isinstance(tpl, dict) or not str(tpl.get("user", "")).strip():
            raise ValueError(f"Missing [templates.{task}] in prompt TOML")

    label_maps = prompt_cfg.get("label_maps", {})
    for task in selected_tasks:
        if task in {"positive-level", "negative-level"}:
            label_map = label_maps.get(task, {})
            if not isinstance(label_map, dict) or not label_map:
                raise ValueError(f"Missing [label_maps.{task}] in prompt TOML")

    source_root = Path(args.source_root)
    if not source_root.exists():
        raise FileNotFoundError(f"source_root not found: {source_root}")

    if args.dimension.strip().lower() == "all":
        selected_dimensions = [p.name for p in sorted(source_root.iterdir()) if p.is_dir()]
    else:
        selected_dimensions = [args.dimension.strip()]

    if not selected_dimensions:
        print("No dimension folders selected.")
        return 0

    client = VLLMClient(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_seconds=int(args.timeout_seconds),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    model_dir = Path(args.target_root) / sanitize_model_name(args.model)
    ensure_dir(model_dir)

    file_jobs: List[Tuple[str, Path]] = []
    for dimension in selected_dimensions:
        dim_input_dir = source_root / dimension
        if not dim_input_dir.exists() or not dim_input_dir.is_dir():
            print(f"[warning] dimension folder not found: {dimension}")
            continue
        for input_file in sorted(dim_input_dir.glob("*.json")):
            file_jobs.append((dimension, input_file))

    if not file_jobs:
        print("No input files found for selected dimensions.")
        return 0

    processed_count: Dict[str, int] = {task: 0 for task in selected_tasks}
    skipped_count: Dict[str, int] = {task: 0 for task in selected_tasks}
    ambiguous_count: Dict[str, int] = {task: 0 for task in selected_tasks}
    invalid_count: Dict[str, int] = {task: 0 for task in selected_tasks}
    invalid_rows: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        task: {dim: [] for dim in selected_dimensions} for task in selected_tasks
    }
    last_verbose: Optional[Dict[str, Any]] = None

    def process_one_file(dimension: str, input_file: Path) -> Dict[str, Any]:
        file_result: Dict[str, Any] = {
            "processed": {task: 0 for task in selected_tasks},
            "skipped": {task: 0 for task in selected_tasks},
            "ambiguous": {task: 0 for task in selected_tasks},
            "invalid": {task: 0 for task in selected_tasks},
            "invalid_rows": {task: [] for task in selected_tasks},
            "last_verbose": None,
        }

        output_files: Dict[str, Path] = {
            task: model_dir / task / dimension / input_file.name for task in selected_tasks
        }
        pending_tasks: List[str] = []
        for task, output_file in output_files.items():
            if output_file.exists():
                file_result["skipped"][task] += 1
            else:
                pending_tasks.append(task)

        if not pending_tasks:
            return file_result

        try:
            items = parse_counterfactual_list(input_file)
        except Exception as exc:
            for task in pending_tasks:
                file_result["invalid"][task] += 1
                file_result["invalid_rows"][task].append(
                    {
                        "task": task,
                        "dimension": dimension,
                        "input_file": str(input_file),
                        "error": f"Input parse error: {exc}",
                    }
                )
            return file_result

        for task in pending_tasks:
            output_rows: List[Dict[str, Any]] = []
            local_invalid_rows: List[Dict[str, Any]] = []
            local_ambiguous_count = 0

            label_map = label_maps.get(task, {}) if task in {"positive-level", "negative-level"} else {}
            allowed_labels = get_task_label_options(prompt_cfg, task) if task in {"positive-labels", "negative-labels"} else []

            for idx, item in enumerate(items):
                participant_id = str(item.get("participant_id", "")).strip()
                scenario = ""
                try:
                    scenario = extract_scenario(item)
                    system_prompt, user_prompt = build_prompt(prompt_cfg, task, scenario)
                    response = client.chat(system_prompt, user_prompt)

                    if task in {"positive-level", "negative-level"}:
                        score, label, score_err = resolve_score(
                            response.content,
                            label_map,
                            min_score=0,
                            max_score=6,
                        )
                        if score is None:
                            local_invalid_rows.append(
                                {
                                    "task": task,
                                    "dimension": dimension,
                                    "input_file": str(input_file),
                                    "index": idx,
                                    "participant_id": participant_id,
                                    "error": score_err or "Score resolve failed",
                                    "raw_output": response.content,
                                }
                            )
                            output_rows.append(
                                {
                                    "participant_id": participant_id,
                                    "scenario": scenario,
                                    "score": None,
                                    "label": "",
                                }
                            )
                        else:
                            output_rows.append(
                                {
                                    "participant_id": participant_id,
                                    "scenario": scenario,
                                    "score": score,
                                    "label": label or "",
                                }
                            )
                            file_result["last_verbose"] = {
                                "task": task,
                                "dimension": dimension,
                                "file": str(input_file),
                                "participant_id": participant_id,
                                "scenario": scenario,
                                "output": {"score": score, "label": label or ""},
                            }
                    else:
                        if not allowed_labels:
                            labels: List[str] = []
                            is_ambiguous = True
                        else:
                            labels, is_ambiguous = resolve_label_list(response.content, allowed_labels)
                        output_rows.append(
                            {
                                "participant_id": participant_id,
                                "scenario": scenario,
                                "labels": labels,
                            }
                        )
                        if is_ambiguous:
                            local_ambiguous_count += 1
                        file_result["last_verbose"] = {
                            "task": task,
                            "dimension": dimension,
                            "file": str(input_file),
                            "participant_id": participant_id,
                            "scenario": scenario,
                            "output": {"labels": labels},
                        }
                except Exception as exc:
                    local_invalid_rows.append(
                        {
                            "task": task,
                            "dimension": dimension,
                            "input_file": str(input_file),
                            "index": idx,
                            "participant_id": participant_id,
                            "error": f"Item execution error: {exc}",
                        }
                    )
                    if task in {"positive-level", "negative-level"}:
                        output_rows.append(
                            {
                                "participant_id": participant_id,
                                "scenario": scenario,
                                "score": None,
                                "label": "",
                            }
                        )
                    else:
                        output_rows.append(
                            {
                                "participant_id": participant_id,
                                "scenario": scenario,
                                "labels": [],
                            }
                        )

            write_json(output_files[task], output_rows)
            file_result["processed"][task] += 1
            file_result["ambiguous"][task] += local_ambiguous_count
            if local_invalid_rows:
                file_result["invalid"][task] += 1
                file_result["invalid_rows"][task].extend(local_invalid_rows)

        return file_result

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [executor.submit(process_one_file, dim, fpath) for dim, fpath in file_jobs]
        with tqdm(total=len(futures), desc="Processing files", unit="file") as pbar:
            for future in as_completed(futures):
                result = future.result()

                for task in selected_tasks:
                    processed_count[task] += int(result["processed"][task])
                    skipped_count[task] += int(result["skipped"][task])
                    ambiguous_count[task] += int(result["ambiguous"][task])
                    invalid_count[task] += int(result["invalid"][task])

                    for row in result["invalid_rows"][task]:
                        row_dimension = str(row.get("dimension", "")).strip()
                        if row_dimension not in invalid_rows[task]:
                            invalid_rows[task][row_dimension] = []
                        invalid_rows[task][row_dimension].append(row)

                if result["last_verbose"] is not None:
                    last_verbose = result["last_verbose"]

                pbar.update(1)

    for task in selected_tasks:
        for dimension, rows in invalid_rows[task].items():
            invalid_path = model_dir / task / f"invalid_samples_{dimension}.jsonl"
            append_jsonl(invalid_path, rows)

    for task in selected_tasks:
        print(
            f"[summary] task={task} processed={processed_count[task]} "
            f"skipped={skipped_count[task]} ambiguous={ambiguous_count[task]} "
            f"invalid={invalid_count[task]}"
        )

    if args.verbose and last_verbose:
        print("[verbose] last task:", last_verbose["task"])
        print("[verbose] last dimension:", last_verbose["dimension"])
        print("[verbose] last file:", last_verbose["file"])
        print("[verbose] last participant_id:", last_verbose["participant_id"])
        print("[verbose] last scenario:", last_verbose["scenario"])
        print("[verbose] last output:", json.dumps(last_verbose["output"], ensure_ascii=False))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}")
        raise SystemExit(1)
