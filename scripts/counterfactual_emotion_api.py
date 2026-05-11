#!/usr/bin/env python3
"""
Usage:
  python scripts/counterfactual_emotion_api.py \
    --provider openrouter \
    --model anthropic/claude-sonnet-4.6 \
    --task all \
    --source_root data/counterfactual \
    --target_root output/first_person/counterfactual_emotion \
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

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


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
    reasoning: Optional[List[str]] = None


def extract_message_text_from_output_items(output_items: Any) -> str:
    if not isinstance(output_items, list):
        return ""

    for output_item in output_items:
        if not isinstance(output_item, dict):
            continue
        content_blocks = output_item.get("content", [])
        if not isinstance(content_blocks, list):
            continue
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            block_text = block.get("text")
            if isinstance(block_text, str) and block_text.strip():
                return block_text
    return ""


def extract_reasoning_from_output_items(output_items: Any) -> List[str]:
    if not isinstance(output_items, list):
        return []

    summary: List[str] = []
    content_texts: List[str] = []
    for output_item in output_items:
        if not isinstance(output_item, dict):
            continue
        if str(output_item.get("type", "")).strip().lower() != "reasoning":
            continue

        raw_summary = output_item.get("summary")
        if isinstance(raw_summary, list):
            for item in raw_summary:
                if isinstance(item, str) and item.strip():
                    summary.append(item.strip())
                elif isinstance(item, dict):
                    item_text = item.get("text")
                    if isinstance(item_text, str) and item_text.strip():
                        summary.append(item_text.strip())

        raw_content = output_item.get("content")
        if isinstance(raw_content, list):
            for content_item in raw_content:
                if not isinstance(content_item, dict):
                    continue
                item_text = content_item.get("text")
                if isinstance(item_text, str) and item_text.strip():
                    content_texts.append(item_text.strip())

    if summary:
        return summary
    if content_texts:
        return content_texts
    return []


class BaseLLMClient:
    def chat(self, system_prompt: str, user_prompt: str) -> ChatResponse:
        raise NotImplementedError


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: Dict[str, Any], model: Optional[str]) -> None:
        self.api_key = str(config.get("api_key", ""))
        self.model = model or str(config.get("model", ""))
        self.timeout_seconds = int(config.get("timeout_seconds", 120))
        self.base_url = str(config.get("base_url", "")).strip()
        self.reasoning_effort = str(config.get("reasoning_effort", "none")).strip()

        temperature_value = config.get("temperature", None)
        self.temperature: Optional[float]
        if temperature_value is None or str(temperature_value).strip() == "":
            self.temperature = None
        else:
            self.temperature = float(temperature_value)

        missing = []
        if not self.api_key:
            missing.append("api_key")
        if not self.model:
            missing.append("model")
        if missing:
            raise ValueError(f"Missing openai config fields: {', '.join(missing)}")

        if OpenAI is None:
            raise RuntimeError("Missing dependency 'openai'. Install with: pip install openai")

        client_kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout_seconds,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = OpenAI(**client_kwargs)

    def chat(self, system_prompt: str, user_prompt: str) -> ChatResponse:
        request_payload: Dict[str, Any] = {
            "model": self.model,
            "input": user_prompt,
        }
        if system_prompt.strip():
            request_payload["instructions"] = system_prompt
        if self.reasoning_effort:
            request_payload["reasoning"] = {
                "effort": self.reasoning_effort,
                "summary": "detailed",
            }
        if self.temperature is not None:
            request_payload["temperature"] = self.temperature

        try:
            response = self.client.responses.create(**request_payload)
        except Exception as exc:
            raise RuntimeError(f"OpenAI Responses API error: {exc}") from exc

        data: Dict[str, Any]
        if hasattr(response, "model_dump"):
            data = response.model_dump()
        elif isinstance(response, dict):
            data = response
        else:
            data = {"raw": str(response)}

        reasoning = extract_reasoning_from_output_items(data.get("output", []))

        content = getattr(response, "output_text", None)
        if not isinstance(content, str) or not content.strip():
            content = extract_message_text_from_output_items(data.get("output", []))

        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected content type in response: {type(content)}")
        if not content.strip():
            raise RuntimeError(f"Empty content in OpenAI response: {data}")

        return ChatResponse(content=content, raw=data, reasoning=reasoning or None)


class OpenRouterClient(BaseLLMClient):
    def __init__(self, config: Dict[str, Any], model: Optional[str]) -> None:
        self.api_key = str(config.get("api_key", "")).strip()
        self.model = model or str(config.get("model", "")).strip()
        self.timeout_seconds = int(config.get("timeout_seconds", 120))
        self.base_url = str(config.get("base_url", "https://openrouter.ai/api/v1")).strip().rstrip("/")
        self.reasoning_effort = str(config.get("reasoning_effort", "none")).strip()

        max_output_tokens_value = config.get("max_output_tokens", None)
        self.max_output_tokens: Optional[int]
        if max_output_tokens_value is None or str(max_output_tokens_value).strip() == "":
            self.max_output_tokens = None
        else:
            self.max_output_tokens = int(max_output_tokens_value)

        temperature_value = config.get("temperature", None)
        self.temperature: Optional[float]
        if temperature_value is None or str(temperature_value).strip() == "":
            self.temperature = None
        else:
            self.temperature = float(temperature_value)

        missing = []
        if not self.api_key:
            missing.append("api_key")
        if not self.model:
            missing.append("model")
        if missing:
            raise ValueError(f"Missing openrouter config fields: {', '.join(missing)}")

        if requests is None:
            raise RuntimeError("Missing dependency 'requests'. Install with: pip install requests")

    def chat(self, system_prompt: str, user_prompt: str) -> ChatResponse:
        request_payload: Dict[str, Any] = {
            "model": self.model,
            "input": user_prompt,
        }
        if system_prompt.strip():
            request_payload["instructions"] = system_prompt
        if self.reasoning_effort:
            request_payload["reasoning"] = {"effort": self.reasoning_effort}
        if self.max_output_tokens is not None:
            request_payload["max_output_tokens"] = self.max_output_tokens
        if self.temperature is not None:
            request_payload["temperature"] = self.temperature

        endpoint = f"{self.base_url}/responses"
        try:
            http_resp = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=request_payload,
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            raise RuntimeError(f"OpenRouter Responses API request error: {exc}") from exc

        if not http_resp.ok:
            raise RuntimeError(
                f"OpenRouter Responses API error: status={http_resp.status_code}, body={http_resp.text}"
            )

        try:
            data = http_resp.json()
        except ValueError as exc:
            raise RuntimeError(f"OpenRouter response is not valid JSON: {exc}; body={http_resp.text}") from exc

        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected OpenRouter response type: {type(data)}")

        output_items = data.get("output", [])
        reasoning = extract_reasoning_from_output_items(output_items)

        content = data.get("output_text")
        if not isinstance(content, str) or not content.strip():
            content = extract_message_text_from_output_items(output_items)

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"Empty content in OpenRouter response: {data}")

        return ChatResponse(content=content, raw=data, reasoning=reasoning or None)


def build_client(config: Dict[str, Any], provider: str, model: str) -> BaseLLMClient:
    providers = config.get("providers", {})
    if provider == "openai":
        return OpenAIClient(providers.get("openai", {}), model=model)
    if provider == "openrouter":
        return OpenRouterClient(providers.get("openrouter", {}), model=model)
    raise ValueError(f"Unsupported provider: {provider}. Use openai or openrouter")


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-") or "model"


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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Counterfactual emotion API evaluator")
    parser.add_argument("--task", type=str, default="all", help="Task name or all")
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="Model provider: openai or openrouter",
    )
    parser.add_argument("--model", type=str, default="gpt-5.2", help="Model name used in output path")
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
    parser.add_argument("--verbose", type=str2bool, default=True, help="Print last sample info")
    parser.add_argument(
        "--config_path",
        type=str,
        default="project_api_keys.toml",
        help="Path to API key TOML config",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="scripts/prompts/baseline_prompt.toml",
        help="Path to prompt TOML config",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    task_value = args.task.strip().lower()
    if task_value == "all":
        selected_tasks = TASKS
    else:
        if task_value not in TASKS:
            raise ValueError(f"Invalid --task: {args.task}. Valid: all or {', '.join(TASKS)}")
        selected_tasks = [task_value]

    source_root = Path(args.source_root)
    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f"source_root not found: {source_root}")

    if args.dimension.strip().lower() == "all":
        selected_dimensions = [p.name for p in sorted(source_root.iterdir()) if p.is_dir()]
    else:
        selected_dimensions = [args.dimension.strip()]

    if not selected_dimensions:
        print("No dimension folders selected.")
        return 0

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

    config = load_toml(Path(args.config_path))
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

    model_name = args.model.strip()
    client = build_client(config, provider=args.provider.strip().lower(), model=model_name)

    model_dir = Path(args.target_root) / sanitize_model_name(model_name)
    ensure_dir(model_dir)

    processed_count: Dict[str, int] = {task: 0 for task in selected_tasks}
    skipped_count: Dict[str, int] = {task: 0 for task in selected_tasks}
    ambiguous_count: Dict[str, int] = {task: 0 for task in selected_tasks}
    invalid_count: Dict[str, int] = {task: 0 for task in selected_tasks}
    last_verbose: Optional[Dict[str, Any]] = None

    total_units = len(file_jobs) * len(selected_tasks)
    progress_bar = None
    if tqdm is not None:
        progress_bar = tqdm(total=total_units, desc="Processing", unit="task", dynamic_ncols=True)
    else:
        print("[info] tqdm is not installed; running without progress bar.")

    def update_progress(step: int = 1) -> None:
        if progress_bar is None:
            return
        progress_bar.update(step)
        progress_bar.set_postfix(
            {
                "ok": sum(processed_count.values()),
                "skip": sum(skipped_count.values()),
                "amb": sum(ambiguous_count.values()),
                "invalid": sum(invalid_count.values()),
            }
        )

    try:
        for dimension, input_file in file_jobs:
            pending_tasks: List[str] = []
            output_files: Dict[str, Path] = {}
            for task in selected_tasks:
                output_path = model_dir / task / dimension / input_file.name
                output_files[task] = output_path
                if output_path.exists():
                    skipped_count[task] += 1
                    update_progress()
                else:
                    pending_tasks.append(task)

            if not pending_tasks:
                continue

            try:
                items = parse_counterfactual_list(input_file)
            except Exception as exc:
                for task in pending_tasks:
                    invalid_path = model_dir / task / f"invalid_samples_{dimension}.jsonl"
                    append_jsonl(
                        invalid_path,
                        [
                            {
                                "task": task,
                                "dimension": dimension,
                                "input_file": str(input_file),
                                "error": f"Input parse error: {exc}",
                            }
                        ],
                    )
                    invalid_count[task] += 1
                    update_progress()
                continue

            for task in pending_tasks:
                output_rows: List[Dict[str, Any]] = []
                local_invalid_rows: List[Dict[str, Any]] = []
                local_ambiguous = 0

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
                                row: Dict[str, Any] = {
                                    "participant_id": participant_id,
                                    "scenario": scenario,
                                    "score": score,
                                    "label": label or "",
                                }
                                if response.reasoning:
                                    row["reasoning"] = response.reasoning
                                output_rows.append(row)
                                last_verbose = {
                                    "task": task,
                                    "dimension": dimension,
                                    "file": str(input_file),
                                    "participant_id": participant_id,
                                    "scenario": scenario,
                                    "output": row,
                                }
                        else:
                            if not allowed_labels:
                                labels: List[str] = []
                                is_ambiguous = True
                            else:
                                labels, is_ambiguous = resolve_label_list(response.content, allowed_labels)

                            row = {
                                "participant_id": participant_id,
                                "scenario": scenario,
                                "labels": labels,
                            }
                            if response.reasoning:
                                row["reasoning"] = response.reasoning
                            output_rows.append(row)

                            if is_ambiguous:
                                local_ambiguous += 1

                            last_verbose = {
                                "task": task,
                                "dimension": dimension,
                                "file": str(input_file),
                                "participant_id": participant_id,
                                "scenario": scenario,
                                "output": row,
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
                processed_count[task] += 1
                ambiguous_count[task] += local_ambiguous
                if local_invalid_rows:
                    invalid_path = model_dir / task / f"invalid_samples_{dimension}.jsonl"
                    append_jsonl(invalid_path, local_invalid_rows)
                    invalid_count[task] += 1
                update_progress()
    finally:
        if progress_bar is not None:
            progress_bar.close()

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
