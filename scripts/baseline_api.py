#!/usr/bin/env python3
"""
Usage:
  python scripts/baseline_api.py \
    --provider openrouter \
    --model anthropic/claude-sonnet-4.6 \
    --task core-appraisals \
    --source_folder data/first_person \
    --target_folder output/first_person/baseline/claude-sonnet-4.6 \
    --verbose false

Notes:
- --task: one of appraisals, positive-level, negative-level, positive-labels,
    negative-labels, core-appraisals, or all.
- Existing output files will be skipped automatically.
- Default output path: output/first_person/baseline/<model>/<task>/<sample_id>.json
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
    "appraisals",
    "positive-level",
    "negative-level",
    "positive-labels",
    "negative-labels",
    "core-appraisals",
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


def extract_open_text_answer(text: str) -> str:
    cleaned = strip_code_fence(text).strip()
    if not cleaned:
        return ""

    marker_match = re.search(r"my\s*answer\s*:\s*(.*)$", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if marker_match:
        candidate = marker_match.group(1).strip()
    else:
        candidate = cleaned

    return candidate.strip().strip("`").strip()


def extract_first_json_block(text: str) -> Optional[str]:
    """use when expecting a JSON object; not be called now"""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def parse_json_from_model(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """use when expecting a JSON object; not be called now"""
    cleaned = strip_code_fence(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed, None
        return None, "Top-level JSON is not an object"
    except json.JSONDecodeError:
        block = extract_first_json_block(cleaned)
        if not block:
            return None, "No JSON object found in response"
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                return parsed, None
            return None, "Extracted JSON is not an object"
        except json.JSONDecodeError as exc:
            return None, f"JSON decode error: {exc}"


def get_thinking_field(payload: Dict[str, Any]) -> Optional[str]:
    for key in ("thinking", "reason", "rationale"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def try_parse_int_from_text(text: str, min_score: int, max_score: int) -> Optional[int]:
    matches = re.findall(r"\\b(\\d+)\\b", text)
    for match in matches:
        score = int(match)
        if min_score <= score <= max_score:
            return score
    return None


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


def resolve_label_list(raw_text: str, allowed_labels: List[str]) -> Tuple[Optional[List[str]], Optional[str]]:
    answer = extract_answer_text(raw_text)
    if not answer:
        return None, "Empty answer"

    if normalize_match_text(answer) == "none":
        return [], None

    normalized_to_label = {normalize_match_text(label): label for label in allowed_labels}

    raw_parts = [part.strip().strip("`\"'").rstrip("。.!?,") for part in answer.split(";")]
    parts = [part for part in raw_parts if part]
    if not parts:
        return None, "No labels found"

    resolved: List[str] = []
    seen = set()
    for part in parts:
        normalized = normalize_match_text(part)
        canonical = normalized_to_label.get(normalized)
        if not canonical:
            return None, f"Unknown label group: {part}"
        if canonical not in seen:
            seen.add(canonical)
            resolved.append(canonical)

    return resolved, None


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\\n")


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


def list_input_files(source_folder: Optional[str], single_file: Optional[str]) -> List[Path]:
    if source_folder:
        folder = Path(source_folder)
        if not folder.exists():
            raise FileNotFoundError(f"source_folder not found: {source_folder}")
        files = sorted(folder.rglob("*.json"))
        return [file for file in files if file.is_file()]

    if single_file:
        file_path = Path(single_file)
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"single_file not found: {single_file}")
        return [file_path]

    raise ValueError("Either --source_folder or --single_file must be provided")


def build_prompts_for_task(
    prompt_cfg: Dict[str, Any],
    task: str,
    scenario: str,
    question: Optional[str] = None,
    statement: Optional[str] = None,
) -> Tuple[str, str]:
    templates = prompt_cfg.get("templates", {})
    if task not in templates:
        raise ValueError(f"Prompt template missing for task: {task}")

    task_tpl = templates[task]
    system_prompt = str(task_tpl.get("system", "")).strip()
    user_tpl = str(task_tpl.get("user", "")).strip()
    if not user_tpl:
        raise ValueError(f"Prompt content missing for task: {task}")

    user_prompt = user_tpl.format(
        scenario=scenario,
        question=question or "",
        statement=statement or question or "",
    )
    return system_prompt, user_prompt


def parse_sample(file_path: Path) -> Tuple[str, str]:
    with file_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    sample_id = file_path.stem
    scenario = (
        payload.get("story_collection", {})
        .get("final_scenario", "")
    )
    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError("Missing story_collection.final_scenario")
    return sample_id, scenario.strip()


def process_appraisals(
    client: BaseLLMClient,
    prompt_cfg: Dict[str, Any],
    scenario: str,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    dim_map = prompt_cfg.get("appraisals", {}).get("dimension_to_statement")
    if not isinstance(dim_map, dict) or not dim_map:
        dim_map = prompt_cfg.get("appraisals", {}).get("dimension_to_question", {})
    label_map = prompt_cfg.get("label_maps", {}).get("appraisals", {})

    result: Dict[str, Any] = {}
    issues: List[Dict[str, Any]] = []

    for dimension, statement in dim_map.items():
        system_prompt, user_prompt = build_prompts_for_task(
            prompt_cfg,
            "appraisals",
            scenario,
            statement=str(statement),
        )

        response = client.chat(system_prompt, user_prompt)
        score, label, score_err = resolve_score(response.content, label_map, min_score=1, max_score=5)
        if score is None:
            issues.append(
                {
                    "dimension": dimension,
                    "error": score_err or "Score resolve failed",
                    "raw_output": response.content,
                }
            )
            continue

        entry = {
            "score": score,
            "label": label or "",
        }
        if response.reasoning:
            entry["reasoning"] = response.reasoning

        result[str(dimension)] = entry

    if issues:
        return None, issues

    if len(result) != len(dim_map):
        return None, [{"error": "Missing appraisal dimensions in output"}]

    return result, []


def process_core_appraisals(
    client: BaseLLMClient,
    prompt_cfg: Dict[str, Any],
    scenario: str,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    questions = prompt_cfg.get("core-appraisals", {}).get("questions", {})
    if not isinstance(questions, dict) or not questions:
        return None, [{"error": "Missing core-appraisals.questions in prompt config"}]

    result: Dict[str, Any] = {}
    issues: List[Dict[str, Any]] = []

    for dimension, question in questions.items():
        system_prompt, user_prompt = build_prompts_for_task(
            prompt_cfg,
            "core-appraisals",
            scenario,
            question=str(question),
        )

        response = client.chat(system_prompt, user_prompt)
        answer = extract_open_text_answer(response.content)
        if not answer:
            issues.append(
                {
                    "dimension": dimension,
                    "error": "Empty answer",
                    "raw_output": response.content,
                }
            )
            continue

        entry: Dict[str, Any] = {
            "question": str(question),
            "answer": answer,
        }
        if response.reasoning:
            entry["reasoning"] = response.reasoning
        result[str(dimension)] = entry

    if issues:
        return None, issues

    if len(result) != len(questions):
        return None, [{"error": "Missing core appraisal questions in output"}]

    return result, []


def process_level_task(
    client: BaseLLMClient,
    prompt_cfg: Dict[str, Any],
    scenario: str,
    task: str,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    label_map = prompt_cfg.get("label_maps", {}).get(task, {})

    system_prompt, user_prompt = build_prompts_for_task(prompt_cfg, task, scenario)
    response = client.chat(system_prompt, user_prompt)

    score, label, score_err = resolve_score(response.content, label_map, min_score=0, max_score=6)
    if score is None:
        return None, [{"error": score_err or "Score resolve failed", "raw_output": response.content}]

    output = {
        "score": score,
        "label": label or "",
    }
    if response.reasoning:
        output["reasoning"] = response.reasoning
    return output, []


def process_labels_task(
    client: BaseLLMClient,
    prompt_cfg: Dict[str, Any],
    scenario: str,
    task: str,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    system_prompt, user_prompt = build_prompts_for_task(prompt_cfg, task, scenario)
    response = client.chat(system_prompt, user_prompt)

    allowed_labels = get_task_label_options(prompt_cfg, task)
    if not allowed_labels:
        return None, [{"error": f"Missing label options in prompt config for task: {task}"}]

    labels, label_err = resolve_label_list(response.content, allowed_labels)
    if labels is None:
        return None, [{"error": label_err or "Label resolve failed", "raw_output": response.content}]

    output = {"labels": labels}
    if response.reasoning:
        output["reasoning"] = response.reasoning

    return output, []


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="First-person baseline API evaluator")
    parser.add_argument("--task", type=str, default="all", help="Task name or all")
    parser.add_argument("--source_folder", type=str, default=None, help="Input folder with json files")
    parser.add_argument("--single_file", type=str, default=None, help="Single json file for quick test")
    parser.add_argument("--target_folder", type=str, default=None, help="Custom output base folder")
    parser.add_argument("--verbose", type=str2bool, default=True, help="Print last sample info")
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="Model provider: openai or openrouter",
    )
    parser.add_argument("--model", type=str, default="gpt-5.2", help="Model name used in output path")
    parser.add_argument(
        "--config_path",
        type=str,
        default="project_api_keys.toml",
        help="Path to API key TOML config; default is project root project_api_keys.toml",
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

    if args.source_folder and args.single_file:
        print("[warning] --source_folder is set; --single_file will be ignored.")

    files = list_input_files(args.source_folder, None if args.source_folder else args.single_file)
    if not files:
        print("No input files found.")
        return 0

    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config_path)
    prompt_path = Path(args.prompt_path)
    config = load_toml(config_path)
    prompt_cfg = load_toml(prompt_path)

    provider_name = args.provider
    model_name = args.model.strip()
    client = build_client(config, provider=provider_name, model=model_name)

    if args.target_folder:
        model_base_dir = Path(args.target_folder)
    else:
        model_base_dir = repo_root / "output" / "first_person" / "baseline" / model_name

    ensure_dir(model_base_dir)

    processed_count: Dict[str, int] = {task: 0 for task in selected_tasks}
    skipped_count: Dict[str, int] = {task: 0 for task in selected_tasks}
    invalid_count: Dict[str, int] = {task: 0 for task in selected_tasks}

    total_units = len(files) * len(selected_tasks)
    progress_bar = None
    if tqdm is not None:
        progress_bar = tqdm(total=total_units, desc="Evaluating", unit="task", dynamic_ncols=True)
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
                "invalid": sum(invalid_count.values()),
            }
        )

    last_verbose: Optional[Dict[str, Any]] = None
    try:
        for input_path in files:
            sample_id = input_path.stem
            try:
                sample_id, scenario = parse_sample(input_path)
            except Exception as exc:
                for task in selected_tasks:
                    invalid_path = model_base_dir / f"invalid_samples_{task}.jsonl"
                    append_jsonl(
                        invalid_path,
                        [
                            {
                                "sample_id": sample_id,
                                "input_file": str(input_path),
                                "task": task,
                                "error": f"Input parse error: {exc}",
                            }
                        ],
                    )
                    invalid_count[task] += 1
                update_progress(step=len(selected_tasks))
                continue

            for task in selected_tasks:
                task_output_file = model_base_dir / task / f"{sample_id}.json"
                if task_output_file.exists():
                    skipped_count[task] += 1
                    update_progress()
                    continue

                issues: List[Dict[str, Any]] = []
                output: Optional[Dict[str, Any]] = None

                try:
                    if task == "appraisals":
                        output, issues = process_appraisals(client, prompt_cfg, scenario)
                    elif task == "core-appraisals":
                        output, issues = process_core_appraisals(client, prompt_cfg, scenario)
                    elif task in {"positive-level", "negative-level"}:
                        output, issues = process_level_task(client, prompt_cfg, scenario, task)
                    elif task in {"positive-labels", "negative-labels"}:
                        output, issues = process_labels_task(client, prompt_cfg, scenario, task)
                    else:
                        issues = [{"error": f"Unsupported task: {task}"}]
                except Exception as exc:
                    issues = [{"error": f"Task execution error: {exc}"}]

                if issues or output is None:
                    invalid_path = model_base_dir / f"invalid_samples_{task}.jsonl"
                    issue_rows = []
                    for issue in issues:
                        issue_rows.append(
                            {
                                "sample_id": sample_id,
                                "input_file": str(input_path),
                                "task": task,
                                **issue,
                            }
                        )
                    if not issue_rows:
                        issue_rows = [
                            {
                                "sample_id": sample_id,
                                "input_file": str(input_path),
                                "task": task,
                                "error": "Unknown invalid sample",
                            }
                        ]
                    append_jsonl(invalid_path, issue_rows)
                    invalid_count[task] += 1
                    update_progress()
                    continue

                write_json(task_output_file, output)
                processed_count[task] += 1

                last_verbose = {
                    "sample_id": sample_id,
                    "text": scenario,
                    "output": output,
                    "task": task,
                }
                update_progress()
    finally:
        if progress_bar is not None:
            progress_bar.close()

    for task in selected_tasks:
        print(
            f"[summary] task={task} processed={processed_count[task]} "
            f"skipped={skipped_count[task]} invalid={invalid_count[task]}"
        )

    if args.verbose and last_verbose:
        print("[verbose] last task:", last_verbose["task"])
        print("[verbose] last sample_id:", last_verbose["sample_id"])
        print("[verbose] last text:", last_verbose["text"])
        print("[verbose] last output:", json.dumps(last_verbose["output"], ensure_ascii=False))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}")
        raise SystemExit(1)
