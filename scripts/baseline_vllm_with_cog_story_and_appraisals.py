#!/usr/bin/env python3
"""
Usage:
    python scripts/baseline_vllm_with_cog_story_and_appraisals.py \
    --model lzw1008/Emollama-chat-13b \
    --task all \
    --source_folder data/first_person \
    --target_folder output/first_person/baseline_with_cog_story_and_appraisals/Emollama-chat-13b/run_2 \
    --base_url http://localhost:8001 \
    --num_threads 4 \
    --max_tokens 512 \
    --verbose false

Notes:
- --task: one of positive-level, negative-level, positive-labels,
    negative-labels, or all.
- Existing output files will be skipped automatically.
- Default output path: output/first_person/baseline_with_cog_story_and_appraisals/<model>
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


DIMENSION_TO_INPUT_APPRAISAL_STATEMENT = {
    "relevance.general": "This situation matters to me.",
    "relevance.urgency": "I need to do something about this situation right away.",
    "relevance.goals": "This situation relates to my goals and plans.",
    "relevance.bodily motives": "This situation relates to my physical well-being.",
    "relevance.social motives": "This situation involves people who matter to me.",
    "relevance.identity motives": "This situation concerns who I am and what I stand for.",
    "certainty.construal": "It is clear to me what is going on in this situation.",
    "certainty.outlook": "I know what will come next in this situation.",
    "certainty.predictability": "I saw this situation coming.",
    "certainty.novelty": "This is a new kind of situation for me.",
    "congruence.general": "This is a good situation.",
    "congruence.outlook": "This situation will get better with time.",
    "congruence.positive prediction error": "This situation is better than I expected.",
    "congruence.negative prediction error": "This situation is worse than I expected.",
    "control.general": "This situation is under my control.",
    "control.select": "I can decide whether to stay in this situation or leave it.",
    "control.vicarious": "Someone can handle this situation for me.",
    "control.effortful": "I have to exert effort in this situation.",
    "accountability.self": "I am responsible for this situation.",
    "accountability.other": "Someone else is responsible for this situation.",
    "accountability.intentionality": "This situation was caused intentionally.",
    "accountability.fairness": "This situation is fair and deserved.",
}


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


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class ChatResponse:
    content: str
    raw: Dict[str, Any]


class BaseLLMClient:
    def chat(self, system_prompt: str, user_prompt: str) -> ChatResponse:
        raise NotImplementedError


class VLLMClient(BaseLLMClient):
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
    appraisals: str = "",
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
        appraisals=appraisals,
        question=question or "",
        statement=statement or question or "",
    )
    return system_prompt, user_prompt


def parse_sample(file_path: Path) -> Tuple[str, str, Dict[str, Any]]:
    with file_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    sample_id = file_path.stem
    scenario = payload.get("cognitive_questions", {}).get("final_scenario", "")
    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError("Missing cognitive_questions.final_scenario")

    appraisal_ratings = payload.get("appraisal_ratings", {})
    if not isinstance(appraisal_ratings, dict) or not appraisal_ratings:
        raise ValueError("Missing appraisal_ratings")

    return sample_id, scenario.strip(), appraisal_ratings


def build_appraisals_text(prompt_cfg: Dict[str, Any], appraisal_ratings: Dict[str, Any]) -> str:
    dim_to_statement = prompt_cfg.get("appraisals", {}).get("dimension_to_statement", {})
    if not isinstance(dim_to_statement, dict) or not dim_to_statement:
        raise ValueError("Missing appraisals.dimension_to_statement in prompt config")

    rendered_statements: List[str] = []
    for dimension, prompt_statement in dim_to_statement.items():
        input_statement = DIMENSION_TO_INPUT_APPRAISAL_STATEMENT.get(str(dimension))
        if not input_statement:
            raise ValueError(f"Missing dimension mapping for: {dimension}")

        raw_opinion = appraisal_ratings.get(input_statement)
        if raw_opinion is None:
            raise ValueError(
                f"Missing appraisal rating for statement: {input_statement} (dimension: {dimension})"
            )

        opinion = str(raw_opinion).strip().lower()
        rendered = str(prompt_statement).replace("{opinion}", opinion).strip()
        if not rendered:
            raise ValueError(f"Empty rendered appraisal statement for dimension: {dimension}")
        rendered_statements.append(rendered)

    return " ".join(rendered_statements)


def process_level_task(
    client: BaseLLMClient,
    prompt_cfg: Dict[str, Any],
    scenario: str,
    appraisals: str,
    task: str,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    label_map = prompt_cfg.get("label_maps", {}).get(task, {})

    system_prompt, user_prompt = build_prompts_for_task(prompt_cfg, task, scenario, appraisals=appraisals)
    response = client.chat(system_prompt, user_prompt)

    score, label, score_err = resolve_score(response.content, label_map, min_score=0, max_score=6)
    if score is None:
        return None, [{"error": score_err or "Score resolve failed", "raw_output": response.content}]

    output = {
        "score": score,
        "label": label or "",
    }
    return output, []


def process_labels_task(
    client: BaseLLMClient,
    prompt_cfg: Dict[str, Any],
    scenario: str,
    appraisals: str,
    task: str,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], bool]:
    system_prompt, user_prompt = build_prompts_for_task(prompt_cfg, task, scenario, appraisals=appraisals)
    response = client.chat(system_prompt, user_prompt)

    allowed_labels = get_task_label_options(prompt_cfg, task)
    if not allowed_labels:
        # Keep invalid reserved for query failures; treat this as ambiguous empty prediction.
        return {"labels": []}, [], True

    labels, is_ambiguous = resolve_label_list(response.content, allowed_labels)

    output = {"labels": labels}
    return output, [], is_ambiguous


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-") or "model"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="First-person baseline evaluator via vLLM")
    parser.add_argument("--task", type=str, default="all", help="Task name or all")
    parser.add_argument("--source_folder", type=str, default=None, help="Input folder with json files")
    parser.add_argument("--single_file", type=str, default=None, help="Single json file for quick test")
    parser.add_argument("--target_folder", type=str, default=None, help="Custom output base folder")
    parser.add_argument("--verbose", type=str2bool, default=True, help="Print last sample info")
    parser.add_argument("--model", type=str, required=True, help="Model name served by vLLM")
    parser.add_argument("--base_url", type=str, default="http://localhost:8001", help="vLLM base URL")
    parser.add_argument("--api_key", type=str, default="", help="Optional API key for vLLM endpoint")
    parser.add_argument("--timeout_seconds", type=int, default=120, help="HTTP timeout in seconds")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Optional max tokens")
    parser.add_argument("--temperature", type=float, default=0.2, help="Optional temperature")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of worker threads")
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="scripts/prompts/baseline_with_appraisal_prompt.toml",
        help="Path to prompt TOML config",
    )
    return parser


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

    if args.source_folder and args.single_file:
        print("[warning] --source_folder is set; --single_file will be ignored.")

    files = list_input_files(args.source_folder, None if args.source_folder else args.single_file)
    if not files:
        print("No input files found.")
        return 0

    repo_root = Path(__file__).resolve().parents[1]
    prompt_path = Path(args.prompt_path)
    prompt_cfg = load_toml(prompt_path)

    model_name = args.model.strip()
    client = VLLMClient(
        model=model_name,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_seconds=int(args.timeout_seconds),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    if args.target_folder:
        model_base_dir = Path(args.target_folder)
    else:
        model_base_dir = repo_root / "output" / "first_person" / "baseline_with_cog_story_and_appraisals" / sanitize_model_name(model_name)

    ensure_dir(model_base_dir)

    processed_count: Dict[str, int] = {task: 0 for task in selected_tasks}
    skipped_count: Dict[str, int] = {task: 0 for task in selected_tasks}
    ambiguous_count: Dict[str, int] = {task: 0 for task in selected_tasks}
    invalid_count: Dict[str, int] = {task: 0 for task in selected_tasks}
    invalid_rows_by_task: Dict[str, List[Dict[str, Any]]] = {task: [] for task in selected_tasks}
    last_verbose: Optional[Dict[str, Any]] = None

    def process_one_sample(input_path: Path) -> Dict[str, Any]:
        sample_result: Dict[str, Any] = {
            "processed": {task: 0 for task in selected_tasks},
            "skipped": {task: 0 for task in selected_tasks},
            "ambiguous": {task: 0 for task in selected_tasks},
            "invalid": {task: 0 for task in selected_tasks},
            "invalid_rows": {task: [] for task in selected_tasks},
            "last_verbose": None,
        }

        sample_id = input_path.stem
        try:
            sample_id, scenario, appraisal_ratings = parse_sample(input_path)
            appraisals_text = build_appraisals_text(prompt_cfg, appraisal_ratings)
        except Exception as exc:
            for task in selected_tasks:
                sample_result["invalid"][task] += 1
                sample_result["invalid_rows"][task].append(
                    {
                        "sample_id": sample_id,
                        "input_file": str(input_path),
                        "task": task,
                        "error": f"Input parse error: {exc}",
                    }
                )
            return sample_result

        for task in selected_tasks:
            task_output_file = model_base_dir / task / f"{sample_id}.json"
            if task_output_file.exists():
                sample_result["skipped"][task] += 1
                continue

            issues: List[Dict[str, Any]] = []
            output: Optional[Dict[str, Any]] = None
            is_ambiguous = False

            try:
                if task in {"positive-level", "negative-level"}:
                    output, issues = process_level_task(client, prompt_cfg, scenario, appraisals_text, task)
                elif task in {"positive-labels", "negative-labels"}:
                    output, issues, is_ambiguous = process_labels_task(client, prompt_cfg, scenario, appraisals_text, task)
                else:
                    issues = [{"error": f"Unsupported task: {task}"}]
            except Exception as exc:
                issues = [{"error": f"Task execution error: {exc}"}]

            if issues or output is None:
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

                sample_result["invalid"][task] += 1
                sample_result["invalid_rows"][task].extend(issue_rows)
                continue

            write_json(task_output_file, output)
            sample_result["processed"][task] += 1
            if task in {"positive-labels", "negative-labels"} and is_ambiguous:
                sample_result["ambiguous"][task] += 1
            sample_result["last_verbose"] = {
                "sample_id": sample_id,
                "text": scenario,
                "output": output,
                "task": task,
            }

        return sample_result

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [executor.submit(process_one_sample, input_path) for input_path in files]
        with tqdm(total=len(futures), desc="Processing samples", unit="sample") as pbar:
            for future in as_completed(futures):
                result = future.result()

                for task in selected_tasks:
                    processed_count[task] += int(result["processed"][task])
                    skipped_count[task] += int(result["skipped"][task])
                    ambiguous_count[task] += int(result["ambiguous"][task])
                    invalid_count[task] += int(result["invalid"][task])
                    invalid_rows_by_task[task].extend(result["invalid_rows"][task])

                if result["last_verbose"] is not None:
                    last_verbose = result["last_verbose"]

                pbar.update(1)

    for task in selected_tasks:
        invalid_path = model_base_dir / f"invalid_samples_{task}.jsonl"
        append_jsonl(invalid_path, invalid_rows_by_task[task])

    for task in selected_tasks:
        print(
            f"[summary] task={task} processed={processed_count[task]} "
            f"skipped={skipped_count[task]} ambiguous={ambiguous_count[task]} "
            f"invalid={invalid_count[task]}"
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
