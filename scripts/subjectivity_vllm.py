#!/usr/bin/env python3
"""Run the four subjectivity-modeling conditions with a vLLM server.

Example:
  python scripts/subjectivity_vllm.py \
    --model Qwen/Qwen3.5-9B \
    --task appraisals-persona \
    --source_folder data/first_person_500 \
    --third_person_folder data/third_person \
    --target_folder output/subjectivity/Qwen-3.5-9B \
    --base_url http://localhost:8001 \
    --max_tokens 2048 \
    --temperature 0.7 \
    --num_samples 5 \
    --max_concurrency 10 \
    --verbose false

Existing sample files are skipped so interrupted runs can be resumed.
"""
import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import tomllib
except ImportError:
    import tomli as tomllib

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TASKS = [
    "appraisals",
    "appraisals-persona",
    "appraisals-third-person-scenario",
    "appraisals-persona-third-person-scenario",
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


class AsyncVLLMClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        timeout_seconds: int,
        max_tokens: Optional[int],
        temperature: Optional[float],
        semaphore: Optional[asyncio.Semaphore] = None,
        on_request_done: Optional[Callable[[], None]] = None,
    ) -> None:
        if AsyncOpenAI is None:
            raise RuntimeError("Missing dependency 'openai'. Install with: pip install openai")

        self.model = model.strip()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.semaphore = semaphore
        self.on_request_done = on_request_done

        if not self.model:
            raise ValueError("Missing model name")
        if not base_url.strip():
            raise ValueError("Missing base_url")

        self._client = AsyncOpenAI(
            base_url=base_url.strip().rstrip("/") + "/v1",
            api_key=api_key.strip() or "dummy",
            timeout=float(timeout_seconds),
        )

    async def chat(self, system_prompt: str, user_prompt: str) -> ChatResponse:
        messages: List[Dict[str, str]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        try:
            if self.semaphore is None:
                response = await self._client.chat.completions.create(**kwargs)
            else:
                async with self.semaphore:
                    response = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise RuntimeError(f"vLLM request error: {exc}") from exc
        finally:
            if self.on_request_done is not None:
                self.on_request_done()

        raw = response.model_dump()
        content = ""
        if response.choices:
            msg = response.choices[0].message
            if msg and msg.content:
                content = msg.content

        if not content.strip():
            raise RuntimeError(f"Empty content in vLLM response: {raw}")

        return ChatResponse(content=content, raw=raw)


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
    persona: str = "",
    third_person_scenario: str = "",
) -> Tuple[str, str]:
    templates = prompt_cfg.get("templates", {})

    task_tpl = templates[task]
    system_prompt = str(task_tpl.get("system", "")).strip()
    user_tpl = str(task_tpl.get("user", "")).strip()
    if not user_tpl:
        raise ValueError(f"Prompt content missing for task: {task}")

    user_prompt = user_tpl.format(
        scenario=scenario,
        question=question or "",
        statement=statement or question or "",
        persona=persona,
        third_person_scenario=third_person_scenario,
    )

    return system_prompt, user_prompt


def _primary_record(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        if not payload:
            raise ValueError("Payload list is empty")
        first_record = payload[0]
        if not isinstance(first_record, dict):
            raise ValueError("Payload list must contain dictionaries")
        return first_record
    raise ValueError("Payload must be a dictionary or a list of dictionaries")


def _record_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        if not all(isinstance(record, dict) for record in payload):
            raise ValueError("Payload list must contain dictionaries")
        return payload
    raise ValueError("Payload must be a dictionary or a list of dictionaries")


def _coalesce_dict(source: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    for key in keys:
        value = source.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _format_demographics(demographics: Dict[str, Any]) -> str:
    age = str(demographics.get("Age", "")).strip()
    sex = str(demographics.get("Sex", "")).strip()
    ethnicity = str(demographics.get("Ethnicity simplified", "")).strip()
    residence = str(demographics.get("Country of residence", "")).strip()

    descriptor_bits: List[str] = []
    if age and age.isdigit():
        descriptor_bits.append(f"{age}-year-old")
    if ethnicity:
        descriptor_bits.append(ethnicity.lower())

    person_noun = "person"
    sex_lower = sex.lower()
    if sex_lower == "male":
        person_noun = "man"
    elif sex_lower == "female":
        person_noun = "woman"
    elif sex:
        descriptor_bits.append(sex_lower)

    if not descriptor_bits and person_noun == "person" and not residence:
        return ""

    subject_text = " ".join(descriptor_bits + [person_noun]).strip() or "person"
    sentence = f"You are a {subject_text}"
    if residence:
        sentence += f" living in {_format_country_name(residence)}"
    return sentence + "."


def _format_country_name(country: str) -> str:
    normalized = country.strip()
    article_countries = {
        "United States": "the United States",
        "United Kingdom": "the United Kingdom",
        "Netherlands": "the Netherlands",
        "Philippines": "the Philippines",
        "Czech Republic": "the Czech Republic",
        "Dominican Republic": "the Dominican Republic",
        "United Arab Emirates": "the United Arab Emirates",
    }
    return article_countries.get(normalized, normalized)


def _describe_big_five(scores: Dict[str, Any]) -> str:
    if not scores:
        return ""

    trait_phrases = [
        (
            "Extraversion",
            [
                "very reserved",
                "somewhat reserved",
                "balanced in sociability",
                "fairly outgoing",
                "very outgoing",
            ],
        ),
        (
            "Agreeableness",
            [
                "very skeptical",
                "somewhat guarded",
                "balanced between warmth and skepticism",
                "fairly warm and cooperative",
                "very warm and trusting",
            ],
        ),
        (
            "Conscientiousness",
            [
                "very spontaneous",
                "somewhat unstructured",
                "moderately organized",
                "fairly organized and dependable",
                "highly organized and disciplined",
            ],
        ),
        (
            "Neuroticism",
            [
                "very calm",
                "fairly calm",
                "moderately reactive",
                "fairly prone to stress",
                "very prone to stress",
            ],
        ),
        (
            "Openness",
            [
                "very conventional",
                "somewhat conventional",
                "balanced between practicality and curiosity",
                "fairly curious and imaginative",
                "very curious and imaginative",
            ],
        ),
    ]

    available_parts: List[str] = []
    for score_key, phrases in trait_phrases:
        raw_value = scores.get(score_key)
        if not isinstance(raw_value, (int, float)):
            continue
        score = max(1, min(10, int(raw_value)))
        band_index = min((score - 1) // 2, len(phrases) - 1)
        available_parts.append(phrases[band_index])

    if not available_parts:
        return ""

    if len(available_parts) == 1:
        trait_text = available_parts[0]
    else:
        trait_text = ", ".join(available_parts[:-1]) + f", and {available_parts[-1]}"

    return f"Your Big Five personality profile suggests that you are {trait_text}."


def extract_persona(payload: Any) -> str:
    record = _primary_record(payload)

    demographics = _coalesce_dict(record, "demographics")
    big_five_scores = _coalesce_dict(record, "BigFive_scores", "BigFive_Scores")

    demographic_sentence = _format_demographics(demographics)
    big_five_sentence = _describe_big_five(big_five_scores)

    sentences = [
        sentence
        for sentence in [demographic_sentence, big_five_sentence]
        if sentence
    ]

    if not sentences:
        raise ValueError("No persona information found in payload")

    return " ".join(sentences)


def extract_third_person_scenario(payload: Any) -> str:
    record = _primary_record(payload)
    scenario = _coalesce_dict(record, "cognitive_questions", "story_collection").get("final_scenario", "")
    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError("Missing final_scenario")
    return scenario.strip()


def parse_sample(
    file_path: Path,
    return_personas: bool = False,
    return_third_person_scenarios: bool = False,
) -> Tuple[str, str] | Tuple[str, str, List[str]]:

    with file_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    record = _primary_record(payload)
    sample_id = file_path.stem
    scenario = _coalesce_dict(record, "story_collection", "cognitive_questions").get("final_scenario", "")
    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError("Missing final_scenario")

    if return_personas:
        return sample_id, scenario.strip(), [extract_persona(record_item) for record_item in _record_list(payload)]

    if return_third_person_scenarios:
        return sample_id, scenario.strip(), [extract_third_person_scenario(record_item) for record_item in _record_list(payload)]

    return sample_id, scenario.strip()


async def process_payloads(
    task: str,
    client: AsyncVLLMClient,
    prompt_cfg: Dict[str, Any],
    scenario: str,
    persona: str,
    third_person_scenario: str,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    dim_map = prompt_cfg.get("appraisals", {}).get("dimension_to_statement")
    if not isinstance(dim_map, dict) or not dim_map:
        dim_map = prompt_cfg.get("appraisals", {}).get("dimension_to_question", {})
    label_map = prompt_cfg.get("label_maps", {}).get("appraisals", {})

    result: Dict[str, Any] = {}
    issues: List[Dict[str, Any]] = []

    for dimension, statement in dim_map.items():
        system_prompt, user_prompt = build_prompts_for_task(
            prompt_cfg=prompt_cfg,
            task=task,
            scenario=scenario,
            statement=str(statement),
            persona=persona,
            third_person_scenario=third_person_scenario,
        )

        response = await client.chat(system_prompt, user_prompt)
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
        result[str(dimension)] = entry

    if issues:
        return None, issues

    if len(result) != len(dim_map):
        return None, [{"error": "Missing appraisal dimensions in output"}]

    return result, []


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-") or "model"


def requests_per_run(prompt_cfg: Dict[str, Any], task: str) -> int:
    if task in TASKS:
        dim_map = prompt_cfg.get("appraisals", {}).get("dimension_to_statement")
        if not isinstance(dim_map, dict) or not dim_map:
            dim_map = prompt_cfg.get("appraisals", {}).get("dimension_to_question", {})
        if isinstance(dim_map, dict) and dim_map:
            return len(dim_map)
        return 1

    return 1


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CAREBench subjectivity-modeling inference via vLLM"
    )
    parser.add_argument("--task", type=str, required=True, choices=TASKS)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--source_folder",
        type=str,
        help="Matched first-person input folder, normally data/first_person_500",
    )
    input_group.add_argument(
        "--single_file",
        type=str,
        help="Single first-person JSON file for a quick test",
    )
    parser.add_argument(
        "--third_person_folder",
        type=str,
        default="data/third_person",
        help="Folder containing the five matched observer records per narrative",
    )
    parser.add_argument("--target_folder", type=str, default=None, help="Custom output base folder")
    parser.add_argument("--verbose", type=str2bool, default=True, help="Print last sample info")
    parser.add_argument("--model", type=str, required=True, help="Model name served by vLLM")
    parser.add_argument("--base_url", type=str, default="http://localhost:8001", help="vLLM base URL")
    parser.add_argument("--api_key", type=str, default="", help="Optional API key for vLLM endpoint")
    parser.add_argument("--timeout_seconds", type=int, default=120, help="HTTP timeout in seconds")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Optional max tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=10,
        help="Maximum number of concurrent vLLM requests",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of predictions generated for each narrative",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="scripts/prompts/subjectivity_prompt.toml",
        help="Path to the subjectivity-modeling prompt configuration",
    )
    return parser


async def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.max_concurrency <= 0:
        raise ValueError("--max_concurrency must be >= 1")
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be >= 1")

    if tqdm is None:
        raise RuntimeError("Missing dependency 'tqdm'. Install with: pip install tqdm")

    selected_tasks = [args.task]

    files = list_input_files(args.source_folder, args.single_file)
    if not files:
        print("No input files found.")
        return 0

    repo_root = Path(__file__).resolve().parents[1]
    prompt_path = Path(args.prompt_path)
    prompt_cfg = load_toml(prompt_path)

    model_name = args.model.strip()
    vllm_url = args.base_url.strip().rstrip("/") + "/v1"
    print(f"[info] vLLM endpoint: {vllm_url}  model: {model_name}  temperature: {args.temperature:g}", flush=True)

    client = AsyncVLLMClient(
        model=model_name,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_seconds=int(args.timeout_seconds),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        semaphore=asyncio.Semaphore(args.max_concurrency),
    )

    if args.target_folder:
        model_base_dir = Path(args.target_folder)
    else:
        folder_name = sanitize_model_name(model_name)
        model_base_dir = repo_root / "output" / "subjectivity" / folder_name

    ensure_dir(model_base_dir)

    processed_count: Dict[str, int] = {task: 0 for task in selected_tasks}
    skipped_count: Dict[str, int] = {task: 0 for task in selected_tasks}
    invalid_count: Dict[str, int] = {task: 0 for task in selected_tasks}
    invalid_rows_by_task: Dict[str, List[Dict[str, Any]]] = {task: [] for task in selected_tasks}

    num_samples = args.num_samples

    # Assigned inside the tqdm context before any coroutine runs.
    pbar_runs: Any = None

    async def process_one_sample(input_path: Path) -> Dict[str, Any]:
        sample_result: Dict[str, Any] = {
            "processed": {task: 0 for task in selected_tasks},
            "skipped": {task: 0 for task in selected_tasks},
            "invalid": {task: 0 for task in selected_tasks},
            "invalid_rows": {task: [] for task in selected_tasks},
            "last_verbose": None,
        }

        sample_id = input_path.stem

        try:
            sample_id, scenario = parse_sample(input_path)
            third_person_personas, third_person_scenarios = [], []
            if args.task in {"appraisals-persona", "appraisals-persona-third-person-scenario"}:
                third_person_path = Path(args.third_person_folder) / f"{sample_id}.json"
                _, _, third_person_personas = parse_sample(
                    file_path=third_person_path,
                    return_personas=True,
                )
            if args.task in {"appraisals-third-person-scenario", "appraisals-persona-third-person-scenario"}:
                third_person_path = Path(args.third_person_folder) / f"{sample_id}.json"
                _, _, third_person_scenarios = parse_sample(
                    file_path=third_person_path,
                    return_third_person_scenarios=True,
                )
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

        async def run_and_track(
            i: int,
            task: str,
            scenario: str,
            third_person_personas: List[str],
            third_person_scenarios: List[str],
        ) -> Any:
            cur_persona = third_person_personas[i % len(third_person_personas)] if third_person_personas else ""
            cur_third_person_scenario = (
                third_person_scenarios[i % len(third_person_scenarios)] if third_person_scenarios else ""
            )

            return await process_payloads(
                task=task,
                client=client,
                prompt_cfg=prompt_cfg,
                scenario=scenario,
                persona=cur_persona,
                third_person_scenario=cur_third_person_scenario,
            )

        for task in selected_tasks:
            task_output_file = model_base_dir / task / f"{sample_id}.json"
            if task_output_file.exists():
                sample_result["skipped"][task] += 1
                continue

            run_results = await asyncio.gather(
                *[
                    run_and_track(
                        i,
                        task,
                        scenario,
                        third_person_personas,
                        third_person_scenarios,
                    ) for i in range(num_samples)
                ],
                return_exceptions=True,
            )

            outputs: List[Dict[str, Any]] = []
            issue_rows: List[Dict[str, Any]] = []
            for run in run_results:
                if isinstance(run, BaseException):
                    error_str = str(run)
                    print(f"[error] {sample_id} task={task}: {error_str}", file=sys.stderr, flush=True)
                    issue_rows.append(
                        {
                            "sample_id": sample_id,
                            "input_file": str(input_path),
                            "task": task,
                            "error": error_str,
                        }
                    )
                    continue
                out, issues = run
                if issues or out is None:
                    for issue in issues:
                        issue_rows.append(
                            {
                                "sample_id": sample_id,
                                "input_file": str(input_path),
                                "task": task,
                                **issue,
                            }
                        )
                    if not issues:
                        issue_rows.append(
                            {
                                "sample_id": sample_id,
                                "input_file": str(input_path),
                                "task": task,
                                "error": "Unknown invalid sample",
                            }
                        )
                else:
                    outputs.append(out)

            if len(outputs) != num_samples:
                sample_result["invalid"][task] += 1
                sample_result["invalid_rows"][task].extend(issue_rows)
                sample_result["invalid_rows"][task].append(
                    {
                        "sample_id": sample_id,
                        "input_file": str(input_path),
                        "task": task,
                        "error": (
                            f"Expected {num_samples} complete predictions, "
                            f"received {len(outputs)}"
                        ),
                    }
                )
                continue

            write_json(task_output_file, outputs)
            sample_result["processed"][task] += 1
            sample_result["last_verbose"] = {
                "sample_id": sample_id,
                "text": scenario,
                "output": outputs,
                "task": task,
            }

        return sample_result

    last_verbose: Optional[Dict[str, Any]] = None
    total_requests = 0
    for input_path in files:
        sample_id = input_path.stem
        for task in selected_tasks:
            task_output_file = model_base_dir / task / f"{sample_id}.json"
            if task_output_file.exists():
                continue
            total_requests += requests_per_run(prompt_cfg, task) * num_samples

    with tqdm(total=len(files), desc="Samples", unit="sample", position=0, leave=True) as pbar_samples:
        with tqdm(total=total_requests, desc="LLM requests", unit="req", position=1, leave=True) as pbar_runs:
            client.on_request_done = lambda: pbar_runs.update(1)
            futures = [asyncio.ensure_future(process_one_sample(f)) for f in files]
            for fut in asyncio.as_completed(futures):
                result = await fut
                for task in selected_tasks:
                    processed_count[task] += int(result["processed"][task])
                    skipped_count[task] += int(result["skipped"][task])
                    invalid_count[task] += int(result["invalid"][task])
                    invalid_rows_by_task[task].extend(result["invalid_rows"][task])

                if result["last_verbose"] is not None:
                    last_verbose = result["last_verbose"]

                pbar_samples.update(1)

    for task in selected_tasks:
        invalid_path = model_base_dir / f"invalid_samples_{task}.jsonl"
        append_jsonl(invalid_path, invalid_rows_by_task[task])

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
        raise SystemExit(asyncio.run(main()))
    except Exception as exc:
        print(f"[error] {exc}")
        raise SystemExit(1)
