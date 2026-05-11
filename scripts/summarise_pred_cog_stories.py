#!/usr/bin/env python3
"""
Usage:
  python scripts/summarise_pred_cog_stories.py \
    --provider openai \
    --model gpt-4 \
    --source_folder data/first_person \
    --core_appraisals_folder output/first_person/baseline/Emollama-chat-13b/run_2/core-appraisals \
    --target_folder output/first_person/baseline/Emollama-chat-13b/run_2/pred-cog-stories \
    --skip_existing
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import tomllib

try:
    import requests
except ImportError:
    requests = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DIMENSIONS = ["relevance", "congruence", "accountability", "control", "certainty"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_toml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"TOML file not found: {path}")
    with path.open("rb") as f:
        return tomllib.load(f)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return data


def get_previous_story(source_payload: Dict[str, Any]) -> Optional[str]:
    value = (
        source_payload.get("cognitive_questions", {})
        .get("summary_answers", {})
        .get("previousStory")
    )
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def get_dimension_answer(core_payload: Dict[str, Any], dimension: str) -> Optional[str]:
    value = core_payload.get(dimension, {}).get("answer")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def to_prompt_value(value: Optional[str]) -> str:
    if value is None:
        return "null"
    return value


def find_prompt_path(cli_prompt_path: Optional[str]) -> Path:
    if cli_prompt_path:
        return Path(cli_prompt_path)

    default_candidates = [
        Path("scripts/prompts/summarsie_pred_cog_stories.toml"),
        Path("scripts/prompts/summarise_pred_cog_stories.toml"),
    ]
    for candidate in default_candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Prompt file not found. Tried scripts/prompts/summarsie_pred_cog_stories.toml "
        "and scripts/prompts/summarise_pred_cog_stories.toml"
    )


def extract_template(prompt_cfg: Dict[str, Any]) -> str:
    template = prompt_cfg.get("templates", {}).get("summarise", {}).get("user")
    if not isinstance(template, str) or not template.strip():
        raise ValueError("Missing templates.summarise.user in prompt TOML")
    return template


def render_user_prompt(
    template: str,
    previous_story: Optional[str],
    relevance: Optional[str],
    congruence: Optional[str],
    accountability: Optional[str],
    control: Optional[str],
    certainty: Optional[str],
) -> str:
    return template.format(
        previousStory=to_prompt_value(previous_story),
        relevance=to_prompt_value(relevance),
        congruence=to_prompt_value(congruence),
        accountability=to_prompt_value(accountability),
        control=to_prompt_value(control),
        certainty=to_prompt_value(certainty),
    )


def postprocess_final_scenario(text: str) -> str:
    return text.replace("\n\n", " ").strip()


@dataclass
class ChatResponse:
    content: str
    raw: Dict[str, Any]


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


class BaseLLMClient:
    def chat(self, system_prompt: str, user_prompt: str) -> ChatResponse:
        raise NotImplementedError


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: Dict[str, Any], model: Optional[str]) -> None:
        self.api_key = str(config.get("api_key", "")).strip()
        self.model = (model or str(config.get("model", ""))).strip()
        self.timeout_seconds = int(config.get("timeout_seconds", 120))
        self.base_url = str(config.get("base_url", "")).strip()
        self.temperature = 0.1

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

        content = getattr(response, "output_text", None)
        if not isinstance(content, str) or not content.strip():
            content = extract_message_text_from_output_items(data.get("output", []))

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"Empty content in OpenAI response: {data}")

        return ChatResponse(content=content.strip(), raw=data)


class OpenRouterClient(BaseLLMClient):
    def __init__(self, config: Dict[str, Any], model: Optional[str]) -> None:
        self.api_key = str(config.get("api_key", "")).strip()
        self.model = (model or str(config.get("model", ""))).strip()
        self.timeout_seconds = int(config.get("timeout_seconds", 120))
        self.base_url = str(config.get("base_url", "https://openrouter.ai/api/v1")).strip().rstrip("/")
        self.temperature = 0.1

        max_output_tokens_value = config.get("max_output_tokens", None)
        self.max_output_tokens: Optional[int]
        if max_output_tokens_value is None or str(max_output_tokens_value).strip() == "":
            self.max_output_tokens = None
        else:
            self.max_output_tokens = int(max_output_tokens_value)

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
        if self.temperature is not None:
            request_payload["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            request_payload["max_output_tokens"] = self.max_output_tokens

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

        content = data.get("output_text")
        if not isinstance(content, str) or not content.strip():
            content = extract_message_text_from_output_items(data.get("output", []))

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"Empty content in OpenRouter response: {data}")

        return ChatResponse(content=content.strip(), raw=data)


def build_client(config: Dict[str, Any], provider: str, model: str) -> BaseLLMClient:
    providers = config.get("providers", {})
    if provider == "openai":
        return OpenAIClient(providers.get("openai", {}), model=model)
    if provider == "openrouter":
        return OpenRouterClient(providers.get("openrouter", {}), model=model)
    raise ValueError(f"Unsupported provider: {provider}. Use openai or openrouter")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarise predicted cognitive stories into final scenarios")
    parser.add_argument("--provider", type=str, required=True, choices=["openai", "openrouter"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--source_folder", type=str, required=True)
    parser.add_argument("--core_appraisals_folder", type=str, required=True)
    parser.add_argument("--target_folder", type=str, required=True)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip samples whose target file already exists",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="",
        help=(
            "Optional prompt TOML path. Default tries "
            "scripts/prompts/summarsie_pred_cog_stories.toml then "
            "scripts/prompts/summarise_pred_cog_stories.toml"
        ),
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="project_api_keys.toml",
        help="Path to API key TOML config; default is project root project_api_keys.toml",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if tqdm is None:
        raise RuntimeError("Missing dependency 'tqdm'. Install with: pip install tqdm")

    source_folder = Path(args.source_folder)
    core_folder = Path(args.core_appraisals_folder)
    target_folder = Path(args.target_folder)

    if not source_folder.exists() or not source_folder.is_dir():
        raise FileNotFoundError(f"source_folder not found: {source_folder}")
    if not core_folder.exists() or not core_folder.is_dir():
        raise FileNotFoundError(f"core_appraisals_folder not found: {core_folder}")

    ensure_dir(target_folder)

    prompt_cfg = load_toml(find_prompt_path(args.prompt_path.strip() or None))
    template = extract_template(prompt_cfg)
    config_path = Path(args.config_path)
    api_cfg = load_toml(config_path)
    client = build_client(api_cfg, provider=args.provider, model=args.model)

    source_files = sorted([p for p in source_folder.rglob("*.json") if p.is_file()])

    summary = {
        "total_source_files": len(source_files),
        "generated": 0,
        "skipped_existing": 0,
        "missing_core_file": 0,
        "missing_previous_story": 0,
        "missing_dimension_answers_total": 0,
        "missing_dimension_answers_by_dimension": {dim: 0 for dim in DIMENSIONS},
        "errors": 0,
    }

    for source_file in tqdm(source_files, desc="Summarising pred cog stories", unit="file"):
        sample_name = source_file.name
        sample_id = source_file.stem
        target_file = target_folder / sample_name

        if args.skip_existing and target_file.exists() and target_file.is_file():
            summary["skipped_existing"] += 1
            continue

        core_file = core_folder / sample_name
        if not core_file.exists() or not core_file.is_file():
            summary["missing_core_file"] += 1
            continue

        try:
            source_payload = load_json(source_file)
            core_payload = load_json(core_file)

            previous_story = get_previous_story(source_payload)
            if previous_story is None:
                summary["missing_previous_story"] += 1

            dimension_answers: Dict[str, Optional[str]] = {}
            for dim in DIMENSIONS:
                ans = get_dimension_answer(core_payload, dim)
                dimension_answers[dim] = ans
                if ans is None:
                    summary["missing_dimension_answers_total"] += 1
                    summary["missing_dimension_answers_by_dimension"][dim] += 1

            user_prompt = render_user_prompt(
                template=template,
                previous_story=previous_story,
                relevance=dimension_answers["relevance"],
                congruence=dimension_answers["congruence"],
                accountability=dimension_answers["accountability"],
                control=dimension_answers["control"],
                certainty=dimension_answers["certainty"],
            )
            response = client.chat(system_prompt="", user_prompt=user_prompt)
            output_scenario = response.content
            final_scenario = postprocess_final_scenario(output_scenario)

            output_payload = {
                "sample_id": sample_id,
                "source_file": str(source_file),
                "core_appraisals_file": str(core_file),
                "inputs": {
                    "previousStory": previous_story,
                    "relevance": dimension_answers["relevance"],
                    "congruence": dimension_answers["congruence"],
                    "accountability": dimension_answers["accountability"],
                    "control": dimension_answers["control"],
                    "certainty": dimension_answers["certainty"],
                },
                "output_scenario": output_scenario,
                "final_scenario": final_scenario,
            }

            ensure_dir(target_file.parent)
            with target_file.open("w", encoding="utf-8") as f:
                json.dump(output_payload, f, ensure_ascii=False, indent=2)

            summary["generated"] += 1
        except Exception as exc:
            summary["errors"] += 1
            print(f"[ERROR] {sample_name}: {exc}")

    print("\n===== Summary =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
