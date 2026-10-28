# CAREBench

**CAREBench: Evaluating LLMs' Emotion Understanding by Assessing Cognitive Appraisal Reasoning**

This repository provides the data-preparation, inference, evaluation, and visualization code for reproducing the experiments reported in the paper. It supports API and locally served models, and includes the prompt configurations, figure-generation scripts, and generated paper figures.

## Data

The dataset is available on [Hugging Face](https://huggingface.co/datasets/zhaoyuesun/CAREBench) under the [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) license. Download the two released JSON Lines files and place them under the project root as follows:

```text
data/
└── raw/
    ├── first_person.jsonl
    └── third_person.jsonl
```

Each line in `first_person.jsonl` contains one first-person annotation with an explicit `sample_id`:

```json
{"sample_id":"...","participant_id":"...","story_collection":{},"cognitive_questions":{},"appraisal_ratings":{},"emotion_labels":{},"personality_questionnaire":{},"BigFive_scores":{},"demographics":{}}
```

Each line in `third_person.jsonl` contains one observer annotation. Five records share each `sample_id`, with `observer_index` ranging from 1 to 5:

```json
{"sample_id":"...","observer_index":1,"participant_id":"...","cognitive_questions":{},"appraisal_ratings":{},"emotion_labels":{},"personality_questionnaire":{},"BigFive_scores":{},"demographics":{}}
```

Convert the released files into the per-narrative layout used by the experiment scripts:

```bash
python scripts/prepare_data.py \
  --first-person-file data/raw/first_person.jsonl \
  --third-person-file data/raw/third_person.jsonl \
  --output-root data
```

The command creates four experiment-ready directories:

| Directory | Contents and use |
|---|---|
| `data/first_person/` | All 1,000 first-person samples; input to the baseline inference and model evaluation scripts. |
| `data/first_person_500/` | The 500 first-person samples with observer annotations; gold annotations for `evaluate_human.py`. |
| `data/third_person/` | Five observer annotations for each of the 500 matched narratives; input to `evaluate_human.py`. |
| `data/counterfactual/` | Intervention samples organized by appraisal dimension; input to the counterfactual inference and evaluation scripts. |

Each directory contains one JSON file per narrative. Existing files are preserved by default; pass `--overwrite` only when the prepared data should be replaced.

## Repository Structure

```text
CAREBench/
├── README.md
├── LICENSE
├── requirements.txt       # Python dependencies and versions
├── llama2.jinja           # chat template used for EmoLLaMa
├── scripts/
│   ├── prepare_data.py    # convert the released JSONL files for the experiments
│   ├── prompts/           # task instructions, label maps, and output templates
│   ├── baseline_*.py      # API- and vLLM-based inference
│   ├── counterfactual_*.py
│   ├── subjectivity_vllm.py  # subjectivity-modeling inference
│   ├── evaluate*.py       # aggregate, per-sample, observer, and intervention evaluation
│   └── summarise_pred_cog_stories.py
├── vis/                   # scripts for generating the paper figures
│   ├── plot_data_statistics.py
│   ├── plot_human_evaluation.py
│   ├── plot_input_effects.py
│   └── plot_intervention_results.py
└── figures/               # generated figures reported in the paper
```

`data/` is created by `scripts/prepare_data.py`, and experiment predictions and evaluation results are written to `output/`. Neither generated directory is included in this repository.

## Environment Setup

The experiments, evaluation, and figure generation were run in a single Conda environment using Python 3.10.20. The exact package versions are recorded in `requirements.txt`. Create the environment from the repository root:

```bash
conda create -n carebench python=3.10.20
conda activate carebench
python -m pip install -r requirements.txt
```

For the experiments reported in the paper, we accessed GPT-5.2 and Claude-Sonnet-4.6 via API and deployed the open-source models on a local server equipped with an NVIDIA H100 80GB GPU, serving them through vLLM with 10 concurrent threads.

## Models

The experiments reported in the paper use the following model versions:

| Model | Model identifier |
|---|---|
| GPT-5.2 | `gpt-5.2` |
| Claude-Sonnet-4.6 | `anthropic/claude-sonnet-4.6` |
| Qwen 3.5-9B | `Qwen/Qwen3.5-9B` |
| Llama 3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` |
| PsyLLM-8B | `GMLHUHE/PsyLLM-8B` |
| EmoLLaMa-chat-13B | `lzw1008/Emollama-chat-13b` |

API credentials are not distributed with this repository. API inference requires users to create `project_api_keys.toml` locally; this file is excluded from version control. Use the following structure and replace the placeholders with your own credentials:

```toml
[providers.openai]
api_key = "YOUR_OPENAI_API_KEY"
timeout_seconds = 120
reasoning_effort = "none"
temperature = 0.2

[providers.openrouter]
api_key = "YOUR_OPENROUTER_API_KEY"
base_url = "https://openrouter.ai/api/v1"
timeout_seconds = 120
reasoning_effort = "none"
temperature = 0.2
```

The model version is selected with `--model`; only credentials and provider settings are stored in this local file.

## Serving Open-Source Models with vLLM

Activate `carebench` on the GPU server and serve the required model on port 8001. Run one model at a time.

```bash
# Qwen 3.5-9B
nohup vllm serve Qwen/Qwen3.5-9B \
  --port 8001 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gdn-prefill-backend triton \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  > vllm.log 2>&1 &

# Llama 3.1-8B-Instruct
nohup vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 8001 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  > vllm.log 2>&1 &

# PsyLLM-8B
nohup vllm serve GMLHUHE/PsyLLM-8B \
  --port 8001 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  > vllm.log 2>&1 &

# EmoLLaMa-chat-13B
nohup vllm serve lzw1008/Emollama-chat-13b \
  --port 8001 \
  --tensor-parallel-size 1 \
  --max-model-len 2048 \
  --chat-template llama2.jinja \
  > vllm.log 2>&1 &
```

Some model repositories require users to accept their licenses and authenticate with Hugging Face before downloading weights. No access credentials are included in this repository.

Verify the server before running an experiment:

```bash
curl http://localhost:8001/v1/models
```

In a second shell on the same server, activate `carebench` before launching an experiment. The `*_vllm.py` clients accept the server root through `--base_url` and append `/v1/chat/completions` internally. Use:

```bash
--base_url http://localhost:8001
```

## Quick Start

All commands below are run from the repository root.

### API inference

The following example evaluates all first-person tasks using an OpenAI model:

```bash
python scripts/baseline_api.py \
  --provider openai \
  --model gpt-5.2 \
  --task all \
  --source_folder data/first_person \
  --target_folder output/first_person/baseline/gpt-5.2 \
  --verbose false
```

To run a single sample before launching a full experiment, replace `--source_folder` with `--single_file path/to/sample.json`.

Existing per-sample prediction files are skipped automatically, so interrupted inference runs can be resumed with the same command.

### Local-model inference with vLLM

Start a vLLM server with an OpenAI-compatible API, then run the corresponding client:

```bash
python scripts/baseline_vllm.py \
  --model Qwen/Qwen3.5-9B \
  --task all \
  --source_folder data/first_person \
  --target_folder output/first_person/baseline/Qwen-3.5-9B/run_1 \
  --base_url http://localhost:8001 \
  --num_threads 10 \
  --max_tokens 2048 \
  --temperature 0.2 \
  --verbose false
```

The model name passed to `--model` must match the model exposed by the vLLM server.

### Evaluation

Evaluate a completed prediction directory against first-person annotations:

```bash
python scripts/evaluate.py \
  --gold_folder data/first_person \
  --pred_folder output/first_person/baseline/gpt-5.2 \
  --prompt_path scripts/prompts/baseline_prompt.toml \
  --output_file results.json
```

This writes:

```text
output/first_person/baseline/gpt-5.2/results.json
```

## Experiment Scripts

| Experiment | API script | vLLM script | Evaluation |
|---|---|---|---|
| Situation-only baseline | `scripts/baseline_api.py` | `scripts/baseline_vllm.py` | `scripts/evaluate.py` |
| Appraisal-rating conditioning | `scripts/baseline_api_with_appraisals.py` | `scripts/baseline_vllm_with_appraisals.py` | `scripts/evaluate.py` |
| Appraisal-reasoning conditioning | `scripts/baseline_api_with_cog_story.py` | `scripts/baseline_vllm_with_cog_story.py` | `scripts/evaluate.py` |
| Reasoning and rating conditioning | `scripts/baseline_api_with_cog_story_and_appraisals.py` | `scripts/baseline_vllm_with_cog_story_and_appraisals.py` | `scripts/evaluate.py` |
| Model-generated reasoning | `scripts/baseline_api_with_pred_cog_story.py` | `scripts/baseline_vllm_with_pred_cog_story.py` | `scripts/evaluate.py` |
| First-person single-dimension reasoning | -- | `scripts/baseline_vllm_with_single_core_appraisal.py` | `scripts/evaluate.py` |
| Model-generated single-dimension reasoning | -- | `scripts/baseline_vllm_with_pred_single_core_appraisal.py` | `scripts/evaluate.py` |
| Appraisal intervention | `scripts/counterfactual_api.py` | `scripts/counterfactual_vllm.py` | `scripts/evaluate_counterfactual.py` |
| Emotion intervention | `scripts/counterfactual_emotion_api.py` | `scripts/counterfactual_emotion_vllm.py` | `scripts/evaluate_counterfactual_emotion.py` |
| Human-observer baseline | -- | -- | `scripts/evaluate_human.py` |
| Subjectivity modeling | -- | `scripts/subjectivity_vllm.py` | `scripts/evaluate_subjectivity.py` |

`scripts/evaluate_per_sample.py` produces sample-level emotion results. `scripts/summarise_pred_cog_stories.py` converts model-generated dimension-specific appraisal answers into the `final_scenario` input used by the model-generated-reasoning condition.

Each script begins with a complete example command and documents its expected input and output paths.

After the corresponding predictions have been generated, run the observer and intervention evaluations as follows:

```bash
# Third-person observer annotations against the matched first-person annotations
python scripts/evaluate_human.py \
  --third_person_folder data/third_person \
  --gold_folder data/first_person_500 \
  --output_dir output/human_eval

# RQ3 appraisal intervention
python scripts/evaluate_counterfactual.py \
  --model Qwen-3.5-9B

# RQ3 emotion intervention
python scripts/evaluate_counterfactual_emotion.py \
  --model Qwen-3.5-9B
```

RQ4 evaluates four conditions. Each command produces five predictions per narrative at temperature 0.7:

```bash
for task in \
  appraisals \
  appraisals-persona \
  appraisals-third-person-scenario \
  appraisals-persona-third-person-scenario
do
  python scripts/subjectivity_vllm.py \
    --model Qwen/Qwen3.5-9B \
    --task "$task" \
    --source_folder data/first_person_500 \
    --third_person_folder data/third_person \
    --target_folder output/subjectivity/Qwen-3.5-9B \
    --base_url http://localhost:8001 \
    --temperature 0.7 \
    --num_samples 5 \
    --max_concurrency 10 \
    --verbose false
done

python scripts/evaluate_subjectivity.py \
  --model_folder output/subjectivity/Qwen-3.5-9B \
  --gold_folder data/third_person
```

The four task directories correspond to repeated sampling, persona conditioning, appraisal-reasoning conditioning, and their combination. `scripts/evaluate_subjectivity.py` reports Mean MAE, Variance MAE, Wasserstein distance, Spearman correlation, and Precision@1/3/5/10 in `output/subjectivity/<model>/results.json`.

## Experimental Settings

The paper uses zero-shot prompting and evaluates each task independently. For RQ1--RQ3, the sampling temperature is 0.2; API models are evaluated once, while locally served models are evaluated over five independent runs and their results are averaged. For RQ4, each condition produces five predictions per scenario with a temperature of 0.7. Extended thinking is disabled for models that support it.

Prompt templates, appraisal statements, label maps, and output constraints are stored under `scripts/prompts/`. Command-line arguments override model, input, output, and decoding settings without changing these templates.

## Figure Generation

The visualization scripts read the prepared dataset or the evaluation results under `output/` and write PDFs to `figures/`:

```bash
# Dataset statistics
python vis/plot_data_statistics.py \
  --data-root data \
  --output-dir figures

# RQ2 input effects
python vis/plot_input_effects.py \
  --results-root output/first_person \
  --output-dir figures

# RQ3 appraisal and emotion interventions
python vis/plot_intervention_results.py \
  --results-root output/first_person \
  --output-dir figures

# Manual human evaluation
python vis/plot_human_evaluation.py \
  --human-eval-dir data/human_eval \
  --output-dir figures
```

`plot_human_evaluation.py` expects the two manually annotated evaluation batches under `data/human_eval/batch_1/` and `data/human_eval/batch_3/`. The reference PDFs generated for the paper are included in `figures/`.

## Outputs

The main scripts use the following output layout:

```text
output/
├── first_person/
│   ├── baseline*/                 # predictions and aggregate task metrics
│   ├── counterfactual/            # appraisal-intervention predictions and correlations
│   └── counterfactual_emotion/    # emotion-intervention predictions and correlations
├── subjectivity/
│   └── <model>/
│       ├── appraisals*/           # five predictions per narrative and condition
│       └── results.json           # distribution and disagreement metrics
└── human_eval/
    └── results.json              # first-person versus observer evaluation
```

Inference scripts write one JSON prediction per sample and task. `scripts/evaluate.py` writes aggregate metrics to `results.json`, while `scripts/evaluate_per_sample.py` writes `results_samples.json` inside the selected prediction directory. The two intervention evaluators write per-dimension correlation results inside the corresponding model directory.

## Citation

```bibtex
@inproceedings{sun2026carebench,
  title     = {{CAREBench}: Evaluating {LLM}s' Emotion Understanding by Assessing Cognitive Appraisal Reasoning},
  author    = {Sun, Zhaoyue and Xu, Hainiu and Uusberg, Andero and Gross, James J. and Slovak, Petr and He, Yulan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

The code in this repository is released under the [MIT License](LICENSE). The CAREBench dataset is distributed separately under the CC BY-NC-ND 4.0 license.
