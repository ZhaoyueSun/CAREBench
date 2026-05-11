# Scripts Overview

This directory contains runnable scripts and prompt configs for the following paper:
**CAREBench: Evaluating LLMs' Emotion Understanding by Assessing Cognitive Appraisal Reasoning**


## Scripts

### Inference Scripts:
- baseline_api.py: API-based baseline runner for step-by-step tasks (`appraisals`, emotion levels, emotion labels, `core-appraisals`, or `all`).
- baseline_api_with_appraisals.py: API-based runner that uses appraisal rating contexts to predict emotion tasks (`positive-level`, `negative-level`, `positive-labels`, `negative-labels`).
- baseline_api_with_cog_story.py: API-based runner that conditions on story with core-appraisal context for appraisal/emotion tasks.
- baseline_api_with_cog_story_and_appraisals.py: API-based runner that conditions on both core-appraisal and appraisal signals for emotion-task prediction.
- baseline_api_with_pred_cog_story.py: API-based runner that reads stories with model-generated core-appraisal reasonings (`final_scenario`) and runs appraisal/emotion tasks.
- counterfactual_api.py: API-based counterfactual evaluator that runs appraisal scoring over `data/counterfactual/<dimension>/*.json`.
- counterfactual_emotion_api.py: API-based counterfactual evaluator for emotion tasks only (levels and labels).
- summarise_pred_cog_stories.py: Builds coherent narrative from structured predicted core-appraisal answers.
- baseline_vllm_*.py: corresponding scripts for vLLM runner.


### Evaluation Scripts
- evaluate_counterfactual.py: Counterfactual appraisal evaluator that reports per-dimension correlation between model deltas and human deltas.
- evaluate_counterfactual_emotion.py: Counterfactual emotion evaluator for level/label deltas with per-dimension correlations.
- evaluate_human.py: Human-eval aggregator that compares third-person annotations to first-person gold and writes global evaluation outputs.
- evaluate_per_sample.py: First-person evaluator that outputs per-sample metrics instead of only aggregate summaries.


## Scripts/prompts

- baseline_prompt.toml: Main baseline prompt pack (appraisal statements, emotion levels, emotion labels, and core-appraisal question templates + label maps).
- baseline_with_appraisal_prompt.toml: Prompt pack for runs that inject appraisal context before predicting emotion tasks.
- baseline_with_single_core_appraisal.toml: Prompt pack for single-core-appraisal-conditioned runs plus emotion-task templates.
- counterfactual_prompt.toml: Prompt pack for counterfactual appraisal scoring (dimension statements + appraisal label map).
- summarsie_pred_cog_stories.toml: Prompt template used to shaping structured core-appraisal answers into coherent stories.
