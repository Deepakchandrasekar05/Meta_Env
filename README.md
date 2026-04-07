# Meta Ads Attribution OpenEnv Environment

A production-style reinforcement learning environment for Meta Ads optimization under attribution uncertainty.

## Project Layout

- `meta_ads_env/`: environment, simulator, reward logic, typed models, graders.
- `data/`: easy/medium/hard task pools.
- `baseline/`: baseline policy and runner.
- `evaluation/`: metrics and optional LLM-based grader.
- `app/`: Gradio demo UI for HF Spaces.

## Quickstart

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run baseline episode:

```bash
python baseline/run_baseline.py
```

4. Launch demo UI:

```bash
python app/app.py
```

## Submission Inference (Required)

The required submission script is [inference.py](inference.py) in the repository root.

Set these environment variables before running it:

- `API_BASE_URL` (LLM API endpoint)
- `MODEL_NAME` (model identifier)
- `HF_TOKEN` (API key)

Run:

```bash
python inference.py
```

The script emits strict structured stdout logs in this format:

- `[START] ...`
- `[STEP] ...`
- `[END] ...`

And runs all three tasks end-to-end.

## Pre-Submission Validator

Use the included validator:

```bash
bash scripts/validate-submission.sh <your_space_url> .
```

## OpenEnv Config

Environment wiring is defined in `openenv.yaml`:
- Observation model: `meta_ads_env.models.Observation`
- Action model: `meta_ads_env.models.Action`
- Reward model: `meta_ads_env.models.Reward`
- Entry point: `meta_ads_env.env:MetaAdsAttributionEnv`

## Task Levels

- `easy_attribution_window`
- `medium_pixel_recovery`
- `hard_full_attribution_audit`
