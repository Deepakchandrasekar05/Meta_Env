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

## OpenEnv Config

Environment wiring is defined in `openenv.yaml`:
- Observation model: `meta_ads_env.models.Observation`
- Action model: `meta_ads_env.models.Action`
- Reward model: `meta_ads_env.models.Reward`
- Entry point: `meta_ads_env.env:MetaAdsEnv`

## Task Levels

- `easy`: cleaner signal, lower attribution noise.
- `medium`: mixed signal and moderate delay.
- `hard`: high attribution ambiguity and delayed conversion patterns.
