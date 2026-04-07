"""
Root inference script required by submission validator.

This script:
- Uses OpenAI client with API_BASE_URL, MODEL_NAME, HF_TOKEN
- Runs all tasks in the environment
- Emits strict structured logs: [START], [STEP], [END]
- Ensures final per-task score is clamped to [0, 1]
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

from openai import OpenAI

from meta_ads_env import MetaAdsAttributionEnv
from meta_ads_env.models import Action
from meta_ads_env.tasks import TASK_REGISTRY


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
BENCHMARK = "meta_ads_attribution_openenv"
MAX_TOKENS = 300
TEMPERATURE = 0.0

VALID_ACTIONS = {
    "adjust_attribution_window",
    "enable_conversions_api",
    "adjust_budget_allocation",
    "change_bid_strategy",
    "add_utm_tracking",
    "segment_audience",
    "enable_aggregated_event_measurement",
    "pause_underperforming_adsets",
    "reallocate_to_top_performers",
    "no_op",
}

ACTION_ALIASES = {
    "extend_attribution_window": "adjust_attribution_window",
    "set_attribution_window": "adjust_attribution_window",
    "update_attribution_window": "adjust_attribution_window",
    "enable_aem": "enable_aggregated_event_measurement",
    "enable_aggregated_event_measure": "enable_aggregated_event_measurement",
    "pause_underperforming": "pause_underperforming_adsets",
    "reallocate_budget_to_top_performers": "reallocate_to_top_performers",
}

SYSTEM_PROMPT = """
You are an expert Meta Ads strategist and data analyst.
You are operating inside a reinforcement-learning environment that simulates
Meta Ads attribution degradation.

Return ONLY JSON (no markdown):
{"action_type": "<available_action>", "parameters": {}, "reasoning": "<one sentence>"}
""".strip()


def _bool_str(value: bool) -> str:
    return str(value).lower()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={_bool_str(done)} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={_bool_str(success)} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _parse_action(raw: str) -> Action:
    text = (raw or "").strip()

    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    text = text.strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return Action(action_type="no_op", parameters={}, reasoning="parse_error")

    action_type = str(payload.get("action_type", "no_op")).strip()
    action_type = ACTION_ALIASES.get(action_type, action_type)

    if action_type not in VALID_ACTIONS:
        action_type = "no_op"

    parameters = payload.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}

    reasoning = payload.get("reasoning", "")
    if reasoning is None:
        reasoning = ""

    try:
        return Action(
            action_type=action_type,
            parameters=parameters,
            reasoning=str(reasoning),
        )
    except Exception:
        return Action(action_type="no_op", parameters={}, reasoning="validation_error")


def _infer_next_action(client: OpenAI, model: str, observation_context: str) -> Action:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": observation_context},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    content = completion.choices[0].message.content or ""
    return _parse_action(content)


def run_task(client: OpenAI, task_id: str) -> int:
    env = MetaAdsAttributionEnv(task_id=task_id)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()

        while not obs.done:
            step_num = steps_taken + 1
            error = None

            try:
                action = _infer_next_action(client, MODEL_NAME, obs.context)
                action_str = action.action_type
                obs, reward, done, _ = env.step(action)
                reward_value = float(reward.total)
            except Exception as exc:
                action_str = "no_op"
                reward_value = 0.0
                done = True
                error = str(exc)

            rewards.append(reward_value)
            steps_taken = step_num
            log_step(step=step_num, action=action_str, reward=reward_value, done=done, error=error)

            if done:
                break

        result = env.grade_episode()
        score = min(max(float(result.score), 0.0), 1.0)
        success = bool(result.passed)

    except Exception:
        # Keep mandatory [END] contract even if task setup fails.
        pass
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return 0


def main() -> int:
    if not API_KEY:
        # Keep behavior explicit for validators/users.
        raise EnvironmentError("HF_TOKEN (or OPENAI_API_KEY/API_KEY) is required for inference")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_id in TASK_REGISTRY:
        rc = run_task(client, task_id)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
