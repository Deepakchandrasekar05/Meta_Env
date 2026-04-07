"""
evaluation/llm_grader.py — LLM-as-judge grader for qualitative scoring.

Scores the agent's REASONING quality on top of the programmatic score.
Uses a rubric to evaluate whether the agent correctly diagnosed the root cause.
"""

from __future__ import annotations
import json
import os
from typing import List

from openai import OpenAI

RUBRIC = """
You are evaluating an AI agent's performance on a Meta Ads attribution recovery task.

Score the agent's trajectory from 0.0 to 1.0 on the following rubric:

1.0 — Agent correctly identified ALL root causes (wrong attribution window, pixel signal loss,
       budget misallocation) and applied the right fixes in a logical order with clear reasoning.

0.75 — Agent identified the primary issue and fixed it, but missed secondary issues or
        applied fixes in a suboptimal order.

0.50 — Agent showed partial understanding of the problem and applied some correct actions,
        but reasoning was vague or steps were redundant.

0.25 — Agent took some valid actions but clearly did not understand the root causes.
        Mixed correct and incorrect reasoning.

0.0  — Agent failed to diagnose any issue correctly. Applied irrelevant or harmful actions.

Return ONLY a JSON object:
{"score": 0.0, "rationale": "one paragraph explanation"}
"""


class LLMGrader:
    def __init__(self, model: str = "gpt-4o"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
        self.model  = model

    def grade_trajectory(
        self,
        task_id: str,
        history: List[dict],
        initial_context: str,
        final_context: str,
    ) -> dict:
        """Score the agent's full trajectory."""

        steps_text = "\n".join(
            f"Step {s['step']}: action={s['action']}, reward={s['reward']:.4f}, effects={s['effects']}"
            for s in history
        )

        prompt = f"""
Task: {task_id}

INITIAL STATE:
{initial_context}

AGENT TRAJECTORY:
{steps_text}

FINAL STATE:
{final_context}

Please evaluate the agent's performance using the rubric.
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": RUBRIC},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        try:
            return json.loads(raw)
        except Exception:
            return {"score": 0.0, "rationale": "Parse error"}