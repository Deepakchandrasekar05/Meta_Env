"""
baseline_agent.py — LLM-powered baseline agent using OpenAI API.

Uses the environment's natural-language context observation to prompt
an LLM to select the next action, then parses the response back into
an Action model. Demonstrates the full agent loop.
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, Optional

from openai import OpenAI

from meta_ads_env.models import Action

SYSTEM_PROMPT = """
You are an expert Meta Ads strategist and data analyst.
You are operating inside a reinforcement-learning environment that simulates
a real Meta Ads campaign suffering from attribution degradation.

Your goal: maximise the TRUE return on ad spend (ROAS) by fixing the
attribution and signal issues that cause Meta's algorithm to optimise on
incomplete data.

At each step you will receive a natural-language observation describing:
- Campaign metrics (reported vs true conversions, ROAS, CPA)
- Attribution window in use
- Pixel signal quality and iOS traffic percentage
- Which mitigations are already enabled (CAPI, AEM, UTM)
- Adset-level breakdowns
- A list of available actions

You must respond with ONLY a JSON object (no markdown, no explanation) in this format:
{
  "action_type": "<one of the available actions>",
  "parameters": { <action-specific params or empty dict> },
  "reasoning": "<one sentence explaining why>"
}

Available actions and their parameters:
- adjust_attribution_window: {"window": "7d_click" | "28d_click" | "7d_click_1d_view"}
- enable_conversions_api: {}
- enable_aggregated_event_measurement: {}
- add_utm_tracking: {}
- adjust_budget_allocation: {"shifts": {"adset_id": new_budget_usd, ...}}
- pause_underperforming_adsets: {"roas_threshold": 1.0}
- reallocate_to_top_performers: {"amount": 2000}
- change_bid_strategy: {"strategy": "value_optimisation" | "cost_cap"}
- segment_audience: {}
- no_op: {}

Prioritise actions in this order:
1. Fix attribution window if it is 1d_click (too narrow)
2. Enable Conversions API if missing (biggest signal recovery)
3. Enable AEM if CAPI is on but AEM is off
4. Pause adsets with true ROAS < 1.0
5. Reallocate budget to top performers
6. no_op only if all issues are resolved
"""


class BaselineAgent:
    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
        self.model  = model

    def act(self, observation_context: str) -> Action:
        """Given the natural-language observation, return an Action."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": observation_context},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback to no_op if response is malformed
            return Action(action_type="no_op", parameters={}, reasoning="Parse error — defaulting to no_op")

        return Action(
            action_type=parsed.get("action_type", "no_op"),
            parameters=parsed.get("parameters", {}),
            reasoning=parsed.get("reasoning", ""),
        )