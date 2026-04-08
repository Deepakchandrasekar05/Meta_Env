"""
baseline_agent.py — LLM-powered baseline agent using OpenAI API.

Uses the environment's natural-language context observation to prompt
an LLM to select the next action, then parses the response back into
an Action model. Demonstrates the full agent loop.
"""

from __future__ import annotations
import json
import os
import re
from typing import Optional

from openai import OpenAI

from meta_ads_env.models import Action

SYSTEM_PROMPT = """
You are an expert Meta Ads strategist and data analyst.
You are operating inside a reinforcement-learning environment that simulates
a real Meta Ads campaign suffering from attribution degradation.

Your goal: maximise the TRUE return on ad spend (ROAS) by fixing the
attribution and signal issues that cause Meta's algorithm to optimise on
incomplete data, AND by optimizing budget allocation.

At each step you will receive a natural-language observation describing:
- Campaign metrics (reported vs true conversions, ROAS, CPA)
- Attribution window in use
- Pixel signal quality and iOS traffic percentage
- Which mitigations are already enabled (CAPI, AEM, UTM)
- Adset-level breakdowns with individual ROAS
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
1. Fix attribution window if it is 1d_click (too narrow) - use 7d_click or 28d_click
2. Enable Conversions API if missing (biggest signal recovery) - check if iOS >40%
3. Enable AEM if CAPI is on but AEM is off (additional iOS recovery)
4. Pause adsets with true ROAS < 1.0 (they lose money) - check adset-level true_roas
5. Reallocate budget to top performers with true ROAS > 2.5x
6. no_op ONLY if ALL issues are resolved and ALL adsets are profitable

IMPORTANT: Do NOT use no_op until you've checked ALL adset-level true_roas values 
and paused/reallocated any underperforming ones!
"""


class BaselineAgent:
    def __init__(self, model: str | None = None):
        api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("API_BASE_URL")
        # Keep baseline deterministic/offline by default; opt in via BASELINE_USE_LLM=true.
        self.use_llm = os.environ.get("BASELINE_USE_LLM", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.client = OpenAI(api_key=api_key, base_url=base_url) if (api_key and self.use_llm) else None
        self.model = model or os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
        self.action_history: list[str] = []
        self.reallocation_count = 0
        self.last_gap: float | None = None

    def _parse_state(self, context: str) -> dict:
        def has(pattern: str) -> bool:
            return re.search(pattern, context, flags=re.IGNORECASE) is not None

        def extract_float(pattern: str, default: float = 0.0) -> float:
            match = re.search(pattern, context, flags=re.IGNORECASE)
            return float(match.group(1)) if match else default

        issues_remaining = ""
        match = re.search(r"Issues remaining:\s*(.*)", context)
        if match:
            issues_remaining = match.group(1)

        adset_roas_pairs: list[tuple[float, float]] = []
        for m in re.finditer(
            r"\(ACTIVE\): .*?Reported ROAS: ([0-9]+\.?[0-9]*)x \| True ROAS: ([0-9]+\.?[0-9]*)x",
            context,
            flags=re.IGNORECASE,
        ):
            adset_roas_pairs.append((float(m.group(1)), float(m.group(2))))

        underperformer_count = sum(1 for rep, tru in adset_roas_pairs if rep < 1.0 and tru < 1.1)

        return {
            "step": int(extract_float(r"Step\s+(\d+)/", 0)),
            "max_steps": int(extract_float(r"Step\s+\d+/(\d+)", 10)),
            "window_1d": has(r"Attribution window:\s*1d_click"),
            "capi_on": has(r"Conversions API:\s*ON"),
            "aem_on": has(r"AEM:\s*ON"),
            "utm_on": has(r"UTM:\s*ON"),
            "modeled": has(r"Reporting mode:\s*modeled"),
            "tracking_investigated": ("tracking_investigated" not in issues_remaining),
            "uncertainty_reintroduced": has(r"Uncertainty reintroduced:\s*YES"),
            "tracking_reliability": extract_float(r"Tracking reliability .*?:\s*(\d+)%", 50.0) / 100.0,
            "gap": extract_float(r"Attribution gap:\s*(\d+\.?\d*)%", 0.0) / 100.0,
            "pending_events": int(extract_float(r"Pending delayed conversions:\s*(\d+)", 0)),
            "released_this_step": int(extract_float(r"Delayed conversions released this step:\s*(\d+)", 0)),
            "issues_remaining": issues_remaining,
            "needs_pause_fix": "paused_bad_adsets" in issues_remaining,
            "underperformer_count": underperformer_count,
            "has_underperformer": underperformer_count > 0,
        }

    def _already_used_too_often(self, action_type: str, cap: int = 2) -> bool:
        return self.action_history.count(action_type) >= cap

    def _next_planned_action(self, state: dict) -> Action:
        if state["step"] == 0:
            self.action_history = []
            self.reallocation_count = 0
            self.last_gap = state["gap"]

        gap_improvement = 0.0
        if self.last_gap is not None:
            gap_improvement = self.last_gap - state["gap"]
        self.last_gap = state["gap"]

        resolved_all = "[]" in state["issues_remaining"]
        near_converged = state["gap"] <= 0.06 and state["pending_events"] <= 1
        low_marginal_gain = state["step"] >= 3 and gap_improvement < 0.01
        low_release_signal = state["released_this_step"] <= 1
        core_unresolved = any(
            token in state["issues_remaining"]
            for token in [
                "attribution_window",
                "conversions_api",
                "aem",
                "modeled_reporting",
                "paused_bad_adsets",
                "budget_allocation",
            ]
        )

        if not core_unresolved and state["released_this_step"] <= 1:
            return Action(action_type="no_op", parameters={}, reasoning="Core fixes complete and delayed gains saturated")

        if resolved_all or near_converged or (low_marginal_gain and low_release_signal and not core_unresolved):
            return Action(action_type="no_op", parameters={}, reasoning="Converged or marginal gains are exhausted")

        # 1) Investigate uncertainty first if reliability is weak.
        if (
            state["tracking_reliability"] < 0.70
            and (
                (not state["tracking_investigated"] and not self._already_used_too_often("investigate_attribution", 1))
                or (state["uncertainty_reintroduced"] and not self._already_used_too_often("investigate_attribution", 2))
            )
        ):
            return Action(
                action_type="investigate_attribution",
                parameters={},
                reasoning="Investigate tracking first to reduce uncertainty before optimization",
            )

        # 2) Fix attribution window before scaling decisions.
        if state["window_1d"]:
            return Action(
                action_type="adjust_attribution_window",
                parameters={"window": "7d_click"},
                reasoning="Move from 1d_click to 7d_click to capture delayed conversions",
            )

        # 3) Recover tracking stack before budget optimization.
        if not state["capi_on"]:
            return Action(
                action_type="enable_conversions_api",
                parameters={},
                reasoning="Enable CAPI to improve server-side recoverability under iOS constraints",
            )
        if state["capi_on"] and not state["aem_on"]:
            return Action(
                action_type="enable_aggregated_event_measurement",
                parameters={},
                reasoning="Enable AEM to improve modeled attribution under privacy limits",
            )

        # 4) Switch reporting mode when uncertainty remains high with pending delayed events.
        if (state["pending_events"] > 0 or state["gap"] > 0.35) and not state["modeled"]:
            return Action(
                action_type="switch_to_modeled_conversions",
                parameters={},
                reasoning="Use modeled reporting to interpret lagged and partially observed conversions",
            )

        # 5) Stabilize by pausing clear losers before scaling.
        if (
            (state["has_underperformer"] or state["needs_pause_fix"])
            and not self._already_used_too_often("pause_underperforming_adsets", 2)
        ):
            return Action(
                action_type="pause_underperforming_adsets",
                parameters={"roas_threshold": 1.4},
                reasoning="Pause adsets with both low reported and low true ROAS before scaling",
            )

        # 6) Controlled reallocation once per episode.
        if self.reallocation_count < 1 and not state["has_underperformer"] and not self._already_used_too_often("reallocate_to_top_performers", 1):
            self.reallocation_count += 1
            return Action(
                action_type="reallocate_to_top_performers",
                parameters={"amount": 1000.0},
                reasoning="Single strategic budget shift to top performers",
            )

        # 7) Promote only after attribution stack is stabilized.
        if (
            state["capi_on"]
            and state["aem_on"]
            and state["modeled"]
            and not state["has_underperformer"]
            and not self._already_used_too_often("promote_ad", 2)
            and not (state["step"] >= 6 and low_release_signal)
        ):
            return Action(
                action_type="promote_ad",
                parameters={},
                reasoning="Scale after tracking quality and attribution interpretation are stabilized",
            )

        # Fallback action that is low risk and non-redundant.
        return Action(action_type="add_utm_tracking", parameters={}, reasoning="Low-risk attribution observability improvement")

    def _llm_action(self, observation_context: str) -> Optional[Action]:
        if self.client is None:
            return None
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": observation_context},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)
            return Action(
                action_type=parsed.get("action_type", "no_op"),
                parameters=parsed.get("parameters", {}),
                reasoning=parsed.get("reasoning", ""),
            )
        except Exception:
            # Any provider/network/quota/parse failure silently falls back to planner.
            return None

    def act(self, observation_context: str) -> Action:
        """Given the natural-language observation, return an Action."""
        state = self._parse_state(observation_context)
        planned = self._next_planned_action(state)

        # Optionally accept LLM output only if it does not violate anti-repetition constraints.
        llm_action = self._llm_action(observation_context)
        if llm_action is not None and not self._already_used_too_often(llm_action.action_type, cap=3):
            action = llm_action
        else:
            action = planned

        self.action_history.append(action.action_type)
        return action