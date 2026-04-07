"""
models.py — Typed Pydantic models for Meta Ads Attribution OpenEnv.
Full OpenEnv spec: Observation, Action, Reward, State.
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# Sub-models
# ─────────────────────────────────────────────────────────────

class AdSetMetrics(BaseModel):
    adset_id: str
    adset_name: str
    budget: float
    spent: float
    impressions: int
    link_clicks: int
    reported_conversions: int
    true_conversions: int
    reported_roas: float
    true_roas: float
    audience_segment: str           # e.g. "retargeting", "lookalike", "broad"
    is_paused: bool = False


class CampaignData(BaseModel):
    campaign_id: str
    campaign_name: str
    objective: str                  # "CONVERSIONS" | "TRAFFIC" | "AWARENESS"
    total_budget: float
    budget_spent: float
    impressions: int
    link_clicks: int
    reported_conversions: int
    true_conversions: int
    reported_cpa: float
    true_cpa: float
    reported_roas: float
    true_roas: float
    attribution_window: str         # "1d_click" | "7d_click" | "28d_click" | "1d_view"
    pixel_signal_quality: float     # 0.0–1.0
    ios_traffic_pct: float          # 0.0–1.0
    conversions_api_enabled: bool
    aem_enabled: bool               # Aggregated Event Measurement
    utm_tracking: bool
    adsets: List[AdSetMetrics] = []


# ─────────────────────────────────────────────────────────────
# Observation
# ─────────────────────────────────────────────────────────────

class Observation(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    step_count: int
    max_steps: int
    campaign_data: CampaignData
    reported_conversions: int
    estimated_true_conversions: int
    attribution_gap_pct: float      # (true - reported) / true  → 0 means perfect
    pixel_signal_quality: float
    ios_traffic_pct: float
    budget_remaining: float
    roas_reported: float
    roas_true: float
    available_actions: List[str]
    context: str                    # natural-language summary for LLM agents
    done: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "easy_attribution_window",
                "difficulty": "easy",
                "step_count": 0,
                "max_steps": 5,
                "reported_conversions": 42,
                "estimated_true_conversions": 110,
                "attribution_gap_pct": 0.618,
                "pixel_signal_quality": 0.85,
                "ios_traffic_pct": 0.30,
                "roas_reported": 1.4,
                "roas_true": 3.7,
            }
        }


# ─────────────────────────────────────────────────────────────
# Action
# ─────────────────────────────────────────────────────────────

ActionType = Literal[
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
]

VALID_ATTRIBUTION_WINDOWS = ["1d_click", "7d_click", "28d_click", "1d_view", "7d_click_1d_view"]
VALID_BID_STRATEGIES = ["lowest_cost", "cost_cap", "bid_cap", "value_optimisation"]


class Action(BaseModel):
    action_type: ActionType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    reasoning: Optional[str] = None  # agent can explain its choice (used by LLM grader)

    class Config:
        json_schema_extra = {
            "example": {
                "action_type": "adjust_attribution_window",
                "parameters": {"window": "7d_click"},
                "reasoning": "The current 1-day window is too narrow and misses purchases that happen 2–7 days after the click.",
            }
        }


# ─────────────────────────────────────────────────────────────
# Reward
# ─────────────────────────────────────────────────────────────

class RewardComponents(BaseModel):
    attribution_accuracy: float = 0.0   # 0–0.35  improvement in gap closure
    roas_improvement:     float = 0.0   # 0–0.25
    signal_quality_gain:  float = 0.0   # 0–0.25
    action_validity:      float = 0.0   # 0–0.10  correct action for context
    step_efficiency:      float = 0.0   # 0–0.05  bonus for fewer steps


class Reward(BaseModel):
    total: float = Field(ge=0.0, le=1.0)
    components: RewardComponents
    explanation: str = ""


# ─────────────────────────────────────────────────────────────
# Environment State (internal)
# ─────────────────────────────────────────────────────────────

class EnvState(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    step_count: int = 0
    max_steps: int = 10
    campaign: CampaignData
    cumulative_reward: float = 0.0
    done: bool = False
    history: List[Dict[str, Any]] = Field(default_factory=list)
    issues_resolved: List[str] = Field(default_factory=list)
    issues_remaining: List[str] = Field(default_factory=list)