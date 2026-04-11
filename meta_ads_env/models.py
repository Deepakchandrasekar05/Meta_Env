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
    capi_coverage: float = 0.0        # estimated server-side event coverage 0.0-1.0
    aem_enabled: bool               # Aggregated Event Measurement
    utm_tracking: bool
    modeled_conversions_enabled: bool = False
    attribution_reporting_mode: Literal["observed", "modeled"] = "observed"
    server_signal_quality: float = 0.0
    pixel_match_quality: float = 0.0
    conversion_delay_index: float = 1.0
    avg_conversion_delay_days: float = 4.0
    adsets: List[AdSetMetrics] = Field(default_factory=list)


class PendingConversion(BaseModel):
    source_adset_id: str
    clicks: int
    expected_conversions: int
    value: int = 0
    delay_days_remaining: int
    original_delay_days: int


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
    attribution_confidence: float = 0.0
    capi_coverage: float = 0.0
    pixel_match_quality: float = 0.0
    conversion_delay_index: float = 1.0
    avg_conversion_delay_days: float = 4.0
    pending_delayed_conversions: int = 0
    modeled_conversions_accumulated: int = 0
    tracked_conversions_accumulated: int = 0
    delayed_conversion_release_events: int = 0
    cumulative_delayed_conversions: int = 0
    issues_resolved_count: int = 0
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
    "promote_ad",
    "reduce_budget",
    "investigate_attribution",
    "switch_to_modeled_conversions",
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
    timing_quality:       float = 0.0   # decision timing under delayed feedback
    uncertainty_handling: float = 0.0   # investigation / modeled use when needed
    redundancy_penalty:   float = 0.0   # repeated or low-value action penalty
    long_term_gain:       float = 0.0   # forward trajectory quality
    issue_resolution_progress: float = 0.0  # progress toward full issue closure


class Reward(BaseModel):
    total: float = Field(ge=-1.0, le=1.0)
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
    day: int = 0
    action_counts: Dict[str, int] = Field(default_factory=dict)
    pending_delayed_conversions: List[PendingConversion] = Field(default_factory=list)
    pending_conversions: List[PendingConversion] = Field(default_factory=list)
    hidden_delayed_conversions: List[PendingConversion] = Field(default_factory=list)
    delayed_true_conversions_total: int = 0
    delayed_reported_conversions_total: int = 0
    tracked_conversions_total: int = 0
    modeled_conversions_total: int = 0
    growth_momentum: float = 1.0
    tracking_reliability: float = 0.5
    attribution_investigation_level: float = 0.0
    attribution_gap_history: List[float] = Field(default_factory=list)
    roas_history: List[float] = Field(default_factory=list)
    signal_quality_history: List[float] = Field(default_factory=list)
    optimal_steps_hint: int = 4
    optimal_steps: int = 4
    scenario_delay_range: List[int] = Field(default_factory=lambda: [2, 7])
    tracking_investigated: bool = False
    uncertainty_reintroduced: bool = False
    hidden_conversions_pool: int = 0
    delayed_conversion_release_last_step: int = 0
    tracked_conversion_release_last_step: int = 0
    modeled_conversion_release_last_step: int = 0
    convergence_reached: bool = False
    conversion_rate_range: List[float] = Field(default_factory=lambda: [0.08, 0.12])
    max_generated_conversions_per_step: int = 35
    max_released_conversions_per_step: int = 35
    target_true_conversions: int = 250
    delayed_reward_buffer: float = 0.0
    delayed_reward_released_last_step: float = 0.0
    terminal_bonus_last_step: float = 0.0
    risk_events: List[str] = Field(default_factory=list)
    budget_optimization_multiplier: float = 1.0
    confidence_score: float = 0.5
    attribution_confidence: float = 0.5
    episode_risk_initialized: bool = False
    easy_meaningful_actions_taken: int = 0
    reasoning_log: List[str] = Field(default_factory=list)
    early_wrong_decision: bool = False
    recovered_after_wrong_decision: bool = False
    random_seed: int = 42
    last_action_type: str = ""
    repeated_action_count: int = 0
    action_impact_memory: Dict[str, float] = Field(default_factory=dict)
    convergence_stagnation_count: int = 0
    episode_rare_events: List[str] = Field(default_factory=list)