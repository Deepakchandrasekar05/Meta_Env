from typing import Literal

from pydantic import BaseModel, Field


class Observation(BaseModel):
    impressions: int = Field(ge=0)
    link_clicks: int = Field(ge=0)
    tracked_conversions: int = Field(ge=0)
    modeled_conversions: int = Field(ge=0)
    cost_per_result: float = Field(ge=0.0)
    days_since_launch: int = Field(ge=0)
    conversion_delay_avg_days: float = Field(ge=0.0)
    ios_traffic_percent: float = Field(ge=0.0, le=100.0)
    pixel_match_quality: float = Field(ge=0.0, le=1.0)
    capi_coverage: float = Field(ge=0.0, le=1.0)
    attribution_window_days: int = Field(ge=1)
    historical_account_performance: float = Field(ge=0.0, le=1.0)
    industry_avg_delay_days: float = Field(ge=0.0)


class Action(BaseModel):
    action: Literal[
        "promote_ad",
        "keep_learning",
        "reduce_budget",
        "investigate_attribution",
        "switch_to_modeled_conversions",
    ]


class Reward(BaseModel):
    reward: float
    reason: str
