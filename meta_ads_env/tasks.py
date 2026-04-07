"""
tasks.py — Task definitions for Meta Ads Attribution OpenEnv.

Three tasks, each returning an EnvState ready for reset():
  1. easy_attribution_window   — fix 1d_click → 7d_click
  2. medium_pixel_recovery     — recover Pixel signal via CAPI + AEM
  3. hard_full_attribution_audit — multi-layer audit + budget optimisation
"""

from __future__ import annotations
from typing import Dict

from meta_ads_env.models import AdSetMetrics, CampaignData, EnvState
from meta_ads_env.simulator import (
    compute_pixel_quality,
    compute_reported_conversions,
    compute_roas,
)

AVG_ORDER_VALUE = 75.0   # USD; used consistently across all tasks


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — EASY
# Problem: attribution window is 1d_click → misses 62% of real conversions
# Fix:     adjust_attribution_window → 7d_click
# ─────────────────────────────────────────────────────────────────────────────

def make_easy_task() -> EnvState:
    pixel_quality = compute_pixel_quality(
        ios_traffic_pct=0.25,   # moderate iOS share
        conversions_api=False,
        aem_enabled=False,
        utm_tracking=True,      # UTM at least works
    )
    true_conv = 180
    attr_window = "1d_click"
    reported_conv = compute_reported_conversions(true_conv, attr_window, pixel_quality)
    spend = 4500.0

    campaign = CampaignData(
        campaign_id="camp_easy_001",
        campaign_name="Spring Sale — Conversions Campaign",
        objective="CONVERSIONS",
        total_budget=5000.0,
        budget_spent=spend,
        impressions=220_000,
        link_clicks=3850,
        reported_conversions=reported_conv,
        true_conversions=true_conv,
        reported_cpa=round(spend / reported_conv, 2) if reported_conv else 9999,
        true_cpa=round(spend / true_conv, 2),
        reported_roas=compute_roas(reported_conv, AVG_ORDER_VALUE, spend),
        true_roas=compute_roas(true_conv, AVG_ORDER_VALUE, spend),
        attribution_window=attr_window,
        pixel_signal_quality=pixel_quality,
        ios_traffic_pct=0.25,
        conversions_api_enabled=False,
        aem_enabled=False,
        utm_tracking=True,
        adsets=[
            AdSetMetrics(
                adset_id="adset_retargeting",
                adset_name="Retargeting",
                budget=1500.0, spent=1400.0,
                impressions=55_000, link_clicks=1100,
                reported_conversions=12, true_conversions=32,
                reported_roas=compute_roas(12, AVG_ORDER_VALUE, 1400),
                true_roas=compute_roas(32, AVG_ORDER_VALUE, 1400),
                audience_segment="retargeting",
            ),
            AdSetMetrics(
                adset_id="adset_lookalike",
                adset_name="Lookalike 1%",
                budget=2250.0, spent=2100.0,
                impressions=110_000, link_clicks=2000,
                reported_conversions=25, true_conversions=92,
                reported_roas=compute_roas(25, AVG_ORDER_VALUE, 2100),
                true_roas=compute_roas(92, AVG_ORDER_VALUE, 2100),
                audience_segment="lookalike_1pct",
            ),
            AdSetMetrics(
                adset_id="adset_broad",
                adset_name="Broad Interest",
                budget=1250.0, spent=1000.0,
                impressions=55_000, link_clicks=750,
                reported_conversions=8, true_conversions=56,
                reported_roas=compute_roas(8, AVG_ORDER_VALUE, 1000),
                true_roas=compute_roas(56, AVG_ORDER_VALUE, 1000),
                audience_segment="broad_interest",
            ),
        ],
    )

    return EnvState(
        task_id="easy_attribution_window",
        difficulty="easy",
        step_count=0,
        max_steps=5,
        campaign=campaign,
        issues_remaining=["attribution_window"],
        issues_resolved=[],
    )


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — MEDIUM
# Problem: heavy iOS traffic (55%) + no CAPI + no AEM → pixel signal is ~33%
# Fix:     enable_conversions_api + enable_aggregated_event_measurement
# ─────────────────────────────────────────────────────────────────────────────

def make_medium_task() -> EnvState:
    ios_pct = 0.55
    pixel_quality = compute_pixel_quality(
        ios_traffic_pct=ios_pct,
        conversions_api=False,
        aem_enabled=False,
        utm_tracking=False,
    )
    true_conv = 240
    attr_window = "7d_click"
    reported_conv = compute_reported_conversions(true_conv, attr_window, pixel_quality)
    spend = 8_200.0

    campaign = CampaignData(
        campaign_id="camp_medium_001",
        campaign_name="iOS-Heavy Apparel Campaign",
        objective="CONVERSIONS",
        total_budget=9_000.0,
        budget_spent=spend,
        impressions=380_000,
        link_clicks=7_600,
        reported_conversions=reported_conv,
        true_conversions=true_conv,
        reported_cpa=round(spend / reported_conv, 2) if reported_conv else 9999,
        true_cpa=round(spend / true_conv, 2),
        reported_roas=compute_roas(reported_conv, AVG_ORDER_VALUE, spend),
        true_roas=compute_roas(true_conv, AVG_ORDER_VALUE, spend),
        attribution_window=attr_window,
        pixel_signal_quality=pixel_quality,
        ios_traffic_pct=ios_pct,
        conversions_api_enabled=False,
        aem_enabled=False,
        utm_tracking=False,
        adsets=[
            AdSetMetrics(
                adset_id="adset_retargeting",
                adset_name="Retargeting (iOS heavy)",
                budget=2_700.0, spent=2_500.0,
                impressions=90_000, link_clicks=2_000,
                reported_conversions=int(60 * pixel_quality),
                true_conversions=60,
                reported_roas=compute_roas(int(60 * pixel_quality), AVG_ORDER_VALUE, 2_500),
                true_roas=compute_roas(60, AVG_ORDER_VALUE, 2_500),
                audience_segment="retargeting",
            ),
            AdSetMetrics(
                adset_id="adset_lookalike",
                adset_name="Lookalike 2%",
                budget=4_050.0, spent=3_800.0,
                impressions=200_000, link_clicks=4_000,
                reported_conversions=int(120 * pixel_quality),
                true_conversions=120,
                reported_roas=compute_roas(int(120 * pixel_quality), AVG_ORDER_VALUE, 3_800),
                true_roas=compute_roas(120, AVG_ORDER_VALUE, 3_800),
                audience_segment="lookalike_2pct",
            ),
            AdSetMetrics(
                adset_id="adset_broad",
                adset_name="Broad Interest",
                budget=2_250.0, spent=1_900.0,
                impressions=90_000, link_clicks=1_600,
                reported_conversions=int(60 * pixel_quality),
                true_conversions=60,
                reported_roas=compute_roas(int(60 * pixel_quality), AVG_ORDER_VALUE, 1_900),
                true_roas=compute_roas(60, AVG_ORDER_VALUE, 1_900),
                audience_segment="broad_interest",
            ),
        ],
    )

    return EnvState(
        task_id="medium_pixel_recovery",
        difficulty="medium",
        step_count=0,
        max_steps=7,
        campaign=campaign,
        issues_remaining=["conversions_api", "aem"],
        issues_resolved=[],
    )


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — HARD
# Problems: 1d_click window + high iOS (60%) + no CAPI/AEM + bad budget split
# Fix:      all of the above + pause low-ROAS adset + reallocate budget
# ─────────────────────────────────────────────────────────────────────────────

def make_hard_task() -> EnvState:
    ios_pct = 0.60
    pixel_quality = compute_pixel_quality(
        ios_traffic_pct=ios_pct,
        conversions_api=False,
        aem_enabled=False,
        utm_tracking=False,
    )
    true_conv = 310
    attr_window = "1d_click"
    reported_conv = compute_reported_conversions(true_conv, attr_window, pixel_quality)
    spend = 14_500.0

    campaign = CampaignData(
        campaign_id="camp_hard_001",
        campaign_name="Q4 Multi-Product Full Funnel",
        objective="CONVERSIONS",
        total_budget=16_000.0,
        budget_spent=spend,
        impressions=620_000,
        link_clicks=12_400,
        reported_conversions=reported_conv,
        true_conversions=true_conv,
        reported_cpa=round(spend / reported_conv, 2) if reported_conv else 9999,
        true_cpa=round(spend / true_conv, 2),
        reported_roas=compute_roas(reported_conv, AVG_ORDER_VALUE, spend),
        true_roas=compute_roas(true_conv, AVG_ORDER_VALUE, spend),
        attribution_window=attr_window,
        pixel_signal_quality=pixel_quality,
        ios_traffic_pct=ios_pct,
        conversions_api_enabled=False,
        aem_enabled=False,
        utm_tracking=False,
        adsets=[
            AdSetMetrics(
                adset_id="adset_retargeting",
                adset_name="Retargeting (warm)",
                budget=2_400.0, spent=2_200.0,
                impressions=70_000, link_clicks=1_800,
                reported_conversions=int(80 * 0.38 * pixel_quality),
                true_conversions=80,
                reported_roas=compute_roas(int(80 * 0.38 * pixel_quality), AVG_ORDER_VALUE, 2_200),
                true_roas=compute_roas(80, AVG_ORDER_VALUE, 2_200),
                audience_segment="retargeting",
            ),
            AdSetMetrics(
                adset_id="adset_lookalike",
                adset_name="Lookalike 1%",
                budget=3_600.0, spent=3_400.0,
                impressions=160_000, link_clicks=5_000,
                reported_conversions=int(130 * 0.38 * pixel_quality),
                true_conversions=130,
                reported_roas=compute_roas(int(130 * 0.38 * pixel_quality), AVG_ORDER_VALUE, 3_400),
                true_roas=compute_roas(130, AVG_ORDER_VALUE, 3_400),
                audience_segment="lookalike_1pct",
            ),
            AdSetMetrics(
                adset_id="adset_broad",
                adset_name="Broad Interest (wasted)",
                budget=6_500.0, spent=6_200.0,   # OVER-budgeted, low ROAS
                impressions=280_000, link_clicks=4_200,
                reported_conversions=int(60 * 0.38 * pixel_quality),
                true_conversions=60,
                reported_roas=compute_roas(int(60 * 0.38 * pixel_quality), AVG_ORDER_VALUE, 6_200),
                true_roas=compute_roas(60, AVG_ORDER_VALUE, 6_200),
                audience_segment="broad_interest",
            ),
            AdSetMetrics(
                adset_id="adset_interest",
                adset_name="Interest Targeting",
                budget=3_500.0, spent=2_700.0,
                impressions=110_000, link_clicks=1_400,
                reported_conversions=int(40 * 0.38 * pixel_quality),
                true_conversions=40,
                reported_roas=compute_roas(int(40 * 0.38 * pixel_quality), AVG_ORDER_VALUE, 2_700),
                true_roas=compute_roas(40, AVG_ORDER_VALUE, 2_700),
                audience_segment="interest",
            ),
        ],
    )

    return EnvState(
        task_id="hard_full_attribution_audit",
        difficulty="hard",
        step_count=0,
        max_steps=10,
        campaign=campaign,
        issues_remaining=[
            "attribution_window",
            "conversions_api",
            "aem",
            "budget_allocation",
            "paused_bad_adsets",
        ],
        issues_resolved=[],
    )


# ─── Registry ────────────────────────────────────────────────────────────────

TASK_REGISTRY: Dict[str, callable] = {
    "easy_attribution_window":    make_easy_task,
    "medium_pixel_recovery":      make_medium_task,
    "hard_full_attribution_audit": make_hard_task,
}


def get_task(task_id: str) -> EnvState:
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{task_id}'. Valid: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[task_id]()