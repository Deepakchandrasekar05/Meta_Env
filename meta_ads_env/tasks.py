"""
tasks.py — Task definitions for Meta Ads Attribution OpenEnv.

Three tasks, each returning an EnvState ready for reset():
  1. easy_attribution_window   — fix 1d_click → 7d_click
  2. medium_pixel_recovery     — recover Pixel signal via CAPI + AEM
  3. hard_full_attribution_audit — multi-layer audit + budget optimisation
"""

from __future__ import annotations
import random
from typing import Dict

from meta_ads_env.models import AdSetMetrics, CampaignData, EnvState, PendingConversion
from meta_ads_env.simulator import (
    compute_pixel_quality,
    compute_reported_conversions,
    compute_server_signal_quality,
    compute_tracking_reliability,
    compute_roas,
)

AVG_ORDER_VALUE = 75.0   # USD; used consistently across all tasks


def _build_hidden_delayed_events(total_hidden: int, adset_ids: list[str], min_delay: int = 2, max_delay: int = 7) -> list[PendingConversion]:
    if total_hidden <= 0 or not adset_ids:
        return []
    span = max(max_delay - min_delay + 1, 1)
    events: list[PendingConversion] = []
    remaining = total_hidden
    bucket_idx = 0
    while remaining > 0:
        delay = min_delay + (bucket_idx % span)
        adset_id = adset_ids[bucket_idx % len(adset_ids)]
        chunk = min(remaining, max(3, total_hidden // (span * 2)))
        events.append(
            PendingConversion(
                source_adset_id=adset_id,
                clicks=chunk,
                expected_conversions=chunk,
                value=chunk,
                delay_days_remaining=0,
                original_delay_days=delay,
            )
        )
        remaining -= chunk
        bucket_idx += 1
    return events


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — EASY
# Problem: attribution window is 1d_click → misses 62% of real conversions
# Fix:     adjust_attribution_window → 7d_click
# ─────────────────────────────────────────────────────────────────────────────

def make_easy_task() -> EnvState:
    rng = random.Random()
    ios_pct = min(max(0.25 + rng.uniform(-0.025, 0.025), 0.15), 0.40)
    pixel_quality = compute_pixel_quality(
        ios_traffic_pct=ios_pct,   # moderate iOS share
        conversions_api=False,
        aem_enabled=False,
        utm_tracking=True,      # UTM at least works
    )
    true_conv = 180
    attr_window = "1d_click"
    reported_conv = compute_reported_conversions(true_conv, attr_window, pixel_quality)
    severity = min(max(rng.uniform(0.92, 1.08), 0.85), 1.15)
    reported_conv = max(int(round(reported_conv * (2.0 - severity))), 1)
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
        ios_traffic_pct=ios_pct,
        conversions_api_enabled=False,
        aem_enabled=False,
        utm_tracking=True,
        modeled_conversions_enabled=False,
        attribution_reporting_mode="observed",
        server_signal_quality=compute_server_signal_quality(False, False, True),
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

    tracking_rel = max(compute_tracking_reliability(campaign, investigation_level=0.0), 0.68)

    return EnvState(
        task_id="easy_attribution_window",
        difficulty="easy",
        step_count=0,
        max_steps=5,
        campaign=campaign,
        issues_remaining=[
            "attribution_window",
            "tracking_investigated",
        ],
        issues_resolved=[],
        day=0,
        growth_momentum=1.05,
        tracking_reliability=tracking_rel,
        attribution_investigation_level=0.0,
        optimal_steps_hint=3,
        scenario_delay_range=[2, 3],
        hidden_conversions_pool=max(true_conv - reported_conv, 0),
        conversion_rate_range=[0.08, 0.12],
        max_generated_conversions_per_step=34,
        max_released_conversions_per_step=40,
        target_true_conversions=260,
        hidden_delayed_conversions=_build_hidden_delayed_events(
            max(true_conv - reported_conv, 0),
            [a.adset_id for a in campaign.adsets],
            2,
            7,
        ),
        attribution_gap_history=[(true_conv - reported_conv) / true_conv],
        roas_history=[campaign.reported_roas],
        signal_quality_history=[tracking_rel],
    )


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — MEDIUM
# Problem: heavy iOS traffic (55%) + no CAPI + no AEM → pixel signal is ~33%
# Fix:     enable_conversions_api + enable_aggregated_event_measurement
# ─────────────────────────────────────────────────────────────────────────────

def make_medium_task() -> EnvState:
    rng = random.Random()
    ios_pct = min(max(0.55 + rng.uniform(-0.05, 0.05), 0.40), 0.70)
    pixel_quality = compute_pixel_quality(
        ios_traffic_pct=ios_pct,
        conversions_api=False,
        aem_enabled=False,
        utm_tracking=False,
    )
    true_conv = 240
    attr_window = "7d_click"
    reported_conv = compute_reported_conversions(true_conv, attr_window, pixel_quality)
    severity = min(max(rng.uniform(0.90, 1.10), 0.82), 1.18)
    reported_conv = max(int(round(reported_conv * (2.0 - severity))), 1)
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
        modeled_conversions_enabled=False,
        attribution_reporting_mode="observed",
        server_signal_quality=compute_server_signal_quality(False, False, False),
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

    tracking_rel = compute_tracking_reliability(campaign, investigation_level=0.0)

    return EnvState(
        task_id="medium_pixel_recovery",
        difficulty="medium",
        step_count=0,
        max_steps=7,
        campaign=campaign,
        issues_remaining=[
            "conversions_api",
            "aem",
            "tracking_investigated",
            "modeled_reporting",
        ],
        issues_resolved=[],
        day=0,
        growth_momentum=1.0,
        tracking_reliability=tracking_rel,
        attribution_investigation_level=0.0,
        optimal_steps_hint=4,
        scenario_delay_range=[2, 5],
        hidden_conversions_pool=max(true_conv - reported_conv, 0),
        conversion_rate_range=[0.08, 0.12],
        max_generated_conversions_per_step=34,
        max_released_conversions_per_step=32,
        target_true_conversions=340,
        hidden_delayed_conversions=_build_hidden_delayed_events(
            max(true_conv - reported_conv, 0),
            [a.adset_id for a in campaign.adsets],
            2,
            7,
        ),
        attribution_gap_history=[(true_conv - reported_conv) / true_conv],
        roas_history=[campaign.reported_roas],
        signal_quality_history=[tracking_rel],
    )


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — HARD
# Problems: 1d_click window + high iOS (60%) + no CAPI/AEM + bad budget split
# Fix:      all of the above + pause low-ROAS adset + reallocate budget
# ─────────────────────────────────────────────────────────────────────────────

def make_hard_task() -> EnvState:
    rng = random.Random()
    ios_pct = min(max(0.60 + rng.uniform(-0.06, 0.06), 0.45), 0.78)
    pixel_quality = compute_pixel_quality(
        ios_traffic_pct=ios_pct,
        conversions_api=False,
        aem_enabled=False,
        utm_tracking=False,
    )
    true_conv = 310
    attr_window = "1d_click"
    reported_conv = compute_reported_conversions(true_conv, attr_window, pixel_quality)
    severity = min(max(rng.uniform(0.88, 1.14), 0.78), 1.24)
    reported_conv = max(int(round(reported_conv * (2.0 - severity))), 1)
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
        modeled_conversions_enabled=False,
        attribution_reporting_mode="observed",
        server_signal_quality=compute_server_signal_quality(False, False, False),
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

    tracking_rel = compute_tracking_reliability(campaign, investigation_level=0.0)

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
            "tracking_investigated",
            "modeled_reporting",
        ],
        issues_resolved=[],
        day=0,
        growth_momentum=0.92,
        tracking_reliability=tracking_rel,
        attribution_investigation_level=0.0,
        optimal_steps_hint=6,
        scenario_delay_range=[3, 7],
        hidden_conversions_pool=max(true_conv - reported_conv, 0),
        conversion_rate_range=[0.08, 0.12],
        max_generated_conversions_per_step=48,
        max_released_conversions_per_step=36,
        target_true_conversions=470,
        hidden_delayed_conversions=_build_hidden_delayed_events(
            max(true_conv - reported_conv, 0),
            [a.adset_id for a in campaign.adsets],
            2,
            7,
        ),
        attribution_gap_history=[(true_conv - reported_conv) / true_conv],
        roas_history=[campaign.reported_roas],
        signal_quality_history=[tracking_rel],
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