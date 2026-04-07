"""
simulator.py — Campaign simulation engine.

Simulates how Meta Ads attribution degrades due to:
  - Narrow attribution windows (1d_click vs 7d_click vs 28d_click)
  - iOS14+ Pixel signal loss
  - Missing Conversions API / AEM
  - Poor budget allocation across ad sets

The simulator tracks a "ground truth" conversion count and applies
degradation multipliers to produce the "reported" numbers the agent sees —
mirroring exactly what advertisers experience in Meta Ads Manager.
"""

from __future__ import annotations
import copy
import random
from typing import Dict, List, Tuple

from meta_ads_env.models import (
    Action, AdSetMetrics, CampaignData, EnvState, Reward, RewardComponents
)


# ─── Attribution window coverage multipliers ─────────────────────────────────
# Fraction of true conversions that fall inside each window (industry averages)
WINDOW_COVERAGE: Dict[str, float] = {
    "1d_click":            0.38,
    "7d_click":            0.72,
    "7d_click_1d_view":    0.80,
    "28d_click":           0.92,
    "1d_view":             0.20,
}

# ─── Pixel signal quality sources ────────────────────────────────────────────
# Each active mitigation adds signal recovery
def compute_pixel_quality(
    ios_traffic_pct: float,
    conversions_api: bool,
    aem_enabled: bool,
    utm_tracking: bool,
) -> float:
    """
    Base signal quality degrades with iOS traffic.
    Each mitigation partially recovers it.
    Returns quality in [0.0, 1.0].
    """
    base = 1.0 - (ios_traffic_pct * 0.70)   # iOS wipes up to 70% of pixel signal
    base = max(base, 0.15)                   # pixel alone floors at 15%

    recovery = 0.0
    if conversions_api:
        recovery += 0.30   # CAPI is the biggest single recovery lever
    if aem_enabled:
        recovery += 0.15   # AEM adds modelled conversions on top
    if utm_tracking:
        recovery += 0.05   # UTM helps with analytics cross-check

    quality = min(base + recovery, 1.0)
    return round(quality, 4)


def compute_reported_conversions(
    true_conversions: int,
    attribution_window: str,
    pixel_quality: float,
) -> int:
    """
    Applies window coverage and pixel signal degradation to ground-truth conversions
    to produce what Meta Ads Manager actually reports.
    """
    window_factor = WINDOW_COVERAGE.get(attribution_window, 0.72)
    reported = int(true_conversions * window_factor * pixel_quality)
    return max(reported, 0)


def compute_roas(conversions: int, avg_order_value: float, spend: float) -> float:
    if spend <= 0:
        return 0.0
    revenue = conversions * avg_order_value
    return round(revenue / spend, 3)


# ─── Ad-set helpers ──────────────────────────────────────────────────────────

def build_adsets(
    campaign: CampaignData,
    avg_order_value: float,
    seed: int = 42,
) -> List[AdSetMetrics]:
    """Generate 3 realistic ad sets for a campaign."""
    rng = random.Random(seed)
    segments = [
        ("retargeting",  0.25, 3.5),   # segment, budget_fraction, true_roas_multiplier
        ("lookalike_1pct", 0.45, 2.0),
        ("broad_interest", 0.30, 1.2),
    ]
    adsets = []
    for seg, frac, roas_mult in segments:
        budget = round(campaign.total_budget * frac, 2)
        spent  = round(budget * rng.uniform(0.80, 0.99), 2)
        impressions = int(spent * rng.uniform(800, 1200))
        link_clicks = int(impressions * rng.uniform(0.008, 0.025))
        true_conv   = int(spent / (campaign.true_cpa / roas_mult + 0.01))
        rep_conv    = compute_reported_conversions(
            true_conv, campaign.attribution_window, campaign.pixel_signal_quality
        )
        true_roas   = compute_roas(true_conv, avg_order_value, spent)
        rep_roas    = compute_roas(rep_conv, avg_order_value, spent)

        adsets.append(AdSetMetrics(
            adset_id=f"adset_{seg}",
            adset_name=seg.replace("_", " ").title(),
            budget=budget,
            spent=spent,
            impressions=impressions,
            link_clicks=link_clicks,
            reported_conversions=rep_conv,
            true_conversions=true_conv,
            reported_roas=rep_roas,
            true_roas=true_roas,
            audience_segment=seg,
            is_paused=False,
        ))
    return adsets


# ─── Action application ──────────────────────────────────────────────────────

class SimulationEngine:
    """
    Applies an Action to an EnvState and returns
    (new_state, reward, done, info).
    """

    def apply(
        self,
        state: EnvState,
        action: Action,
        avg_order_value: float = 75.0,
    ) -> Tuple[EnvState, Reward, bool, Dict]:
        new_state = state.model_copy(deep=True)
        c = new_state.campaign
        reward_components = RewardComponents()
        info: Dict = {"action_applied": action.action_type, "effects": []}

        # ── Snapshot before ──────────────────────────────────────────────
        before_gap = _attribution_gap(c)
        before_signal = c.pixel_signal_quality
        before_roas = c.reported_roas

        # ── Apply action ─────────────────────────────────────────────────
        valid = True

        if action.action_type == "adjust_attribution_window":
            window = action.parameters.get("window", "7d_click")
            if window in WINDOW_COVERAGE and window != c.attribution_window:
                c.attribution_window = window
                info["effects"].append(f"Attribution window changed to {window}")
                if "attribution_window" not in new_state.issues_resolved:
                    new_state.issues_resolved.append("attribution_window")
            else:
                valid = False
                info["effects"].append("Invalid or unchanged attribution window")

        elif action.action_type == "enable_conversions_api":
            if not c.conversions_api_enabled:
                c.conversions_api_enabled = True
                info["effects"].append("Conversions API enabled")
                if "conversions_api" not in new_state.issues_resolved:
                    new_state.issues_resolved.append("conversions_api")
            else:
                valid = False
                info["effects"].append("Conversions API already enabled")

        elif action.action_type == "enable_aggregated_event_measurement":
            if not c.aem_enabled:
                c.aem_enabled = True
                info["effects"].append("AEM enabled")
                if "aem" not in new_state.issues_resolved:
                    new_state.issues_resolved.append("aem")
            else:
                valid = False
                info["effects"].append("AEM already enabled")

        elif action.action_type == "add_utm_tracking":
            if not c.utm_tracking:
                c.utm_tracking = True
                info["effects"].append("UTM parameters added")
                if "utm_tracking" not in new_state.issues_resolved:
                    new_state.issues_resolved.append("utm_tracking")
            else:
                valid = False
                info["effects"].append("UTM tracking already enabled")

        elif action.action_type == "adjust_budget_allocation":
            shifts = action.parameters.get("shifts", {})
            for adset_id, new_budget in shifts.items():
                for ads in c.adsets:
                    if ads.adset_id == adset_id and not ads.is_paused:
                        ads.budget = max(0.0, float(new_budget))
                        info["effects"].append(f"Budget for {adset_id} → ${new_budget}")
            if "budget_allocation" not in new_state.issues_resolved:
                new_state.issues_resolved.append("budget_allocation")

        elif action.action_type == "pause_underperforming_adsets":
            threshold = action.parameters.get("roas_threshold", 1.0)
            paused = []
            for ads in c.adsets:
                if ads.true_roas < threshold and not ads.is_paused:
                    ads.is_paused = True
                    paused.append(ads.adset_id)
            info["effects"].append(f"Paused adsets: {paused}")
            if paused and "paused_bad_adsets" not in new_state.issues_resolved:
                new_state.issues_resolved.append("paused_bad_adsets")

        elif action.action_type == "reallocate_to_top_performers":
            active = [a for a in c.adsets if not a.is_paused]
            if len(active) < 2:
                valid = False
                info["effects"].append("Not enough active adsets to reallocate")
            else:
                top = max(active, key=lambda a: a.true_roas)
                realloc_amt = action.parameters.get("amount", 500.0)
                top.budget += float(realloc_amt)
                info["effects"].append(f"Reallocated ${realloc_amt} to {top.adset_id}")
                if "budget_reallocation" not in new_state.issues_resolved:
                    new_state.issues_resolved.append("budget_reallocation")

        elif action.action_type == "change_bid_strategy":
            strategy = action.parameters.get("strategy", "lowest_cost")
            info["effects"].append(f"Bid strategy → {strategy}")
            if "bid_strategy" not in new_state.issues_resolved:
                new_state.issues_resolved.append("bid_strategy")

        elif action.action_type == "segment_audience":
            info["effects"].append("Audience segmentation applied")
            if "audience_segmentation" not in new_state.issues_resolved:
                new_state.issues_resolved.append("audience_segmentation")

        elif action.action_type == "no_op":
            info["effects"].append("No action taken")
            valid = False  # Penalise doing nothing

        # ── Recompute derived fields ──────────────────────────────────────
        c.pixel_signal_quality = compute_pixel_quality(
            c.ios_traffic_pct,
            c.conversions_api_enabled,
            c.aem_enabled,
            c.utm_tracking,
        )
        c.reported_conversions = compute_reported_conversions(
            c.true_conversions,
            c.attribution_window,
            c.pixel_signal_quality,
        )
        c.reported_roas = compute_roas(c.reported_conversions, avg_order_value, c.budget_spent)
        c.true_roas     = compute_roas(c.true_conversions,     avg_order_value, c.budget_spent)
        c.reported_cpa  = (c.budget_spent / c.reported_conversions) if c.reported_conversions > 0 else 9999
        c.true_cpa      = (c.budget_spent / c.true_conversions)     if c.true_conversions > 0     else 9999

        # ── Compute reward ────────────────────────────────────────────────
        after_gap    = _attribution_gap(c)
        after_signal = c.pixel_signal_quality
        after_roas   = c.reported_roas

        # 1. Attribution accuracy improvement (max 0.35)
        gap_delta = max(before_gap - after_gap, 0.0)
        reward_components.attribution_accuracy = round(min(gap_delta * 0.70, 0.35), 4)

        # 2. ROAS improvement (max 0.25)
        roas_delta = max(after_roas - before_roas, 0.0)
        reward_components.roas_improvement = round(min(roas_delta * 0.08, 0.25), 4)

        # 3. Signal quality gain (max 0.25)
        sig_delta = max(after_signal - before_signal, 0.0)
        reward_components.signal_quality_gain = round(min(sig_delta * 0.50, 0.25), 4)

        # 4. Action validity (0.10 for valid, 0 for no-op / invalid)
        reward_components.action_validity = 0.10 if valid else 0.0

        # 5. Step efficiency bonus (0.05 if done quickly)
        if new_state.step_count < new_state.max_steps // 2:
            reward_components.step_efficiency = 0.05
        else:
            reward_components.step_efficiency = 0.0

        total = (
            reward_components.attribution_accuracy
            + reward_components.roas_improvement
            + reward_components.signal_quality_gain
            + reward_components.action_validity
            + reward_components.step_efficiency
        )
        total = round(min(total, 1.0), 4)

        reward = Reward(
            total=total,
            components=reward_components,
            explanation=(
                f"Gap closed by {gap_delta:.2%}, signal ↑{sig_delta:.2%}, "
                f"ROAS ↑{roas_delta:.2f}x. Action {'valid' if valid else 'invalid/redundant'}."
            ),
        )

        # ── Advance step ──────────────────────────────────────────────────
        new_state.step_count += 1
        new_state.cumulative_reward += total
        new_state.history.append({
            "step": new_state.step_count,
            "action": action.action_type,
            "reward": total,
            "effects": info["effects"],
        })

        # ── Check done ────────────────────────────────────────────────────
        done = (
            new_state.step_count >= new_state.max_steps
            or _all_issues_resolved(new_state)
        )
        new_state.done = done

        return new_state, reward, done, info


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _attribution_gap(c: CampaignData) -> float:
    if c.true_conversions == 0:
        return 0.0
    return max((c.true_conversions - c.reported_conversions) / c.true_conversions, 0.0)


def _all_issues_resolved(state: EnvState) -> bool:
    remaining = set(state.issues_remaining) - set(state.issues_resolved)
    return len(remaining) == 0