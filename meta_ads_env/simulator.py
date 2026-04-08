"""Campaign simulation engine with delayed causal attribution dynamics."""

from __future__ import annotations
import math
import random
from typing import Dict, List, Tuple

from meta_ads_env.models import (
    Action,
    AdSetMetrics,
    CampaignData,
    EnvState,
    PendingConversion,
    Reward,
    RewardComponents,
)


# ─── Attribution window coverage multipliers ─────────────────────────────────
# Fraction of true conversions that fall inside each window (industry averages)
WINDOW_COVERAGE: Dict[str, float] = {
    "1d_click": 0.30,
    "7d_click": 0.78,
    "7d_click_1d_view": 0.86,
    "28d_click": 0.92,
    "1d_view": 0.20,
}

WINDOW_DAYS: Dict[str, int] = {
    "1d_click": 1,
    "7d_click": 7,
    "7d_click_1d_view": 7,
    "28d_click": 28,
    "1d_view": 1,
}

SEGMENT_PARAMS: Dict[str, Dict[str, float]] = {
    "retargeting": {"ctr": 0.027, "cvr": 0.11, "imp_per_usd": 780.0},
    "lookalike_1pct": {"ctr": 0.019, "cvr": 0.068, "imp_per_usd": 980.0},
    "lookalike_2pct": {"ctr": 0.017, "cvr": 0.056, "imp_per_usd": 1010.0},
    "broad_interest": {"ctr": 0.012, "cvr": 0.020, "imp_per_usd": 1120.0},
    "interest": {"ctr": 0.013, "cvr": 0.028, "imp_per_usd": 1080.0},
}

SEGMENT_DECAY: Dict[str, float] = {
    "retargeting": 0.006,
    "lookalike_1pct": 0.008,
    "lookalike_2pct": 0.009,
    "broad_interest": 0.014,
    "interest": 0.012,
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


def compute_server_signal_quality(
    conversions_api: bool,
    aem_enabled: bool,
    utm_tracking: bool,
) -> float:
    quality = 0.05
    if conversions_api:
        quality += 0.55
    if aem_enabled:
        quality += 0.18
    if utm_tracking:
        quality += 0.08
    return round(min(quality, 0.95), 4)


def compute_attribution_confidence(
    pixel_match_quality: float,
    capi_coverage: float,
    ios_traffic_pct: float,
) -> float:
    ios_component = 1.0 - min(max(ios_traffic_pct * 100.0, 0.0), 100.0) / 100.0
    confidence = (pixel_match_quality * 0.4) + (capi_coverage * 0.4) + (ios_component * 0.2)
    return round(min(max(confidence, 0.0), 1.0), 4)


def compute_tracking_reliability(campaign: CampaignData, investigation_level: float) -> float:
    # Pixel degrades with iOS while server-side partially recovers signal.
    pixel_weight = 0.70
    server_weight = 0.30
    base = campaign.pixel_signal_quality * pixel_weight + campaign.server_signal_quality * server_weight
    base = (base * 0.80) + (campaign.capi_coverage * 0.20)
    recovery = min(investigation_level, 1.0) * 0.18
    return round(min(base + recovery, 0.98), 4)


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

def build_adsets(campaign: CampaignData, avg_order_value: float, seed: int = 42) -> List[AdSetMetrics]:
    # Kept for backward compatibility for external callers.
    _ = (avg_order_value, seed)
    return campaign.adsets


# ─── Action application ──────────────────────────────────────────────────────

class SimulationEngine:
    """
    Applies an Action to an EnvState and returns
    (new_state, reward, done, info).
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

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

        # Deterministic noise source: same task + same action trajectory => same outputs.
        deterministic_seed = (
            int(new_state.random_seed)
            + ((new_state.day + 1) * 997)
            + ((new_state.step_count + 1) * 211)
            + (sum(ord(ch) for ch in action.action_type) * 3)
            + (len(new_state.history) * 17)
        )
        self.rng.seed(deterministic_seed)

        # ── Snapshot before ──────────────────────────────────────────────
        before_gap = _attribution_gap(c)
        before_signal = state.tracking_reliability
        before_roas = c.reported_roas
        before_momentum = state.growth_momentum
        before_issue_fraction = _issue_resolution_fraction(state)
        converged_before = _is_converged(state)

        # ── Apply action ─────────────────────────────────────────────────
        valid = True
        action_count = new_state.action_counts.get(action.action_type, 0) + 1
        new_state.action_counts[action.action_type] = action_count
        prev_action = new_state.history[-1]["action"] if new_state.history else ""
        prev2_action = new_state.history[-2]["action"] if len(new_state.history) >= 2 else ""
        same_as_previous = action.action_type == prev_action and action.action_type != ""
        repeat_count = 1
        if same_as_previous:
            repeat_count = 2
            if action.action_type == prev2_action:
                repeat_count = 3
        if action.action_type != "no_op":
            new_state.easy_meaningful_actions_taken += 1
        diminishing = _diminishing_returns(action_count)
        timing_bonus = 0.0
        uncertainty_bonus = 0.0
        delayed_release_bonus = 0.0
        stable_stack = _is_stack_stable(new_state)

        if action.action_type == "promote_ad":
            if not stable_stack:
                valid = False
                timing_bonus = -0.10
                uncertainty_bonus = min(uncertainty_bonus, -0.04)
                info["effects"].append("Promotion blocked: stack not stabilized")
            else:
                lift = 0.18 * diminishing
                if new_state.tracking_reliability < 0.72:
                    # Risky scaling under low confidence can backfire later.
                    lift *= 0.65
                    timing_bonus -= 0.06
                    new_state.risk_events.append("early_scale_risk")
                    info["effects"].append("Risk event: early scaling under low tracking confidence")
                    new_state.budget_optimization_multiplier = max(new_state.budget_optimization_multiplier - 0.10, 0.80)
                if not new_state.tracking_investigated:
                    lift *= 0.75
                    timing_bonus -= 0.05
                    uncertainty_bonus -= 0.03
                    new_state.risk_events.append("promote_before_tracking_fix")
                    info["effects"].append("Promotion before tracking fix reduced future conversion quality")
                new_state.growth_momentum = min(new_state.growth_momentum + lift, 1.8)
                info["effects"].append(f"Promotion lift applied (+{lift:.2f} momentum)")
                timing_bonus = 0.12

        elif action.action_type == "reduce_budget":
            scale = action.parameters.get("scale", 0.85)
            scale = min(max(float(scale), 0.60), 0.98)
            for ads in c.adsets:
                if not ads.is_paused:
                    ads.budget = round(ads.budget * scale, 2)
            new_state.growth_momentum = max(new_state.growth_momentum - 0.12 * diminishing, 0.55)
            info["effects"].append(f"Budgets scaled by {scale:.2f}; growth momentum reduced")
            timing_bonus = -0.03 if new_state.day <= 2 else 0.04
            if (not new_state.tracking_investigated) and new_state.day <= 2:
                new_state.early_wrong_decision = True
                timing_bonus -= 0.08
                info["effects"].append("Wrong early decision: budget reduced before attribution diagnosis")

        elif action.action_type == "investigate_attribution":
            if new_state.tracking_investigated and not new_state.uncertainty_reintroduced:
                valid = False
                uncertainty_bonus = -0.06
                info["effects"].append("Redundant investigation (no new uncertainty)")
            else:
                gain = 0.22 * diminishing
                new_state.attribution_investigation_level = min(
                    new_state.attribution_investigation_level + gain, 1.0
                )
                new_state.tracking_investigated = True
                new_state.uncertainty_reintroduced = False
                info["effects"].append(f"Attribution investigation depth +{gain:.2f}")
                uncertainty_bonus = 0.09
                if "tracking_investigated" not in new_state.issues_resolved:
                    new_state.issues_resolved.append("tracking_investigated")
                if new_state.early_wrong_decision:
                    new_state.recovered_after_wrong_decision = True
                    info["effects"].append("Recovery path activated after early budget misstep")

        elif action.action_type == "switch_to_modeled_conversions":
            if c.attribution_reporting_mode == "modeled":
                valid = False
                uncertainty_bonus = -0.03
                info["effects"].append("Modeled reporting already active")
            else:
                c.modeled_conversions_enabled = True
                c.attribution_reporting_mode = "modeled"
                new_state.growth_momentum = min(new_state.growth_momentum + (0.06 * diminishing), 1.9)
                recovered_total = new_state.tracked_conversions_total + new_state.modeled_conversions_total
                if recovered_total > 0:
                    target_modeled_share = 0.30 if c.aem_enabled else 0.22
                    desired_modeled = int(round(recovered_total * target_modeled_share))
                    reclassify = min(new_state.tracked_conversions_total, max(desired_modeled - new_state.modeled_conversions_total, 0))
                    if reclassify > 0:
                        new_state.tracked_conversions_total -= reclassify
                        new_state.modeled_conversions_total += reclassify
                        info["effects"].append(f"Reclassified {reclassify} recovered conversions into modeled bucket")
                info["effects"].append("Reporting switched to modeled conversions")
                uncertainty_bonus = 0.06
                if (not c.conversions_api_enabled) or (not c.aem_enabled) or (not new_state.tracking_investigated):
                    # Premature modeled mode leads to noisier optimization and weaker signal trust.
                    timing_bonus -= 0.12
                    uncertainty_bonus -= 0.08
                    new_state.growth_momentum = max(new_state.growth_momentum - 0.06, 0.55)
                    info["effects"].append("Modeled reporting switched too early; quality penalty applied")
                if not c.aem_enabled:
                    uncertainty_bonus -= 0.06
                    new_state.risk_events.append("modeled_before_aem")
                    info["effects"].append("Modeled reporting before AEM introduced noisy signal")
                if "modeled_reporting" not in new_state.issues_resolved:
                    new_state.issues_resolved.append("modeled_reporting")

        if action.action_type == "adjust_attribution_window":
            window = action.parameters.get("window", "7d_click")
            if window in WINDOW_COVERAGE and window != c.attribution_window:
                c.attribution_window = window
                new_state.growth_momentum = min(new_state.growth_momentum + (0.05 * diminishing), 1.9)
                info["effects"].append(f"Attribution window changed to {window}")
                if "attribution_window" not in new_state.issues_resolved:
                    new_state.issues_resolved.append("attribution_window")
            else:
                valid = False
                info["effects"].append("Invalid or unchanged attribution window")

        elif action.action_type == "enable_conversions_api":
            if not c.conversions_api_enabled:
                c.conversions_api_enabled = True
                new_state.growth_momentum = min(new_state.growth_momentum + (0.08 * diminishing), 1.9)
                info["effects"].append("Conversions API enabled")
                if "conversions_api" not in new_state.issues_resolved:
                    new_state.issues_resolved.append("conversions_api")
            else:
                valid = False
                info["effects"].append("Conversions API already enabled")

        elif action.action_type == "enable_aggregated_event_measurement":
            if not c.aem_enabled:
                c.aem_enabled = True
                new_state.growth_momentum = min(new_state.growth_momentum + (0.07 * diminishing), 1.9)
                info["effects"].append("AEM enabled")
                if "aem" not in new_state.issues_resolved:
                    new_state.issues_resolved.append("aem")
            else:
                valid = False
                info["effects"].append("AEM already enabled")

        elif action.action_type == "add_utm_tracking":
            if not c.utm_tracking:
                c.utm_tracking = True
                new_state.growth_momentum = min(new_state.growth_momentum + (0.04 * diminishing), 1.9)
                info["effects"].append("UTM parameters added")
                if "utm_tracking" not in new_state.issues_resolved:
                    new_state.issues_resolved.append("utm_tracking")
            else:
                valid = False
                info["effects"].append("UTM tracking already enabled")

        elif action.action_type == "adjust_budget_allocation":
            shifts = action.parameters.get("shifts", {})
            moved_any = False
            for adset_id, new_budget in shifts.items():
                for ads in c.adsets:
                    if ads.adset_id == adset_id and not ads.is_paused:
                        ads.budget = max(0.0, float(new_budget))
                        moved_any = True
                        info["effects"].append(f"Budget for {adset_id} → ${new_budget}")
            if moved_any and "budget_allocation" not in new_state.issues_resolved:
                new_state.issues_resolved.append("budget_allocation")
            if not moved_any:
                valid = False
            if "paused_bad_adsets" in new_state.issues_remaining and "paused_bad_adsets" not in new_state.issues_resolved:
                timing_bonus -= 0.05
            if moved_any and ((not new_state.tracking_investigated) or (not c.conversions_api_enabled)):
                timing_bonus -= 0.12
                new_state.growth_momentum = max(new_state.growth_momentum - 0.05, 0.55)
                new_state.risk_events.append("premature_budget_shift")
                info["effects"].append("Budget shift before attribution fixes reduced future efficiency")
            if moved_any:
                # Budget mix changes compound over future steps once tracking quality is usable.
                compounding_gain = 0.14 if new_state.tracking_investigated else 0.08
                new_state.budget_optimization_multiplier = min(
                    new_state.budget_optimization_multiplier + (compounding_gain * diminishing),
                    1.85,
                )

        elif action.action_type == "pause_underperforming_adsets":
            threshold = action.parameters.get("roas_threshold", 1.0)
            paused = []
            wasted_spend_cut = 0.0
            for ads in c.adsets:
                # Only pause when both observed and true performance are weak.
                reported_cutoff = max(float(threshold) - 0.25, 0.80)
                if ads.true_roas < float(threshold) and ads.reported_roas < reported_cutoff and not ads.is_paused:
                    ads.is_paused = True
                    paused.append(ads.adset_id)
                    wasted_spend_cut += ads.spent * 0.06
            info["effects"].append(f"Paused adsets: {paused}")
            if paused and "paused_bad_adsets" not in new_state.issues_resolved:
                new_state.issues_resolved.append("paused_bad_adsets")
            if paused:
                momentum_gain = min(0.04 + (0.02 * len(paused)), 0.12)
                new_state.growth_momentum = min(new_state.growth_momentum + momentum_gain, 1.9)
                c.budget_spent = max(c.budget_spent - wasted_spend_cut, 0.0)
                new_state.budget_optimization_multiplier = min(
                    new_state.budget_optimization_multiplier + (0.10 * diminishing),
                    1.85,
                )
                info["effects"].append(f"Waste reduction applied (${wasted_spend_cut:.0f})")
            if not paused:
                valid = False

        elif action.action_type == "reallocate_to_top_performers":
            active = [a for a in c.adsets if not a.is_paused]
            if len(active) < 2:
                valid = False
                info["effects"].append("Not enough active adsets to reallocate")
            else:
                top = max(active, key=lambda a: a.true_roas)
                low = min(active, key=lambda a: a.true_roas)
                realloc_amt = action.parameters.get("amount", 500.0)
                realloc_amt = float(realloc_amt)
                donor_amt = min(realloc_amt, max(low.budget * 0.4, 0.0))
                low.budget = max(0.0, low.budget - donor_amt)
                top.budget += donor_amt
                # Reallocation has delayed compounding impact on future conversion potential.
                new_state.growth_momentum = min(new_state.growth_momentum + (0.09 * diminishing), 1.9)
                info["effects"].append(f"Reallocated ${realloc_amt} to {top.adset_id}")
                if "budget_reallocation" not in new_state.issues_resolved:
                    new_state.issues_resolved.append("budget_reallocation")
                if "budget_allocation" not in new_state.issues_resolved:
                    new_state.issues_resolved.append("budget_allocation")
                if "paused_bad_adsets" in new_state.issues_remaining and "paused_bad_adsets" not in new_state.issues_resolved:
                    timing_bonus -= 0.05
                if (not new_state.tracking_investigated) or (not c.conversions_api_enabled):
                    timing_bonus -= 0.14
                    new_state.growth_momentum = max(new_state.growth_momentum - 0.06, 0.55)
                    new_state.risk_events.append("premature_reallocation")
                    info["effects"].append("Premature reallocation penalty: attribution stack not ready")
                else:
                    new_state.budget_optimization_multiplier = min(
                        new_state.budget_optimization_multiplier + (0.18 * diminishing),
                        1.95,
                    )

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
            valid = False

        if (not new_state.tracking_investigated) and action.action_type in {
            "promote_ad",
            "adjust_budget_allocation",
            "reallocate_to_top_performers",
            "switch_to_modeled_conversions",
        }:
            timing_bonus -= 0.08
            uncertainty_bonus -= 0.04

        # Confidence estimate drives risk penalties when acting under poor observability.
        unresolved = len(set(new_state.issues_remaining) - set(new_state.issues_resolved))
        c.capi_coverage = min(max(c.capi_coverage + (0.28 if c.conversions_api_enabled else -0.03), 0.0), 1.0)
        c.pixel_match_quality = min(max((0.70 * c.pixel_signal_quality) + (0.30 * new_state.tracking_reliability), 0.0), 1.0)
        new_state.attribution_confidence = compute_attribution_confidence(
            pixel_match_quality=c.pixel_match_quality,
            capi_coverage=c.capi_coverage,
            ios_traffic_pct=c.ios_traffic_pct,
        )
        new_state.confidence_score = min(
            max(
                (0.80 * new_state.attribution_confidence)
                + (0.15 * (1.0 - _attribution_gap(c)))
                + (0.05 * (1.0 - min(unresolved / 8.0, 1.0))),
                0.0,
            ),
            1.0,
        )
        if new_state.confidence_score < 0.55 and action.action_type in {"promote_ad", "reallocate_to_top_performers", "adjust_budget_allocation"}:
            timing_bonus -= 0.06
            uncertainty_bonus -= 0.05
            new_state.risk_events.append("low_confidence_risk_penalty")

        # Long-term state updates from tracking stack.
        c.pixel_signal_quality = compute_pixel_quality(
            c.ios_traffic_pct,
            c.conversions_api_enabled,
            c.aem_enabled,
            c.utm_tracking,
        )
        c.server_signal_quality = compute_server_signal_quality(
            c.conversions_api_enabled,
            c.aem_enabled,
            c.utm_tracking,
        )
        new_state.tracking_reliability = compute_tracking_reliability(
            c,
            new_state.attribution_investigation_level,
        )

        # Causal daily transition: impressions -> clicks -> delayed conversions -> observed signals.
        self._simulate_day(new_state, avg_order_value)

        # ── Compute reward ────────────────────────────────────────────────
        after_gap    = _attribution_gap(c)
        after_signal = new_state.tracking_reliability
        after_roas   = c.reported_roas
        after_momentum = new_state.growth_momentum

        gap_delta = max(before_gap - after_gap, 0.0)
        roas_delta = after_roas - before_roas
        sig_delta = max(after_signal - before_signal, 0.0)
        momentum_delta = after_momentum - before_momentum
        issue_progress = max(_issue_resolution_fraction(new_state) - before_issue_fraction, 0.0)
        early_factor = max(0.2, 1.0 - (new_state.day / max(new_state.max_steps, 1)))
        delayed_recovery = min(new_state.delayed_conversion_release_last_step / max(new_state.max_released_conversions_per_step, 1), 1.0)

        reward_components.attribution_accuracy = round(min(gap_delta * (0.80 + 0.32 * early_factor), 0.58), 4)
        reward_components.roas_improvement = round(max(min(roas_delta * 0.12, 0.30), -0.18), 4)
        reward_components.signal_quality_gain = round(min(sig_delta * 0.60, 0.25), 4)
        reward_components.action_validity = 0.10 if valid else -0.06
        reward_components.step_efficiency = 0.05 if new_state.step_count <= new_state.optimal_steps_hint else -0.03
        reward_components.timing_quality = round(timing_bonus + (0.08 * delayed_recovery), 4)
        reward_components.uncertainty_handling = round(uncertainty_bonus, 4)
        reward_components.long_term_gain = round(max(min((momentum_delta * 0.26) + (0.08 * delayed_recovery), 0.14), -0.10), 4)
        reward_components.issue_resolution_progress = round(min(issue_progress * 0.18, 0.12), 4)
        reward_components.redundancy_penalty = round(_redundancy_penalty(action.action_type, action_count), 4)

        # Penalize redundant actions with stronger recency awareness (last 2 steps).
        recent_redundancy = 0.0
        if action.action_type in {prev_action, prev2_action} and action.action_type:
            recent_redundancy -= 0.03
        reward_components.redundancy_penalty = round(
            reward_components.redundancy_penalty + recent_redundancy,
            4,
        )

        # Encourage action diversity until convergence is reached.
        if (not converged_before) and action.action_type not in {prev_action, prev2_action}:
            reward_components.uncertainty_handling = round(reward_components.uncertainty_handling + 0.02, 4)

        if new_state.confidence_score < 0.50 and action.action_type not in {"investigate_attribution", "no_op"}:
            reward_components.uncertainty_handling = round(reward_components.uncertainty_handling - 0.05, 4)
            reward_components.redundancy_penalty = round(reward_components.redundancy_penalty - 0.04, 4)

        if valid and gap_delta < 0.005 and sig_delta < 0.005 and delayed_recovery < 0.10:
            reward_components.redundancy_penalty = round(reward_components.redundancy_penalty - 0.06, 4)

        if action.action_type == "adjust_attribution_window" and new_state.difficulty == "easy":
            reward_components.timing_quality = round(reward_components.timing_quality - 0.04, 4)

        if new_state.day <= 2 and not new_state.tracking_investigated and action.action_type != "investigate_attribution":
            reward_components.redundancy_penalty = round(reward_components.redundancy_penalty - 0.08, 4)

        if action.action_type == "promote_ad" and not stable_stack:
            reward_components.redundancy_penalty = round(reward_components.redundancy_penalty - 0.04, 4)

        if action.action_type == "promote_ad" and action_count > 2:
            reward_components.redundancy_penalty = round(reward_components.redundancy_penalty - min(0.02 * (action_count - 2), 0.08), 4)

        if action.action_type == "promote_ad" and roas_delta < 0.08:
            reward_components.redundancy_penalty = round(reward_components.redundancy_penalty - 0.04, 4)

        if _all_issues_resolved(new_state) and action.action_type != "no_op":
            reward_components.redundancy_penalty = round(reward_components.redundancy_penalty - 0.06, 4)

        if "paused_bad_adsets" in new_state.issues_remaining and "paused_bad_adsets" not in new_state.issues_resolved:
            if action.action_type in {"promote_ad", "reallocate_to_top_performers"}:
                reward_components.redundancy_penalty = round(reward_components.redundancy_penalty - 0.05, 4)

        if new_state.day >= 6 and "paused_bad_adsets" in new_state.issues_remaining and "paused_bad_adsets" not in new_state.issues_resolved:
            if action.action_type != "pause_underperforming_adsets":
                reward_components.redundancy_penalty = round(reward_components.redundancy_penalty - 0.04, 4)

        if converged_before and action.action_type != "no_op":
            reward_components.redundancy_penalty = round(reward_components.redundancy_penalty - 0.08, 4)

        if new_state.day >= (new_state.max_steps - 2) and action.action_type in {"add_utm_tracking", "segment_audience", "change_bid_strategy"}:
            reward_components.redundancy_penalty = round(reward_components.redundancy_penalty - 0.05, 4)

        reward_components.timing_quality = round(
            reward_components.timing_quality + _ordering_bonus(new_state, action.action_type) + delayed_release_bonus + (0.03 * early_factor),
            4,
        )

        # Slight reward-scale stochasticity prevents a single rigid trajectory from always dominating.
        reward_scale = self.rng.uniform(0.96, 1.04)

        immediate_reward = (
            (0.45 * reward_components.action_validity)
            + (0.45 * reward_components.step_efficiency)
            + (0.50 * reward_components.timing_quality)
            + (0.35 * reward_components.uncertainty_handling)
            + (0.60 * reward_components.redundancy_penalty)
        )
        step_penalty = 0.02
        immediate_reward -= step_penalty
        delayed_credit = (
            (0.80 * reward_components.attribution_accuracy)
            + (0.80 * reward_components.roas_improvement)
            + (0.70 * reward_components.signal_quality_gain)
            + (0.60 * reward_components.long_term_gain)
            + (0.55 * reward_components.issue_resolution_progress)
        )

        new_state.delayed_reward_buffer = round(new_state.delayed_reward_buffer + delayed_credit, 6)
        delayed_reward = max(min(new_state.delayed_reward_buffer * 0.45, 0.70), -0.40)
        new_state.delayed_reward_buffer = round(new_state.delayed_reward_buffer - delayed_reward, 6)
        new_state.delayed_reward_released_last_step = round(delayed_reward, 4)

        terminal_bonus = 0.0
        done_candidate = (
            (new_state.step_count + 1) >= new_state.max_steps
            or (_all_issues_resolved(new_state) and (new_state.day + 1) >= 2)
            or _is_converged(new_state)
        )

        # Penalize unnecessary steps beyond the task's optimal budget if not done.
        if ((new_state.step_count + 1) > new_state.optimal_steps) and (not done_candidate):
            immediate_reward -= 0.05
        if done_candidate:
            final_gap_pred = after_gap
            roas_gain_pred = max((c.true_roas - state.campaign.true_roas) / max(state.campaign.true_roas, 0.01), 0.0)
            signal_pred = new_state.tracking_reliability
            efficiency_pred = max(0.0, 1.0 - ((new_state.step_count + 1) / max(new_state.max_steps, 1)))
            terminal_bonus = max(
                min(
                    (0.30 * max(1.0 - final_gap_pred, 0.0))
                    + (0.30 * min(roas_gain_pred, 1.5))
                    + (0.20 * signal_pred)
                    + (0.20 * efficiency_pred),
                    0.90,
                ),
                0.0,
            )
            if (new_state.step_count + 1) <= 6:
                terminal_bonus += 0.08
            elif (new_state.step_count + 1) >= (new_state.max_steps - 1):
                terminal_bonus -= 0.08
            if converged_before and action.action_type != "no_op":
                terminal_bonus -= 0.05
        new_state.terminal_bonus_last_step = round(terminal_bonus, 4)

        raw_total = (immediate_reward + delayed_reward + terminal_bonus) * reward_scale

        # Apply diminishing returns on repeated local behavior.
        repeat_scale = max(0.55, 1.0 - (0.10 * max(repeat_count - 1, 0)))
        raw_total *= repeat_scale

        # Repeated ineffective actions should be negative (not neutral).
        ineffective_repeat = same_as_previous and (gap_delta <= 0.001) and (sig_delta <= 0.001) and (roas_delta <= 0.0)
        if ineffective_repeat:
            raw_total -= 0.05

        if new_state.early_wrong_decision and (not new_state.recovered_after_wrong_decision):
            raw_total -= 0.03
        if new_state.recovered_after_wrong_decision:
            raw_total += 0.02

        # Medium tasks are intentionally capped to avoid unrealistically high step rewards.
        if new_state.difficulty == "medium" and raw_total > 0.5:
            raw_total *= 0.8

        # Smooth squash into [-1, 1] so penalties can produce small negative rewards.
        total = math.tanh(0.9 * raw_total)
        total = round(min(max(total, -0.95), 0.95), 4)

        reward = Reward(
            total=total,
            components=reward_components,
            explanation=(
                f"day={new_state.day} gap_delta={gap_delta:.2%} signal_delta={sig_delta:.2%} "
                f"roas_delta={roas_delta:.2f} momentum={new_state.growth_momentum:.2f} "
                f"repeat_count={action_count} immediate={immediate_reward:.3f} delayed={delayed_reward:.3f} terminal={terminal_bonus:.3f} reward_scale={reward_scale:.3f} raw_total={raw_total:.3f} confidence={new_state.confidence_score:.2f}"
            ),
        )

        # ── Advance step ──────────────────────────────────────────────────
        new_state.step_count += 1
        new_state.day += 1
        new_state.cumulative_reward += total
        new_state.attribution_gap_history.append(_attribution_gap(c))
        new_state.roas_history.append(c.reported_roas)
        new_state.signal_quality_history.append(new_state.tracking_reliability)
        new_state.history.append({
            "step": new_state.step_count,
            "day": new_state.day,
            "action": action.action_type,
            "reasoning": action.reasoning or "",
            "reward": total,
            "immediate_reward": round(immediate_reward, 4),
            "delayed_reward": round(delayed_reward, 4),
            "terminal_bonus": round(terminal_bonus, 4),
            "delayed_release": new_state.delayed_conversion_release_last_step,
            "tracked_release": new_state.tracked_conversion_release_last_step,
            "modeled_release": new_state.modeled_conversion_release_last_step,
            "risk_events": [e for e in info["effects"] if "penalty" in e.lower() or "risk" in e.lower() or "reduced" in e.lower()],
            "effects": info["effects"],
        })
        if action.reasoning:
            new_state.reasoning_log.append(action.reasoning)
        new_state.risk_events.extend([e for e in info["effects"] if "penalty" in e.lower() or "risk" in e.lower() or "reduced" in e.lower()])

        # ── Check done ────────────────────────────────────────────────────
        min_steps_required = 3 if new_state.difficulty == "easy" else 2
        easy_action_gate = (
            new_state.difficulty != "easy"
            or new_state.easy_meaningful_actions_taken >= 3
        )
        done = (
            new_state.step_count >= new_state.max_steps
            or (_all_issues_resolved(new_state) and new_state.day >= min_steps_required and easy_action_gate)
            or (_is_converged(new_state) and easy_action_gate)
        )
        if done and _is_converged(new_state):
            new_state.convergence_reached = True
            new_state.cumulative_reward += 0.06
        new_state.done = done

        return new_state, reward, done, info

    def _simulate_day(self, state: EnvState, avg_order_value: float) -> None:
        campaign = state.campaign
        day_seed = state.day + 1

        if not state.episode_risk_initialized:
            state.episode_risk_initialized = True
            if self.rng.random() < self.rng.uniform(0.15, 0.25):
                event = self.rng.choice(["tracking_drop", "modeled_noise", "delayed_spike"])
                state.risk_events.append(event)
                if event == "tracking_drop":
                    state.tracking_reliability = max(state.tracking_reliability - self.rng.uniform(0.10, 0.20), 0.15)
                elif event == "delayed_spike":
                    spike_pool = max(int(state.hidden_conversions_pool * self.rng.uniform(0.10, 0.18)), 6)
                    if state.campaign.adsets:
                        source_ids = [a.adset_id for a in state.campaign.adsets if not a.is_paused] or [state.campaign.adsets[0].adset_id]
                        state.pending_delayed_conversions.append(
                            PendingConversion(
                                source_adset_id=self.rng.choice(source_ids),
                                clicks=spike_pool,
                                expected_conversions=spike_pool,
                                value=spike_pool,
                                delay_days_remaining=2,
                                original_delay_days=2,
                            )
                        )

        if state.difficulty == "easy":
            reliability_amp = 0.015
            delay_jitter_amp = 0.022
        elif state.difficulty == "medium":
            reliability_amp = 0.035
            delay_jitter_amp = 0.040
        else:
            reliability_amp = 0.050
            delay_jitter_amp = 0.055

        reliability_noise = _deterministic_noise((day_seed * 97) + len(campaign.adsets), reliability_amp)
        effective_tracking_reliability = min(max(state.tracking_reliability + reliability_noise, 0.20), 0.99)

        # Controlled stochasticity: small day-to-day drift and rare disruptive events.
        campaign.ios_traffic_pct = min(max(campaign.ios_traffic_pct + self.rng.uniform(-0.01, 0.01), 0.10), 0.85)
        if "tracking_drop" not in state.risk_events and self.rng.random() < (0.20 if state.difficulty == "hard" else 0.15):
            drop = self.rng.uniform(0.10, 0.16)
            effective_tracking_reliability = max(effective_tracking_reliability - drop, 0.15)
            state.risk_events.append("tracking_drop")

        if not campaign.conversions_api_enabled and self.rng.random() < 0.22:
            effective_tracking_reliability = max(effective_tracking_reliability - self.rng.uniform(0.02, 0.06), 0.15)
            state.risk_events.append("signal_degradation_event")

        state.tracking_reliability = effective_tracking_reliability

        daily_impressions = 0
        daily_clicks = 0
        daily_spend = 0.0
        released_total = 0
        tracked_release = 0
        modeled_release = 0
        remaining_generation_cap = _budgeted_step_cap(state, phase="generate")
        remaining_release_cap = _budgeted_step_cap(state, phase="release")

        min_delay, max_delay = state.scenario_delay_range[0], state.scenario_delay_range[1]

        for idx, adset in enumerate(campaign.adsets):
            if adset.is_paused:
                continue

            params = SEGMENT_PARAMS.get(adset.audience_segment, SEGMENT_PARAMS["broad_interest"])
            decay = SEGMENT_DECAY.get(adset.audience_segment, 0.010)
            age_penalty = max(0.65, 1.0 - (state.day * decay))
            remaining_budget = max(campaign.total_budget - campaign.budget_spent - daily_spend, 0.0)
            active_count = max(len([a for a in campaign.adsets if not a.is_paused]), 1)
            fair_share = remaining_budget / active_count
            daily_budget = min(max(adset.budget * 0.08, 0.0), fair_share)
            momentum_factor = max(min(state.growth_momentum, 1.8), 0.5)
            impressions = int(daily_budget * params["imp_per_usd"] * momentum_factor * age_penalty)
            effective_ctr = min(max(params["ctr"] * 1.08 * age_penalty, 0.01), 0.03)
            clicks = int(impressions * effective_ctr)
            conversion_probability = min(
                max(params["cvr"] * age_penalty * (1.0 + 0.08 * (state.growth_momentum - 1.0)), 0.002),
                0.22,
            )
            optimization_quality = 0.86 + (0.40 * state.tracking_reliability)
            attribution_learning = 0.90 + (0.26 * WINDOW_COVERAGE.get(campaign.attribution_window, 0.72))
            conversion_probability = min(
                conversion_probability * optimization_quality * attribution_learning * state.budget_optimization_multiplier,
                0.28,
            )
            if state.campaign.attribution_window == "1d_click" and state.day >= 1:
                conversion_probability *= 0.92
            if state.day >= 4 and not campaign.conversions_api_enabled:
                conversion_probability *= 0.92
            if state.day >= 5 and not state.tracking_investigated:
                conversion_probability *= 0.90
            low_rate, high_rate = state.conversion_rate_range[0], state.conversion_rate_range[1]
            calibration_mid = (low_rate + high_rate) * 0.5
            conversion_probability = min(max(conversion_probability, low_rate), high_rate)
            conversion_probability = 0.65 * conversion_probability + 0.35 * calibration_mid

            delay_span = max(max_delay - min_delay + 1, 1)
            conversions_by_delay: Dict[int, int] = {}
            for click_i in range(clicks):
                if remaining_generation_cap <= 0:
                    break
                seed = (day_seed * 131) + ((idx + 1) * 17) + (click_i * 7)
                sample = _deterministic_bucket(seed)
                if sample < conversion_probability:
                    delay_pick = _deterministic_bucket(seed * 3 + 11)
                    delay = min_delay
                    cumulative = 0.0
                    weights = []
                    for d in range(delay_span):
                        delay_day = min_delay + d
                        weight_noise = _deterministic_noise(seed + (d * 53), delay_jitter_amp)
                        weights.append(max(_delay_weight(delay_day) + weight_noise, 0.01))
                    weight_sum = sum(weights)
                    for d_idx, weight in enumerate(weights):
                        cumulative += weight / weight_sum
                        if delay_pick <= cumulative:
                            delay = min_delay + d_idx
                            break
                    conversions_by_delay[delay] = conversions_by_delay.get(delay, 0) + 1
                    remaining_generation_cap -= 1

            if clicks > 80 and not conversions_by_delay and remaining_generation_cap > 0:
                fallback_delay = min_delay + ((day_seed + idx) % delay_span)
                conversions_by_delay[fallback_delay] = 1
                remaining_generation_cap -= 1

            for delay, conv_count in conversions_by_delay.items():
                state.pending_delayed_conversions.append(
                    PendingConversion(
                        source_adset_id=adset.adset_id,
                        clicks=clicks,
                        expected_conversions=conv_count,
                        value=conv_count,
                        delay_days_remaining=delay,
                        original_delay_days=delay,
                    )
                )

            adset.spent = round(adset.spent + daily_budget, 2)
            adset.impressions += impressions
            adset.link_clicks += clicks

            daily_impressions += impressions
            daily_clicks += clicks
            daily_spend += daily_budget

        campaign.impressions += daily_impressions
        campaign.link_clicks += daily_clicks
        campaign.budget_spent = round(min(campaign.total_budget, campaign.budget_spent + daily_spend), 2)

        matured: List[PendingConversion] = []
        remaining: List[PendingConversion] = []
        for item in state.pending_delayed_conversions:
            updated = item.model_copy(deep=True)
            updated.delay_days_remaining -= 1
            if updated.delay_days_remaining <= 0:
                matured.append(updated)
            else:
                remaining.append(updated)
        state.pending_delayed_conversions = remaining
        state.pending_conversions = list(state.pending_delayed_conversions)

        for item in matured:
            if remaining_release_cap <= 0:
                state.pending_delayed_conversions.append(
                    PendingConversion(
                        source_adset_id=item.source_adset_id,
                        clicks=item.clicks,
                        expected_conversions=item.expected_conversions,
                        value=item.value,
                        delay_days_remaining=1,
                        original_delay_days=item.original_delay_days,
                    )
                )
                continue

            adset = next((a for a in campaign.adsets if a.adset_id == item.source_adset_id), None)
            if adset is None:
                continue

            true_conv = min(item.expected_conversions, remaining_release_cap)
            spillover = max(item.expected_conversions - true_conv, 0)
            if spillover > 0:
                state.pending_delayed_conversions.append(
                    PendingConversion(
                        source_adset_id=item.source_adset_id,
                        clicks=item.clicks,
                        expected_conversions=spillover,
                        value=spillover,
                        delay_days_remaining=1,
                        original_delay_days=item.original_delay_days,
                    )
                )
            remaining_release_cap -= true_conv
            campaign.true_conversions += true_conv
            adset.true_conversions += true_conv
            state.delayed_true_conversions_total += true_conv

            observed, modeled = _materialize_observed_signals(
                true_conversions=true_conv,
                delay_days=item.original_delay_days,
                attribution_window=campaign.attribution_window,
                tracking_reliability=effective_tracking_reliability,
                modeled_enabled=campaign.modeled_conversions_enabled,
                aem_enabled=campaign.aem_enabled,
            )

            if state.difficulty == "easy" and campaign.attribution_window in {"7d_click", "7d_click_1d_view", "28d_click"}:
                easy_boost = 2.80 if state.tracking_investigated else 2.20
                observed = min(max(int(round(observed * easy_boost)), observed), max(true_conv - modeled, 0))

            if campaign.modeled_conversions_enabled and ("modeled_noise" not in state.risk_events) and self.rng.random() < 0.15:
                extra_modeled = int(round(modeled * self.rng.uniform(0.10, 0.22)))
                if extra_modeled > 0:
                    modeled += extra_modeled
                    state.risk_events.append("modeled_noise")
                    state.risk_events.append("modeled_overestimation")

            if campaign.modeled_conversions_enabled and "modeled_noise" in state.risk_events and state.day <= 2:
                distortion = self.rng.uniform(0.85, 1.15)
                modeled = max(int(round(modeled * distortion)), 0)

            campaign.reported_conversions += observed + modeled
            adset.reported_conversions += observed + modeled
            state.delayed_reported_conversions_total += observed
            state.tracked_conversions_total += observed
            state.modeled_conversions_total += modeled
            hidden = max(true_conv - observed - modeled, 0)
            state.hidden_conversions_pool += hidden
            if hidden > 0:
                state.hidden_delayed_conversions.append(
                    PendingConversion(
                        source_adset_id=item.source_adset_id,
                        clicks=item.clicks,
                        expected_conversions=hidden,
                        value=hidden,
                        delay_days_remaining=0,
                        original_delay_days=item.original_delay_days,
                    )
                )
            released_total += true_conv
            tracked_release += observed
            modeled_release += modeled

        state.delayed_conversion_release_last_step = released_total
        state.tracked_conversion_release_last_step = tracked_release
        state.modeled_conversion_release_last_step = modeled_release
        if state.difficulty == "easy":
            hidden_release_cap = max(int(state.max_released_conversions_per_step * 2.00), 60)
        else:
            hidden_release_cap = min(
                max(int(state.max_released_conversions_per_step * 0.75), 12),
                max(int(state.max_released_conversions_per_step), 1),
            )
        newly_visible_hidden = _reveal_currently_visible_hidden_events(
            state,
            release_cap=max(remaining_release_cap, hidden_release_cap),
        )
        if newly_visible_hidden > 0:
            state.delayed_true_conversions_total += newly_visible_hidden
            state.tracked_conversion_release_last_step += newly_visible_hidden
        state.uncertainty_reintroduced = (
            state.day >= 3
            and state.tracking_investigated
            and state.tracking_reliability < 0.55
            and len(state.pending_delayed_conversions) > 3
            and not campaign.modeled_conversions_enabled
        )

        if not campaign.conversions_api_enabled and state.day >= 2:
            state.tracking_reliability = max(state.tracking_reliability - 0.015, 0.20)
        if action_decay := state.action_counts.get("promote_ad", 0):
            if action_decay > 2:
                state.growth_momentum = max(state.growth_momentum - (0.01 * (action_decay - 2)), 0.50)

        delayed_recovery_ratio = min(
            state.delayed_conversion_release_last_step / max(state.max_released_conversions_per_step, 1),
            1.0,
        )
        attribution_lift = max(WINDOW_COVERAGE.get(campaign.attribution_window, 0.72) - WINDOW_COVERAGE.get("1d_click", 0.30), 0.0)
        signal_lift = max(effective_tracking_reliability - 0.45, 0.0)
        recovered_value_multiplier = min(1.0 + (0.75 * delayed_recovery_ratio) + (0.55 * attribution_lift) + (0.40 * signal_lift), 2.35)
        true_value_multiplier = min(1.0 + (0.45 * delayed_recovery_ratio) + (0.30 * attribution_lift) + (0.25 * signal_lift), 1.85)
        reported_value = avg_order_value * recovered_value_multiplier
        true_value = avg_order_value * true_value_multiplier

        for adset in campaign.adsets:
            adset.reported_roas = compute_roas(adset.reported_conversions, reported_value, adset.spent)
            adset.true_roas = compute_roas(adset.true_conversions, true_value, adset.spent)

        campaign.reported_roas = compute_roas(campaign.reported_conversions, reported_value, campaign.budget_spent)
        campaign.true_roas = compute_roas(campaign.true_conversions, true_value, campaign.budget_spent)
        campaign.reported_cpa = (
            round(campaign.budget_spent / campaign.reported_conversions, 2)
            if campaign.reported_conversions > 0
            else 9999
        )
        campaign.true_cpa = (
            round(campaign.budget_spent / campaign.true_conversions, 2)
            if campaign.true_conversions > 0
            else 9999
        )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _attribution_gap(c: CampaignData) -> float:
    if c.true_conversions == 0:
        return 0.0
    return max((c.true_conversions - c.reported_conversions) / c.true_conversions, 0.0)


def _all_issues_resolved(state: EnvState) -> bool:
    remaining = set(state.issues_remaining) - set(state.issues_resolved)
    return len(remaining) == 0


def _budgeted_step_cap(state: EnvState, phase: str) -> int:
    if phase == "generate":
        base_cap = max(int(state.max_generated_conversions_per_step), 1)
    else:
        base_cap = max(int(state.max_released_conversions_per_step), 1)

    low_target = max(int(0.6 * state.target_true_conversions), 1)
    preferred_high_target = max(int(1.2 * state.target_true_conversions), low_target + 1)
    hard_upper_target = max(int(2.0 * state.target_true_conversions), preferred_high_target + 1)
    current_true = int(state.campaign.true_conversions)
    progress = min(max(current_true / preferred_high_target, 0.0), 1.0)

    if current_true >= hard_upper_target:
        return 0

    if current_true < low_target:
        return base_cap

    # Taper per-step conversions as trajectory approaches preferred realism bound.
    low_progress = low_target / preferred_high_target
    taper = 1.0 - ((progress - low_progress) / max(1.0 - low_progress, 1e-6))
    taper = min(max(taper, 0.20), 1.0)
    return max(int(round(base_cap * taper)), 1)


def _deterministic_bucket(value: int, modulo: int = 1000) -> float:
    return (value % modulo) / float(modulo)


def _deterministic_noise(seed: int, amplitude: float) -> float:
    return (_deterministic_bucket(seed, 10000) - 0.5) * 2.0 * amplitude


def _delay_weight(delay_days: int) -> float:
    if delay_days <= 3:
        return 0.60
    if delay_days <= 5:
        return 0.28
    return 0.12


def _issue_resolution_fraction(state: EnvState) -> float:
    required = set(state.issues_remaining)
    if not required:
        return 1.0
    solved = len(required & set(state.issues_resolved))
    return solved / len(required)


def _materialize_observed_signals(
    true_conversions: int,
    delay_days: int,
    attribution_window: str,
    tracking_reliability: float,
    modeled_enabled: bool,
    aem_enabled: bool,
) -> Tuple[int, int]:
    window_days = WINDOW_DAYS.get(attribution_window, 7)
    if delay_days > window_days:
        observed_base = 0
    else:
        observed_base = int(round(true_conversions * WINDOW_COVERAGE.get(attribution_window, 0.72)))

    observed = int(round(observed_base * tracking_reliability))
    unattributed = max(true_conversions - observed, 0)

    modeled = 0
    if modeled_enabled:
        modeled_factor = 0.42 + (0.12 if aem_enabled else 0.0)
        modeled = int(round(unattributed * modeled_factor))

        # Keep modeled share meaningful once modeled reporting is active.
        total_recovered = observed + modeled
        if total_recovered > 0:
            target_modeled_share = 0.30 if aem_enabled else 0.22
            required_modeled = int(round(total_recovered * target_modeled_share))
            if modeled < required_modeled:
                shift = min(observed, required_modeled - modeled)
                modeled += shift
                observed -= shift

    return max(observed, 0), max(modeled, 0)


def _diminishing_returns(action_count: int) -> float:
    return 1.0 / (1.0 + 0.65 * max(action_count - 1, 0))


def _redundancy_penalty(action_type: str, action_count: int) -> float:
    if action_type == "no_op":
        return -0.05
    if action_count <= 1:
        return 0.0
    return -min(0.02 * (action_count - 1), 0.12)


def _is_stack_stable(state: EnvState) -> bool:
    c = state.campaign
    return (
        c.attribution_window in {"7d_click", "7d_click_1d_view", "28d_click"}
        and c.conversions_api_enabled
        and c.aem_enabled
        and c.attribution_reporting_mode == "modeled"
        and "paused_bad_adsets" in state.issues_resolved
    )


def _is_converged(state: EnvState) -> bool:
    c = state.campaign
    major_remaining = {"attribution_window", "conversions_api", "aem", "paused_bad_adsets", "budget_allocation"}
    unresolved_major = major_remaining - set(state.issues_resolved)
    roas_stable = True
    if len(state.roas_history) >= 3:
        tail = state.roas_history[-3:]
        roas_stable = (max(tail) - min(tail)) <= 0.12
    return (
        _attribution_gap(c) < 0.10
        and state.tracking_reliability >= 0.92
        and roas_stable
        and len(unresolved_major) == 0
    )


def _ordering_bonus(state: EnvState, action_type: str) -> float:
    # Multiple valid paths: A is ideal, B is strong alternative, C is risky but recoverable.
    strategy_paths = [
        (
            [
            "investigate_attribution",
            "adjust_attribution_window",
            "enable_conversions_api",
            "enable_aggregated_event_measurement",
            "switch_to_modeled_conversions",
            "pause_underperforming_adsets",
            "reallocate_to_top_performers",
            "promote_ad",
            ],
            0.016,
        ),
        (
            [
            "enable_conversions_api",
            "enable_aggregated_event_measurement",
            "switch_to_modeled_conversions",
            "investigate_attribution",
            "adjust_attribution_window",
            "pause_underperforming_adsets",
            "reallocate_to_top_performers",
            "promote_ad",
            ],
            0.012,
        ),
        (
            [
            "investigate_attribution",
            "promote_ad",
            "adjust_attribution_window",
            "enable_conversions_api",
            "enable_aggregated_event_measurement",
            "switch_to_modeled_conversions",
            "pause_underperforming_adsets",
            "reallocate_to_top_performers",
            ],
            0.008,
        ),
    ]

    if action_type == "no_op":
        return -0.01

    history_actions = [step.get("action", "") for step in state.history]
    best_bonus = -0.02
    for path, base_bonus in strategy_paths:
        if action_type not in path:
            continue
        idx = path.index(action_type)
        prereq = set(path[:idx])
        already = set(history_actions)
        matched = len(prereq & already)
        missing = max(len(prereq) - matched, 0)
        candidate = base_bonus + (0.005 * matched) - (0.014 * missing)
        if idx <= 2:
            candidate += 0.006
        best_bonus = max(best_bonus, candidate)

    if action_type in {"promote_ad", "reallocate_to_top_performers", "adjust_budget_allocation"} and not state.tracking_investigated:
        best_bonus -= 0.03

    return round(max(min(best_bonus, 0.06), -0.08), 4)


def _reveal_currently_visible_hidden_events(state: EnvState, release_cap: int) -> int:
    if not state.hidden_delayed_conversions:
        return 0

    window_days = WINDOW_DAYS.get(state.campaign.attribution_window, 7)
    released = 0
    remaining_hidden: List[PendingConversion] = []
    for evt in state.hidden_delayed_conversions:
        if release_cap <= 0:
            remaining_hidden.append(evt)
            continue
        if evt.original_delay_days <= window_days:
            observed, modeled = _materialize_observed_signals(
                true_conversions=evt.expected_conversions,
                delay_days=evt.original_delay_days,
                attribution_window=state.campaign.attribution_window,
                tracking_reliability=state.tracking_reliability,
                modeled_enabled=state.campaign.modeled_conversions_enabled,
                aem_enabled=state.campaign.aem_enabled,
            )
            if state.difficulty == "easy" and state.campaign.attribution_window in {"7d_click", "7d_click_1d_view", "28d_click"}:
                easy_boost = 2.60 if state.tracking_investigated else 2.00
                observed = min(max(int(round(observed * easy_boost)), observed), max(evt.expected_conversions - modeled, 0))
            newly_visible = observed + modeled
            if newly_visible > release_cap:
                scale = release_cap / max(newly_visible, 1)
                observed = int(round(observed * scale))
                modeled = min(release_cap - observed, int(round(modeled * scale)))
                newly_visible = observed + modeled
            if newly_visible > 0:
                released += newly_visible
                release_cap -= newly_visible
                state.campaign.reported_conversions += newly_visible
                state.tracked_conversions_total += observed
                state.modeled_conversions_total += modeled
                state.delayed_reported_conversions_total += observed
                state.hidden_conversions_pool = max(state.hidden_conversions_pool - newly_visible, 0)
                if state.campaign.adsets:
                    adset = next((a for a in state.campaign.adsets if a.adset_id == evt.source_adset_id), None)
                    if adset is not None:
                        adset.reported_conversions += newly_visible
            still_hidden = max(evt.expected_conversions - newly_visible, 0)
            if still_hidden > 0:
                remaining_hidden.append(
                    PendingConversion(
                        source_adset_id=evt.source_adset_id,
                        clicks=evt.clicks,
                        expected_conversions=still_hidden,
                        value=still_hidden,
                        delay_days_remaining=0,
                        original_delay_days=evt.original_delay_days,
                    )
                )
        else:
            remaining_hidden.append(evt)

    state.hidden_delayed_conversions = remaining_hidden
    return released