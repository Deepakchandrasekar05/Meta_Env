"""
env.py — MetaAdsAttributionEnv: the main OpenEnv environment class.

Public API (OpenEnv spec):
    env = MetaAdsAttributionEnv(task_id="easy_attribution_window")
    obs  = env.reset()
    obs, reward, done, info = env.step(action)
    state = env.state()
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from meta_ads_env.models import Action, EnvState, Observation, Reward
from meta_ads_env.simulator import (
    SimulationEngine,
    WINDOW_COVERAGE,
    _attribution_gap,
    compute_pixel_quality,
    compute_server_signal_quality,
    compute_tracking_reliability,
)
from meta_ads_env.tasks import TASK_REGISTRY, get_task
from meta_ads_env.grader import grade, TaskResult

AVG_ORDER_VALUE = 75.0

AVAILABLE_ACTIONS: List[str] = [
    "investigate_attribution",
    "switch_to_modeled_conversions",
    "promote_ad",
    "reduce_budget",
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


class MetaAdsAttributionEnv:
    """
    OpenEnv-compliant environment for Meta Ads attribution recovery.

    Simulates the real-world problem of Meta Ads under-reporting conversions
    due to narrow attribution windows, iOS Pixel degradation, and missing
    server-side signals. An agent must diagnose and remediate these issues.
    """

    metadata = {
        "name": "meta-ads-attribution-env",
        "version": "1.0.0",
        "tasks": list(TASK_REGISTRY.keys()),
    }

    def __init__(self, task_id: str = "easy_attribution_window"):
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task '{task_id}'. Choose from: {list(TASK_REGISTRY)}")
        self.task_id   = task_id
        self._engine   = SimulationEngine()
        self._state: Optional[EnvState] = None
        self._initial_gap:    float = 0.0
        self._initial_signal: float = 0.0
        self._initial_true_roas: float = 0.0

    # ─── reset ───────────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset the environment to its initial state and return the first observation."""
        self._state = get_task(self.task_id)
        self._apply_reset_randomization()
        c = self._state.campaign
        self._initial_gap      = _attribution_gap(c)
        self._initial_signal   = self._state.tracking_reliability
        self._initial_true_roas = c.true_roas
        return self._build_observation()

    # ─── step ────────────────────────────────────────────────────────────────

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Apply an action to the environment.

        Returns:
            observation: current environment state as Observation
            reward:      Reward model with total and component breakdown
            done:        True if episode is over
            info:        auxiliary diagnostic information
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        new_state, reward, done, info = self._engine.apply(
            self._state, action, avg_order_value=AVG_ORDER_VALUE
        )
        self._state = new_state
        obs = self._build_observation()
        return obs, reward, done, info

    # ─── state ───────────────────────────────────────────────────────────────

    def state(self) -> EnvState:
        """Return the full internal environment state."""
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    # ─── grade ───────────────────────────────────────────────────────────────

    def grade_episode(self) -> TaskResult:
        """
        Score the completed episode with the task-specific grader.
        Call after done=True.
        """
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        kwargs: Dict[str, Any] = {}
        if self._state.task_id == "easy_attribution_window":
            kwargs["initial_gap"] = self._initial_gap
        elif self._state.task_id == "medium_pixel_recovery":
            kwargs["initial_signal"] = self._initial_signal
        elif self._state.task_id == "hard_full_attribution_audit":
            kwargs["initial_gap"]       = self._initial_gap
            kwargs["initial_signal"]    = self._initial_signal
            kwargs["initial_true_roas"] = self._initial_true_roas
        return grade(self._state, **kwargs)

    # ─── helpers ─────────────────────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        s = self._state
        c = s.campaign
        true_c  = c.true_conversions
        rep_c   = c.reported_conversions
        gap_pct = _attribution_gap(c)
        confidence = min(max(s.confidence_score, 0.0), 1.0)
        obs_noise = 0.02 + (0.03 * (1.0 - confidence))
        observed_gap = min(max(gap_pct + self._engine.rng.uniform(-obs_noise, obs_noise), 0.0), 1.0)
        observed_signal = min(max(s.tracking_reliability + self._engine.rng.uniform(-obs_noise, obs_noise), 0.0), 1.0)
        observed_ios = min(max(c.ios_traffic_pct + self._engine.rng.uniform(-0.03, 0.03), 0.0), 1.0)
        observed_factor = max(
            WINDOW_COVERAGE.get(c.attribution_window, 0.72) * max(observed_signal, 0.10),
            0.10,
        )
        inferred_true = int(rep_c / observed_factor)

        # Build adset breakdown for agent
        adset_info = []
        if c.adsets:
            adset_info.append("\n\nAdset Performance Breakdown:")
            for adset in c.adsets:
                status = "PAUSED" if adset.is_paused else "ACTIVE"
                adset_info.append(
                    f"  • {adset.adset_name} ({status}): "
                    f"Spent ${adset.spent:,.0f}/{adset.budget:,.0f} | "
                    f"Reported ROAS: {adset.reported_roas:.2f}x | "
                    f"True ROAS: {adset.true_roas:.2f}x | "
                    f"Segment: {adset.audience_segment}"
                )
        adset_context = "\n".join(adset_info)

        context = (
            f"Campaign '{c.campaign_name}' | Objective: {c.objective}\n"
            f"Simulation day: {s.day}\n"
            f"Spend: ${c.budget_spent:,.0f} / ${c.total_budget:,.0f}\n"
            f"Reported conversions: {rep_c} | Estimated true conversions: {inferred_true}\n"
            f"Attribution gap (estimated): {observed_gap:.1%} of conversions are UNTRACKED\n"
            f"Attribution window: {c.attribution_window}\n"
            f"Pixel signal quality (estimated): {min(max(c.pixel_signal_quality + self._engine.rng.uniform(-0.02, 0.02), 0.0), 1.0):.0%}  "
            f"(iOS traffic est.: {observed_ios:.0%})\n"
            f"Pixel match quality: {c.pixel_match_quality:.0%} | CAPI coverage: {c.capi_coverage:.0%}\n"
            f"Server-side signal quality: {c.server_signal_quality:.0%}\n"
            f"Attribution confidence: {s.attribution_confidence:.0%}\n"
            f"Tracking reliability (estimated observability): {observed_signal:.0%}\n"
            f"State confidence score: {s.confidence_score:.0%}\n"
            f"Tracking investigated: {'YES' if s.tracking_investigated else 'NO'} | "
            f"Uncertainty reintroduced: {'YES' if s.uncertainty_reintroduced else 'NO'}\n"
            f"Conversions API: {'ON' if c.conversions_api_enabled else 'OFF'}  |  "
            f"AEM: {'ON' if c.aem_enabled else 'OFF'}  |  "
            f"UTM: {'ON' if c.utm_tracking else 'OFF'}\n"
            f"Reporting mode: {c.attribution_reporting_mode}\n"
            f"Pending delayed conversions: {len(s.pending_delayed_conversions)} events\n"
            f"Delayed conversions released this step: {s.delayed_conversion_release_last_step}\n"
            f"Cumulative delayed conversions: {s.delayed_true_conversions_total}\n"
            f"Tracked conversions accumulated: {s.tracked_conversions_total}\n"
            f"Modeled conversions accumulated: {s.modeled_conversions_total}\n"
            f"Delayed reward buffer: {s.delayed_reward_buffer:.3f} | Released this step: {s.delayed_reward_released_last_step:.3f}\n"
            f"Terminal bonus (last): {s.terminal_bonus_last_step:.3f}\n"
            f"Recent risk events: {s.risk_events[-3:] if s.risk_events else []}\n"
            f"Reported ROAS: {c.reported_roas:.2f}x  |  True ROAS: {c.true_roas:.2f}x\n"
            f"{adset_context}\n"
            f"Step {s.step_count}/{s.max_steps}\n"
            f"Diagnostic summary: signal_quality={observed_signal:.0%}, "
            f"attribution_gap={observed_gap:.1%}, confidence={s.confidence_score:.0%}\n"
            f"Operational health: budget_multiplier={s.budget_optimization_multiplier:.2f}, "
            f"convergence_stagnation={s.convergence_stagnation_count}"
        )

        return Observation(
            task_id=s.task_id,
            difficulty=s.difficulty,
            step_count=s.step_count,
            max_steps=s.max_steps,
            campaign_data=c,
            reported_conversions=rep_c,
            estimated_true_conversions=inferred_true,
            attribution_gap_pct=round(observed_gap, 4),
            pixel_signal_quality=round(min(max(c.pixel_signal_quality + self._engine.rng.uniform(-0.02, 0.02), 0.0), 1.0), 4),
            ios_traffic_pct=round(observed_ios, 4),
            budget_remaining=c.total_budget - c.budget_spent,
            roas_reported=c.reported_roas,
            roas_true=c.true_roas,
            attribution_confidence=s.attribution_confidence,
            capi_coverage=c.capi_coverage,
            pixel_match_quality=c.pixel_match_quality,
            conversion_delay_index=c.conversion_delay_index,
            avg_conversion_delay_days=c.avg_conversion_delay_days,
            pending_delayed_conversions=len(s.pending_delayed_conversions),
            modeled_conversions_accumulated=s.modeled_conversions_total,
            tracked_conversions_accumulated=s.tracked_conversions_total,
            delayed_conversion_release_events=s.delayed_conversion_release_last_step,
            cumulative_delayed_conversions=s.delayed_true_conversions_total,
            issues_resolved_count=len(set(s.issues_resolved)),
            available_actions=AVAILABLE_ACTIONS,
            context=context,
            done=s.done,
        )

    def _apply_reset_randomization(self) -> None:
        if self._state is None:
            return
        s = self._state
        c = s.campaign

        ios_jitter = self._engine.rng.uniform(-0.05, 0.05)
        c.ios_traffic_pct = min(max(c.ios_traffic_pct + ios_jitter, 0.10), 0.90)

        base_gap = _attribution_gap(c)
        gap_scale = self._engine.rng.uniform(0.90, 1.10)
        target_gap = min(max(base_gap * gap_scale, 0.02), 0.92)
        target_reported = int(round(c.true_conversions * (1.0 - target_gap)))
        c.reported_conversions = min(max(target_reported, 0), c.true_conversions)
        c.reported_cpa = round(c.budget_spent / c.reported_conversions, 2) if c.reported_conversions > 0 else 9999
        c.true_cpa = round(c.budget_spent / c.true_conversions, 2) if c.true_conversions > 0 else 9999
        c.reported_roas = round((c.reported_conversions * AVG_ORDER_VALUE) / c.budget_spent, 3) if c.budget_spent > 0 else 0.0
        c.true_roas = round((c.true_conversions * AVG_ORDER_VALUE) / c.budget_spent, 3) if c.budget_spent > 0 else 0.0

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

        signal_noise = self._engine.rng.uniform(-0.08, 0.08)
        s.tracking_reliability = min(
            max(compute_tracking_reliability(c, s.attribution_investigation_level) + signal_noise, 0.15),
            0.98,
        )

    @staticmethod
    def action_space() -> Dict[str, Any]:
        return {
            "type": "discrete+params",
            "actions": AVAILABLE_ACTIONS,
            "parameters": {
                "promote_ad": {},
                "reduce_budget": {"scale": "float (0.60-0.98)"},
                "investigate_attribution": {},
                "switch_to_modeled_conversions": {},
                "adjust_attribution_window": {"window": ["1d_click", "7d_click", "7d_click_1d_view", "28d_click", "1d_view"]},
                "adjust_budget_allocation":  {"shifts": "dict[adset_id, new_budget_usd]"},
                "change_bid_strategy":       {"strategy": ["lowest_cost", "cost_cap", "bid_cap", "value_optimisation"]},
                "pause_underperforming_adsets": {"roas_threshold": "float"},
                "reallocate_to_top_performers": {"amount": "float (USD)"},
            },
        }

    @staticmethod
    def observation_space() -> Dict[str, Any]:
        return Observation.model_json_schema()