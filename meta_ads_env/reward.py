"""Reward utilities for episode-level scoring and trajectory penalties."""

from __future__ import annotations
from typing import List

from meta_ads_env.models import EnvState


MAX_COMPONENTS = {
    "attribution_accuracy": 0.35,
    "signal_quality_gain":  0.25,
    "roas_improvement":     0.25,
    "action_validity":      0.10,
    "step_efficiency":      0.05,
}


def compute_episode_reward(
    final_state: EnvState,
    initial_true_roas: float,
    initial_gap: float,
    initial_signal: float,
) -> float:
    """
    Episode-level (terminal) reward computed after all steps.
    Returns 0.0–1.0 score used by the grader.
    """
    c = final_state.campaign

    # 1. Attribution gap closure (% of gap that was closed)
    if initial_gap > 0:
        final_gap = max(
            (c.true_conversions - c.reported_conversions) / c.true_conversions, 0
        )
        gap_closed = max(initial_gap - final_gap, 0) / initial_gap
    else:
        gap_closed = 1.0

    # 2. Signal quality recovery (tracking reliability, not just pixel quality)
    if initial_signal < 1.0:
        signal_recovered = max(final_state.tracking_reliability - initial_signal, 0) / (1.0 - initial_signal)
    else:
        signal_recovered = 1.0

    # 3. ROAS improvement ratio
    if initial_true_roas > 0 and c.true_roas > initial_true_roas:
        roas_ratio = min((c.true_roas - initial_true_roas) / initial_true_roas, 1.0)
    else:
        roas_ratio = 0.0

    # 4. Issues resolved fraction
    issues_total = len(set(final_state.issues_remaining))
    issues_fixed = len(set(final_state.issues_resolved) & set(final_state.issues_remaining))
    issues_fraction = (issues_fixed / issues_total) if issues_total > 0 else 1.0

    # 5. Action efficiency and redundancy quality
    action_efficiency = 1.0 - min(
        max(final_state.step_count - final_state.optimal_steps_hint, 0) / max(final_state.max_steps, 1),
        1.0,
    )
    redundancy_penalty = max(-penalise_trajectory(final_state.history), 0.0)

    # Weighted final score
    score = (
        gap_closed         * MAX_COMPONENTS["attribution_accuracy"]
        + signal_recovered * MAX_COMPONENTS["signal_quality_gain"]
        + roas_ratio        * MAX_COMPONENTS["roas_improvement"]
        + issues_fraction   * MAX_COMPONENTS["action_validity"]
        + action_efficiency * MAX_COMPONENTS["step_efficiency"]
        - redundancy_penalty * 0.05
    )

    return round(min(score, 1.0), 4)


def penalise_trajectory(history: List[dict]) -> float:
    """
    Scans action history and returns a total penalty (negative float).
    """
    penalty = 0.0
    seen_actions: List[str] = []

    previous_action = ""
    repeated_streak = 0

    for step in history:
        act = step.get("action", "")

        if act == "no_op":
            penalty -= 0.05

        if act in seen_actions:
            penalty -= 0.02

        if act == previous_action and act:
            repeated_streak += 1
            penalty -= min(0.01 * repeated_streak, 0.05)
        else:
            repeated_streak = 0

        if act == "reduce_budget" and previous_action == "promote_ad":
            penalty -= 0.015

        if act == "promote_ad" and previous_action == "reduce_budget":
            penalty -= 0.015

        previous_action = act
        seen_actions.append(act)

    return round(penalty, 4)