"""
reward.py — Reward function with partial-progress signals.

Reward breakdown (total sums to ≤ 1.0):
  0.35  Attribution accuracy improvement  (gap between true & reported closes)
  0.25  Signal quality gain               (pixel quality rises)
  0.25  ROAS improvement                  (reported ROAS approaches true ROAS)
  0.10  Action validity                   (valid, non-redundant action)
  0.05  Step efficiency                   (bonus for resolving early)

Penalties:
  -0.05 per no_op action
  -0.02 per repeated identical action
"""

from __future__ import annotations
from typing import List

from meta_ads_env.models import Action, EnvState, Reward, RewardComponents


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

    # 2. Signal quality recovery
    if initial_signal < 1.0:
        signal_recovered = max(c.pixel_signal_quality - initial_signal, 0) / (1.0 - initial_signal)
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

    # Weighted final score
    score = (
        gap_closed        * MAX_COMPONENTS["attribution_accuracy"]
        + signal_recovered * MAX_COMPONENTS["signal_quality_gain"]
        + roas_ratio       * MAX_COMPONENTS["roas_improvement"]
        + issues_fraction  * (MAX_COMPONENTS["action_validity"] + MAX_COMPONENTS["step_efficiency"])
    )

    return round(min(score, 1.0), 4)


def penalise_trajectory(history: List[dict]) -> float:
    """
    Scans action history and returns a total penalty (negative float).
    """
    penalty = 0.0
    seen_actions: List[str] = []

    for step in history:
        act = step.get("action", "")
        if act == "no_op":
            penalty -= 0.05
        if act in seen_actions:
            penalty -= 0.02
        seen_actions.append(act)

    return round(penalty, 4)