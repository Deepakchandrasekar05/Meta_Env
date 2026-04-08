"""
grader.py — Programmatic agent graders for all three tasks.

Each grader receives a completed EnvState and returns a TaskResult
with a 0.0–1.0 score, pass/fail verdict, and breakdown.

Task criteria:
  EASY   — window changed to 7d_click (or better). Score by gap closure.
  MEDIUM — CAPI + AEM enabled. Score by signal quality achieved.
  HARD   — all 5 issues resolved; score by weighted composite.
"""

from __future__ import annotations
from typing import Dict, List
from pydantic import BaseModel

from meta_ads_env.models import EnvState
from meta_ads_env.reward import penalise_trajectory
from meta_ads_env.simulator import _attribution_gap, compute_pixel_quality


PASS_THRESHOLD = 0.60   # minimum score to "pass" a task


def _trajectory_metrics(state: EnvState, initial_gap: float, initial_signal: float, initial_true_roas: float) -> Dict[str, float]:
    c = state.campaign

    final_gap = _attribution_gap(c)
    gap_reduction = (max(initial_gap - final_gap, 0) / initial_gap) if initial_gap > 0 else 1.0

    final_signal = state.tracking_reliability
    signal_recovery = (
        max(final_signal - initial_signal, 0) / max(1.0 - initial_signal, 0.01)
        if initial_signal < 1.0
        else 1.0
    )

    roas_improvement = (
        max(c.true_roas - initial_true_roas, 0) / max(initial_true_roas, 0.01)
        if initial_true_roas > 0
        else 0.0
    )

    efficiency = max(0.0, 1.0 - (state.step_count / max(state.max_steps, 1)))
    action_efficiency = 1.0 - min(max(state.step_count - state.optimal_steps_hint, 0) / max(state.max_steps, 1), 1.0)
    redundancy_penalty = max(-penalise_trajectory(state.history), 0.0)

    return {
        "gap_reduction": round(min(max(gap_reduction, 0.0), 1.0), 4),
        "signal_recovery": round(min(max(signal_recovery, 0.0), 1.0), 4),
        "roas_improvement": round(min(max(roas_improvement, 0.0), 1.0), 4),
        "efficiency": round(efficiency, 4),
        "action_efficiency": round(action_efficiency, 4),
        "redundancy_penalty": round(redundancy_penalty, 4),
        "issues_resolved_count": float(len(set(state.issues_resolved))),
    }


class TaskResult(BaseModel):
    task_id: str
    difficulty: str
    score: float                    # 0.0 – 1.0
    passed: bool
    breakdown: Dict[str, float]
    feedback: List[str]
    steps_used: int
    cumulative_reward: float


# ─────────────────────────────────────────────────────────────────────────────
# EASY grader
# ─────────────────────────────────────────────────────────────────────────────

def grade_easy(state: EnvState, initial_gap: float = 0.62) -> TaskResult:
    c = state.campaign
    feedback: List[str] = []

    # Primary criterion: attribution window changed to ≥ 7d_click
    window_ok = c.attribution_window in {"7d_click", "7d_click_1d_view", "28d_click"}
    window_score = 1.0 if window_ok else 0.0
    if window_ok:
        feedback.append(f"✅ Attribution window correctly set to '{c.attribution_window}'")
    else:
        feedback.append(f"❌ Attribution window still '{c.attribution_window}' — should be 7d_click or wider")

    metrics = _trajectory_metrics(
        state,
        initial_gap=initial_gap,
        initial_signal=state.signal_quality_history[0] if state.signal_quality_history else state.tracking_reliability,
        initial_true_roas=state.campaign.true_roas,
    )

    gap_closed = metrics["gap_reduction"]
    if gap_closed >= 0.50:
        feedback.append(f"✅ Attribution gap reduced by {gap_closed:.0%}")
    else:
        feedback.append(f"⚠️  Attribution gap only reduced by {gap_closed:.0%}")

    # Efficiency
    efficiency = metrics["efficiency"]
    feedback.append(f"ℹ️  Completed in {state.step_count}/{state.max_steps} steps")

    score = round(
        max(
            (window_score * 0.50)
            + (gap_closed * 0.30)
            + (metrics["signal_recovery"] * 0.05)
            + (metrics["action_efficiency"] * 0.15)
            - (metrics["redundancy_penalty"] * 0.10),
            0.0,
        ),
        4,
    )

    return TaskResult(
        task_id=state.task_id,
        difficulty=state.difficulty,
        score=score,
        passed=score >= PASS_THRESHOLD,
        breakdown={
            "window_correct":   window_score,
            "gap_closed":       round(gap_closed, 4),
            "efficiency":       round(efficiency, 4),
            "signal_recovery": metrics["signal_recovery"],
            "action_efficiency": metrics["action_efficiency"],
            "redundant_action_penalty": metrics["redundancy_penalty"],
            "issues_resolved_count": metrics["issues_resolved_count"],
        },
        feedback=feedback,
        steps_used=state.step_count,
        cumulative_reward=state.cumulative_reward,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MEDIUM grader
# ─────────────────────────────────────────────────────────────────────────────

def grade_medium(state: EnvState, initial_signal: float = 0.325) -> TaskResult:
    c = state.campaign
    feedback: List[str] = []

    # Primary: CAPI enabled (biggest lever)
    capi_score = 1.0 if c.conversions_api_enabled else 0.0
    if c.conversions_api_enabled:
        feedback.append("✅ Conversions API enabled")
    else:
        feedback.append("❌ Conversions API NOT enabled — biggest signal recovery lever missed")

    # Secondary: AEM enabled
    aem_score = 1.0 if c.aem_enabled else 0.0
    if c.aem_enabled:
        feedback.append("✅ Aggregated Event Measurement enabled")
    else:
        feedback.append("⚠️  AEM not enabled — modelled conversions unavailable")

    metrics = _trajectory_metrics(
        state,
        initial_gap=state.attribution_gap_history[0] if state.attribution_gap_history else _attribution_gap(c),
        initial_signal=initial_signal,
        initial_true_roas=state.campaign.true_roas,
    )

    # Signal quality achieved
    achieved_signal = state.tracking_reliability
    optimal_signal  = compute_pixel_quality(c.ios_traffic_pct, True, True, True)
    signal_fraction = (achieved_signal - initial_signal) / max(optimal_signal - initial_signal, 0.01)
    signal_fraction = round(min(max(signal_fraction, 0), 1), 4)
    feedback.append(
        f"ℹ️  Signal quality: {initial_signal:.0%} → {achieved_signal:.0%} "
        f"(optimal: {optimal_signal:.0%})"
    )

    efficiency = metrics["efficiency"]

    score = round(
        max(
            capi_score * 0.40
            + aem_score * 0.25
            + signal_fraction * 0.25
            + metrics["action_efficiency"] * 0.10
            + metrics["roas_improvement"] * 0.08
            - metrics["redundancy_penalty"] * 0.08,
            0.0,
        ),
        4,
    )

    return TaskResult(
        task_id=state.task_id,
        difficulty=state.difficulty,
        score=score,
        passed=score >= PASS_THRESHOLD,
        breakdown={
            "capi_enabled":    capi_score,
            "aem_enabled":     aem_score,
            "signal_recovery": signal_fraction,
            "efficiency":      round(efficiency, 4),
            "roas_improvement": metrics["roas_improvement"],
            "action_efficiency": metrics["action_efficiency"],
            "redundant_action_penalty": metrics["redundancy_penalty"],
            "issues_resolved_count": metrics["issues_resolved_count"],
        },
        feedback=feedback,
        steps_used=state.step_count,
        cumulative_reward=state.cumulative_reward,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HARD grader
# ─────────────────────────────────────────────────────────────────────────────

def grade_hard(
    state: EnvState,
    initial_gap: float    = 0.785,
    initial_signal: float = 0.280,
    initial_true_roas: float = 1.61,
) -> TaskResult:
    c = state.campaign
    feedback: List[str] = []
    issues_required = {
        "attribution_window",
        "conversions_api",
        "aem",
        "modeled_reporting",
        "tracking_investigated",
        "budget_allocation",
        "paused_bad_adsets",
    }
    resolved = set(state.issues_resolved) & issues_required

    checks: Dict[str, float] = {}

    # 1. Attribution window
    w_ok = c.attribution_window in {"7d_click", "7d_click_1d_view", "28d_click"}
    checks["attribution_window"] = 1.0 if w_ok else 0.0
    feedback.append(("✅" if w_ok else "❌") + f" Attribution window: {c.attribution_window}")

    # 2. Conversions API
    checks["conversions_api"] = 1.0 if c.conversions_api_enabled else 0.0
    feedback.append(("✅" if c.conversions_api_enabled else "❌") + " Conversions API")

    # 3. AEM
    checks["aem"] = 1.0 if c.aem_enabled else 0.0
    feedback.append(("✅" if c.aem_enabled else "❌") + " AEM")

    # 4. Budget allocation — did agent touch budgets or pause bad adsets?
    paused_any = any(a.is_paused for a in c.adsets)
    checks["paused_bad_adsets"] = 1.0 if paused_any else 0.0
    feedback.append(("✅" if paused_any else "❌") + " Paused under-performing adsets")

    checks["tracking_investigated"] = 1.0 if state.tracking_investigated else 0.0
    feedback.append(("✅" if state.tracking_investigated else "❌") + " Tracking investigated")

    checks["modeled_reporting"] = 1.0 if c.attribution_reporting_mode == "modeled" else 0.0
    feedback.append(("✅" if c.attribution_reporting_mode == "modeled" else "❌") + " Modeled reporting enabled")

    # 5. Budget reallocation
    budget_reallocated = "budget_allocation" in state.issues_resolved or "budget_reallocation" in state.issues_resolved
    checks["budget_allocation"] = 1.0 if budget_reallocated else 0.0
    feedback.append(("✅" if budget_reallocated else "❌") + " Budget reallocated to top performers")

    metrics = _trajectory_metrics(
        state,
        initial_gap=initial_gap,
        initial_signal=initial_signal,
        initial_true_roas=initial_true_roas,
    )

    gap_closed = metrics["gap_reduction"]
    sig_recovery = metrics["signal_recovery"]
    roas_gain = metrics["roas_improvement"]

    feedback.append(
        f"ℹ️  Gap closed: {gap_closed:.0%} | Signal: {initial_signal:.0%}→{c.pixel_signal_quality:.0%} | "
        f"True ROAS: {initial_true_roas:.2f}→{c.true_roas:.2f}"
    )

    issues_fraction = len(resolved) / len(issues_required)
    efficiency = metrics["efficiency"]

    critical_missing_penalty = (
        (1.0 - checks["paused_bad_adsets"]) * 0.15
        + (1.0 - checks["tracking_investigated"]) * 0.07
        + (1.0 - checks["modeled_reporting"]) * 0.08
    )

    score = round(
        max(
            issues_fraction * 0.40
            + gap_closed    * 0.20
            + sig_recovery  * 0.15
            + roas_gain     * 0.15
            + metrics["action_efficiency"] * 0.10
            - metrics["redundancy_penalty"] * 0.10,
            - critical_missing_penalty,
            0.0,
        ),
        4,
    )

    return TaskResult(
        task_id=state.task_id,
        difficulty=state.difficulty,
        score=score,
        passed=score >= PASS_THRESHOLD,
        breakdown={
            **{f"issue_{k}": v for k, v in checks.items()},
            "issues_fraction": round(issues_fraction, 4),
            "gap_closed":      round(gap_closed, 4),
            "signal_recovery": round(sig_recovery, 4),
            "roas_gain":       round(roas_gain, 4),
            "efficiency":      round(efficiency, 4),
            "action_efficiency": metrics["action_efficiency"],
            "redundant_action_penalty": metrics["redundancy_penalty"],
            "critical_missing_penalty": round(critical_missing_penalty, 4),
            "issues_resolved_count": metrics["issues_resolved_count"],
        },
        feedback=feedback,
        steps_used=state.step_count,
        cumulative_reward=state.cumulative_reward,
    )


# ─── Dispatcher ──────────────────────────────────────────────────────────────

GRADERS = {
    "easy_attribution_window":     grade_easy,
    "medium_pixel_recovery":       grade_medium,
    "hard_full_attribution_audit": grade_hard,
}


def grade(state: EnvState, **kwargs) -> TaskResult:
    grader_fn = GRADERS.get(state.task_id)
    if grader_fn is None:
        raise ValueError(f"No grader for task '{state.task_id}'")
    return grader_fn(state, **kwargs)