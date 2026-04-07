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
from meta_ads_env.reward import compute_episode_reward, penalise_trajectory
from meta_ads_env.simulator import _attribution_gap, compute_pixel_quality


PASS_THRESHOLD = 0.60   # minimum score to "pass" a task


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

    # Secondary: how much of the gap was closed
    final_gap = _attribution_gap(c)
    gap_closed = max(initial_gap - final_gap, 0) / initial_gap if initial_gap > 0 else 1.0
    if gap_closed >= 0.50:
        feedback.append(f"✅ Attribution gap reduced by {gap_closed:.0%}")
    else:
        feedback.append(f"⚠️  Attribution gap only reduced by {gap_closed:.0%}")

    # Efficiency
    efficiency = max(0, 1 - (state.step_count / state.max_steps))
    feedback.append(f"ℹ️  Completed in {state.step_count}/{state.max_steps} steps")

    trajectory_penalty = penalise_trajectory(state.history)

    score = round(
        max(
            (window_score * 0.55) + (gap_closed * 0.35) + (efficiency * 0.10) + trajectory_penalty,
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
            "trajectory_penalty": trajectory_penalty,
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

    # Signal quality achieved
    achieved_signal = c.pixel_signal_quality
    optimal_signal  = compute_pixel_quality(c.ios_traffic_pct, True, True, True)
    signal_fraction = (achieved_signal - initial_signal) / max(optimal_signal - initial_signal, 0.01)
    signal_fraction = round(min(max(signal_fraction, 0), 1), 4)
    feedback.append(
        f"ℹ️  Signal quality: {initial_signal:.0%} → {achieved_signal:.0%} "
        f"(optimal: {optimal_signal:.0%})"
    )

    efficiency = max(0, 1 - (state.step_count / state.max_steps))
    trajectory_penalty = penalise_trajectory(state.history)

    score = round(
        max(
            capi_score * 0.40
            + aem_score * 0.25
            + signal_fraction * 0.25
            + efficiency * 0.10
            + trajectory_penalty,
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
            "trajectory_penalty": trajectory_penalty,
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
    issues_required = {"attribution_window", "conversions_api", "aem", "budget_allocation", "paused_bad_adsets"}
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

    # 5. Budget reallocation
    budget_reallocated = "budget_allocation" in state.issues_resolved or "budget_reallocation" in state.issues_resolved
    checks["budget_allocation"] = 1.0 if budget_reallocated else 0.0
    feedback.append(("✅" if budget_reallocated else "❌") + " Budget reallocated to top performers")

    # Signal and ROAS improvement
    final_gap    = _attribution_gap(c)
    gap_closed   = max(initial_gap - final_gap, 0) / initial_gap if initial_gap > 0 else 1.0
    sig_recovery = max(c.pixel_signal_quality - initial_signal, 0) / max(1.0 - initial_signal, 0.01)
    roas_gain    = min(max(c.true_roas - initial_true_roas, 0) / initial_true_roas, 1.0) if initial_true_roas > 0 else 0

    feedback.append(
        f"ℹ️  Gap closed: {gap_closed:.0%} | Signal: {initial_signal:.0%}→{c.pixel_signal_quality:.0%} | "
        f"True ROAS: {initial_true_roas:.2f}→{c.true_roas:.2f}"
    )

    issues_fraction = len(resolved) / len(issues_required)
    trajectory_penalty = penalise_trajectory(state.history)
    efficiency = max(0, 1 - (state.step_count / state.max_steps))

    score = round(
        max(
            issues_fraction * 0.40
            + gap_closed    * 0.20
            + sig_recovery  * 0.15
            + roas_gain     * 0.15
            + efficiency    * 0.10
            + trajectory_penalty,
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
            "trajectory_penalty": trajectory_penalty,
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