"""
run_baseline.py — Reproducible baseline evaluation across all 3 tasks.

Usage:
    OPENAI_API_KEY=sk-... python baseline/run_baseline.py

Produces a score table and saves results to baseline_results.json.
"""

from __future__ import annotations
import json
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meta_ads_env import MetaAdsAttributionEnv
from baseline.baseline_agent import BaselineAgent


TASKS = [
    "easy_attribution_window",
    "medium_pixel_recovery",
    "hard_full_attribution_audit",
]


def run_task(task_id: str, agent: BaselineAgent, verbose: bool = True) -> dict:
    env = MetaAdsAttributionEnv(task_id=task_id)
    obs = env.reset()

    if verbose:
        print(f"\n{'='*60}")
        print(f"TASK: {task_id.upper()}")
        print(f"{'='*60}")
        print(obs.context)
        print()

    total_reward = 0.0
    step = 0

    while not obs.done:
        action = agent.act(obs.context)

        if verbose:
            print(f"  Step {step+1}: {action.action_type}  params={action.parameters}")
            print(f"           Reasoning: {action.reasoning}")

        obs, reward, done, info = env.step(action)
        total_reward += reward.total

        if verbose:
            print(f"           Reward: {reward.total:.4f}  ({reward.explanation})")
            print(f"           Effects: {info['effects']}")

        step += 1
        if done:
            break

    result = env.grade_episode()

    if verbose:
        print(f"\n── Episode Summary ──────────────────────────────")
        print(f"  Score:  {result.score:.4f}  ({'PASS ✅' if result.passed else 'FAIL ❌'})")
        print(f"  Steps:  {result.steps_used}/{env._state.max_steps}")
        print(f"  Cumulative reward: {result.cumulative_reward:.4f}")
        print("  Breakdown:")
        for k, v in result.breakdown.items():
            print(f"    {k}: {v}")
        print("  Feedback:")
        for fb in result.feedback:
            print(f"    {fb}")

    return {
        "task_id":            result.task_id,
        "difficulty":         result.difficulty,
        "score":              result.score,
        "passed":             result.passed,
        "steps_used":         result.steps_used,
        "cumulative_reward":  result.cumulative_reward,
        "breakdown":          result.breakdown,
        "feedback":           result.feedback,
    }


def main():
    print("Meta Ads Attribution OpenEnv — Baseline Evaluation")
    print("Model: gpt-4o-mini  |  Tasks: 3\n")

    try:
        agent = BaselineAgent(model="gpt-4o-mini")
    except EnvironmentError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    all_results = []
    for task_id in TASKS:
        result = run_task(task_id, agent, verbose=True)
        all_results.append(result)

    # Summary table
    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task':<35} {'Score':>7}  {'Pass':>6}  {'Steps':>6}")
    print("-" * 60)
    for r in all_results:
        tag = "✅" if r["passed"] else "❌"
        print(f"{r['task_id']:<35} {r['score']:>7.4f}  {tag:>6}  {r['steps_used']:>6}")

    avg = sum(r["score"] for r in all_results) / len(all_results)
    print("-" * 60)
    print(f"{'AVERAGE':<35} {avg:>7.4f}")
    print()

    out_path = os.path.join(os.path.dirname(__file__), "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": all_results, "average_score": round(avg, 4)}, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()