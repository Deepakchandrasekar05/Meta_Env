"""
evaluation/metrics.py — Aggregate metrics across episodes and tasks.
"""

from __future__ import annotations
from typing import List, Dict
import statistics


def summarise_results(results: List[Dict]) -> Dict:
    scores = [r["score"] for r in results]

    def _mean_from_breakdown(key: str) -> float:
        vals = [r.get("breakdown", {}).get(key) for r in results if key in r.get("breakdown", {})]
        if not vals:
            return 0.0
        return round(statistics.mean(vals), 4)

    return {
        "mean_score":   round(statistics.mean(scores), 4),
        "median_score": round(statistics.median(scores), 4),
        "min_score":    round(min(scores), 4),
        "max_score":    round(max(scores), 4),
        "pass_rate":    round(sum(1 for r in results if r["passed"]) / len(results), 4),
        "mean_gap_reduction": _mean_from_breakdown("gap_closed"),
        "mean_signal_recovery": _mean_from_breakdown("signal_recovery"),
        "mean_roas_improvement": _mean_from_breakdown("roas_gain") or _mean_from_breakdown("roas_improvement"),
        "mean_action_efficiency": _mean_from_breakdown("action_efficiency"),
        "mean_redundancy_penalty": _mean_from_breakdown("redundant_action_penalty"),
        "by_difficulty": {
            diff: round(
                statistics.mean(r["score"] for r in results if r["difficulty"] == diff) or 0, 4
            )
            for diff in ["easy", "medium", "hard"]
            if any(r["difficulty"] == diff for r in results)
        },
    }