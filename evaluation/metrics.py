"""
evaluation/metrics.py — Aggregate metrics across episodes and tasks.
"""

from __future__ import annotations
from typing import List, Dict
import statistics


def summarise_results(results: List[Dict]) -> Dict:
    scores = [r["score"] for r in results]
    return {
        "mean_score":   round(statistics.mean(scores), 4),
        "median_score": round(statistics.median(scores), 4),
        "min_score":    round(min(scores), 4),
        "max_score":    round(max(scores), 4),
        "pass_rate":    round(sum(1 for r in results if r["passed"]) / len(results), 4),
        "by_difficulty": {
            diff: round(
                statistics.mean(r["score"] for r in results if r["difficulty"] == diff) or 0, 4
            )
            for diff in ["easy", "medium", "hard"]
            if any(r["difficulty"] == diff for r in results)
        },
    }