from typing import Dict, List


def grade_episode(total_reward: float, action_log: List[str]) -> Dict[str, str | float]:
    if total_reward >= 8:
        band = "excellent"
        summary = "Strong policy decisions with robust attribution handling."
    elif total_reward >= 4:
        band = "good"
        summary = "Generally good decisions with room for better timing."
    elif total_reward >= 0:
        band = "fair"
        summary = "Mixed decisions. Attribution uncertainty not handled consistently."
    else:
        band = "poor"
        summary = "Policy underperforms and likely misclassifies ad quality."

    investigate_rate = (
        action_log.count("investigate_attribution") / len(action_log) if action_log else 0.0
    )

    return {
        "score": total_reward,
        "band": band,
        "investigation_rate": round(investigate_rate, 3),
        "summary": summary,
    }
