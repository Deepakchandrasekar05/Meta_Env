from typing import Dict, List


def aggregate_rewards(episode_rewards: List[float]) -> Dict[str, float]:
    if not episode_rewards:
        return {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}

    return {
        "count": float(len(episode_rewards)),
        "mean": float(sum(episode_rewards) / len(episode_rewards)),
        "min": float(min(episode_rewards)),
        "max": float(max(episode_rewards)),
    }
