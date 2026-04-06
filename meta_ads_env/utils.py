from typing import Dict


def format_observation(obs: Dict[str, float | int]) -> str:
    keys = [
        "impressions",
        "link_clicks",
        "tracked_conversions",
        "modeled_conversions",
        "cost_per_result",
        "days_since_launch",
    ]
    return " | ".join(f"{k}={obs[k]}" for k in keys)
