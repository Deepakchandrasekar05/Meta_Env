import random
from copy import deepcopy
from typing import Any, Dict


class AdSimulator:
    def simulate(self, state: Dict[str, Any], action: str) -> Dict[str, Any]:
        new_state = deepcopy(state)
        obs = new_state["observation"]

        # Base traffic drift.
        obs["impressions"] += random.randint(200, 1200)
        obs["link_clicks"] += random.randint(10, 80)

        # True performance drives latent conversion lift.
        if state["true_performance"] == "good":
            obs["tracked_conversions"] += random.randint(1, 5)
        else:
            obs["tracked_conversions"] += random.randint(0, 2)

        # Tracking reliability determines measurement quality.
        if state["tracking_reliability"] == "low":
            obs["tracked_conversions"] += random.randint(0, 1)
            obs["modeled_conversions"] += random.randint(1, 6)
        else:
            obs["tracked_conversions"] += random.randint(2, 6)
            obs["modeled_conversions"] += random.randint(0, 2)

        # Action-dependent behavior shifts.
        if action == "promote_ad":
            obs["cost_per_result"] = max(0.1, obs["cost_per_result"] - random.uniform(0.0, 0.4))
        elif action == "reduce_budget":
            obs["impressions"] = max(0, int(obs["impressions"] * 0.9))
            obs["link_clicks"] = max(0, int(obs["link_clicks"] * 0.92))
        elif action == "investigate_attribution":
            obs["pixel_match_quality"] = min(1.0, obs["pixel_match_quality"] + random.uniform(0.02, 0.08))
            obs["capi_coverage"] = min(1.0, obs["capi_coverage"] + random.uniform(0.05, 0.12))
        elif action == "switch_to_modeled_conversions":
            obs["tracked_conversions"] += random.randint(0, 1)
            obs["modeled_conversions"] += random.randint(3, 8)

        # Update time.
        new_state["days_since_launch"] += 1
        obs["days_since_launch"] = new_state["days_since_launch"]

        return new_state
