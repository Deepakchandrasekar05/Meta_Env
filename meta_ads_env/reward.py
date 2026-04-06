from typing import Any, Dict, Tuple


class RewardCalculator:
    def calculate(
        self, state: Dict[str, Any], action: str, next_state: Dict[str, Any]
    ) -> Tuple[float, str]:
        true_perf = state["true_performance"]
        tracking = state["tracking_reliability"]

        if action == "promote_ad" and true_perf == "good":
            return 1.0, "Correctly promoted good ad"

        if action == "reduce_budget" and true_perf == "bad":
            return 1.0, "Correctly reduced bad ad"

        if action == "reduce_budget" and true_perf == "good":
            return -1.0, "Reduced a good ad"

        if action == "promote_ad" and true_perf == "bad":
            return -1.0, "Promoted a bad ad"

        if action == "investigate_attribution" and tracking == "low":
            return 0.6, "Correctly detected attribution issue"

        if action == "switch_to_modeled_conversions" and tracking == "low":
            return 0.8, "Used modeled conversions correctly"

        if action == "keep_learning":
            return 0.3, "Waited for more data"

        # Small penalty for actions that are not catastrophic but not ideal.
        return -0.2, "Suboptimal decision"
