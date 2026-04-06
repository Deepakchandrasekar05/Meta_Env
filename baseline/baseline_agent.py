from meta_ads_env.models import Observation


class BaselineAgent:
    def act(self, observation: Observation) -> str:
        if observation.ios_traffic_percent > 50 and observation.pixel_match_quality < 0.5:
            return "investigate_attribution"

        if observation.tracked_conversions > 20:
            return "promote_ad"

        if observation.days_since_launch < 3:
            return "keep_learning"

        if observation.modeled_conversions > observation.tracked_conversions + 5:
            return "switch_to_modeled_conversions"

        return "reduce_budget"
