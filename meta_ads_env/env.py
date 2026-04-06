from typing import Any, Dict, Tuple

from .models import Action, Observation, Reward
from .reward import RewardCalculator
from .simulator import AdSimulator
from .tasks import load_task


class MetaAdsEnv:
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.simulator = AdSimulator()
        self.reward_calc = RewardCalculator()
        self.current_state: Dict[str, Any] | None = None
        self.done = False

    def reset(self) -> Observation:
        self.current_state = load_task(self.task_name)
        self.done = False
        return Observation(**self.current_state["observation"])

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.current_state is None:
            raise RuntimeError("Environment is not reset. Call reset() before step().")
        if self.done:
            raise RuntimeError("Episode already finished. Call reset() to start a new episode.")

        next_state = self.simulator.simulate(self.current_state, action.action)
        reward_value, reason = self.reward_calc.calculate(
            self.current_state,
            action.action,
            next_state,
        )

        self.current_state = next_state

        if next_state["days_since_launch"] > 14:
            self.done = True

        observation = Observation(**next_state["observation"])
        reward = Reward(reward=reward_value, reason=reason)

        info = {
            "task": self.task_name,
            "true_performance": next_state["true_performance"],
            "tracking_reliability": next_state["tracking_reliability"],
        }
        return observation, reward, self.done, info

    def state(self) -> Dict[str, Any]:
        if self.current_state is None:
            raise RuntimeError("Environment is not reset. Call reset() first.")
        return self.current_state
