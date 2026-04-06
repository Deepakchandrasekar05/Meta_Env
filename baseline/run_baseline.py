import sys
from pathlib import Path

# Allow running as `python baseline/run_baseline.py` without package install.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.baseline_agent import BaselineAgent
from meta_ads_env.env import MetaAdsEnv
from meta_ads_env.grader import grade_episode
from meta_ads_env.models import Action


def main() -> None:
    env = MetaAdsEnv(task_name="easy")
    agent = BaselineAgent()

    obs = env.reset()
    done = False
    total_reward = 0.0
    action_log: list[str] = []

    while not done:
        action_str = agent.act(obs)
        action_log.append(action_str)
        obs, reward, done, _ = env.step(Action(action=action_str))
        total_reward += reward.reward

    grade = grade_episode(total_reward, action_log)
    print(f"Total Reward: {total_reward:.2f}")
    print("Grade:", grade)


if __name__ == "__main__":
    main()
