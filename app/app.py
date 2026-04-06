import os
import sys
from pathlib import Path

import gradio as gr

# Allow running as `python app/app.py` without package install.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.baseline_agent import BaselineAgent
from meta_ads_env.env import MetaAdsEnv
from meta_ads_env.models import Action


def run_demo(task_name: str) -> tuple[str, float]:
    env = MetaAdsEnv(task_name=task_name)
    agent = BaselineAgent()
    obs = env.reset()

    done = False
    total_reward = 0.0
    trace: list[str] = []

    while not done:
        action = agent.act(obs)
        obs, reward, done, _ = env.step(Action(action=action))
        total_reward += reward.reward
        trace.append(f"day={obs.days_since_launch} action={action} reward={reward.reward:+.2f}")

    return "\n".join(trace), total_reward


with gr.Blocks(title="Meta Ads Attribution RL Demo") as demo:
    gr.Markdown("# Meta Ads Attribution OpenEnv Demo")
    gr.Markdown("Run a baseline policy across easy/medium/hard simulated attribution tasks.")

    task = gr.Dropdown(choices=["easy", "medium", "hard"], value="easy", label="Task")
    run = gr.Button("Run Baseline Episode")
    trace_output = gr.Textbox(label="Episode Trace", lines=12)
    reward_output = gr.Number(label="Total Reward")

    run.click(fn=run_demo, inputs=[task], outputs=[trace_output, reward_output])


if __name__ == "__main__":
    host = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name=host, server_port=port)
