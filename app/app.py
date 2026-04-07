import os
import sys
from pathlib import Path

import gradio as gr

# Allow running as `python app/app.py` without package install.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.baseline_agent import BaselineAgent
from meta_ads_env import MetaAdsAttributionEnv


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and os.getenv(key) is None:
            os.environ[key] = value


def run_demo(task_id: str) -> tuple[str, float]:
    _load_env_file(ROOT / ".env")

    try:
        env = MetaAdsAttributionEnv(task_id=task_id)
        agent = BaselineAgent()
        obs = env.reset()
    except Exception as exc:
        return (
            "Unable to start demo. Ensure .env contains HF_TOKEN, API_BASE_URL, and MODEL_NAME. "
            f"Details: {exc}",
            0.0,
        )

    total_reward = 0.0
    trace: list[str] = []
    step = 0

    try:
        while not obs.done:
            action = agent.act(obs.context)
            obs, reward, done, info = env.step(action)
            step += 1

            total_reward += reward.total
            trace.append(
                f"step={step} action={action.action_type} "
                f"reward={reward.total:+.2f} effects={info.get('effects', [])}"
            )
            if done:
                break
    except Exception as exc:
        trace.append(f"runtime_error={exc}")

    return "\n".join(trace), total_reward


with gr.Blocks(title="Meta Ads Attribution RL Demo") as demo:
    gr.Markdown("# Meta Ads Attribution OpenEnv Demo")
    gr.Markdown("Run a baseline policy across all attribution tasks.")

    task = gr.Dropdown(
        choices=[
            "easy_attribution_window",
            "medium_pixel_recovery",
            "hard_full_attribution_audit",
        ],
        value="easy_attribution_window",
        label="Task",
    )
    run = gr.Button("Run Baseline Episode")
    trace_output = gr.Textbox(label="Episode Trace", lines=12)
    reward_output = gr.Number(label="Total Reward")

    run.click(fn=run_demo, inputs=[task], outputs=[trace_output, reward_output])


if __name__ == "__main__":
    host = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name=host, server_port=port)
