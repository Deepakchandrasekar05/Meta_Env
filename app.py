from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gradio as gr

# Allow running as `python app/app.py` without package install.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meta_ads_env import MetaAdsAttributionEnv
from meta_ads_env.models import Action
from meta_ads_env.tasks import TASK_REGISTRY

env: MetaAdsAttributionEnv | None = None

ACTION_CHOICES = [
    "investigate_attribution",
    "switch_to_modeled_conversions",
    "promote_ad",
    "reduce_budget",
    "adjust_attribution_window",
    "enable_conversions_api",
    "adjust_budget_allocation",
    "change_bid_strategy",
    "add_utm_tracking",
    "segment_audience",
    "enable_aggregated_event_measurement",
    "pause_underperforming_adsets",
    "reallocate_to_top_performers",
    "no_op",
]


def reset_env(task_id: str) -> str:
    global env
    env = MetaAdsAttributionEnv(task_id=task_id)
    obs = env.reset()

    return json.dumps(
        {
            "event": "reset",
            "task": task_id,
            "observation": obs.model_dump(),
            "done": obs.done,
        },
        indent=2,
    )


def step_env(action_type: str, reasoning: str) -> str:
    global env
    if env is None:
        return "Please click RESET first"

    action_type = (action_type or "").strip()
    if not action_type:
        return "Please select an action type"
    if action_type not in ACTION_CHOICES:
        return f"Invalid action_type '{action_type}'. Choose one of: {', '.join(ACTION_CHOICES)}"

    try:
        action = Action(
            action_type=action_type,
            parameters={},
            reasoning=reasoning,
        )

        obs, reward, done, info = env.step(action)

        return json.dumps(
            {
                "event": "step",
                "action": action_type,
                "observation": obs.model_dump(),
                "reward": reward.model_dump(),
                "done": done,
                "info": info,
            },
            indent=2,
        )

    except Exception as exc:
        return f"Error: {exc}"


def get_state() -> str:
    global env
    if env is None:
        return "No active session"

    return json.dumps(env.state().model_dump(), indent=2)


with gr.Blocks(title="Meta Ads RL Playground") as demo:
    gr.Markdown("## Meta Ads Attribution RL Playground")

    task_dropdown = gr.Dropdown(
        choices=list(TASK_REGISTRY.keys()),
        value="easy_attribution_window",
        label="Select Task",
    )

    with gr.Row():
        reset_btn = gr.Button("Reset", variant="primary")
        step_btn = gr.Button("Step", variant="secondary")
        state_btn = gr.Button("Get State")

    action_input = gr.Dropdown(
        choices=ACTION_CHOICES,
        value="investigate_attribution",
        label="Action Type",
    )

    reasoning_input = gr.Textbox(
        label="Reasoning",
        placeholder="Explain why you chose this action",
    )

    output_box = gr.Code(
        label="Output (JSON)",
        language="json",
    )

    reset_btn.click(reset_env, inputs=task_dropdown, outputs=output_box)
    step_btn.click(step_env, inputs=[action_input, reasoning_input], outputs=output_box)
    state_btn.click(get_state, outputs=output_box)


if __name__ == "__main__":
    host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    port = int(os.getenv("GRADIO_SERVER_PORT", "8000"))
    demo.launch(server_name=host, server_port=port)
