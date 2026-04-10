from __future__ import annotations

import json
import os
from threading import Lock
from uuid import uuid4

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import uvicorn

from meta_ads_env import MetaAdsAttributionEnv
from meta_ads_env.models import Action
from meta_ads_env.tasks import TASK_REGISTRY

app = FastAPI(title="meta-ads-attribution-env-server")
_SESSIONS: dict[str, MetaAdsAttributionEnv] = {}
_LOCK = Lock()
_GRADIO_ENV: MetaAdsAttributionEnv | None = None

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
    global _GRADIO_ENV
    _GRADIO_ENV = MetaAdsAttributionEnv(task_id=task_id)
    obs = _GRADIO_ENV.reset()

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
    global _GRADIO_ENV
    if _GRADIO_ENV is None:
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

        obs, reward, done, info = _GRADIO_ENV.step(action)

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


def get_state_gradio() -> str:
    global _GRADIO_ENV
    if _GRADIO_ENV is None:
        return "No active session"

    return json.dumps(_GRADIO_ENV.state().model_dump(), indent=2)


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
    state_btn.click(get_state_gradio, outputs=output_box)


app = gr.mount_gradio_app(app, demo, path="/web")



class ResetRequest(BaseModel):
    task_id: str = "easy_attribution_window"
    session_id: str | None = None


class StepRequest(BaseModel):
    session_id: str
    action_type: str
    parameters: dict = Field(default_factory=dict)
    reasoning: str | None = None


class GradeRequest(BaseModel):
    session_id: str


def _get_session(session_id: str) -> MetaAdsAttributionEnv:
    with _LOCK:
        env = _SESSIONS.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    return env


def _obs_payload(obs) -> dict:
    return obs.model_dump()


def _reward_payload(reward) -> dict:
    return reward.model_dump()


# @app.get("/")
# def root() -> RedirectResponse:
#     return RedirectResponse(url="/web", status_code=307)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "meta-ads-attribution-env",
        "sessions": len(_SESSIONS),
    }


@app.get("/tasks")
def tasks() -> dict:
    return {"tasks": list(TASK_REGISTRY.keys())}


@app.post("/reset")
def reset_episode(req: ResetRequest) -> dict:
    if req.task_id not in TASK_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{req.task_id}'. Valid: {list(TASK_REGISTRY.keys())}",
        )

    env = MetaAdsAttributionEnv(task_id=req.task_id)
    obs = env.reset()

    session_id = req.session_id or str(uuid4())
    with _LOCK:
        _SESSIONS[session_id] = env

    return {
        "session_id": session_id,
        "task_id": req.task_id,
        "observation": _obs_payload(obs),
        "done": obs.done,
    }


@app.post("/step")
def step_episode(req: StepRequest) -> dict:
    env = _get_session(req.session_id)
    try:
        action = Action(
            action_type=req.action_type,
            parameters=req.parameters or {},
            reasoning=req.reasoning,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid action payload: {exc}") from exc

    try:
        obs, reward, done, info = env.step(action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result: dict = {
        "session_id": req.session_id,
        "observation": _obs_payload(obs),
        "reward": _reward_payload(reward),
        "done": done,
        "info": info,
    }

    if done:
        result["grade"] = env.grade_episode().model_dump()

    return result


@app.get("/state/{session_id}")
def get_state(session_id: str) -> dict:
    env = _get_session(session_id)
    return {
        "session_id": session_id,
        "state": env.state().model_dump(),
    }


@app.post("/grade")
def grade_episode(req: GradeRequest) -> dict:
    env = _get_session(req.session_id)
    return {
        "session_id": req.session_id,
        "grade": env.grade_episode().model_dump(),
    }


@app.delete("/session/{session_id}")
def delete_session(session_id: str) -> dict:
    with _LOCK:
        existed = _SESSIONS.pop(session_id, None) is not None
    if not existed:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    return {"session_id": session_id, "deleted": True}


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    # Use the in-process app object so direct execution via python server/app.py works.
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
