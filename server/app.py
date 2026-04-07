from __future__ import annotations

import os
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from meta_ads_env import MetaAdsAttributionEnv
from meta_ads_env.models import Action
from meta_ads_env.tasks import TASK_REGISTRY

app = FastAPI(title="meta-ads-attribution-env-server")
_SESSIONS: dict[str, MetaAdsAttributionEnv] = {}
_LOCK = Lock()


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


@app.get("/")
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
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
