"""HTTP client helper for the Meta Ads attribution environment server.

This file is intentionally placed at repository root to match the default
`openenv init` scaffold shape.
"""

from __future__ import annotations

import json
from typing import Any
from urllib import error, parse, request


class MetaAdsEnvClient:
    """Lightweight client for the FastAPI server in `server/app.py`."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def _call(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Accept": "application/json"}

        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(url=url, method=method.upper(), data=data, headers=headers)

        try:
            with request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{exc.code} {exc.reason}: {body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Unable to reach server at {url}: {exc.reason}") from exc

    def health(self) -> dict[str, Any]:
        return self._call("GET", "/health")

    def tasks(self) -> dict[str, Any]:
        return self._call("GET", "/tasks")

    def reset(self, task_id: str = "easy_attribution_window", session_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"task_id": task_id}
        if session_id:
            payload["session_id"] = session_id
        return self._call("POST", "/reset", payload)

    def step(
        self,
        session_id: str,
        action_type: str,
        parameters: dict[str, Any] | None = None,
        reasoning: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "session_id": session_id,
            "action_type": action_type,
            "parameters": parameters or {},
            "reasoning": reasoning,
        }
        return self._call("POST", "/step", payload)

    def state(self, session_id: str) -> dict[str, Any]:
        path = "/state/" + parse.quote(session_id, safe="")
        return self._call("GET", path)

    def grade(self, session_id: str) -> dict[str, Any]:
        return self._call("POST", "/grade", {"session_id": session_id})
