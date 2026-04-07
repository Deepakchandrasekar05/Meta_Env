"""utils.py — Shared helpers."""
from __future__ import annotations
import json
from typing import Any


def pretty_json(obj: Any) -> str:
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    return json.dumps(obj, indent=2, default=str)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))