import json
import random
from pathlib import Path
from typing import Any, Dict, List


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _task_file_for_name(task_name: str) -> Path:
    mapping = {
        "easy": DATA_DIR / "easy_ads.json",
        "medium": DATA_DIR / "medium_ads.json",
        "hard": DATA_DIR / "hard_ads.json",
    }
    try:
        return mapping[task_name]
    except KeyError as exc:
        valid = ", ".join(mapping.keys())
        raise ValueError(f"Unknown task '{task_name}'. Valid tasks: {valid}") from exc


def load_task(task_name: str) -> Dict[str, Any]:
    task_file = _task_file_for_name(task_name)
    with task_file.open("r", encoding="utf-8") as handle:
        tasks: List[Dict[str, Any]] = json.load(handle)

    if not tasks:
        raise ValueError(f"No tasks available in {task_file}")

    return random.choice(tasks)
