import os
from typing import Any, Dict

from openai import OpenAI


def llm_grade_episode(summary: Dict[str, Any]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY not set. Skipping LLM grading."

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    prompt = (
        "You are grading an RL policy for Meta Ads attribution optimization. "
        "Provide a concise grade with strengths, weaknesses, and one recommendation.\n\n"
        f"Episode summary: {summary}"
    )

    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.2,
    )
    return response.output_text
