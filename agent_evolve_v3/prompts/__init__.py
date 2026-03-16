from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent


def load_planning_prompt() -> str:
    return (_PROMPTS_DIR / "planning.md").read_text(encoding="utf-8")


def load_implementation_prompt() -> str:
    return (_PROMPTS_DIR / "implementation.md").read_text(encoding="utf-8")
