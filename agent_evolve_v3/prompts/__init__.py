from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent


def _load_prompt(*, name: str, baseline: str | None = None) -> str:
    if baseline:
        baseline_path = _PROMPTS_DIR / baseline / name
        if baseline_path.exists():
            return baseline_path.read_text(encoding="utf-8")

    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def load_planning_prompt(*, baseline: str | None = None) -> str:
    return _load_prompt(name="planning.md", baseline=baseline)


def load_implementation_prompt(*, baseline: str | None = None) -> str:
    return _load_prompt(name="implementation.md", baseline=baseline)


def load_failure_investigation_prompt(*, baseline: str | None = None) -> str:
    return _load_prompt(name="failure_investigation.md", baseline=baseline)
