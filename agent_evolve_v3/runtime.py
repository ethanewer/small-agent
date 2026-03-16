# pyright: reportUnusedCallResult=false

from __future__ import annotations

import json
from pathlib import Path
import subprocess


def run_command(
    *,
    command: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def record_completed_process(
    *,
    output_path: Path,
    completed: subprocess.CompletedProcess[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def run_cursor_agent(
    *,
    workspace_path: Path,
    prompt_text: str,
    cursor_model: str,
) -> subprocess.CompletedProcess[str]:
    command = [
        "agent",
        "--print",
        "--force",
        "--trust",
        "--sandbox",
        "disabled",
        "--workspace",
        str(workspace_path),
        "--model",
        cursor_model,
        prompt_text,
    ]
    return run_command(command=command, cwd=workspace_path)


def run_planner_agent(
    *,
    workspace_path: Path,
    prompt_text: str,
    cursor_model: str,
) -> subprocess.CompletedProcess[str]:
    return run_cursor_agent(
        workspace_path=workspace_path,
        prompt_text=prompt_text,
        cursor_model=cursor_model,
    )


def run_implementation_agent(
    *,
    workspace_path: Path,
    prompt_text: str,
    cursor_model: str,
) -> subprocess.CompletedProcess[str]:
    return run_cursor_agent(
        workspace_path=workspace_path,
        prompt_text=prompt_text,
        cursor_model=cursor_model,
    )


def run_workspace_validation(
    *,
    workspace_path: Path,
    model_key: str,
) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    return run_command(
        command=[
            "uv",
            "run",
            "python",
            "-m",
            "agent_evolve_v3.service_cli",
            "validate",
            "--workspace",
            str(workspace_path),
            "--model-key",
            model_key,
        ],
        cwd=repo_root,
    )
