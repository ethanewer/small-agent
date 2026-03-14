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


def run_cursor_refiner(
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
            "agent_evolve_v2.service_cli",
            "validate",
            "--workspace",
            str(workspace_path),
            "--model-key",
            model_key,
        ],
        cwd=repo_root,
    )


def run_workspace_critique(
    *,
    workspace_path: Path,
    max_failures: int,
    max_successes: int,
) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    return run_command(
        command=[
            "uv",
            "run",
            "python",
            "-m",
            "agent_evolve_v2.service_cli",
            "critique",
            "--workspace",
            str(workspace_path),
            "--max-failures",
            str(max_failures),
            "--max-successes",
            str(max_successes),
        ],
        cwd=repo_root,
    )
