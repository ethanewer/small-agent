# pyright: reportUnusedCallResult=false, reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile


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
            "agent_evolve_v3.services.cli",
            "validate",
            "--workspace",
            str(workspace_path),
            "--model-key",
            model_key,
        ],
        cwd=repo_root,
    )


def run_failure_investigation_agent(
    *,
    prompt_text: str,
    cursor_model: str,
    run_0_agent_log: str,
    run_1_agent_log: str,
    run_0_verifier: str,
    run_1_verifier: str,
    run_0_exception: str = "",
    run_1_exception: str = "",
    core_agent_source: str,
) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix="agent-evolve-v3-failure-inv-") as tmpdir:
        work_dir = Path(tmpdir)
        (work_dir / "run_0_agent_log.txt").write_text(run_0_agent_log, encoding="utf-8")
        (work_dir / "run_1_agent_log.txt").write_text(run_1_agent_log, encoding="utf-8")
        (work_dir / "run_0_verifier.txt").write_text(run_0_verifier, encoding="utf-8")
        (work_dir / "run_1_verifier.txt").write_text(run_1_verifier, encoding="utf-8")
        (work_dir / "run_0_exception.txt").write_text(run_0_exception, encoding="utf-8")
        (work_dir / "run_1_exception.txt").write_text(run_1_exception, encoding="utf-8")
        (work_dir / "core_agent.py").write_text(core_agent_source, encoding="utf-8")

        completed = run_cursor_agent(
            workspace_path=work_dir,
            prompt_text=prompt_text,
            cursor_model=cursor_model,
        )

        output_path = work_dir / "output.json"
        if output_path.exists():
            try:
                data = json.loads(output_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
            except (json.JSONDecodeError, OSError):
                pass

        return {
            "task_name": "",
            "general_failure_reason": "infrastructure_error",
            "task_specific_explanation": f"Investigation agent failed (rc={completed.returncode})",
            "consistency": "both_same_failure",
            "suggested_fix_category": "not_fixable_by_agent",
        }
