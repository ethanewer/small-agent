# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportExplicitAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import inspect
import json
import os
from pathlib import Path
import shlex
import shutil
from typing import Any

from agent_evolve_v2.service_runtime import (
    build_runtime_env_payload,
    stage_remote_bundle,
)

_base_agent_cls: type[object] = object
try:
    _harbor_base_module = importlib.import_module(name="harbor.agents.base")
    candidate = getattr(_harbor_base_module, "BaseAgent", object)
    if isinstance(candidate, type):
        _base_agent_cls = candidate
except Exception:
    _base_agent_cls = object

HarborBaseAgent = _base_agent_cls
REMOTE_WORKSPACE_ROOT = "/tmp/agent-evolve-v2-workspace"


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _environment_exec(
    *,
    environment: Any,
    command: str,
    cwd: str | None,
    env: dict[str, str] | None,
    timeout_sec: int,
) -> Any:
    executor = getattr(environment, "exec", None)
    if not callable(executor):
        raise RuntimeError("Harbor environment does not expose an exec method.")
    attempts = [
        {
            "command": command,
            "cwd": cwd,
            "env": env,
            "timeout_sec": timeout_sec,
        },
        {
            "command": command,
            "cwd": cwd,
            "env": env,
        },
        {"command": command},
    ]
    for kwargs in attempts:
        try:
            return await _maybe_await(executor(**kwargs))
        except TypeError:
            continue
    return await _maybe_await(executor(command))


def _extract_exec_fields(exec_result: Any) -> tuple[int, str, str]:
    def _as_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    if exec_result is None:
        return 0, "", ""
    if isinstance(exec_result, tuple):
        if len(exec_result) == 3:
            first, second, third = exec_result
            return int(first), _as_text(second), _as_text(third)
        if len(exec_result) == 2:
            first, second = exec_result
            return int(first), _as_text(second), ""
    if isinstance(exec_result, str):
        return 0, exec_result, ""
    if isinstance(exec_result, dict):
        return (
            int(
                exec_result.get(
                    "exit_code",
                    exec_result.get("returncode", exec_result.get("return_code", 0)),
                )
            ),
            _as_text(exec_result.get("stdout", "")),
            _as_text(exec_result.get("stderr", "")),
        )
    return (
        int(
            getattr(
                exec_result,
                "exit_code",
                getattr(
                    exec_result, "returncode", getattr(exec_result, "return_code", 0)
                ),
            )
        ),
        _as_text(getattr(exec_result, "stdout", "")),
        _as_text(getattr(exec_result, "stderr", "")),
    )


def _safe_setattr(*, obj: Any, name: str, value: Any) -> None:
    try:
        setattr(obj, name, value)
    except Exception:
        return


def _set_context_metadata(
    *,
    context: Any,
    model_key: str,
    run_command: str,
    exit_code: int,
    stdout: str,
    stderr: str,
) -> None:
    metadata = getattr(context, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
    payload = {
        "workspace_agent": {
            "model_key": model_key,
            "run_command": run_command,
            "success": exit_code == 0,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
        },
        "exit_code": exit_code,
    }
    metadata.update(json.loads(json.dumps(payload)))
    _safe_setattr(obj=context, name="metadata", value=metadata)


async def _environment_upload_dir(
    *,
    environment: Any,
    source_dir: Path,
    target_dir: str,
) -> None:
    uploader = getattr(environment, "upload_dir", None)
    if not callable(uploader):
        raise RuntimeError("Harbor environment does not expose upload_dir.")
    try:
        for kwargs in (
            {"source_dir": source_dir, "target_dir": target_dir},
            {"source_dir": str(source_dir), "target_dir": target_dir},
        ):
            try:
                await _maybe_await(uploader(**kwargs))
                return
            except TypeError:
                continue
        await _maybe_await(uploader(str(source_dir), target_dir))
    finally:
        shutil.rmtree(path=source_dir, ignore_errors=True)


class WorkspaceHarborAgent(HarborBaseAgent):
    def __init__(
        self,
        *,
        logs_dir: str | Path | None = None,
        model_name: str | None = None,
        extra_env: dict[str, str] | None = None,
        **kwargs: object,
    ) -> None:
        try:
            base_init_kwargs: dict[str, object] = dict(kwargs)
            if logs_dir is not None:
                base_init_kwargs["logs_dir"] = logs_dir
            if model_name is not None:
                base_init_kwargs["model_name"] = model_name
            super().__init__(**base_init_kwargs)
        except TypeError:
            try:
                super().__init__()
            except Exception:
                pass
        if extra_env:
            os.environ.update(extra_env)

    @staticmethod
    def name() -> str:
        return "agent-evolve-v2-workspace"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: Any) -> None:
        await self._ensure_workspace_available(environment=environment)

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        await self._ensure_workspace_available(environment=environment)
        model_key = str(getattr(self, "model_name", "") or "").strip()
        if not model_key:
            model_key = os.environ.get("WORKSPACE_MODEL_KEY", "").strip()
        if not model_key:
            _set_context_metadata(
                context=context,
                model_key="",
                run_command="",
                exit_code=1,
                stdout="",
                stderr="WORKSPACE_MODEL_KEY is required.",
            )
            raise RuntimeError("WORKSPACE_MODEL_KEY is required.")

        repo_root = _require_env_path(env_name="AGENT_EVOLVE_V2_REPO_ROOT")
        env_overrides = build_runtime_env_payload(
            repo_root=repo_root,
            model_key=model_key,
            final_message_enabled=False,
        )
        for key in (
            "OPENROUTER_API_KEY",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "SMALL_AGENT_CA_BUNDLE",
        ):
            value = os.environ.get(key)
            if value:
                env_overrides[key] = value
        command = (
            f"python3 {shlex.quote(REMOTE_WORKSPACE_ROOT + '/remote_runner.py')} "
            f"--no-final-message "
            f"{shlex.quote(instruction.strip())}"
        )
        exec_result = await _environment_exec(
            environment=environment,
            command=command,
            cwd=None,
            env=env_overrides,
            timeout_sec=60 * 60,
        )
        exit_code, stdout, stderr = _extract_exec_fields(exec_result=exec_result)
        _set_context_metadata(
            context=context,
            model_key=model_key,
            run_command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
        )
        if exit_code != 0:
            details = stderr.strip() or stdout.strip() or "no output captured"
            raise RuntimeError(
                f"Workspace harness exited with exit_code={exit_code}: {details}"
            )

    async def _ensure_workspace_available(self, *, environment: Any) -> None:
        workspace_root = _require_env_path(env_name="AGENT_EVOLVE_V2_WORKSPACE_PATH")
        repo_root = _require_env_path(env_name="AGENT_EVOLVE_V2_REPO_ROOT")
        bundle_root = stage_remote_bundle(
            workspace_root=workspace_root,
            repo_root=repo_root,
        )
        await _environment_upload_dir(
            environment=environment,
            source_dir=bundle_root,
            target_dir=REMOTE_WORKSPACE_ROOT,
        )
        bootstrap_command = (
            "set -e; "
            "export DEBIAN_FRONTEND=noninteractive; "
            "if ! command -v python3 >/dev/null 2>&1; then "
            "apt-get update && apt-get install -y --no-install-recommends "
            "python3 python3-pip ca-certificates; "
            "fi; "
            "if ! python3 -m pip --version >/dev/null 2>&1; then "
            "apt-get update && apt-get install -y --no-install-recommends "
            "python3-pip ca-certificates; "
            "fi; "
            "if ! command -v tmux >/dev/null 2>&1; then "
            "apt-get update && apt-get install -y --no-install-recommends tmux; "
            "fi; "
            "if ! command -v rg >/dev/null 2>&1; then "
            "apt-get update && apt-get install -y --no-install-recommends ripgrep; "
            "fi; "
            'PIP_BREAK_FLAG=""; '
            "if python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages; then "
            'PIP_BREAK_FLAG="--break-system-packages"; '
            "fi; "
            'if ! python3 -c "import anthropic, rich, openai, httpx, truststore" >/dev/null 2>&1; then '
            "python3 -m pip install --disable-pip-version-check --no-input "
            "$PIP_BREAK_FLAG anthropic rich openai httpx truststore || "
            "python3 -m pip install --disable-pip-version-check --no-input "
            "$PIP_BREAK_FLAG "
            "--trusted-host pypi.org --trusted-host files.pythonhosted.org "
            "anthropic rich openai httpx truststore; "
            "fi"
        )
        bootstrap_result = await _environment_exec(
            environment=environment,
            command=bootstrap_command,
            cwd=REMOTE_WORKSPACE_ROOT,
            env=None,
            timeout_sec=300,
        )
        exit_code, stdout, stderr = _extract_exec_fields(exec_result=bootstrap_result)
        if exit_code != 0:
            raise RuntimeError(
                "Workspace bootstrap failed: "
                + (stderr.strip() or stdout.strip() or "no output")
            )


def _require_env_path(*, env_name: str) -> Path:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        raise RuntimeError(f"{env_name} is required.")
    return Path(raw).resolve()
