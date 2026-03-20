from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any

from rich.console import Console  # pyright: ignore[reportMissingImports]


_base_agent_cls: type[object] = object
try:  # pragma: no cover - Harbor is optional in local tests.
    _harbor_base_module = importlib.import_module(name="harbor.agents.base")
    candidate = getattr(_harbor_base_module, "BaseAgent", object)
    if isinstance(candidate, type):
        _base_agent_cls = candidate
except Exception:
    _base_agent_cls = object

HarborBaseAgent = _base_agent_cls

_CLI_ROOT = Path(__file__).resolve().parents[1]
if str(_CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(_CLI_ROOT))

from harbor_config import (  # noqa: E402
    CONFIG_PATH,
    _env_var_name,
    build_runtime_config,
    load_config,
)

_MODEL_ENV_NAME = "SMALL_AGENT_HARBOR_MODEL"
_AGENT_ENV_NAME = "SMALL_AGENT_HARBOR_AGENT"
_REMOTE_CLI_ROOT = "/tmp/small-agent-cli"


def _safe_setattr(obj: Any, name: str, value: Any) -> None:
    try:
        setattr(obj, name, value)
    except Exception:
        return


def _call_if_exists(*, obj: Any, method_name: str, kwargs: dict[str, Any]) -> bool:
    method = getattr(obj, method_name, None)
    if not callable(method):
        return False

    try:
        method(**kwargs)
        return True
    except Exception:
        return False


def _append_context_message(context: Any, message: str) -> None:
    if _call_if_exists(
        obj=context,
        method_name="add_message",
        kwargs={"message": message},
    ):
        return

    if _call_if_exists(
        obj=context,
        method_name="append_message",
        kwargs={"message": message},
    ):
        return

    _call_if_exists(
        obj=context,
        method_name="log",
        kwargs={"message": message},
    )
    metadata = getattr(context, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        _safe_setattr(obj=context, name="metadata", value=metadata)
    if isinstance(metadata, dict):
        logs = metadata.setdefault("small_agent_logs", [])
        if isinstance(logs, list):
            logs.append(message)


def _set_context_result(
    *,
    context: Any,
    success: bool,
    exit_code: int,
    stdout: str,
    stderr: str,
) -> None:
    payload = {
        "success": success,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
    }
    if _call_if_exists(
        obj=context,
        method_name="set_result",
        kwargs={"result": payload},
    ):
        return

    _safe_setattr(obj=context, name="success", value=success)
    _safe_setattr(obj=context, name="exit_code", value=exit_code)
    _safe_setattr(obj=context, name="stdout", value=stdout)
    _safe_setattr(obj=context, name="stderr", value=stderr)

    metadata = getattr(context, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        _safe_setattr(obj=context, name="metadata", value=metadata)
    if isinstance(metadata, dict):
        metadata["small_agent_result"] = payload


def _record_setup_stage(
    *,
    context: Any | None,
    stage: str,
    status: str,
    details: str | None = None,
) -> None:
    if context is None:
        return

    metadata = getattr(context, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        _safe_setattr(obj=context, name="metadata", value=metadata)
    if not isinstance(metadata, dict):
        return

    setup_log = metadata.setdefault("small_agent_setup", {})
    if not isinstance(setup_log, dict):
        return

    payload: dict[str, str] = {"status": status}
    if details:
        payload["details"] = details
    setup_log[stage] = payload


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
    timeout_sec: int | None,
) -> Any:
    executor = getattr(environment, "exec", None)
    if not callable(executor):
        raise RuntimeError("Harbor environment does not expose an exec method.")

    call_attempts: list[dict[str, Any]] = [
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
    for kwargs in call_attempts:
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
        exit_code = int(
            exec_result.get(
                "exit_code",
                exec_result.get("returncode", exec_result.get("return_code", 0)),
            )
        )
        stdout = _as_text(exec_result.get("stdout", ""))
        stderr = _as_text(exec_result.get("stderr", ""))
        return exit_code, stdout, stderr

    exit_code = int(
        getattr(
            exec_result,
            "exit_code",
            getattr(exec_result, "returncode", getattr(exec_result, "return_code", 0)),
        )
    )
    stdout = _as_text(getattr(exec_result, "stdout", ""))
    stderr = _as_text(getattr(exec_result, "stderr", ""))
    return exit_code, stdout, stderr


def _raise_for_exec_failure(*, exec_result: Any, action: str) -> None:
    exit_code, stdout, stderr = _extract_exec_fields(exec_result=exec_result)
    if exit_code == 0:
        return

    details = stderr.strip() or stdout.strip() or "no output captured"
    raise RuntimeError(f"{action} failed with exit_code={exit_code}: {details}")


async def _environment_is_dir(
    *, environment: Any, path: str, max_retries: int = 5
) -> bool:
    for attempt in range(max_retries + 1):
        try:
            checker = getattr(environment, "is_dir", None)
            if callable(checker):
                try:
                    result = await _maybe_await(checker(path=path))
                    return bool(result)
                except TypeError:
                    result = await _maybe_await(checker(path))
                    return bool(result)

            probe = await _environment_exec(
                environment=environment,
                command=f"test -d {shlex.quote(path)}",
                cwd=None,
                env=None,
                timeout_sec=5,
            )
            exit_code, _, _ = _extract_exec_fields(exec_result=probe)
            return exit_code == 0
        except (ProcessLookupError, TimeoutError, OSError):
            if attempt >= max_retries:
                raise
            await asyncio.sleep(float(2**attempt))

    raise AssertionError("unreachable")


_UPLOAD_EXCLUDE_DIRS: set[str] = {
    "sft",
    "harbor",
    ".git",
    ".venv",
    ".ruff_cache",
    ".local",
    "__pycache__",
    "node_modules",
    "config",
    "agent_evolve",
}

_UPLOAD_EXCLUDE_FILES: set[str] = {
    "uv.lock",
    ".env",
}


def _stage_upload_dir(source_dir: Path) -> Path:
    """Create a lightweight staging copy that excludes heavy directories."""
    import shutil
    import tempfile

    staging = Path(tempfile.mkdtemp(prefix="harbor-upload-"))
    for entry in source_dir.iterdir():
        if entry.name in _UPLOAD_EXCLUDE_DIRS and entry.is_dir():
            continue
        if entry.name in _UPLOAD_EXCLUDE_FILES and entry.is_file():
            continue
        dest = staging / entry.name
        if entry.is_dir():
            shutil.copytree(src=entry, dst=dest, symlinks=True)
        else:
            shutil.copy2(src=entry, dst=dest)

    return staging


async def _environment_upload_dir(
    *,
    environment: Any,
    source_dir: Path,
    target_dir: str,
) -> None:
    import shutil

    uploader = getattr(environment, "upload_dir", None)
    if not callable(uploader):
        raise RuntimeError("Harbor environment does not expose an upload_dir method.")

    staged = _stage_upload_dir(source_dir=source_dir)
    try:
        attempts: list[dict[str, Any]] = [
            {"source_dir": staged, "target_dir": target_dir},
            {"source_dir": str(staged), "target_dir": target_dir},
        ]
        for kwargs in attempts:
            try:
                await _maybe_await(uploader(**kwargs))
                return
            except TypeError:
                continue

        await _maybe_await(uploader(str(staged), target_dir))
    finally:
        shutil.rmtree(path=staged, ignore_errors=True)


class SmallAgentHarborAgent(HarborBaseAgent):
    def __init__(
        self,
        *,
        config_path: str | None = None,
        agent_key: str | None = None,
        model_key: str | None = None,
        model_name: str | None = None,
        logs_dir: str | Path | None = None,
        extra_env: dict[str, str] | None = None,
        **_kwargs: object,
    ) -> None:
        try:
            base_init_kwargs: dict[str, object] = dict(_kwargs)
            if logs_dir is not None:
                base_init_kwargs["logs_dir"] = logs_dir
            if model_name is not None:
                base_init_kwargs["model_name"] = model_name
            super().__init__(**base_init_kwargs)
        except TypeError:
            # Local tests may run without Harbor's BaseAgent implementation.
            try:
                super().__init__()
            except Exception:
                pass

        if extra_env:
            os.environ.update(extra_env)

        self._config_path = (
            Path(config_path).resolve() if config_path else Path(CONFIG_PATH).resolve()
        )
        self._forced_agent_key = agent_key
        self._forced_model_key = model_key or model_name

    @staticmethod
    def name() -> str:
        return "small-agent-harbor-external"

    def version(self) -> str | None:
        return "0.1.0"

    def _select_keys(self) -> tuple[str | None, str | None]:
        selected_agent = self._forced_agent_key or os.getenv(_AGENT_ENV_NAME) or None
        selected_model = self._forced_model_key or os.getenv(_MODEL_ENV_NAME) or None
        return selected_agent, selected_model

    async def _ensure_cli_available(
        self,
        *,
        environment: Any,
        context: Any | None = None,
    ) -> None:
        _record_setup_stage(
            context=context,
            stage="check_remote_cli_dir",
            status="started",
        )
        cli_present = await _environment_is_dir(
            environment=environment,
            path=_REMOTE_CLI_ROOT,
        )
        _record_setup_stage(
            context=context,
            stage="check_remote_cli_dir",
            status="ok",
            details=f"exists={cli_present}",
        )
        if not cli_present:
            _record_setup_stage(
                context=context,
                stage="upload_cli_bundle",
                status="started",
            )
            await _environment_upload_dir(
                environment=environment,
                source_dir=_CLI_ROOT,
                target_dir=_REMOTE_CLI_ROOT,
            )
            _record_setup_stage(
                context=context,
                stage="upload_cli_bundle",
                status="ok",
            )

        setup_command = (
            f"if [ -f {shlex.quote(_REMOTE_CLI_ROOT + '/cli.py')} ] && "
            f"[ -f {shlex.quote(_REMOTE_CLI_ROOT + '/config.json')} ]; then "
            "echo 'small-agent cli bundle is available'; "
            "else "
            "echo 'small-agent cli bundle is missing required files' >&2; "
            "exit 1; "
            "fi"
        )
        _record_setup_stage(
            context=context,
            stage="preflight_cli_bundle",
            status="started",
        )
        setup_result = await _environment_exec(
            environment=environment,
            command=setup_command,
            cwd=None,
            env=None,
            timeout_sec=60,
        )
        _raise_for_exec_failure(
            exec_result=setup_result,
            action="Harbor setup preflight",
        )
        _record_setup_stage(
            context=context,
            stage="preflight_cli_bundle",
            status="ok",
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
            "if ! command -v node >/dev/null 2>&1; then "
            "apt-get update && apt-get install -y --no-install-recommends "
            "nodejs ca-certificates; "
            "fi; "
            "if ! command -v tmux >/dev/null 2>&1; then "
            "apt-get update && apt-get install -y --no-install-recommends "
            "tmux; "
            "fi; "
            'PIP_BREAK_FLAG=""; '
            "if python3 -m pip install --help 2>/dev/null | "
            "grep -q -- --break-system-packages; then "
            'PIP_BREAK_FLAG="--break-system-packages"; '
            "fi; "
            'if ! python3 -c "import rich, litellm, tenacity" '
            ">/dev/null 2>&1; then "
            "python3 -m pip install --disable-pip-version-check --no-input "
            "--ignore-installed "
            "$PIP_BREAK_FLAG rich litellm tenacity || "
            "python3 -m pip install --disable-pip-version-check --no-input "
            "--ignore-installed "
            "$PIP_BREAK_FLAG "
            "--trusted-host pypi.org --trusted-host files.pythonhosted.org "
            "rich litellm tenacity; "
            "fi"
        )
        _record_setup_stage(
            context=context,
            stage="bootstrap_python_dependencies",
            status="started",
        )
        bootstrap_result = await _environment_exec(
            environment=environment,
            command=bootstrap_command,
            cwd=_REMOTE_CLI_ROOT,
            env=None,
            timeout_sec=300,
        )
        _raise_for_exec_failure(
            exec_result=bootstrap_result,
            action="Harbor setup dependency bootstrap",
        )
        _record_setup_stage(
            context=context,
            stage="bootstrap_python_dependencies",
            status="ok",
        )

    async def setup(self, environment: Any) -> None:
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                await self._ensure_cli_available(
                    environment=environment,
                    context=None,
                )
                break
            except (ProcessLookupError, TimeoutError, OSError):
                if attempt >= max_retries:
                    raise
                await asyncio.sleep(float(2**attempt))

    async def run(
        self,
        instruction: str,
        environment: Any,
        context: Any,
    ) -> None:
        console = Console(record=True)
        instruction_clean = instruction.strip()
        if not instruction_clean:
            _set_context_result(
                context=context,
                success=False,
                exit_code=1,
                stdout="",
                stderr="Instruction is required.",
            )
            return
        await self._ensure_cli_available(
            environment=environment,
            context=context,
        )

        loaded_config = load_config(path=self._config_path)
        selected_agent, selected_model = self._select_keys()
        active_agent_key = selected_agent or loaded_config.default_agent
        active_model_key = selected_model or loaded_config.default_model

        if active_agent_key not in loaded_config.agents:
            known = ", ".join(sorted(loaded_config.agents.keys()))
            error_message = (
                f"Unknown Harbor agent override '{active_agent_key}'. "
                f"Known config agent keys: {known}"
            )
            _append_context_message(context=context, message=error_message)
            _set_context_result(
                context=context,
                success=False,
                exit_code=1,
                stdout="",
                stderr=error_message,
            )
            return

        if active_model_key not in loaded_config.models:
            known = ", ".join(sorted(loaded_config.models.keys()))
            error_message = (
                f"Unknown Harbor model override '{active_model_key}'. "
                f"Known config model keys: {known}"
            )
            _append_context_message(context=context, message=error_message)
            _set_context_result(
                context=context,
                success=False,
                exit_code=1,
                stdout="",
                stderr=error_message,
            )
            return

        runtime_cfg = build_runtime_config(
            config=loaded_config,
            agent_key=active_agent_key,
            model_key=active_model_key,
            allow_shell_lookup=True,
        )
        _append_context_message(
            context=context,
            message=(
                "Starting small-agent Harbor run "
                f"(agent={active_agent_key}, model={active_model_key})."
            ),
        )
        env_overrides = {
            "OPENAI_MODEL": runtime_cfg.model.model,
            "OPENAI_BASE_URL": runtime_cfg.model.api_base,
            "OPENAI_API_KEY": runtime_cfg.model.api_key,
        }
        model_env_name = _env_var_name(
            config_api_key=loaded_config.models[active_model_key].api_key,
        )
        if model_env_name:
            env_overrides[model_env_name] = runtime_cfg.model.api_key
        run_command = (
            f"python3 {shlex.quote(_REMOTE_CLI_ROOT + '/cli.py')} "
            f"--config {shlex.quote(_REMOTE_CLI_ROOT + '/config.json')} "
            f"--agent {shlex.quote(active_agent_key)} "
            f"--model {shlex.quote(active_model_key)} "
            f"--no-final-message "
            f"{shlex.quote(instruction_clean)}"
        )
        exec_result = await _environment_exec(
            environment=environment,
            command=run_command,
            cwd=None,
            env=env_overrides,
            timeout_sec=int(loaded_config.max_wait_seconds * loaded_config.max_turns),
        )
        exit_code, stdout, stderr = _extract_exec_fields(exec_result=exec_result)
        _append_context_message(
            context=context,
            message=(f"small-agent Harbor run completed with exit_code={exit_code}."),
        )
        metadata_payload = {
            "agent_key": active_agent_key,
            "model_key": active_model_key,
            "run_command": run_command,
            "captured_console": console.export_text(),
        }
        _safe_setattr(obj=context, name="small_agent_metadata", value=metadata_payload)
        metadata = getattr(context, "metadata", None)
        if isinstance(metadata, dict):
            metadata["small_agent"] = json.loads(json.dumps(metadata_payload))

        _set_context_result(
            context=context,
            success=exit_code == 0,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
        )
