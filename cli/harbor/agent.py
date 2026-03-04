from __future__ import annotations

import importlib
import importlib.util
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

cli_file_path = _CLI_ROOT / "cli.py"
spec = importlib.util.spec_from_file_location(
    name="small_agent_cli", location=cli_file_path
)
if spec is None or spec.loader is None:  # pragma: no cover
    raise RuntimeError(f"Unable to import cli module from {cli_file_path}")

_cli_module = importlib.util.module_from_spec(spec=spec)
sys.modules["small_agent_cli"] = _cli_module
spec.loader.exec_module(module=_cli_module)

CONFIG_PATH = getattr(_cli_module, "CONFIG_PATH")
build_runtime_config = getattr(_cli_module, "build_runtime_config")
load_config = getattr(_cli_module, "load_config")
get_agent = getattr(_cli_module, "get_agent")

_MODEL_ENV_NAME = "SMALL_AGENT_HARBOR_MODEL"
_AGENT_ENV_NAME = "SMALL_AGENT_HARBOR_AGENT"


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
    if isinstance(metadata, dict):
        metadata["small_agent_result"] = payload


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
    if exec_result is None:
        return 0, "", ""

    if isinstance(exec_result, tuple):
        if len(exec_result) == 3:
            first, second, third = exec_result
            return int(first), str(second), str(third)

        if len(exec_result) == 2:
            first, second = exec_result
            return int(first), str(second), ""

    if isinstance(exec_result, str):
        return 0, exec_result, ""

    if isinstance(exec_result, dict):
        exit_code = int(exec_result.get("exit_code", exec_result.get("returncode", 0)))
        stdout = str(exec_result.get("stdout", ""))
        stderr = str(exec_result.get("stderr", ""))
        return exit_code, stdout, stderr

    exit_code = int(
        getattr(
            exec_result,
            "exit_code",
            getattr(exec_result, "returncode", 0),
        )
    )
    stdout = str(getattr(exec_result, "stdout", ""))
    stderr = str(getattr(exec_result, "stderr", ""))
    return exit_code, stdout, stderr


def _raise_for_exec_failure(*, exec_result: Any, action: str) -> None:
    exit_code, stdout, stderr = _extract_exec_fields(exec_result=exec_result)
    if exit_code == 0:
        return

    details = stderr.strip() or stdout.strip() or "no output captured"
    raise RuntimeError(f"{action} failed with exit_code={exit_code}: {details}")


class SmallAgentHarborAgent(HarborBaseAgent):
    def __init__(
        self,
        *,
        config_path: str | None = None,
        agent_key: str | None = None,
        model_key: str | None = None,
        model_name: str | None = None,
        logs_dir: str | Path | None = None,
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

    async def setup(self, environment: Any) -> None:
        setup_command = (
            "if [ -x ./cli/run ]; then "
            "echo 'small-agent cli is available'; "
            "else "
            "echo 'missing ./cli/run in workspace root' >&2; "
            "exit 1; "
            "fi"
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

        try:
            get_agent(agent_key=active_agent_key)
        except ValueError as err:
            _append_context_message(context=context, message=str(err))
            _set_context_result(
                context=context,
                success=False,
                exit_code=1,
                stdout="",
                stderr=str(err),
            )
            return

        runtime_cfg = build_runtime_config(
            config=loaded_config,
            agent_key=active_agent_key,
            model_key=active_model_key,
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
        run_command = (
            "./cli/run "
            f"--config {shlex.quote(str(self._config_path))} "
            f"--agent {shlex.quote(active_agent_key)} "
            f"--model {shlex.quote(active_model_key)} "
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
