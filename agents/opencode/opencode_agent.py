# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from agents.core.events import AgentEvent
from agents.core.result import RunResult
from agents.core.sink import EventSink
from agents.core.task import Task
from agents.interface import AgentRuntimeConfig


_ORCHESTRATOR_PATH = Path(__file__).resolve().parent / "orchestrator.ts"
_AGENT_DIR = Path(__file__).resolve().parent
_SDK_NODE_MODULES_CANDIDATES = (
    _AGENT_DIR / "node_modules",
    _AGENT_DIR.parents[1] / ".local" / "tools" / "node_modules",
)

_BUN_EXTRA_DIRS = (
    Path.home() / ".bun" / "bin",
    Path.home() / ".opencode" / "bin",
    Path("/usr/local/bin"),
    Path("/opt/homebrew/bin"),
)


def _augmented_path() -> str:
    path = os.environ.get("PATH", "")
    parts = path.split(os.pathsep) if path else []
    for extra in _BUN_EXTRA_DIRS:
        extra_str = str(extra)
        if extra_str not in parts and extra.is_dir():
            parts.insert(0, extra_str)

    return os.pathsep.join(parts)


def _resolve_bun() -> str:
    found = shutil.which("bun")
    if found:
        return found

    for extra in _BUN_EXTRA_DIRS:
        candidate = extra / "bun"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    return "bun"


def _opencode_model_id(*, model: str, api_base: str) -> str:
    normalized = model.strip()

    if "/" in normalized:
        return normalized

    base_lower = api_base.rstrip("/").lower()
    if base_lower.endswith("api.openai.com/v1"):
        return f"openai/{normalized}"

    if base_lower.endswith("openrouter.ai/api/v1"):
        return f"openrouter/{normalized}"

    return f"openai/{normalized}"


def _compact_single_line(text: str, *, max_chars: int = 160) -> str:
    single_line = " ".join(text.split())
    if len(single_line) <= max_chars:
        return single_line

    return single_line[: max_chars - 3] + "..."


def _render_trajectory(
    *,
    trajectory: list[dict[str, Any]],
    console: Console,
    sink: EventSink | None,
    verbosity: int,
) -> str | None:
    final_text: str | None = None

    for msg in trajectory:
        role = str(msg.get("role", ""))
        text = str(msg.get("text", "")).strip()
        tools = msg.get("tools", [])

        if role == "assistant" and text:
            final_text = text
            if sink:
                sink.emit(
                    event=AgentEvent(
                        event_type="reasoning",
                        payload={"message": text, "source": "opencode"},
                    )
                )

        if isinstance(tools, list):
            for tool_call in tools:
                if not isinstance(tool_call, dict):
                    continue

                tool_name = str(tool_call.get("tool", "tool"))
                tool_input = tool_call.get("input", {})
                tool_output = tool_call.get("output", "")
                tool_status = str(tool_call.get("status", ""))

                input_text = (
                    json.dumps(tool_input, ensure_ascii=True, sort_keys=True)
                    if isinstance(tool_input, dict)
                    else str(tool_input)
                )
                output_text = str(tool_output).strip() if tool_output else ""

                if verbosity == 0:
                    console.print(f"{tool_name}: {_compact_single_line(input_text)}")
                    if output_text:
                        console.print(_compact_single_line(output_text))
                    console.print("─" * max(20, console.width), style="dim")
                else:
                    console.print(f"{tool_name}: {input_text}")
                    if output_text:
                        console.print(output_text)
                    else:
                        console.print("[no detailed output]")
                    console.print("─" * max(20, console.width), style="dim")

                if sink:
                    sink.emit(
                        event=AgentEvent(
                            event_type="tool_call",
                            payload={
                                "name": tool_name,
                                "arguments": tool_input,
                                "status": tool_status,
                            },
                        )
                    )
                    if output_text:
                        sink.emit(
                            event=AgentEvent(
                                event_type="tool_result",
                                payload={
                                    "name": tool_name,
                                    "output": output_text,
                                },
                            )
                        )

    return final_text


class OpencodeAgent:
    def run(self, instruction: str, cfg: AgentRuntimeConfig, console: Console) -> int:
        result = self.run_task(
            task=Task.from_instruction(instruction=instruction),
            cfg=cfg,
            console=console,
            sink=None,
        )
        return result.exit_code

    def run_task(
        self,
        *,
        task: Task,
        cfg: AgentRuntimeConfig,
        console: Console | None = None,
        sink: EventSink | None = None,
    ) -> RunResult:
        if console is None:
            console = Console()

        options = cfg.agent_config
        verbosity = int(options.get("verbosity", 1))
        trajectory_path = str(options.get("trajectory_path", ""))

        opencode_model = _opencode_model_id(
            model=cfg.model.model,
            api_base=cfg.model.api_base,
        )

        bun_binary = _resolve_bun()

        node_paths = [str(p) for p in _SDK_NODE_MODULES_CANDIDATES if p.is_dir()]
        node_path = (
            os.pathsep.join(node_paths)
            if node_paths
            else str(_AGENT_DIR / "node_modules")
        )

        env: dict[str, str] = {
            "OPENCODE_MODEL": opencode_model,
            "PATH": _augmented_path(),
            "HOME": os.environ.get("HOME", ""),
            "NODE_PATH": node_path,
        }

        if cfg.model.api_key:
            env["OPENAI_API_KEY"] = cfg.model.api_key

        for key in (
            "OPENROUTER_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENAI_BASE_URL",
            "XDG_CONFIG_HOME",
            "XDG_CACHE_HOME",
            "XDG_STATE_HOME",
            "BUN_INSTALL",
        ):
            value = os.environ.get(key)
            if value:
                env[key] = value

        if trajectory_path:
            env["OPENCODE_TRAJECTORY_PATH"] = trajectory_path

        env.update({key: str(val) for key, val in dict(options.get("env", {})).items()})

        try:
            completed = subprocess.run(
                [bun_binary, "run", str(_ORCHESTRATOR_PATH), task.instruction],
                cwd=str(Path.cwd()),
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
        except FileNotFoundError:
            console.print(
                Panel(
                    "bun not found. Install bun (https://bun.sh) and ensure it is on PATH.",
                    title="Agent Error",
                    border_style="red",
                )
            )
            result = RunResult(
                exit_code=1,
                success=False,
                task_id=task.task_id,
                final_message="bun not found",
            )
            if sink:
                sink.emit(
                    event=AgentEvent(
                        event_type="issue",
                        payload={
                            "kind": "missing_binary",
                            "message": "bun not found",
                        },
                    )
                )
                sink.finalize(result=result)
            return result
        except subprocess.CalledProcessError as err:
            stderr_text = str(err.stderr or "").strip()
            stdout_text = str(err.output or "").strip()
            error_detail = stderr_text or stdout_text or str(err)
            console.print(
                Panel(
                    error_detail,
                    title="Agent Error",
                    border_style="red",
                )
            )
            result = RunResult(
                exit_code=1,
                success=False,
                task_id=task.task_id,
                final_message=error_detail,
            )
            if sink:
                sink.emit(
                    event=AgentEvent(
                        event_type="issue",
                        payload={"kind": "subprocess", "message": error_detail},
                    )
                )
                sink.finalize(result=result)
            return result

        stdout = completed.stdout.strip()
        try:
            trajectory = json.loads(stdout)
        except json.JSONDecodeError:
            if verbosity >= 1 and stdout:
                console.print(stdout)
            trajectory = []

        if isinstance(trajectory, list):
            final_text = _render_trajectory(
                trajectory=trajectory,
                console=console,
                sink=sink,
                verbosity=verbosity,
            )
            if final_text:
                console.print(Panel(final_text, title="Done", border_style="green"))

        result = RunResult(
            exit_code=0,
            success=True,
            task_id=task.task_id,
        )
        if sink:
            sink.emit(
                event=AgentEvent(
                    event_type="done",
                    payload={"message": "opencode run completed"},
                )
            )
            sink.finalize(result=result)

        return result
