# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportUnusedCallResult=false, reportImplicitRelativeImport=false, reportImplicitStringConcatenation=false

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

from runtime_types import WorkspaceRunResult, WorkspaceRuntimeConfig

_ORCHESTRATOR_PATH = Path(__file__).resolve().parent / "orchestrator.ts"
_AGENT_DIR = Path(__file__).resolve().parent


def _build_sdk_candidates() -> tuple[Path, ...]:
    candidates = [_AGENT_DIR / "node_modules"]
    if len(_AGENT_DIR.parents) > 3:
        candidates.append(_AGENT_DIR.parents[3] / ".local" / "tools" / "node_modules")
    return tuple(candidates)


_SDK_NODE_MODULES_CANDIDATES = _build_sdk_candidates()

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


def _emit(*parts: str) -> None:
    for part in parts:
        print(part, flush=True)


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


class WorkspaceAgent:
    def run_task(
        self,
        *,
        instruction: str,
        cfg: WorkspaceRuntimeConfig,
        console: object | None = None,
        task_id: str = "adhoc",
    ) -> WorkspaceRunResult:
        del console

        workspace_root = Path(__file__).resolve().parents[1]
        trajectory_path = workspace_root / "outputs" / "trajectory.json"
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)

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
            "OPENCODE_TRAJECTORY_PATH": str(trajectory_path),
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

        try:
            completed = subprocess.run(
                [bun_binary, "run", str(_ORCHESTRATOR_PATH), instruction],
                cwd=str(Path.cwd()),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError:
            _emit('<issue kind="missing_binary">bun not found</issue>')
            return WorkspaceRunResult(
                exit_code=1,
                success=False,
                task_id=task_id,
            )

        if completed.returncode != 0:
            stderr_text = (completed.stderr or "").strip()
            _emit(
                f'<issue kind="subprocess">{escape(stderr_text or "opencode failed")}</issue>'
            )
            return WorkspaceRunResult(
                exit_code=completed.returncode,
                success=False,
                task_id=task_id,
            )

        stdout = completed.stdout.strip()

        try:
            trajectory = json.loads(stdout)
        except json.JSONDecodeError:
            trajectory = []

        if isinstance(trajectory, list):
            for msg in trajectory:
                if not isinstance(msg, dict):
                    continue

                role = str(msg.get("role", ""))
                text = str(msg.get("text", "")).strip()
                tools = msg.get("tools", [])

                if role == "assistant" and text:
                    _emit(
                        "<turn>",
                        "<analysis>",
                        escape(text[:2000]),
                        "</analysis>",
                    )

                if isinstance(tools, list):
                    for tool_call in tools:
                        if not isinstance(tool_call, dict):
                            continue
                        _tool_name = str(tool_call.get("tool", "tool"))
                        tool_input = tool_call.get("input", {})
                        tool_output = tool_call.get("output", "")
                        input_str = (
                            json.dumps(tool_input, ensure_ascii=True)
                            if isinstance(tool_input, dict)
                            else str(tool_input)
                        )
                        output_str = (
                            str(tool_output).strip() if tool_output else "[no output]"
                        )
                        _emit(
                            "<command>",
                            "<input>",
                            escape(input_str[:1000]),
                            "</input>",
                            "<output>",
                            escape(output_str[:2000]),
                            "</output>",
                            "</command>",
                        )

                if role == "assistant" and text:
                    _emit("</turn>", "")

        _emit("<done>opencode run completed</done>")
        return WorkspaceRunResult(
            exit_code=0,
            success=True,
            task_id=task_id,
        )
