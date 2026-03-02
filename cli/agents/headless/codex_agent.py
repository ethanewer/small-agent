from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from agents.headless.util import run_subprocess
from agents.interface import AgentRuntimeConfig


def _sampling_args(sampling_params: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in sampling_params.items():
        if isinstance(value, str):
            args.extend(["--config", f'{key}="{value}"'])
            continue
        if isinstance(value, bool):
            args.extend(["--config", f"{key}={'true' if value else 'false'}"])
            continue
        args.extend(["--config", f"{key}={value}"])
    return args


class CodexHeadlessAgent:
    def run(self, instruction: str, cfg: AgentRuntimeConfig, console: Console) -> int:
        options = cfg.agent_config
        sampling_params = dict(options.get("sampling_params", {}))

        args = ["codex", "exec", "--full-auto", "--sandbox", "danger-full-access"]
        if cfg.model.model:
            args.extend(["--model", cfg.model.model])
        args.extend(_sampling_args(sampling_params=sampling_params))
        args.append(instruction)

        env = {
            "OPENAI_API_KEY": cfg.model.api_key,
            "OPENAI_BASE_URL": cfg.model.api_base,
            "PATH": os.environ.get("PATH", ""),
            "NODE_NO_WARNINGS": "1",
            **{key: str(val) for key, val in dict(options.get("env", {})).items()},
        }

        try:
            run_subprocess(
                args=args,
                cwd=str(Path.cwd()),
                env=env,
                check=True,
            )
        except FileNotFoundError:
            console.print(
                Panel(
                    "codex CLI not found. Install @openai/codex and ensure `codex` is on PATH.",
                    title="Agent Error",
                    border_style="red",
                )
            )
            return 1
        except (subprocess.CalledProcessError, ValueError, TypeError) as err:
            console.print(Panel(str(err), title="Agent Error", border_style="red"))
            return 1

        return 0
