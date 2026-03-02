from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from agents.interface import AgentRuntimeConfig
from agents.local_binary import resolve_agent_binary
from agents.openai_compat import (
    normalize_openai_compatible_model,
    opencode_model_arg,
    opencode_provider_id,
    preflight_agent_model_compatibility,
)
from agents.opencode.util import run_subprocess

MODEL_CATALOG_ERROR_PATTERNS = (
    "ProviderModelNotFoundError",
    "Model not found:",
    "is not a valid model ID",
)


def _render_opencode_error(err: Exception) -> str:
    if isinstance(err, subprocess.CalledProcessError):
        detail = str(err.stderr or err.output or "").strip()
        if not detail:
            detail = str(err)
        if any(pattern in detail for pattern in MODEL_CATALOG_ERROR_PATTERNS):
            return (
                f"{detail}\n\n"
                "Hint: refresh OpenCode model catalog and verify the provider/model ID:\n"
                "- `opencode models --refresh`\n"
                "- `opencode models openrouter`\n"
                "- `opencode models openai`"
            )
        return detail

    return str(err)


class OpencodeAgent:
    def run(self, instruction: str, cfg: AgentRuntimeConfig, console: Console) -> int:
        options = cfg.agent_config
        binary = str(
            options.get("binary") or resolve_agent_binary(default_binary="opencode")
        )
        output_format = str(options.get("output_format", "default"))
        pass_model_arg = bool(options.get("pass_model_arg", False))
        normalized_model = normalize_openai_compatible_model(
            model=cfg.model.model,
            api_base=cfg.model.api_base,
        )
        resolved_opencode_model = opencode_model_arg(
            model=cfg.model.model,
            api_base=cfg.model.api_base,
        )
        compatibility_error = preflight_agent_model_compatibility(
            agent_key="opencode",
            model=cfg.model.model,
            api_base=cfg.model.api_base,
        )
        if compatibility_error:
            console.print(
                Panel(
                    compatibility_error,
                    title="Agent Compatibility Error",
                    border_style="red",
                )
            )
            return 1

        args = [binary, "run", instruction]
        if pass_model_arg:
            args.extend(["--model", resolved_opencode_model])
        if output_format:
            args.extend(["--format", output_format])

        provider_id = opencode_provider_id(api_base=cfg.model.api_base)
        config_content = {
            "model": resolved_opencode_model,
            "provider": {
                provider_id: {
                    "options": {
                        "apiKey": cfg.model.api_key,
                        "baseURL": cfg.model.api_base,
                    }
                }
            },
        }
        env = {
            "OPENAI_MODEL": normalized_model,
            "OPENAI_BASE_URL": cfg.model.api_base,
            "OPENAI_API_BASE": cfg.model.api_base,
            "OPENAI_API_KEY": cfg.model.api_key,
            "OPENROUTER_API_KEY": cfg.model.api_key,
            "OPENCODE_CONFIG_CONTENT": json.dumps(config_content),
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
                fail_patterns=MODEL_CATALOG_ERROR_PATTERNS,
            )
        except FileNotFoundError:
            console.print(
                Panel(
                    "opencode CLI not found. Install opencode and ensure `opencode` is on PATH.",
                    title="Agent Error",
                    border_style="red",
                )
            )
            return 1
        except (subprocess.CalledProcessError, ValueError, TypeError) as err:
            console.print(
                Panel(
                    _render_opencode_error(err), title="Agent Error", border_style="red"
                )
            )
            return 1

        return 0
