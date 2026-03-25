# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportUnusedCallResult=false, reportImplicitRelativeImport=false, reportImplicitStringConcatenation=false

from __future__ import annotations

import json

from agent import run
from agent_types import Config, WorkspaceRunResult, WorkspaceRuntimeConfig


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

        config = Config(
            model=cfg.model.model,
            api_base=cfg.model.api_base,
            api_key=cfg.model.api_key,
            temperature=cfg.model.temperature,
            context_length=cfg.model.context_length,
            extra_params=cfg.model.extra_params,
            max_turns=int(cfg.agent_config.get("max_turns", 50)),
            max_wait_seconds=float(cfg.agent_config.get("max_wait_seconds", 60.0)),
        )

        result = run(instruction=instruction, config=config)

        print(f"<agent_history>{json.dumps(result.history, indent=2)}</agent_history>")

        return WorkspaceRunResult(
            exit_code=result.exit_code,
            success=result.success,
            task_id=task_id,
        )
