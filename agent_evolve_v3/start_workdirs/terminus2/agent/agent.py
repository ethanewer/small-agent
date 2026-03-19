# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportUnusedCallResult=false, reportImplicitRelativeImport=false, reportImplicitStringConcatenation=false

from __future__ import annotations

from xml.sax.saxutils import escape

from core_agent import (
    AgentCallbacks,
    Command,
    Config as CoreConfig,
    ModelConfig as CoreModelConfig,
    ParsedResponse,
    run_agent,
)
from runtime_types import WorkspaceRunResult, WorkspaceRuntimeConfig


def _emit(*parts: str) -> None:
    for part in parts:
        print(part, flush=True)


def _strip_terminal_prefix(output: str) -> str:
    for prefix in ("New Terminal Output:\n", "Current Terminal Screen:\n"):
        if output.startswith(prefix):
            return output[len(prefix) :]
    return output


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

        max_turns = int(cfg.agent_config.get("max_turns", 50))
        max_wait_seconds = float(cfg.agent_config.get("max_wait_seconds", 60.0))
        core_cfg = CoreConfig(
            active_model_key=cfg.model.model,
            active_model=CoreModelConfig(
                model=cfg.model.model,
                api_base=cfg.model.api_base,
                api_key=cfg.model.api_key,
                temperature=cfg.model.temperature,
                context_length=cfg.model.context_length,
                extra_params=cfg.model.extra_params,
            ),
            verbosity=1,
            max_turns=max_turns,
            max_wait_seconds=max_wait_seconds,
        )

        turn_open = False

        def _close_turn() -> None:
            nonlocal turn_open
            if turn_open:
                _emit("</turn>", "")
                turn_open = False

        def on_reasoning(turn: int, parsed: ParsedResponse) -> None:
            nonlocal turn_open
            _close_turn()
            turn_open = True
            _emit(
                f'<turn n="{turn}">',
                "<analysis>",
                escape(parsed.analysis),
                "</analysis>",
                "<plan>",
                escape(parsed.plan),
                "</plan>",
            )

        def on_command_output(command: Command, output: str) -> None:
            raw = _strip_terminal_prefix(output).strip() or "[no output]"
            _emit(
                "<command>",
                "<input>",
                escape(command.keystrokes),
                "</input>",
                "<output>",
                escape(raw),
                "</output>",
                "</command>",
            )

        def on_issue(kind: str, message: str) -> None:
            _emit(f'<issue kind="{escape(kind)}">', escape(message), "</issue>")

        def on_done(done_text: str) -> None:
            _close_turn()
            _emit(f"<done>{escape(done_text)}</done>")

        def on_stopped(stopped_max_turns: int) -> None:
            _close_turn()
            _emit(f'<stopped max_turns="{stopped_max_turns}" />')

        compaction_counts: dict[str, int] = {"proactive": 0, "reactive": 0}

        def on_compaction(kind: str) -> None:
            compaction_counts[kind] = compaction_counts.get(kind, 0) + 1
            _emit(f'<compaction kind="{escape(kind)}" />')

        callbacks = AgentCallbacks(
            on_reasoning=on_reasoning,
            on_command_output=on_command_output,
            on_issue=on_issue,
            on_done=on_done,
            on_stopped=on_stopped,
            on_compaction=on_compaction,
        )
        exit_code = run_agent(
            instruction=instruction,
            cfg=core_cfg,
            api_key=cfg.model.api_key,
            callbacks=callbacks,
        )
        _close_turn()
        total = compaction_counts["proactive"] + compaction_counts["reactive"]
        _emit(
            f'<summary compactions="{total}"'
            + f' proactive="{compaction_counts["proactive"]}"'
            + f' reactive="{compaction_counts["reactive"]}" />'
        )
        return WorkspaceRunResult(
            exit_code=exit_code,
            success=exit_code == 0,
            task_id=task_id,
        )
