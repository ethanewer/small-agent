# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportUnusedCallResult=false, reportImplicitRelativeImport=false, reportImplicitStringConcatenation=false, reportUnannotatedClassAttribute=false, reportPossiblyUnboundVariable=false

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent_types import Config, Logger, RunResult

import litellm
from litellm.utils import get_max_tokens, token_counter
from tenacity import retry, stop_after_attempt

from tmux_session import TmuxSession, start_session

litellm.suppress_debug_info = True


class OutputLengthExceededError(Exception):
    truncated_response: str | None

    def __init__(self, message: str, truncated_response: str | None = None) -> None:
        super().__init__(message)
        self.truncated_response = truncated_response


MAX_OUTPUT_BYTES = 10_000
_TEMPLATES_DIR = Path(__file__).resolve().parent / "prompt_templates"
DEFAULT_SYSTEM_PROMPT_TEMPLATE = (_TEMPLATES_DIR / "system_prompt.txt").read_text()
_TIMEOUT_TEMPLATE = (_TEMPLATES_DIR / "timeout.txt").read_text()


@dataclass
class ModelResult:
    content: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class Command:
    keystrokes: str
    duration: float


@dataclass
class ParsedResponse:
    analysis: str
    plan: str
    commands: list[Command]
    task_complete: bool


def limit_output_length(output: str, max_bytes: int = MAX_OUTPUT_BYTES) -> str:
    if len(output.encode("utf-8")) <= max_bytes:
        return output

    portion_size = max_bytes // 2
    output_bytes = output.encode("utf-8")
    first_portion = output_bytes[:portion_size].decode("utf-8", errors="ignore")
    last_portion = output_bytes[-portion_size:].decode("utf-8", errors="ignore")
    omitted_bytes = (
        len(output_bytes)
        - len(first_portion.encode("utf-8"))
        - len(last_portion.encode("utf-8"))
    )
    return (
        f"{first_portion}\n[... output limited to {max_bytes} bytes; "
        f"{omitted_bytes} interior bytes omitted ...]\n{last_portion}"
    )


@dataclass
class ParseResult:
    parsed: ParsedResponse | None
    error: str
    warning: str


def _extract_json_content(response: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    json_start = -1
    json_end = -1
    brace_count = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(response):
        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            if brace_count == 0:
                json_start = i

            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and json_start != -1:
                json_end = i + 1
                break

    if json_start == -1 or json_end == -1:
        return "", ["No valid JSON object found"]

    before_text = response[:json_start].strip()
    after_text = response[json_end:].strip()
    if before_text:
        warnings.append("Extra text detected before JSON object")

    if after_text:
        warnings.append("Extra text detected after JSON object")

    return response[json_start:json_end], warnings


def _check_field_order(json_content: str, warnings: list[str]) -> None:
    expected_order = ["analysis", "plan", "commands"]
    positions: dict[str, int] = {}
    for field in expected_order:
        match = re.search(rf'"{field}"\s*:', json_content)
        if match:
            positions[field] = match.start()

    if len(positions) < 2:
        return

    present = [f for f in expected_order if f in positions]
    actual = [f for f, _ in sorted(positions.items(), key=lambda x: x[1])]
    if actual != present:
        warnings.append(
            f"Fields appear in wrong order. "
            f"Found: {' -> '.join(actual)}, expected: {' -> '.join(present)}"
        )


def _parse_commands(
    commands_data: list[object],
    warnings: list[str],
) -> tuple[list[Command], str]:
    commands: list[Command] = []

    for i, cmd_data in enumerate(commands_data):
        if not isinstance(cmd_data, dict):
            return [], f"Command {i + 1} must be an object"

        if "keystrokes" not in cmd_data:
            return [], f"Command {i + 1} missing required 'keystrokes' field"

        keystrokes = cmd_data["keystrokes"]
        if not isinstance(keystrokes, str):
            return [], f"Command {i + 1} 'keystrokes' must be a string"

        if "duration" in cmd_data:
            duration_value = cmd_data["duration"]
            if not isinstance(duration_value, (int, float)):
                warnings.append(
                    f"Command {i + 1}: Invalid duration value, using default 1.0"
                )
                duration = 1.0
            else:
                duration = float(duration_value)
        else:
            warnings.append(
                f"Command {i + 1}: Missing duration field, using default 1.0"
            )
            duration = 1.0

        known_fields = {"keystrokes", "duration"}
        unknown = set(cmd_data.keys()) - known_fields
        if unknown:
            warnings.append(
                f"Command {i + 1}: Unknown fields: {', '.join(sorted(unknown))}"
            )

        if i < len(commands_data) - 1 and not keystrokes.endswith("\n"):
            warnings.append(
                f"Command {i + 1} should end with newline when followed "
                + "by another command. Otherwise the two commands will be "
                + "concatenated together on the same line."
            )

        commands.append(Command(keystrokes=keystrokes, duration=float(duration)))

    return commands, ""


def _fix_incomplete_json(response: str) -> str | None:
    brace_count = response.count("{") - response.count("}")
    if brace_count > 0:
        return response + "}" * brace_count

    return None


def _fix_mixed_content(response: str) -> str | None:
    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    matches = re.findall(json_pattern, response, re.DOTALL)
    for match in matches:
        try:
            json.loads(match)
            return match
        except json.JSONDecodeError:
            continue

    return None


def _try_parse_response(response: str) -> ParseResult:
    warnings: list[str] = []

    json_content, extract_warnings = _extract_json_content(response)
    warnings.extend(extract_warnings)

    if not json_content:
        return ParseResult(
            parsed=None,
            error="No valid JSON found in response",
            warning=_format_warnings(warnings),
        )

    try:
        data = json.loads(json_content)
    except json.JSONDecodeError as exc:
        return ParseResult(
            parsed=None,
            error=f"Invalid JSON: {exc}",
            warning=_format_warnings(warnings),
        )

    if not isinstance(data, dict):
        return ParseResult(
            parsed=None,
            error="Response must be a JSON object",
            warning=_format_warnings(warnings),
        )

    missing_fields = [
        field for field in ("analysis", "plan", "commands") if field not in data
    ]
    if missing_fields:
        return ParseResult(
            parsed=None,
            error=f"Missing required fields: {', '.join(missing_fields)}",
            warning=_format_warnings(warnings),
        )

    if not isinstance(data["commands"], list):
        return ParseResult(
            parsed=None,
            error="Field 'commands' must be an array",
            warning=_format_warnings(warnings),
        )

    _check_field_order(json_content, warnings)

    commands_data: list[object] = data["commands"]
    commands, parse_error = _parse_commands(
        commands_data=commands_data,
        warnings=warnings,
    )

    task_complete = _coerce_task_complete(data.get("task_complete", False))
    if parse_error:
        if task_complete:
            warnings.append(parse_error)
            commands = []
        else:
            return ParseResult(
                parsed=None,
                error=parse_error,
                warning=_format_warnings(warnings),
            )

    return ParseResult(
        parsed=ParsedResponse(
            analysis=str(data["analysis"]),
            plan=str(data["plan"]),
            commands=commands,
            task_complete=task_complete,
        ),
        error="",
        warning=_format_warnings(warnings),
    )


def _format_warnings(warnings: list[str]) -> str:
    if not warnings:
        return ""

    return "- " + "\n- ".join(warnings)


def parse_response(text: str) -> ParseResult:
    result = _try_parse_response(text)
    if not result.error:
        return result

    auto_fixes: list[tuple[str, Callable[[str], str | None]]] = [
        (
            "Fixed incomplete JSON by adding missing closing brace",
            _fix_incomplete_json,
        ),
        ("Extracted JSON from mixed content", _fix_mixed_content),
    ]
    for fix_name, fix_fn in auto_fixes:
        fixed = fix_fn(text)
        if fixed is not None:
            corrected = _try_parse_response(fixed)
            if not corrected.error:
                auto_warning = (
                    f"AUTO-CORRECTED: {fix_name} - please fix this in future responses"
                )
                corrected.warning = _combine_warnings(auto_warning, corrected.warning)
                return corrected

    return result


def _combine_warnings(auto_warning: str, existing_warning: str) -> str:
    if existing_warning:
        return f"- {auto_warning}\n{existing_warning}"

    return f"- {auto_warning}"


def _coerce_task_complete(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}

    return False


def build_prompt(
    instruction: str,
    terminal_state: str,
    max_wait_seconds: float,
    system_prompt_template: str = DEFAULT_SYSTEM_PROMPT_TEMPLATE,
) -> str:
    del max_wait_seconds
    return system_prompt_template.format(
        instruction=instruction, terminal_state=terminal_state
    )


def _litellm_model_name(model: str, api_base: str) -> str:
    if "/" in model:
        return model

    if "openai.com" in api_base:
        return f"openai/{model}"

    if "openrouter.ai" in api_base:
        return f"openrouter/{model}"

    return f"openai/{model}"


@retry(stop=stop_after_attempt(3))
def call_model(
    cfg: Config,
    prompt: str,
    history: list[dict[str, str]],
) -> ModelResult:
    model_name = _litellm_model_name(model=cfg.model, api_base=cfg.api_base)
    completion_kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [*history, {"role": "user", "content": prompt}],
        "api_base": cfg.api_base,
        "api_key": cfg.api_key,
    }

    if cfg.temperature is not None:
        completion_kwargs["temperature"] = cfg.temperature

    if cfg.extra_params:
        for k, v in cfg.extra_params.items():
            completion_kwargs[k] = v

    try:
        result = litellm.completion(**completion_kwargs)
        content = result.choices[0].message.content or ""  # pyright: ignore[reportAttributeAccessIssue]
        if result.choices[0].finish_reason == "length":  # pyright: ignore[reportAttributeAccessIssue]
            raise OutputLengthExceededError(
                f"Model {model_name} hit max_tokens limit. Response was truncated.",
                truncated_response=content,
            )

        usage = result.usage  # pyright: ignore[reportAttributeAccessIssue]
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        return ModelResult(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except OutputLengthExceededError:
        raise
    except litellm.ContextWindowExceededError as err:  # pyright: ignore[reportPrivateImportUsage]
        raise RuntimeError(
            f"Context length exceeded for model {model_name}: {err}"
        ) from err


_OUTPUT_LENGTH_ERROR_MSG = (
    "ERROR!! NONE of the actions you just requested were performed "
    "because you exceeded the maximum output length of 4096 tokens. "
    "Your outputs must be less than 4096 tokens. Re-issue this request, "
    "breaking it into chunks each of which is less than 4096 tokens."
)

_MAX_QUERY_ATTEMPTS = 3


def _query_model(
    *,
    cfg: Config,
    prompt: str,
    history: list[dict[str, str]],
    original_instruction: str,
    terminal_state: str,
    logger: Logger | None = None,
) -> ModelResult:
    last_error: Exception | None = None
    current_prompt = prompt

    for _attempt in range(_MAX_QUERY_ATTEMPTS):
        try:
            model_result = call_model(
                cfg=cfg,
                prompt=current_prompt,
                history=history,
            )
            _append_turn_history(
                history=history,
                prompt=current_prompt,
                model_response=model_result.content,
            )
            return model_result
        except OutputLengthExceededError as exc:
            last_error = exc
            truncated = exc.truncated_response or ""

            warnings_text = ""
            try:
                parse_result = parse_response(truncated)
                if parse_result.warning:
                    warnings_text = (
                        f"\n\nParser warnings from your truncated response:\n"
                        f"{parse_result.warning}"
                    )
            except Exception:  # noqa: BLE001
                pass

            error_msg = _OUTPUT_LENGTH_ERROR_MSG
            if warnings_text:
                error_msg += warnings_text

            history.extend(
                [
                    {"role": "user", "content": current_prompt},
                    {"role": "assistant", "content": truncated},
                    {"role": "user", "content": error_msg},
                ]
            )
            current_prompt = error_msg
        except RuntimeError as exc:
            err_msg = str(exc).lower()
            if "context" in err_msg and "length" in err_msg:
                last_error = exc
                _unwind_messages(history=history, cfg=cfg)
                summarized = _summarize_history(
                    call_model_fn=call_model,
                    cfg=cfg,
                    history=history,
                    original_instruction=original_instruction,
                    terminal_state=terminal_state,
                )
                current_prompt = f"{summarized}\n\n{current_prompt}"
                if logger:
                    logger.log(
                        event_type="compaction",
                        payload={"kind": "reactive"},
                    )
                continue

            raise

    raise RuntimeError(
        f"Model query failed after {_MAX_QUERY_ATTEMPTS} attempts: {last_error}"
    )


def execute_command(
    session: TmuxSession, cmd: Command, max_wait_seconds: float
) -> None:
    effective_wait = min(max(cmd.duration, 0.0), max(max_wait_seconds, 0.1))

    if cmd.keystrokes == "":
        time.sleep(effective_wait)
        return

    session.send_keys(
        keys=cmd.keystrokes,
        min_timeout_sec=effective_wait,
    )


def completion_confirmation_message(terminal_output: str) -> str:
    return (
        f"Current terminal state:\n{terminal_output}\n\n"
        "Are you sure you want to mark the task as complete? "
        "This will trigger your solution to be graded and you won't be able to "
        'make any further corrections. If so, include "task_complete": true '
        "in your JSON response again."
    )


def _append_turn_history(
    history: list[dict[str, str]],
    prompt: str,
    model_response: str,
) -> None:
    history.extend(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": model_response},
        ]
    )


_FALLBACK_CONTEXT_LIMIT = 1_000_000
_PROACTIVE_FREE_TOKEN_THRESHOLD = 8_000
_UNWIND_TARGET_FREE_TOKENS = 4_000


def _count_total_tokens(model: str, messages: list[dict[str, str]]) -> int:
    return token_counter(model=model, messages=messages)


def _litellm_model_for_cfg(cfg: Config) -> str:
    return _litellm_model_name(model=cfg.model, api_base=cfg.api_base)


def _get_model_context_limit(cfg: Config) -> int:
    if cfg.context_length and cfg.context_length > 0:
        return cfg.context_length

    try:
        limit = get_max_tokens(_litellm_model_for_cfg(cfg))
        if limit and limit > 0:
            return limit
    except Exception:  # noqa: BLE001
        pass

    return _FALLBACK_CONTEXT_LIMIT


def _unwind_messages(
    history: list[dict[str, str]],
    cfg: Config,
) -> None:
    context_limit = _get_model_context_limit(cfg)
    while len(history) > 1:
        current_tokens = _count_total_tokens(
            model=_litellm_model_for_cfg(cfg), messages=history
        )
        if context_limit - current_tokens >= _UNWIND_TARGET_FREE_TOKENS:
            break

        if len(history) >= 2:
            del history[-2:]
        else:
            break


def _summarize_history(
    *,
    call_model_fn: Callable[..., ModelResult],
    cfg: Config,
    history: list[dict[str, str]],
    original_instruction: str,
    terminal_state: str,
) -> str:
    if not history:
        return original_instruction

    summary_prompt = (
        "You are about to hand off your work to another AI agent. "
        "Please provide a comprehensive summary of what you have "
        "accomplished so far on this task:\n\n"
        f"Original Task: {original_instruction}\n\n"
        "Based on the conversation history, please provide a detailed summary covering:\n"
        "1. Major Actions Completed\n"
        "2. Important Information Learned\n"
        "3. Challenging Problems Addressed\n"
        "4. Current Status\n\n"
        "Be comprehensive and detailed."
    )

    summary_result = call_model_fn(
        cfg=cfg,
        prompt=summary_prompt,
        history=history,
    )

    question_prompt = (
        f"You are picking up work from a previous AI agent on this task:\n\n"
        f"**Original Task:**\n{original_instruction}\n\n"
        f"**Summary from Previous Agent:**\n{summary_result.content}\n\n"
        f"**Current Terminal Screen:**\n{terminal_state}\n\n"
        "Please begin by asking several questions (at least five, more if necessary) "
        "about the current state of the solution that are not answered in the "
        "summary from the prior agent. After you ask these questions you will "
        "be on your own, so ask everything you need to know."
    )

    questions_result = call_model_fn(
        cfg=cfg,
        prompt=question_prompt,
        history=[],
    )

    answers_result = call_model_fn(
        cfg=cfg,
        prompt=(
            "The next agent has a few questions for you, please answer each "
            "of them one by one in detail:\n\n" + questions_result.content
        ),
        history=history,
    )

    first_message = history[0] if history else None
    history.clear()
    if first_message is not None:
        history.append(first_message)

    history.extend(
        [
            {"role": "user", "content": question_prompt},
            {"role": "assistant", "content": questions_result.content},
        ]
    )

    handoff_prompt = (
        "Here are the answers the other agent provided.\n\n"
        + answers_result.content
        + "\n\n"
        + "Continue working on this task from where the previous agent left off."
        " You can no longer ask questions. Please follow the spec to interact with "
        "the terminal."
    )

    return handoff_prompt


def _check_proactive_summarization(
    *,
    call_model_fn: Callable[..., ModelResult],
    cfg: Config,
    history: list[dict[str, str]],
    original_instruction: str,
    terminal_state: str,
) -> str | None:
    context_limit = _get_model_context_limit(cfg)
    current_tokens = _count_total_tokens(
        model=_litellm_model_for_cfg(cfg), messages=history
    )
    free_tokens = context_limit - current_tokens

    if free_tokens < _PROACTIVE_FREE_TOKEN_THRESHOLD:
        return _summarize_history(
            call_model_fn=call_model_fn,
            cfg=cfg,
            history=history,
            original_instruction=original_instruction,
            terminal_state=terminal_state,
        )

    return None


def _execute_turn_commands(
    session: TmuxSession,
    parsed: ParsedResponse,
    max_wait_seconds: float,
    logger: Logger | None,
) -> str:
    for cmd in parsed.commands:
        normalized_duration = min(max(cmd.duration, 0.0), 60.0)
        command = Command(
            keystrokes=cmd.keystrokes,
            duration=normalized_duration,
        )
        try:
            execute_command(
                session=session,
                cmd=command,
                max_wait_seconds=max_wait_seconds,
            )
        except TimeoutError:
            terminal_output = limit_output_length(session.get_incremental_output())
            timeout_msg = _TIMEOUT_TEMPLATE.format(
                timeout_sec=command.duration,
                command=command.keystrokes,
                terminal_state=terminal_output,
            )
            if logger:
                logger.log(
                    event_type="command_output",
                    payload={
                        "keystrokes": command.keystrokes,
                        "duration": command.duration,
                        "output": timeout_msg,
                    },
                )
            return timeout_msg

    terminal_output = session.get_incremental_output()
    if logger and parsed.commands:
        last_cmd = parsed.commands[-1]
        logger.log(
            event_type="command_output",
            payload={
                "keystrokes": last_cmd.keystrokes,
                "duration": min(max(last_cmd.duration, 0.0), 60.0),
                "output": terminal_output,
            },
        )

    return limit_output_length(terminal_output)


def run(
    *,
    instruction: str,
    config: Config,
    logger: Logger | None = None,
    system_prompt_template: str = DEFAULT_SYSTEM_PROMPT_TEMPLATE,
) -> RunResult:
    cfg = config
    session = start_session()
    history: list[dict[str, str]] = []
    pending_completion = False
    terminal_state = session.get_incremental_output()
    prompt = build_prompt(
        instruction=instruction,
        terminal_state=terminal_state,
        max_wait_seconds=cfg.max_wait_seconds,
        system_prompt_template=system_prompt_template,
    )

    try:
        for turn in range(1, cfg.max_turns + 1):
            if not session.is_session_alive():
                break

            summarized = _check_proactive_summarization(
                call_model_fn=call_model,
                cfg=cfg,
                history=history,
                original_instruction=instruction,
                terminal_state=terminal_state,
            )
            if summarized is not None:
                prompt = summarized
                if logger:
                    logger.log(
                        event_type="compaction",
                        payload={"kind": "proactive"},
                    )

            try:
                model_result = _query_model(
                    cfg=cfg,
                    prompt=prompt,
                    history=history,
                    original_instruction=instruction,
                    terminal_state=terminal_state,
                    logger=logger,
                )
            except Exception as err:
                if logger:
                    logger.log(
                        event_type="issue",
                        payload={"kind": "model", "message": str(err)},
                    )

                return RunResult(exit_code=1, success=False, history=history)

            result = parse_response(model_result.content)

            feedback = ""
            if result.error:
                feedback += f"ERROR: {result.error}"
            if result.warning:
                feedback += (
                    f"\nWARNINGS: {result.warning}"
                    if feedback
                    else f"WARNINGS: {result.warning}"
                )

            if result.error:
                prompt = (
                    f"Previous response had parsing errors:\n{feedback}\n\n"
                    "Please fix these issues and provide a proper JSON response."
                )
                if logger:
                    logger.log(
                        event_type="issue",
                        payload={"kind": "parser", "message": result.error},
                    )

                continue

            parsed = result.parsed
            assert parsed is not None

            if logger:
                logger.log(
                    event_type="reasoning",
                    payload={"analysis": parsed.analysis, "plan": parsed.plan},
                    turn=turn,
                )

            terminal_output = _execute_turn_commands(
                session=session,
                parsed=parsed,
                max_wait_seconds=cfg.max_wait_seconds,
                logger=logger,
            )
            terminal_state = terminal_output

            if parsed.task_complete:
                if pending_completion:
                    if logger:
                        logger.log(
                            event_type="done",
                            payload={"message": "Task marked complete."},
                        )

                    return RunResult(exit_code=0, success=True, history=history)

                pending_completion = True
                prompt = completion_confirmation_message(terminal_output)
            else:
                pending_completion = False
                if feedback:
                    prompt = (
                        f"Previous response had warnings:\n{feedback}\n\n"
                        f"{terminal_output}"
                    )
                else:
                    prompt = terminal_output

        if logger:
            logger.log(
                event_type="stopped",
                payload={"max_turns": cfg.max_turns},
            )

        return RunResult(exit_code=1, success=False, history=history)
    finally:
        session.close()
