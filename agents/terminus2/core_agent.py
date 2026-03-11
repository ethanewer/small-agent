from __future__ import annotations

import json
import os
import re
import ssl
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import httpx
import openai
from agents.terminus2.final_summary import ModelResult, build_done_text
from agents.terminus2.tmux_session import TmuxSession, start_session

warnings.filterwarnings(
    action="ignore",
    message=".*certificate verify failed.*",
)

_TLS_CERT_ERROR_MARKERS = (
    "certificate verify failed",
    "certificate_verify_failed",
    "self-signed certificate",
    "unable to get local issuer certificate",
)
_tls_configured = False


class OutputLengthExceededError(Exception):
    truncated_response: str | None

    def __init__(self, message: str, truncated_response: str | None = None) -> None:
        super().__init__(message)
        self.truncated_response = truncated_response


def _configure_tls_trust() -> None:
    global _tls_configured
    if _tls_configured:
        return

    ca_bundle = (
        os.getenv("SMALL_AGENT_CA_BUNDLE")
        or os.getenv("REQUESTS_CA_BUNDLE")
        or os.getenv("SSL_CERT_FILE")
        or ""
    ).strip()
    if ca_bundle:
        resolved_bundle = os.path.expanduser(ca_bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", resolved_bundle)
        os.environ.setdefault("SSL_CERT_FILE", resolved_bundle)
        _tls_configured = True
        return

    try:
        import truststore  # type: ignore

        truststore.inject_into_ssl()
    except Exception:  # noqa: BLE001
        pass

    _tls_configured = True


def _is_tls_certificate_error(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _TLS_CERT_ERROR_MARKERS)


def _tls_error_help_message(api_base: str) -> str:
    return (
        "TLS certificate verification failed while connecting to the model provider.\n"
        f"api_base: {api_base}\n"
        "If your network uses a custom/intercepting CA, set "
        "SMALL_AGENT_CA_BUNDLE=/path/to/ca-bundle.pem (or REQUESTS_CA_BUNDLE / "
        "SSL_CERT_FILE) and rerun the command."
    )


def _make_openai_client(
    *,
    api_base: str,
    api_key: str,
    verify_ssl: bool = True,
) -> openai.OpenAI:
    http_client: httpx.Client | None = None
    if not verify_ssl:
        http_client = httpx.Client(verify=False)  # noqa: S501
    else:
        ca_bundle = os.getenv("REQUESTS_CA_BUNDLE") or os.getenv("SSL_CERT_FILE")
        if ca_bundle:
            ctx = ssl.create_default_context(cafile=ca_bundle)
            http_client = httpx.Client(verify=ctx)

    return openai.OpenAI(
        base_url=api_base,
        api_key=api_key,
        http_client=http_client,
    )


MAX_OUTPUT_BYTES = 10_000
SYSTEM_PROMPT = """You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.

Format your response as JSON with the following structure:

{{
  "analysis": "Analyze the current state based on the terminal output provided. What do you see? What has been accomplished? What still needs to be done?",
  "plan": "Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.",
  "commands": [
    {{
      "keystrokes": "ls -la\\n",
      "duration": 0.1
    }},
    {{
      "keystrokes": "cd project\\n",
      "duration": 0.1
    }}
  ],
  "task_complete": true
}}

Required fields:
- "analysis": Your analysis of the current situation
- "plan": Your plan for the next steps
- "commands": Array of command objects to execute

Optional fields:
- "task_complete": Boolean indicating if the task is complete (defaults to false if not present)

Command object structure:
- "keystrokes": String containing the exact keystrokes to send to the terminal (required)
- "duration": Number of seconds to wait for the command to complete before the next command will be executed (defaults to 1.0 if not present)

IMPORTANT: The text inside "keystrokes" will be used completely verbatim as keystrokes. Write commands exactly as you want them sent to the terminal:
- Most bash commands should end with a newline (\\n) to cause them to execute
- For special key sequences, use tmux-style escape sequences:
  - C-c for Ctrl+C
  - C-d for Ctrl+D

The "duration" attribute specifies the number of seconds to wait for the command to complete (default: 1.0) before the next command will be executed. On immediate tasks (e.g., cd, ls, echo, cat) set a duration of 0.1 seconds. On commands (e.g., gcc, find, rustc) set a duration of 1.0 seconds. On slow commands (e.g., make, python3 [long running script], wget [file]) set an appropriate duration as you determine necessary.

It is better to set a smaller duration than a longer duration. It is always possible to wait again if the prior output has not finished, by running {{"keystrokes": "", "duration": 10.0}} on subsequent requests to wait longer. Never wait longer than 60 seconds; prefer to poll to see intermediate result status.

Important notes:
- Each command's keystrokes are sent exactly as written to the terminal
- Do not include extra whitespace before or after the keystrokes unless it's part of the intended command
- Extra text before or after the JSON will generate warnings but be tolerated
- The JSON must be valid - use proper escaping for quotes and special characters within strings
- Commands array can be empty if you want to wait without taking action

Task Description:
{instruction}

Current terminal state:
{terminal_state}
"""


@dataclass
class ModelConfig:
    model: str
    api_base: str
    api_key: str | None = None
    temperature: float | None = None
    context_length: int | None = None
    extra_params: dict[str, Any] | None = None


@dataclass
class Config:
    active_model_key: str
    active_model: ModelConfig
    verbosity: int = 1
    max_turns: int = 50
    max_wait_seconds: float = 60.0
    final_message_enabled: bool = True


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
    final_message: str | None


@dataclass
class AgentCallbacks:
    on_reasoning: Callable[[int, ParsedResponse], None] | None = None
    on_command_output: Callable[[Command, str], None] | None = None
    on_issue: Callable[[str, str], None] | None = None
    on_done: Callable[[str], None] | None = None
    on_stopped: Callable[[int], None] | None = None


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

    commands_data: list[object] = data["commands"]  # pyright: ignore[reportAny]
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

    try:
        final_message = _normalized_final_message(data.get("final_message"))
    except ValueError as exc:
        return ParseResult(
            parsed=None,
            error=str(exc),
            warning=_format_warnings(warnings),
        )

    return ParseResult(
        parsed=ParsedResponse(
            analysis=str(data["analysis"]),
            plan=str(data["plan"]),
            commands=commands,
            task_complete=task_complete,
            final_message=final_message,
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


def _normalized_final_message(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        return value

    raise ValueError("'final_message' must be a string when provided")


def build_prompt(instruction: str, terminal_state: str, max_wait_seconds: float) -> str:
    del max_wait_seconds
    return SYSTEM_PROMPT.format(instruction=instruction, terminal_state=terminal_state)


def call_model(
    cfg: Config,
    prompt: str,
    history: list[dict[str, str]],
    api_key: str,
) -> ModelResult:
    _configure_tls_trust()
    model_name = cfg.active_model.model
    completion_kwargs: dict[str, Any] = {}

    if cfg.active_model.temperature is not None:
        completion_kwargs["temperature"] = cfg.active_model.temperature

    if cfg.active_model.extra_params:
        completion_kwargs["extra_body"] = cfg.active_model.extra_params

    messages: list[dict[str, str]] = [*history, {"role": "user", "content": prompt}]
    last_error: Exception | None = None
    allow_insecure_tls_retry = True
    verify_ssl = True

    for attempt in range(3):
        try:
            client = _make_openai_client(
                api_base=cfg.active_model.api_base,
                api_key=api_key,
                verify_ssl=verify_ssl,
            )
            result = client.chat.completions.create(  # pyright: ignore[reportCallIssue]
                model=model_name,
                messages=messages,  # pyright: ignore[reportArgumentType]
                **completion_kwargs,
            )
            content = result.choices[0].message.content or ""
            if result.choices[0].finish_reason == "length":
                raise OutputLengthExceededError(
                    f"Model {model_name} hit max_tokens limit. Response was truncated.",
                    truncated_response=content,
                )

            usage = result.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0

            return ModelResult(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except OutputLengthExceededError:
            raise
        except Exception as err:  # noqa: BLE001
            last_error = err
            err_msg = str(err)
            if _is_tls_certificate_error(err_msg):
                if allow_insecure_tls_retry:
                    verify_ssl = False
                    allow_insecure_tls_retry = False
                    continue

                raise RuntimeError(
                    _tls_error_help_message(api_base=cfg.active_model.api_base)
                ) from err

            lowered = err_msg.lower()
            if (
                any(
                    token in lowered
                    for token in ("429", "rate", "timeout", "temporarily")
                )
                and attempt < 2
            ):
                time.sleep(2 * (attempt + 1))
                continue

            break

    raise RuntimeError(f"Model request failed: {last_error}")


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
    api_key: str,
    original_instruction: str,
    terminal_state: str,
) -> ModelResult:
    last_error: Exception | None = None
    current_prompt = prompt

    for _attempt in range(_MAX_QUERY_ATTEMPTS):
        try:
            model_result = call_model(
                cfg=cfg,
                prompt=current_prompt,
                history=history,
                api_key=api_key,
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
                    api_key=api_key,
                    original_instruction=original_instruction,
                    terminal_state=terminal_state,
                )
                current_prompt = f"{summarized}\n\n{current_prompt}"
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


def _estimate_total_tokens(messages: list[dict[str, str]]) -> int:
    """Rough per-message estimate used only during iterative history unwinding."""
    return sum(len(m.get("content", "")) // 4 for m in messages)


def _get_model_context_limit(cfg: Config) -> int:
    if cfg.active_model.context_length and cfg.active_model.context_length > 0:
        return cfg.active_model.context_length

    return _FALLBACK_CONTEXT_LIMIT


def _unwind_messages(
    history: list[dict[str, str]],
    cfg: Config,
) -> None:
    context_limit = _get_model_context_limit(cfg)
    while len(history) > 1:
        current_tokens = _estimate_total_tokens(history)
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
    api_key: str,
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
        api_key=api_key,
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
        api_key=api_key,
    )

    answers_result = call_model_fn(
        cfg=cfg,
        prompt=(
            "The next agent has a few questions for you, please answer each "
            "of them one by one in detail:\n\n" + questions_result.content
        ),
        history=history,
        api_key=api_key,
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
    api_key: str,
    original_instruction: str,
    terminal_state: str,
    current_tokens: int,
) -> str | None:
    context_limit = _get_model_context_limit(cfg)
    free_tokens = context_limit - current_tokens

    if free_tokens < _PROACTIVE_FREE_TOKEN_THRESHOLD:
        return _summarize_history(
            call_model_fn=call_model_fn,
            cfg=cfg,
            history=history,
            api_key=api_key,
            original_instruction=original_instruction,
            terminal_state=terminal_state,
        )

    return None


def _execute_turn_commands(
    session: TmuxSession,
    parsed: ParsedResponse,
    max_wait_seconds: float,
    callbacks: AgentCallbacks,
) -> str:
    for cmd in parsed.commands:
        normalized_duration = min(max(cmd.duration, 0.0), 60.0)
        command = Command(
            keystrokes=cmd.keystrokes,
            duration=normalized_duration,
        )
        execute_command(
            session=session,
            cmd=command,
            max_wait_seconds=max_wait_seconds,
        )
        if callbacks.on_command_output:
            callbacks.on_command_output(command, "")

    terminal_output = session.get_incremental_output()
    return limit_output_length(terminal_output)


def run_agent(
    instruction: str,
    cfg: Config,
    api_key: str,
    callbacks: AgentCallbacks | None = None,
) -> int:
    callbacks = callbacks or AgentCallbacks()
    session = start_session()
    history: list[dict[str, str]] = []
    pending_completion = False
    pending_final_message: str | None = None
    terminal_state = session.get_incremental_output()
    last_prompt_tokens = 0
    prompt = build_prompt(
        instruction=instruction,
        terminal_state=terminal_state,
        max_wait_seconds=cfg.max_wait_seconds,
    )

    try:
        for turn in range(1, cfg.max_turns + 1):
            if not session.is_session_alive():
                break

            summarized = _check_proactive_summarization(
                call_model_fn=call_model,
                cfg=cfg,
                history=history,
                api_key=api_key,
                original_instruction=instruction,
                terminal_state=terminal_state,
                current_tokens=last_prompt_tokens,
            )
            if summarized is not None:
                prompt = summarized

            try:
                model_result = _query_model(
                    cfg=cfg,
                    prompt=prompt,
                    history=history,
                    api_key=api_key,
                    original_instruction=instruction,
                    terminal_state=terminal_state,
                )
            except Exception as err:
                if callbacks.on_issue:
                    callbacks.on_issue("model", str(err))

                return 1

            last_prompt_tokens = (
                model_result.prompt_tokens + model_result.completion_tokens
            )

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
                if callbacks.on_issue:
                    callbacks.on_issue("parser", result.error)

                continue

            parsed = result.parsed
            assert parsed is not None

            if callbacks.on_reasoning:
                callbacks.on_reasoning(turn, parsed)

            terminal_output = _execute_turn_commands(
                session=session,
                parsed=parsed,
                max_wait_seconds=cfg.max_wait_seconds,
                callbacks=callbacks,
            )
            terminal_state = terminal_output

            if parsed.task_complete:
                if parsed.final_message and parsed.final_message.strip():
                    pending_final_message = parsed.final_message.strip()

                if pending_completion:
                    if cfg.final_message_enabled:
                        done_text = build_done_text(
                            call_model_fn=call_model,
                            cfg=cfg,
                            history=history,
                            api_key=api_key,
                            pending_final_message=pending_final_message,
                        )
                        if callbacks.on_done:
                            callbacks.on_done(done_text)

                    return 0

                pending_completion = True
                prompt = completion_confirmation_message(terminal_output)
            else:
                pending_completion = False
                pending_final_message = None
                if feedback:
                    prompt = (
                        f"Previous response had warnings:\n{feedback}\n\n"
                        f"{terminal_output}"
                    )
                else:
                    prompt = terminal_output

        if callbacks.on_stopped:
            callbacks.on_stopped(cfg.max_turns)

        return 1
    finally:
        session.close()
