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
import pexpect
from rich.console import Console

from agents.core.events import AgentEvent
from agents.core.result import RunResult
from agents.core.sink import EventSink
from agents.core.task import Task
from agents.interface import AgentRuntimeConfig

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

PROMPT_SENTINEL = "__EVOLVER_PROMPT__> "
MAX_OUTPUT_BYTES = 10_000
SYSTEM_PROMPT = """You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and terminal output from previously executed commands.

Respond as JSON with this structure:
{
  "analysis": "Current state analysis",
  "plan": "Next-step plan",
  "commands": [
    {
      "keystrokes": "ls -la\\n",
      "duration": 0.1
    }
  ],
  "task_complete": false
}

Required fields:
- analysis (string)
- plan (string)
- commands (array of command objects)

Command object fields:
- keystrokes (string, required)
- duration (number, optional, defaults to 1.0)

Notes:
- "keystrokes" are sent verbatim.
- Use "C-c" for Ctrl+C and "C-d" for Ctrl+D.
- Prefer short durations and poll with empty keystrokes when needed.

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


@dataclass
class Config:
    active_model_key: str
    active_model: ModelConfig
    verbosity: int = 0
    max_turns: int = 50
    max_wait_seconds: float = 60.0


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


@dataclass
class AgentCallbacks:
    on_reasoning: Callable[[int, ParsedResponse], None] | None = None
    on_command_output: Callable[[Command, str], None] | None = None
    on_issue: Callable[[str, str], None] | None = None
    on_done: Callable[[], None] | None = None
    on_stopped: Callable[[int], None] | None = None


def _cfg_int(cfg: AgentRuntimeConfig, key: str, default: int) -> int:
    value = cfg.agent_config.get(key, default)  # pyright: ignore[reportAny]
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float, str)):
        return int(value)

    return default


def _cfg_float(cfg: AgentRuntimeConfig, key: str, default: float) -> float:
    value = cfg.agent_config.get(key, default)  # pyright: ignore[reportAny]
    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float, str)):
        return float(value)

    return default


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

        commands.append(Command(keystrokes=keystrokes, duration=max(duration, 0.0)))

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


def clean_terminal_output(output: str) -> str:
    ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", output).replace("\r", "")


def normalize_command_output(output: str, command: Command) -> str:
    cleaned = clean_terminal_output(output=output)
    command_line = command.keystrokes.strip()
    normalized_lines: list[str] = []

    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped in {PROMPT_SENTINEL.strip(), "%", "'", '"'}:
            continue

        if stripped.startswith("%"):
            continue

        if command_line and stripped == command_line:
            continue

        if stripped.startswith(PROMPT_SENTINEL):
            stripped = stripped.removeprefix(PROMPT_SENTINEL).strip()
            if not stripped:
                continue

        normalized_lines.append(stripped)

    return "\n".join(normalized_lines).strip()


def build_prompt(instruction: str, terminal_state: str, max_wait_seconds: float) -> str:
    del max_wait_seconds
    return SYSTEM_PROMPT.format(instruction=instruction, terminal_state=terminal_state)


def call_model(
    cfg: Config,
    prompt: str,
    history: list[dict[str, str]],
    api_key: str,
) -> str:
    _configure_tls_trust()
    model_name = cfg.active_model.model
    completion_kwargs: dict[str, Any] = {}

    if cfg.active_model.temperature is not None:
        completion_kwargs["temperature"] = cfg.active_model.temperature

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
            content = result.choices[0].message.content
            return content or ""
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


def start_shell() -> pexpect.spawn[str]:
    child = pexpect.spawn(
        "/bin/bash",
        ["--noprofile", "--norc", "-i"],
        encoding="utf-8",
        timeout=15,
        echo=False,
    )
    child.sendline(f"export PS1='{PROMPT_SENTINEL}'")
    child.expect_exact(PROMPT_SENTINEL)
    return child


def execute_command(
    child: pexpect.spawn[str],
    cmd: Command,
    max_wait_seconds: float,
) -> str:
    effective_wait = min(max(cmd.duration, 0.0), max(max_wait_seconds, 0.1))
    if cmd.keystrokes == "":
        time.sleep(effective_wait)
        return ""

    if cmd.keystrokes.strip() == "C-c":
        child.sendcontrol("c")
    elif cmd.keystrokes.strip() == "C-d":
        child.sendcontrol("d")
    else:
        keystrokes = cmd.keystrokes
        if keystrokes.endswith("\n") and keystrokes.count("\n") == 1:
            child.sendline(keystrokes.rstrip("\n"))
        else:
            child.send(keystrokes)

    timeout = min(max(cmd.duration, 0.0) + 2.0, max(max_wait_seconds, 0.1))
    try:
        child.expect_exact(PROMPT_SENTINEL, timeout=timeout)
        raw_output = child.before or ""
    except pexpect.TIMEOUT:
        raw_output = child.before or ""

    return normalize_command_output(output=raw_output, command=cmd)


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


def _count_total_tokens(messages: list[dict[str, str]]) -> int:
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
        current_tokens = _count_total_tokens(history)
        if context_limit - current_tokens >= _UNWIND_TARGET_FREE_TOKENS:
            break

        if len(history) >= 2:
            del history[-2:]
        else:
            break


def _summarize_history(
    *,
    call_model_fn: Callable[..., str],
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

    try:
        summary_response = call_model_fn(
            cfg=cfg,
            prompt=summary_prompt,
            history=history,
            api_key=api_key,
        )
    except Exception:  # noqa: BLE001
        summary_response = "(summary unavailable)"

    handoff_prompt = (
        f"**Original Task:**\n{original_instruction}\n\n"
        f"**Summary from Previous Agent:**\n{summary_response}\n\n"
        f"**Current Terminal Screen:**\n{terminal_state}\n\n"
        "Continue working on this task from where the previous agent left off."
    )

    history.clear()
    return handoff_prompt


def _check_proactive_summarization(
    *,
    call_model_fn: Callable[..., str],
    cfg: Config,
    history: list[dict[str, str]],
    api_key: str,
    original_instruction: str,
    terminal_state: str,
) -> str | None:
    context_limit = _get_model_context_limit(cfg)
    current_tokens = _count_total_tokens(history)
    free_tokens = context_limit - current_tokens

    if free_tokens < _PROACTIVE_FREE_TOKEN_THRESHOLD:
        _unwind_messages(history=history, cfg=cfg)
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
    *,
    child: pexpect.spawn[str],
    parsed: ParsedResponse,
    max_wait_seconds: float,
    callbacks: AgentCallbacks,
) -> str:
    combined_output_parts: list[str] = []
    for cmd in parsed.commands:
        output = execute_command(
            child=child,
            cmd=cmd,
            max_wait_seconds=max_wait_seconds,
        )
        if callbacks.on_command_output:
            callbacks.on_command_output(cmd, output)

        if output:
            combined_output_parts.append(output)

    terminal_output = "\n".join(combined_output_parts).strip()
    return limit_output_length(terminal_output or "[no new output]")


def run_agent(
    *,
    instruction: str,
    cfg: Config,
    api_key: str,
    callbacks: AgentCallbacks | None = None,
) -> int:
    callbacks = callbacks or AgentCallbacks()
    child = start_shell()
    history: list[dict[str, str]] = []
    pending_completion = False
    prompt = build_prompt(
        instruction=instruction,
        terminal_state="Current Terminal Screen:\n(empty)",
        max_wait_seconds=cfg.max_wait_seconds,
    )

    try:
        for turn in range(1, cfg.max_turns + 1):
            summarized = _check_proactive_summarization(
                call_model_fn=call_model,
                cfg=cfg,
                history=history,
                api_key=api_key,
                original_instruction=instruction,
                terminal_state=prompt,
            )
            if summarized is not None:
                prompt = summarized

            try:
                model_response = call_model(
                    cfg=cfg,
                    prompt=prompt,
                    history=history,
                    api_key=api_key,
                )
            except Exception as err:
                if callbacks.on_issue:
                    callbacks.on_issue("model", str(err))

                return 1
            _append_turn_history(
                history=history,
                prompt=prompt,
                model_response=model_response,
            )

            result = parse_response(model_response)

            if result.error:
                prompt = (
                    f"Previous response had parsing errors:\n{result.error}\n\n"
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
                child=child,
                parsed=parsed,
                max_wait_seconds=cfg.max_wait_seconds,
                callbacks=callbacks,
            )

            if parsed.task_complete:
                if pending_completion:
                    if callbacks.on_done:
                        callbacks.on_done()

                    return 0

                pending_completion = True
                prompt = completion_confirmation_message(terminal_output)
            else:
                pending_completion = False
                if result.warning:
                    prompt = (
                        f"Previous response had warnings:\n{result.warning}\n\n"
                        f"{terminal_output}"
                    )
                else:
                    prompt = terminal_output

        if callbacks.on_stopped:
            callbacks.on_stopped(cfg.max_turns)

        return 1
    finally:
        child.close(force=True)


class Agent:
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
        del console
        verbosity = _cfg_int(cfg=cfg, key="verbosity", default=0)
        max_turns = _cfg_int(cfg=cfg, key="max_turns", default=50)
        max_wait_seconds = _cfg_float(
            cfg=cfg,
            key="max_wait_seconds",
            default=60.0,
        )
        core_cfg = Config(
            active_model_key=cfg.model.model,
            active_model=ModelConfig(
                model=cfg.model.model,
                api_base=cfg.model.api_base,
                api_key=cfg.model.api_key,
                temperature=cfg.model.temperature,
                context_length=cfg.model.context_length,
            ),
            verbosity=verbosity,
            max_turns=max_turns,
            max_wait_seconds=max_wait_seconds,
        )

        def on_reasoning(turn: int, parsed: ParsedResponse) -> None:
            if sink:
                sink.emit(
                    event=AgentEvent(
                        event_type="reasoning",
                        turn=turn,
                        payload={"analysis": parsed.analysis, "plan": parsed.plan},
                    )
                )

        def on_command_output(command: Command, output: str) -> None:
            if sink:
                sink.emit(
                    event=AgentEvent(
                        event_type="command_output",
                        payload={
                            "keystrokes": command.keystrokes,
                            "duration": command.duration,
                            "output": output,
                        },
                    )
                )

        def on_issue(kind: str, message: str) -> None:
            if sink:
                sink.emit(
                    event=AgentEvent(
                        event_type="issue",
                        payload={"kind": kind, "message": message},
                    )
                )

        def on_done() -> None:
            if sink:
                sink.emit(
                    event=AgentEvent(
                        event_type="done",
                        payload={"message": "Task marked complete."},
                    )
                )

        def on_stopped(stopped_max_turns: int) -> None:
            if sink:
                sink.emit(
                    event=AgentEvent(
                        event_type="stopped",
                        payload={"max_turns": stopped_max_turns},
                    )
                )

        callbacks = AgentCallbacks(
            on_reasoning=on_reasoning,
            on_command_output=on_command_output,
            on_issue=on_issue,
            on_done=on_done,
            on_stopped=on_stopped,
        )
        exit_code = run_agent(
            instruction=task.instruction,
            cfg=core_cfg,
            api_key=cfg.model.api_key,
            callbacks=callbacks,
        )
        result = RunResult(
            exit_code=exit_code,
            success=exit_code == 0,
            task_id=task.task_id,
        )
        if sink:
            sink.finalize(result=result)

        return result
