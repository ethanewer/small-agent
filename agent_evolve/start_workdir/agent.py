from __future__ import annotations

import contextlib
import io
import json
import os
import re
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, cast

from rich.console import Console

from agents.core.events import AgentEvent
from agents.core.result import RunResult
from agents.core.sink import EventSink
from agents.core.task import Task
from agents.interface import AgentRuntimeConfig

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")
warnings.filterwarnings(
    action="ignore",
    message=".*Failed to fetch remote model cost map.*",
)
warnings.filterwarnings(
    action="ignore",
    message=".*certificate verify failed.*",
)

import litellm  # noqa: E402
import pexpect  # noqa: E402

from litellm import completion  # noqa: E402

_TLS_CERT_ERROR_MARKERS = (
    "certificate verify failed",
    "certificate_verify_failed",
    "self-signed certificate",
    "unable to get local issuer certificate",
)
_tls_configured = False

litellm.suppress_debug_info = True

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
    except Exception:  # noqa: BLE001
        _tls_configured = True
        return

    try:
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


def extract_json_content(response: str) -> str:
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
        raise ValueError("No valid JSON object found in model response")

    return response[json_start:json_end]


def parse_response(text: str) -> ParsedResponse:
    json_payload = extract_json_content(response=text)
    data = json.loads(json_payload)
    missing_fields = [
        field for field in ("analysis", "plan", "commands") if field not in data
    ]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    if not isinstance(data["commands"], list):
        raise ValueError("Field 'commands' must be an array")

    commands: list[Command] = []
    parse_error: str | None = None
    for i, command in enumerate(data["commands"]):
        if not isinstance(command, dict):
            parse_error = f"Command {i + 1} must be an object"
            break

        if "keystrokes" not in command:
            parse_error = f"Command {i + 1} missing required 'keystrokes' field"
            break

        keystrokes = command["keystrokes"]
        if not isinstance(keystrokes, str):
            parse_error = f"Command {i + 1} 'keystrokes' must be a string"
            break

        duration_value = command.get("duration", 1.0)
        duration = (
            float(duration_value) if isinstance(duration_value, (int, float)) else 1.0
        )
        commands.append(Command(keystrokes=keystrokes, duration=max(duration, 0.0)))

    task_complete = _coerce_task_complete(data.get("task_complete", False))
    if parse_error:
        if task_complete:
            commands = []
        else:
            raise ValueError(parse_error)

    return ParsedResponse(
        analysis=str(data["analysis"]),
        plan=str(data["plan"]),
        commands=commands,
        task_complete=task_complete,
    )


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


@contextlib.contextmanager
def suppress_stdio_fd() -> Any:
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(devnull)


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
    if cfg.active_model.api_base.rstrip("/").endswith("openrouter.ai/api/v1"):
        completion_kwargs["custom_llm_provider"] = "openrouter"
        if model_name.startswith("openrouter/"):
            model_name = model_name.removeprefix("openrouter/")

    if cfg.active_model.temperature is not None:
        completion_kwargs["temperature"] = cfg.active_model.temperature

    last_error: Exception | None = None
    allow_insecure_tls_retry = True
    for attempt in range(3):
        try:
            with (
                suppress_stdio_fd(),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                result = completion(
                    model=model_name,
                    api_base=cfg.active_model.api_base,
                    api_key=api_key,
                    messages=history + [{"role": "user", "content": prompt}],
                    **completion_kwargs,
                )
            payload = cast(dict[str, Any], cast(object, result))
            return str(payload["choices"][0]["message"]["content"])
        except Exception as err:  # noqa: BLE001
            last_error = err
            message = str(err).lower()
            if _is_tls_certificate_error(str(err)):
                if allow_insecure_tls_retry:
                    completion_kwargs["ssl_verify"] = False
                    allow_insecure_tls_retry = False
                    continue

                raise RuntimeError(
                    _tls_error_help_message(api_base=cfg.active_model.api_base)
                ) from err

            if (
                any(
                    token in message
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
    prompt = build_prompt(
        instruction=instruction,
        terminal_state="Current Terminal Screen:\n(empty)",
        max_wait_seconds=cfg.max_wait_seconds,
    )

    try:
        for turn in range(1, cfg.max_turns + 1):
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

            try:
                parsed = parse_response(model_response)
            except Exception as err:
                prompt = (
                    "Previous response had parsing errors:\n"
                    f"{err}\n\n"
                    "Please fix these issues and provide a proper JSON response."
                )
                if callbacks.on_issue:
                    callbacks.on_issue("parser", str(err))

                continue

            if callbacks.on_reasoning:
                callbacks.on_reasoning(turn, parsed)

            terminal_output = _execute_turn_commands(
                child=child,
                parsed=parsed,
                max_wait_seconds=cfg.max_wait_seconds,
                callbacks=callbacks,
            )

            if parsed.task_complete:
                if callbacks.on_done:
                    callbacks.on_done()

                return 0

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
