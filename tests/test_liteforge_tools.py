from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import httpx
import pytest

from agents.liteforge.context import Context, ToolCall, ToolResult
from agents.liteforge.provider import (
    _parse_model_string,
    _resolve_openai_model,
    detect_provider,
)
from agents.liteforge.tools import fetch, shell
from agents.liteforge.tools import (
    fs_patch,
    fs_read,
    fs_remove,
    fs_search,
    fs_undo,
    fs_write,
)
from agents.liteforge.tools.executor import ToolExecutor
from agents.liteforge.tools.todo import TodoManager, execute_read, execute_write


def _load_shell_module_from_source() -> types.ModuleType:
    shell_path = (
        Path(__file__).resolve().parent.parent / "agents/liteforge/tools/shell.py"
    )
    spec = importlib.util.spec_from_file_location("liteforge_shell_source", shell_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_provider_module_from_source() -> types.ModuleType:
    provider_path = (
        Path(__file__).resolve().parent.parent / "agents/liteforge/provider.py"
    )
    module_name = "liteforge_provider_source"
    spec = importlib.util.spec_from_file_location(
        module_name,
        provider_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tool_env(tmp_path: Path) -> dict[str, object]:
    return {
        "cwd": str(tmp_path),
        "shell": "/bin/sh",
        "max_read_size": 2000,
    }


def test_write_read_patch_remove_undo_roundtrip(tool_env: dict[str, object]) -> None:
    snapshots: dict[str, str | None] = {}

    created = fs_write.execute(
        args={"file_path": "notes.txt", "content": "hello world"},
        env=tool_env,
        snapshots=snapshots,
    )
    assert "Created file:" in created

    read_result = fs_read.execute(args={"file_path": "notes.txt"}, env=tool_env)
    assert "1:hello world" in read_result

    patched = fs_patch.execute(
        args={
            "file_path": "notes.txt",
            "old_string": "hello world",
            "new_string": "goodbye world",
        },
        env=tool_env,
        snapshots=snapshots,
    )
    assert "Replaced 1 occurrence" in patched

    patched_read = fs_read.execute(args={"file_path": "notes.txt"}, env=tool_env)
    assert "1:goodbye world" in patched_read

    removed = fs_remove.execute(
        args={"path": "notes.txt"},
        env=tool_env,
        snapshots=snapshots,
    )
    assert "Removed file:" in removed

    restored = fs_undo.execute(
        args={"path": "notes.txt"},
        env=tool_env,
        snapshots=snapshots,
    )
    assert "to previous state" in restored

    restored_read = fs_read.execute(args={"file_path": "notes.txt"}, env=tool_env)
    assert "1:goodbye world" in restored_read


def test_read_line_ranges(tool_env: dict[str, object]) -> None:
    snapshots: dict[str, str | None] = {}
    fs_write.execute(
        args={"file_path": "multi.txt", "content": "a\nb\nc\nd"},
        env=tool_env,
        snapshots=snapshots,
    )

    ranged = fs_read.execute(
        args={"file_path": "multi.txt", "start_line": 2, "end_line": 3},
        env=tool_env,
    )
    lines = ranged.splitlines()
    assert lines[0:2] == ["2:b", "3:c"]
    assert "[Showing lines 2-3 of 4 total]" in ranged


def test_read_accepts_string_line_numbers_and_bool(tool_env: dict[str, object]) -> None:
    snapshots: dict[str, str | None] = {}
    fs_write.execute(
        args={"file_path": "multi.txt", "content": "a\nb\nc\nd"},
        env=tool_env,
        snapshots=snapshots,
    )

    ranged = fs_read.execute(
        args={
            "file_path": "multi.txt",
            "start_line": "2",
            "end_line": "3",
            "show_line_numbers": "false",
        },
        env=tool_env,
    )
    assert ranged.splitlines()[0:2] == ["b", "c"]


def test_read_invalid_line_number_returns_error_without_exception(
    tool_env: dict[str, object],
) -> None:
    snapshots: dict[str, str | None] = {}
    fs_write.execute(
        args={"file_path": "multi.txt", "content": "a\nb\nc\nd"},
        env=tool_env,
        snapshots=snapshots,
    )
    executor = ToolExecutor(env=tool_env)
    output, is_error = executor.execute(
        tool_name="read",
        arguments={"file_path": "multi.txt", "start_line": "not-an-int"},
    )
    assert not is_error
    assert "start_line must be an integer" in output


def test_search_accepts_string_offset_and_head_limit(
    monkeypatch: pytest.MonkeyPatch,
    tool_env: dict[str, object],
) -> None:
    def fake_run(*args: object, **kwargs: object) -> types.SimpleNamespace:
        del args, kwargs
        return types.SimpleNamespace(
            stdout="file1.py\nfile2.py\nfile3.py\n",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(fs_search.subprocess, "run", fake_run)
    output = fs_search.execute(
        args={
            "pattern": "README|readme|README.md",
            "offset": "1",
            "head_limit": "1",
        },
        env=tool_env,
    )
    assert output == "file2.py"


def test_search_invalid_head_limit_returns_error_without_exception(
    tool_env: dict[str, object],
) -> None:
    executor = ToolExecutor(env=tool_env)
    output, is_error = executor.execute(
        tool_name="fs_search",
        arguments={"pattern": ".*", "head_limit": "not-an-int"},
    )
    assert not is_error
    assert "head_limit must be an integer" in output


def test_todo_write_and_read_roundtrip() -> None:
    manager = TodoManager()
    write_out = execute_write(
        args={
            "todos": [
                {"id": "a", "content": "first", "status": "pending"},
                {"id": "b", "content": "second", "status": "in_progress"},
            ]
        },
        manager=manager,
    )
    assert "[ ] a: first" in write_out
    assert "[~] b: second" in write_out

    read_out = execute_read(manager=manager)
    assert "[ ] a: first" in read_out
    assert "[~] b: second" in read_out


def test_fetch_converts_html(
    monkeypatch: pytest.MonkeyPatch, tool_env: dict[str, object]
) -> None:
    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/html"}
        text = "<html><body><h1>Hello</h1><p>World</p></body></html>"

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            del exc_type, exc, tb
            return False

        def get(self, url: str) -> FakeResponse:
            if url == "https://example.com/robots.txt":
                r = FakeResponse()
                r.text = ""
                return r
            assert url == "https://example.com"
            return FakeResponse()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    output = fetch.execute(args={"url": "https://example.com"}, env=tool_env)
    assert "Hello" in output
    assert "World" in output


def test_shell_execute_uses_subprocess_and_formats_output(
    monkeypatch: pytest.MonkeyPatch,
    tool_env: dict[str, object],
) -> None:
    def fake_run(*args: object, **kwargs: object) -> types.SimpleNamespace:
        del args, kwargs
        return types.SimpleNamespace(
            stdout="line1\nline2\n",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(shell.subprocess, "run", fake_run)
    output = shell.execute(args={"command": "echo hi"}, env=tool_env)
    assert "line1" in output
    assert "Exit code: 0" in output


def test_shell_rejects_broad_python_process_kills(
    monkeypatch: pytest.MonkeyPatch,
    tool_env: dict[str, object],
) -> None:
    shell_module = _load_shell_module_from_source()
    called = False

    def fake_run(*args: object, **kwargs: object) -> types.SimpleNamespace:
        del args, kwargs
        nonlocal called
        called = True
        return types.SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(shell_module.subprocess, "run", fake_run)
    output = shell_module.execute(args={"command": "pkill -f python"}, env=tool_env)
    assert "Refusing broad process-kill command" in output
    assert called is False


def test_shell_allows_targeted_pid_kill(
    monkeypatch: pytest.MonkeyPatch,
    tool_env: dict[str, object],
) -> None:
    shell_module = _load_shell_module_from_source()

    def fake_run(*args: object, **kwargs: object) -> types.SimpleNamespace:
        del args, kwargs
        return types.SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(shell_module.subprocess, "run", fake_run)
    output = shell_module.execute(args={"command": "kill 12345"}, env=tool_env)
    assert "Exit code: 0" in output


def test_executor_enforces_read_before_patch(tool_env: dict[str, object]) -> None:
    target = Path(str(tool_env["cwd"])) / "file.txt"
    target.write_text("old value\n")

    executor = ToolExecutor(env=tool_env)
    output, is_error = executor.execute(
        tool_name="patch",
        arguments={
            "file_path": "file.txt",
            "old_string": "old value\n",
            "new_string": "new value\n",
        },
    )
    assert not is_error
    assert "must read the file" in output

    read_output, read_err = executor.execute(
        tool_name="read",
        arguments={"file_path": "file.txt"},
    )
    assert not read_err
    assert "old value" in read_output

    patch_output, patch_err = executor.execute(
        tool_name="patch",
        arguments={
            "file_path": "file.txt",
            "old_string": "old value\n",
            "new_string": "new value\n",
        },
    )
    assert not patch_err
    assert "Replaced 1 occurrence" in patch_output
    assert target.read_text() == "new value\n"


def test_executor_overwrite_requires_read(tool_env: dict[str, object]) -> None:
    target = Path(str(tool_env["cwd"])) / "over.txt"
    target.write_text("first\n")
    executor = ToolExecutor(env=tool_env)

    output, is_error = executor.execute(
        tool_name="write",
        arguments={"file_path": "over.txt", "content": "second", "overwrite": True},
    )
    assert not is_error
    assert "must read the file" in output


def test_context_message_conversion_shapes() -> None:
    context = Context()
    context.set_system_messages(texts=["sysA", "sysB"])
    context.add_user_message(text="question")
    tool_call = ToolCall(id="call_1", name="read", arguments={"file_path": "a.txt"})
    context.add_assistant_message(content="running tool", tool_calls=[tool_call])
    context.add_tool_result(
        result=ToolResult(
            tool_call_id="call_1",
            name="read",
            content="1:hello",
            is_error=False,
        )
    )

    api_messages = context.to_api_messages()
    assert api_messages[0]["role"] == "system"
    assert "sysA" in api_messages[0]["content"]
    assert "sysB" in api_messages[0]["content"]
    assert api_messages[1]["role"] == "user"
    assert api_messages[-1]["role"] == "tool"

    system_text, anthropic_messages = context.to_anthropic_messages()
    assert "sysA" in system_text and "sysB" in system_text
    assert anthropic_messages[0]["role"] == "user"
    assert anthropic_messages[1]["role"] == "assistant"
    assert anthropic_messages[2]["role"] == "user"


def test_provider_model_parsing_and_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, model = _parse_model_string(model="openai/gpt-4o-mini")
    assert provider == "openai"
    assert model == "gpt-4o-mini"

    provider, model = _parse_model_string(model="gpt-4.1")
    assert provider == ""
    assert model == "gpt-4.1"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    assert detect_provider() == "anthropic"

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert detect_provider() == "openai"


def test_provider_preserves_openrouter_model_names() -> None:
    assert (
        _resolve_openai_model(
            model="qwen/qwen3-coder-next",
            base_url="https://openrouter.ai/api/v1",
        )
        == "qwen/qwen3-coder-next"
    )
    assert (
        _resolve_openai_model(
            model="openai/gpt-4o-mini",
            base_url="https://api.openai.com/v1",
        )
        == "gpt-4o-mini"
    )
    assert (
        _resolve_openai_model(
            model="z-ai/glm-5",
            base_url="https://example-proxy.invalid/v1",
        )
        == "z-ai/glm-5"
    )


def test_provider_retries_transient_openai_connection_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_module = _load_provider_module_from_source()
    attempts = 0

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self.create)
            )

        def create(self, **kwargs: object) -> types.SimpleNamespace:
            del kwargs
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("Connection error.")
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content="retry ok", tool_calls=None
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=types.SimpleNamespace(
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                ),
            )

    fake_openai = types.ModuleType("openai")
    setattr(fake_openai, "OpenAI", FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr(provider_module.time, "sleep", lambda _seconds: None)

    context = Context()
    context.add_user_message(text="hello")
    response = provider_module.chat(
        context=context,
        model="openai/gpt-4o-mini",
        tools=[],
    )

    assert response.content == "retry ok"
    assert attempts == 2


def test_provider_retries_streaming_openai_connection_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_module = _load_provider_module_from_source()
    attempts = 0

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self.create)
            )

        def create(self, **kwargs: object):
            del kwargs
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("Connection error.")

            def stream():
                yield types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            delta=types.SimpleNamespace(
                                content="retry stream", tool_calls=None
                            ),
                            finish_reason=None,
                        )
                    ]
                )
                yield types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            delta=types.SimpleNamespace(content=None, tool_calls=None),
                            finish_reason="stop",
                        )
                    ]
                )

            return stream()

    fake_openai = types.ModuleType("openai")
    setattr(fake_openai, "OpenAI", FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr(provider_module.time, "sleep", lambda _seconds: None)

    chunks: list[str] = []
    context = Context()
    context.add_user_message(text="hello")
    response = provider_module.chat(
        context=context,
        model="openai/gpt-4o-mini",
        tools=[],
        stream_callback=chunks.append,
    )

    assert response.content == "retry stream"
    assert chunks == ["retry stream"]
    assert attempts == 2
