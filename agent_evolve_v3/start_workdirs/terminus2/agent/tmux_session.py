# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportUnusedCallResult=false, reportUnannotatedClassAttribute=false

from __future__ import annotations

import os
import re
import subprocess
import time


_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class TmuxSession:
    _ENTER_KEYS = {"Enter", "C-m", "KPEnter", "C-j", "^M", "^J"}
    _ENDS_WITH_NEWLINE_PATTERN = r"[\r\n]$"
    _NEWLINE_CHARS = "\r\n"

    def __init__(self, session_name: str) -> None:
        self._session_name = session_name
        self._previous_buffer: str | None = None
        self._closed = False

        result = subprocess.run(
            ["tmux", "-V"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "tmux is not installed. Please install tmux to run this agent."
            )

    def start(self) -> None:
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-x",
                "160",
                "-y",
                "40",
                "-d",
                "-s",
                self._session_name,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            [
                "tmux",
                "set-option",
                "-t",
                self._session_name,
                "history-limit",
                "50000",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

    def _is_enter_key(self, key: str) -> bool:
        return key in self._ENTER_KEYS

    def _ends_with_newline(self, key: str) -> bool:
        return re.search(self._ENDS_WITH_NEWLINE_PATTERN, key) is not None

    def _is_executing_command(self, key: str) -> bool:
        return self._is_enter_key(key) or self._ends_with_newline(key)

    def send_keys(
        self,
        keys: str | list[str],
        *,
        min_timeout_sec: float = 0.0,
    ) -> None:
        if isinstance(keys, str):
            keys = [keys]

        start = time.time()
        subprocess.run(
            ["tmux", "send-keys", "-t", self._session_name, *keys],
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - start
        if elapsed < min_timeout_sec:
            time.sleep(min_timeout_sec - elapsed)

    def capture_pane(self, *, capture_entire: bool = False) -> str:
        extra_args = ["-S", "-"] if capture_entire else []
        result = subprocess.run(
            ["tmux", "capture-pane", "-p", *extra_args, "-t", self._session_name],
            capture_output=True,
            text=True,
        )
        return result.stdout

    def get_incremental_output(self) -> str:
        current_buffer = self.capture_pane(capture_entire=True)

        if self._previous_buffer is None:
            self._previous_buffer = current_buffer
            return f"Current Terminal Screen:\n{self._get_visible_screen()}"

        new_content = self._find_new_content(current_buffer=current_buffer)
        self._previous_buffer = current_buffer

        if new_content is not None:
            if new_content.strip():
                return f"New Terminal Output:\n{new_content}"
            else:
                return f"Current Terminal Screen:\n{self._get_visible_screen()}"
        else:
            return f"Current Terminal Screen:\n{self._get_visible_screen()}"

    def _find_new_content(self, current_buffer: str) -> str | None:
        if self._previous_buffer is None:
            return None

        pb = self._previous_buffer.strip()
        if pb in current_buffer:
            idx = current_buffer.index(pb)
            if "\n" in pb:
                idx = pb.rfind("\n")
            return current_buffer[idx:]

        return None

    def _get_visible_screen(self) -> str:
        return self.capture_pane(capture_entire=False)

    def is_session_alive(self) -> bool:
        result = subprocess.run(
            ["tmux", "has-session", "-t", self._session_name],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        subprocess.run(
            ["tmux", "kill-session", "-t", self._session_name],
            capture_output=True,
            text=True,
        )


def clean_terminal_output(output: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", output).replace("\r", "")


def start_session() -> TmuxSession:
    session_name = f"terminus2-{os.getpid()}"
    session = TmuxSession(session_name=session_name)
    session.start()
    return session
