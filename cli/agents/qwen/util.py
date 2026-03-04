from __future__ import annotations

import errno
import os
import select
import selectors
import signal
import subprocess
import sys
from typing import Callable, Optional, Protocol, Sequence, cast

ArgList = Sequence[str]


class BinaryReadable(Protocol):
    def read(self, n: int = -1) -> bytes: ...


def _safe_write(stream: object, data: bytes) -> None:
    try:
        fileno = stream.fileno()  # type: ignore[attr-defined]
    except Exception:
        fileno = None

    if fileno is not None:
        view = memoryview(data)
        while view:
            try:
                written = os.write(fileno, view)
                view = view[written:]
            except BlockingIOError:
                select.select([], [fileno], [])
            except OSError as err:
                if err.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    select.select([], [fileno], [])
                    continue
                break
        if not view:
            return

    text = data.decode(errors="replace")
    try:
        stream.write(text)  # type: ignore[attr-defined]
        stream.flush()  # type: ignore[attr-defined]
    except Exception:
        return


def run_subprocess(
    *,
    args: ArgList,
    cwd: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
    check: bool = True,
    echo_stdout: bool = True,
    on_stdout_line: Callable[[str], None] | None = None,
) -> int:
    proc = subprocess.Popen(
        args,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        bufsize=0,
    )

    if proc.stdout is not None:
        os.set_blocking(proc.stdout.fileno(), False)
    if proc.stderr is not None:
        os.set_blocking(proc.stderr.fileno(), False)

    selector = selectors.DefaultSelector()
    if proc.stdout is not None:
        selector.register(proc.stdout, selectors.EVENT_READ, data=sys.stdout)
    if proc.stderr is not None:
        selector.register(proc.stderr, selectors.EVENT_READ, data=sys.stderr)

    captured_stdout_chunks: list[str] = []
    captured_stderr_chunks: list[str] = []
    stdout_line_buffer = ""

    try:
        while selector.get_map():
            events = selector.select(timeout=1.0)
            if not events:
                if proc.poll() is None:
                    continue
                events = [
                    (key, selectors.EVENT_READ) for key in selector.get_map().values()
                ]

            for key, _ in events:
                file_obj = key.fileobj
                # selectors stubs allow int/HasFileno, but we register Popen pipes.
                if isinstance(file_obj, int) or not hasattr(file_obj, "read"):
                    try:
                        selector.unregister(file_obj)
                    except Exception:
                        pass
                    continue

                reader = cast(BinaryReadable, file_obj)
                try:
                    chunk = reader.read(8192)
                except BlockingIOError:
                    continue
                except Exception:
                    chunk = b""

                if not chunk:
                    try:
                        selector.unregister(file_obj)
                    except Exception:
                        pass
                    continue

                decoded_chunk = chunk.decode(errors="replace")
                if key.data is sys.stdout:
                    if echo_stdout:
                        _safe_write(stream=key.data, data=chunk)
                    captured_stdout_chunks.append(decoded_chunk)
                    if on_stdout_line is not None:
                        stdout_line_buffer += decoded_chunk
                        while "\n" in stdout_line_buffer:
                            line, stdout_line_buffer = stdout_line_buffer.split("\n", 1)
                            on_stdout_line(line)
                else:
                    _safe_write(stream=key.data, data=chunk)
                    captured_stderr_chunks.append(decoded_chunk)

        if on_stdout_line is not None and stdout_line_buffer:
            on_stdout_line(stdout_line_buffer)

        rc = proc.wait()
    except KeyboardInterrupt:
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        raise
    finally:
        try:
            selector.close()
        except Exception:
            pass
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()

    if check and rc != 0:
        raise subprocess.CalledProcessError(
            returncode=rc,
            cmd=list(args),
            output="".join(captured_stdout_chunks),
            stderr="".join(captured_stderr_chunks),
        )
    return rc
