# pyright: reportUnusedCallResult=false

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> int:
    workspace_root = Path(__file__).resolve().parent
    completed = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "tests",
            "-q",
        ],
        cwd=workspace_root,
        text=True,
        capture_output=True,
        check=False,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
