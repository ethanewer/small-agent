from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
CLI_PATH = PROJECT_ROOT / "cli.py"
CLI_SPEC = importlib.util.spec_from_file_location("cli_live", CLI_PATH)
assert CLI_SPEC and CLI_SPEC.loader
cli = importlib.util.module_from_spec(CLI_SPEC)
sys.modules["cli_live"] = cli
CLI_SPEC.loader.exec_module(cli)


def _local_or_path_binary(name: str) -> str | None:
    local_path = PROJECT_ROOT / ".bin" / name
    if local_path.exists():
        return str(local_path)

    return shutil.which(name)


class TestHeadlessLive(unittest.TestCase):
    def test_qwen_headless_live_if_available(self) -> None:
        if _local_or_path_binary("qwen") is None:
            self.skipTest("qwen binary not found on PATH")
        if not cli.resolve_api_key("OPENROUTER_API_KEY"):
            self.skipTest("OPENROUTER_API_KEY not set")
        env = os.environ.copy()
        env["PATH"] = f"{PROJECT_ROOT / '.bin'}:{env.get('PATH', '')}"

        proc = subprocess.run(
            [
                sys.executable,
                str(CLI_PATH),
                "--agent",
                "qwen",
                "--model",
                "qwen3-coder-next",
                "Reply with exactly: OK",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            env=env,
        )
        combined_output = f"{proc.stdout}\n{proc.stderr}"
        self.assertEqual(proc.returncode, 0, msg=combined_output)
        self.assertIn("Agent: qwen", combined_output)


if __name__ == "__main__":
    unittest.main()
