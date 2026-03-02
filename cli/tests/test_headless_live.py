from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = PROJECT_ROOT / "cli.py"
CLI_SPEC = importlib.util.spec_from_file_location("cli_live", CLI_PATH)
assert CLI_SPEC and CLI_SPEC.loader
cli = importlib.util.module_from_spec(CLI_SPEC)
sys.modules["cli_live"] = cli
CLI_SPEC.loader.exec_module(cli)


class TestHeadlessLive(unittest.TestCase):
    def test_qwen_headless_live_if_available(self) -> None:
        if shutil.which("qwen") is None:
            self.skipTest("qwen binary not found on PATH")
        if not cli.resolve_api_key("OPENROUTER_API_KEY"):
            self.skipTest("OPENROUTER_API_KEY not set")

        proc = subprocess.run(
            [
                sys.executable,
                str(CLI_PATH),
                "--agent",
                "qwen",
                "--model",
                "qwen3.5-35b-a3b",
                "Reply with exactly: OK",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        combined_output = f"{proc.stdout}\n{proc.stderr}"
        self.assertEqual(proc.returncode, 0, msg=combined_output)
        self.assertIn("Agent: qwen", combined_output)


if __name__ == "__main__":
    unittest.main()
