from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
CLI_PATH = PROJECT_ROOT / "cli.py"
CLI_SPEC = importlib.util.spec_from_file_location("cli_live", CLI_PATH)
assert CLI_SPEC and CLI_SPEC.loader
cli = importlib.util.module_from_spec(CLI_SPEC)
sys.modules["cli_live"] = cli
CLI_SPEC.loader.exec_module(cli)

pytestmark = [pytest.mark.requires_api_key]


def _local_or_path_binary(name: str) -> str | None:
    local_path = PROJECT_ROOT / ".local" / "bin" / name
    if local_path.exists():
        return str(local_path)

    return shutil.which(name)


class TestHeadlessLive(unittest.TestCase):
    def test_qwen_headless_live_if_available(self) -> None:
        strict_mode = os.getenv("SMALL_AGENT_STRICT_TESTS") == "1"
        if _local_or_path_binary("qwen") is None:
            if strict_mode:
                self.fail("qwen binary not found on PATH")

            self.skipTest("qwen binary not found on PATH")

        if not cli.resolve_api_key("OPENROUTER_API_KEY"):
            if strict_mode:
                self.fail("OPENROUTER_API_KEY not set")

            self.skipTest("OPENROUTER_API_KEY not set")

        env = os.environ.copy()
        env["PATH"] = f"{PROJECT_ROOT / '.local' / 'bin'}:{env.get('PATH', '')}"

        proc = subprocess.run(
            [
                sys.executable,
                str(CLI_PATH),
                "--agent",
                "qwen",
                "--model",
                "qwen3.5-flash",
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
        if proc.returncode != 0:
            lowered = combined_output.lower()
            network_failure_markers = (
                "certificate_verify_failed",
                "self-signed certificate",
                "connection error",
                "api error",
                "econnreset",
                "etimedout",
                "enotfound",
            )
            if any(marker in lowered for marker in network_failure_markers):
                if strict_mode:
                    self.fail(combined_output)

                self.skipTest(
                    "qwen live call failed due to network/API transport issues"
                )

        self.assertEqual(proc.returncode, 0, msg=combined_output)
        self.assertIn("Agent: qwen", combined_output)


if __name__ == "__main__":
    unittest.main()
