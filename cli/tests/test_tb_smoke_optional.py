from __future__ import annotations

import os
import shutil
import subprocess
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _has_tb() -> bool:
    return shutil.which("tb") is not None


class TestTerminalBenchSmokeOptional(unittest.TestCase):
    def _require_prerequisites(self) -> None:
        strict_mode = os.getenv("SMALL_AGENT_STRICT_TESTS") == "1"
        tb_available = _has_tb()

        if strict_mode:
            self.assertTrue(tb_available, "tb command not available")
            return

        if not tb_available:
            self.skipTest("tb command not available")

    def test_tb_help_smoke(self) -> None:
        self._require_prerequisites()
        # Lightweight smoke: assert tb CLI works in environment.
        proc = subprocess.run(
            ["tb", "--help"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("Usage:", proc.stdout)

    def test_tb_single_task_config_smoke(self) -> None:
        self._require_prerequisites()
        # Keep this very short: only verify CLI startup/version output.
        proc = subprocess.run(
            ["tb", "run", "--help"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("Usage:", proc.stdout)


if __name__ == "__main__":
    unittest.main()
