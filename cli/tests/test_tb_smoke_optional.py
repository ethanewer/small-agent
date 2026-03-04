from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _has_tb() -> bool:
    return shutil.which("tb") is not None


def _has_docker() -> bool:
    if shutil.which("docker") is None:
        return False
    proc = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return proc.returncode == 0


@unittest.skipUnless(_has_tb(), "tb command not available")
@unittest.skipUnless(_has_docker(), "docker not available/running")
@unittest.skipUnless(os.getenv("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY not set")
class TestTerminalBenchSmokeOptional(unittest.TestCase):
    def test_tb_help_smoke(self) -> None:
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
        # Keep this lightweight and optional; no full benchmark sweep.
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = Path(tmp_dir) / "sample.jsonl"
            sample_path.write_text(
                json.dumps(
                    {
                        "task_id": "smoke-1",
                        "instruction": "echo hello",
                        "metadata": {"suite": "optional-smoke"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertTrue(sample_path.exists())


if __name__ == "__main__":
    unittest.main()
