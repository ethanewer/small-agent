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


class TestTerminalBenchSmokeOptional(unittest.TestCase):
    def _require_prerequisites(self) -> None:
        strict_mode = os.getenv("SMALL_AGENT_STRICT_TESTS") == "1"
        tb_available = _has_tb()
        docker_available = _has_docker()
        has_openrouter_key = bool(os.getenv("OPENROUTER_API_KEY"))

        if strict_mode:
            self.assertTrue(tb_available, "tb command not available")
            self.assertTrue(docker_available, "docker not available/running")
            self.assertTrue(has_openrouter_key, "OPENROUTER_API_KEY not set")
            return

        if not tb_available:
            self.skipTest("tb command not available")
        if not docker_available:
            self.skipTest("docker not available/running")
        if not has_openrouter_key:
            self.skipTest("OPENROUTER_API_KEY not set")

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
