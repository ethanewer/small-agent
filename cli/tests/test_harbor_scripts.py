from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestHarborRunnerScripts(unittest.TestCase):
    def _run_script(
        self, script_name: str, *args: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(PROJECT_ROOT / "harbor" / script_name), *args],
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )

    def test_run_smoke_uses_config_defaults(self) -> None:
        proc = self._run_script("run_smoke.sh", "--dry-run")
        self.assertEqual(proc.returncode, 0, msg=f"{proc.stdout}\n{proc.stderr}")
        self.assertIn("Resolved model=qwen3-coder-next agent=terminus-2", proc.stdout)
        self.assertIn("-d terminal-bench-sample@2.0", proc.stdout)
        self.assertIn("SMALL_AGENT_HARBOR_MODEL=qwen3-coder-next", proc.stdout)
        self.assertIn("SMALL_AGENT_HARBOR_AGENT=terminus-2", proc.stdout)

    def test_run_small_accepts_overrides(self) -> None:
        proc = self._run_script(
            "run_small.sh",
            "--model",
            "gpt-5.3-codex",
            "--agent",
            "qwen",
            "--dry-run",
        )
        self.assertEqual(proc.returncode, 0, msg=f"{proc.stdout}\n{proc.stderr}")
        self.assertIn("Resolved model=gpt-5.3-codex agent=qwen", proc.stdout)
        self.assertIn("-d terminal-bench-sample@2.0", proc.stdout)
        self.assertIn("SMALL_AGENT_HARBOR_MODEL=gpt-5.3-codex", proc.stdout)
        self.assertIn("SMALL_AGENT_HARBOR_AGENT=qwen", proc.stdout)

    def test_run_full_rejects_invalid_model(self) -> None:
        proc = self._run_script("run_full.sh", "--model", "missing-model", "--dry-run")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Unknown model key 'missing-model'", proc.stderr)


if __name__ == "__main__":
    unittest.main()
