from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
import re
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestHarborRunnerScripts(unittest.TestCase):
    def _run_script(
        self,
        script_name: str,
        *args: str,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        return subprocess.run(
            [str(PROJECT_ROOT / "harbor" / script_name), *args],
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            check=False,
            env=merged_env,
        )

    def test_run_small_uses_config_defaults(self) -> None:
        proc = self._run_script("run_small.sh", "--dry-run")
        self.assertEqual(proc.returncode, 0, msg=f"{proc.stdout}\n{proc.stderr}")
        self.assertRegex(
            proc.stdout,
            re.compile(
                r"Resolved model=qwen3-coder-next agent=terminus-2 n_concurrent=\d+"
            ),
        )
        self.assertIn("-d terminal-bench-sample@2.0", proc.stdout)
        self.assertRegex(proc.stdout, re.compile(r"--n-concurrent \d+"))
        self.assertIn("--env docker", proc.stdout)
        self.assertIn("--delete", proc.stdout)
        self.assertIn("--no-force-build", proc.stdout)
        self.assertIn("SMALL_AGENT_HARBOR_MODEL=qwen3-coder-next", proc.stdout)
        self.assertIn("SMALL_AGENT_HARBOR_AGENT=terminus-2", proc.stdout)
        self.assertRegex(
            proc.stdout,
            re.compile(r"--jobs-dir [^ ]*/cli/harbor/jobs/[^ ]+"),
        )

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
        self.assertRegex(
            proc.stdout,
            re.compile(r"Resolved model=gpt-5.3-codex agent=qwen n_concurrent=\d+"),
        )
        self.assertIn("-d terminal-bench-sample@2.0", proc.stdout)
        self.assertRegex(proc.stdout, re.compile(r"--n-concurrent \d+"))
        self.assertIn("SMALL_AGENT_HARBOR_MODEL=gpt-5.3-codex", proc.stdout)
        self.assertIn("SMALL_AGENT_HARBOR_AGENT=qwen", proc.stdout)

    def test_run_full_rejects_invalid_model(self) -> None:
        proc = self._run_script("run_full.sh", "--model", "missing-model", "--dry-run")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Unknown model key 'missing-model'", proc.stderr)

    def test_run_full_uses_full_concurrency_default(self) -> None:
        proc = self._run_script("run_full.sh", "--dry-run")
        self.assertEqual(proc.returncode, 0, msg=f"{proc.stdout}\n{proc.stderr}")
        self.assertRegex(
            proc.stdout,
            re.compile(
                r"Resolved model=qwen3-coder-next agent=terminus-2 n_concurrent=\d+"
            ),
        )
        self.assertIn("-d terminal-bench@2.0", proc.stdout)
        self.assertRegex(proc.stdout, re.compile(r"--n-concurrent \d+"))

    def test_run_small_rejects_extra_args(self) -> None:
        proc = self._run_script("run_small.sh", "--dry-run", "--", "--env", "modal")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Unknown argument", proc.stderr)

    def test_run_small_passthroughs_shell_api_key_env(self) -> None:
        proc = self._run_script(
            "run_small.sh",
            "--dry-run",
            env={"OPENROUTER_API_KEY": "test-key"},
        )
        self.assertEqual(proc.returncode, 0, msg=f"{proc.stdout}\n{proc.stderr}")
        self.assertIn("--agent-env OPENROUTER_API_KEY=test-key", proc.stdout)


if __name__ == "__main__":
    unittest.main()
