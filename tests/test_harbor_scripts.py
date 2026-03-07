from __future__ import annotations

import os
import re
import subprocess
import unittest
from pathlib import Path

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

    # ── run_smoke.sh ──────────────────────────────────────────────

    def test_run_smoke_uses_config_defaults(self) -> None:
        proc = self._run_script("run_smoke.sh", "--dry-run")
        self.assertEqual(proc.returncode, 0, msg=f"{proc.stdout}\n{proc.stderr}")
        self.assertRegex(
            proc.stdout,
            re.compile(
                r"Resolved model=qwen3-coder-next agent=terminus-2 n_concurrent=\d+"
            ),
        )
        self.assertIn("-d terminal-bench@2.0", proc.stdout)
        self.assertIn("--n-concurrent 1", proc.stdout)
        self.assertIn("--task-name fix-git", proc.stdout)

    def test_run_smoke_accepts_overrides(self) -> None:
        proc = self._run_script(
            "run_smoke.sh",
            "--model",
            "minimax-m2.5",
            "--agent",
            "qwen",
            "--dry-run",
        )
        self.assertEqual(proc.returncode, 0, msg=f"{proc.stdout}\n{proc.stderr}")
        self.assertIn("SMALL_AGENT_HARBOR_MODEL=minimax-m2.5", proc.stdout)
        self.assertIn("SMALL_AGENT_HARBOR_AGENT=qwen", proc.stdout)

    def test_run_smoke_rejects_extra_args(self) -> None:
        proc = self._run_script("run_smoke.sh", "--dry-run", "--", "--env", "modal")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Unknown argument", proc.stderr)

    def test_run_smoke_passthroughs_shell_api_key_env(self) -> None:
        proc = self._run_script(
            "run_smoke.sh",
            "--dry-run",
            env={"OPENROUTER_API_KEY": "test-key"},
        )
        self.assertEqual(proc.returncode, 0, msg=f"{proc.stdout}\n{proc.stderr}")
        self.assertIn("--agent-env OPENROUTER_API_KEY=test-key", proc.stdout)

    # ── run_debug.sh ──────────────────────────────────────────────

    def test_run_debug_split_1(self) -> None:
        proc = self._run_script("run_debug.sh", "--split", "1", "--dry-run")
        self.assertEqual(proc.returncode, 0, msg=f"{proc.stdout}\n{proc.stderr}")
        self.assertIn("-d terminal-bench@2.0", proc.stdout)
        self.assertIn("--n-concurrent 5", proc.stdout)
        expected = [
            "adaptive-rejection-sampler",
            "break-filter-js-from-html",
            "constraints-scheduling",
            "crack-7z-hash",
            "regex-log",
        ]
        for task in expected:
            self.assertIn(
                f"--task-name {task}",
                proc.stdout,
                msg=f"Missing --task-name filter for {task}",
            )

    def test_run_debug_split_2(self) -> None:
        proc = self._run_script("run_debug.sh", "--split", "2", "--dry-run")
        self.assertEqual(proc.returncode, 0, msg=f"{proc.stdout}\n{proc.stderr}")
        expected = [
            "db-wal-recovery",
            "extract-elf",
            "git-leak-recovery",
            "hf-model-inference",
            "kv-store-grpc",
        ]
        for task in expected:
            self.assertIn(f"--task-name {task}", proc.stdout)

    def test_run_debug_split_3(self) -> None:
        proc = self._run_script("run_debug.sh", "--split", "3", "--dry-run")
        self.assertEqual(proc.returncode, 0, msg=f"{proc.stdout}\n{proc.stderr}")
        expected = [
            "largest-eigenval",
            "modernize-scientific-stack",
            "nginx-request-logging",
            "openssl-selfsigned-cert",
            "pypi-server",
        ]
        for task in expected:
            self.assertIn(f"--task-name {task}", proc.stdout)

    def test_run_debug_split_4(self) -> None:
        proc = self._run_script("run_debug.sh", "--split", "4", "--dry-run")
        self.assertEqual(proc.returncode, 0, msg=f"{proc.stdout}\n{proc.stderr}")
        expected = [
            "build-cython-ext",
            "caffe-cifar-10",
            "chess-best-move",
            "code-from-image",
            "compile-compcert",
        ]
        for task in expected:
            self.assertIn(f"--task-name {task}", proc.stdout)

    def test_run_debug_missing_split_fails(self) -> None:
        proc = self._run_script("run_debug.sh", "--dry-run")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Missing required --split", proc.stderr)

    def test_run_debug_invalid_split_fails(self) -> None:
        proc = self._run_script("run_debug.sh", "--split", "5", "--dry-run")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Invalid split", proc.stderr)

    def test_run_debug_accepts_overrides(self) -> None:
        proc = self._run_script(
            "run_debug.sh",
            "--split",
            "1",
            "--model",
            "minimax-m2.5",
            "--agent",
            "qwen",
            "--dry-run",
        )
        self.assertEqual(proc.returncode, 0, msg=f"{proc.stdout}\n{proc.stderr}")
        self.assertIn("SMALL_AGENT_HARBOR_MODEL=minimax-m2.5", proc.stdout)
        self.assertIn("SMALL_AGENT_HARBOR_AGENT=qwen", proc.stdout)

    # ── run_small_benchmark.sh ────────────────────────────────────

    def test_run_small_benchmark_uses_config_defaults(self) -> None:
        proc = self._run_script("run_small_benchmark.sh", "--dry-run")
        self.assertEqual(proc.returncode, 0, msg=f"{proc.stdout}\n{proc.stderr}")
        self.assertRegex(
            proc.stdout,
            re.compile(
                r"Resolved model=qwen3-coder-next agent=terminus-2 n_concurrent=\d+"
            ),
        )
        self.assertIn("-d terminal-bench@2.0", proc.stdout)
        expected_tasks = [
            "cobol-modernization",
            "fix-git",
            "overfull-hbox",
            "prove-plus-comm",
            "custom-memory-heap-crash",
            "distribution-search",
            "dna-insert",
            "filter-js-from-html",
            "financial-document-processor",
            "gcode-to-text",
            "git-multibranch",
            "headless-terminal",
            "large-scale-text-editing",
            "log-summary-date-ranges",
            "cancel-async-tasks",
        ]
        for task in expected_tasks:
            self.assertIn(
                f"--task-name {task}",
                proc.stdout,
                msg=f"Missing --task-name filter for {task}",
            )
        self.assertEqual(len(expected_tasks), 15)

    def test_run_small_benchmark_accepts_overrides(self) -> None:
        proc = self._run_script(
            "run_small_benchmark.sh",
            "--model",
            "minimax-m2.5",
            "--agent",
            "qwen",
            "--dry-run",
        )
        self.assertEqual(proc.returncode, 0, msg=f"{proc.stdout}\n{proc.stderr}")
        self.assertIn("SMALL_AGENT_HARBOR_MODEL=minimax-m2.5", proc.stdout)
        self.assertIn("SMALL_AGENT_HARBOR_AGENT=qwen", proc.stdout)

    def test_run_small_benchmark_disjoint_from_debug(self) -> None:
        """All run_small_benchmark tasks must be disjoint from all debug splits."""
        small_proc = self._run_script("run_small_benchmark.sh", "--dry-run")
        self.assertEqual(
            small_proc.returncode, 0, msg=f"{small_proc.stdout}\n{small_proc.stderr}"
        )
        debug_tasks: set[str] = set()
        for split in ("1", "2", "3", "4"):
            proc = self._run_script("run_debug.sh", "--split", split, "--dry-run")
            self.assertEqual(proc.returncode, 0, msg=f"{proc.stdout}\n{proc.stderr}")
            for m in re.finditer(r"--task-name (\S+)", proc.stdout):
                debug_tasks.add(m.group(1))

        small_tasks: set[str] = set()
        for m in re.finditer(r"--task-name (\S+)", small_proc.stdout):
            small_tasks.add(m.group(1))

        overlap = small_tasks & debug_tasks
        self.assertEqual(
            overlap,
            set(),
            msg=f"run_small_benchmark tasks overlap with run_debug: {overlap}",
        )

    # ── run_full_benchmark.sh ─────────────────────────────────────

    def test_run_full_benchmark_uses_config_defaults(self) -> None:
        proc = self._run_script("run_full_benchmark.sh", "--dry-run")
        self.assertEqual(proc.returncode, 0, msg=f"{proc.stdout}\n{proc.stderr}")
        self.assertRegex(
            proc.stdout,
            re.compile(
                r"Resolved model=qwen3-coder-next agent=terminus-2 n_concurrent=\d+"
            ),
        )
        self.assertIn("-d terminal-bench@2.0", proc.stdout)
        self.assertRegex(proc.stdout, re.compile(r"--n-concurrent \d+"))
        self.assertIn("--env docker", proc.stdout)
        self.assertIn("--delete", proc.stdout)
        self.assertIn("--no-force-build", proc.stdout)

    def test_run_full_benchmark_rejects_invalid_model(self) -> None:
        proc = self._run_script(
            "run_full_benchmark.sh", "--model", "missing-model", "--dry-run"
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Unknown model key 'missing-model'", proc.stderr)


if __name__ == "__main__":
    unittest.main()
