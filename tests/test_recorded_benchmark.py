from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT / "agent_evolve" / "start_workdir" / "run_recorded_benchmark.py"
)

spec = importlib.util.spec_from_file_location("run_recorded_benchmark", MODULE_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["run_recorded_benchmark"] = mod
spec.loader.exec_module(mod)


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs(unittest.TestCase):
    def test_defaults(self) -> None:
        args = mod.parse_args(["--iteration", "1"])
        self.assertEqual(args.iteration, 1)
        self.assertEqual(args.agent_key, "terminus-2")
        self.assertIsNone(args.model_key)
        self.assertEqual(args.run_label, "run")

    def test_run_label_propagates(self) -> None:
        args = mod.parse_args(["--iteration", "2", "--run-label", "eval"])
        self.assertEqual(args.run_label, "eval")
        self.assertEqual(args.iteration, 2)


# ---------------------------------------------------------------------------
# _next_run_dir
# ---------------------------------------------------------------------------


class TestNextRunDir(unittest.TestCase):
    def test_returns_first_when_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = mod._next_run_dir(
                iter_dir=Path(tmp), label="run", create_dir=False
            )
            self.assertEqual(result.name, "run-0001")

    def test_returns_second_when_first_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "run-0001").mkdir()
            result = mod._next_run_dir(
                iter_dir=Path(tmp), label="run", create_dir=False
            )
            self.assertEqual(result.name, "run-0002")

    def test_respects_custom_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = mod._next_run_dir(
                iter_dir=Path(tmp), label="eval", create_dir=False
            )
            self.assertEqual(result.name, "eval-0001")

    def test_create_dir_makes_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = mod._next_run_dir(iter_dir=Path(tmp), label="run", create_dir=True)
            self.assertTrue(result.is_dir())


# ---------------------------------------------------------------------------
# _resolve_new_job_dir
# ---------------------------------------------------------------------------


class TestResolveNewJobDir(unittest.TestCase):
    def test_returns_last_new_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jobs = Path(tmp)
            (jobs / "old-job").mkdir()
            (jobs / "new-job-a").mkdir()
            (jobs / "new-job-b").mkdir()
            result = mod._resolve_new_job_dir(jobs_root=jobs, before={"old-job"})
            self.assertEqual(result.name, "new-job-b")

    def test_raises_when_no_new_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jobs = Path(tmp)
            (jobs / "existing").mkdir()
            with self.assertRaises(RuntimeError):
                mod._resolve_new_job_dir(jobs_root=jobs, before={"existing"})


# ---------------------------------------------------------------------------
# _is_pid_alive
# ---------------------------------------------------------------------------


class TestIsPidAlive(unittest.TestCase):
    def test_returns_true_for_own_pid(self) -> None:
        self.assertTrue(mod._is_pid_alive(os.getpid()))

    def test_returns_false_for_dead_pid(self) -> None:
        self.assertFalse(mod._is_pid_alive(2_000_000_000))


# ---------------------------------------------------------------------------
# _collect_job_dirs
# ---------------------------------------------------------------------------


class TestCollectJobDirs(unittest.TestCase):
    def test_returns_empty_for_nonexistent_root(self) -> None:
        result = mod._collect_job_dirs(jobs_root=Path("/nonexistent/path/jobs"))
        self.assertEqual(result, set())

    def test_returns_directory_names_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jobs = Path(tmp)
            (jobs / "dir-a").mkdir()
            (jobs / "dir-b").mkdir()
            (jobs / "file.txt").write_text("x", encoding="utf-8")
            result = mod._collect_job_dirs(jobs_root=jobs)
            self.assertEqual(result, {"dir-a", "dir-b"})


# ---------------------------------------------------------------------------
# _iter_root
# ---------------------------------------------------------------------------


class TestIterRoot(unittest.TestCase):
    def test_creates_dirs_and_returns_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "run-root" / "agent_evolve"
            workdir.mkdir(parents=True)
            snap, evals = mod._iter_root(workdir_root=workdir, iteration=3)
            self.assertTrue(snap.is_dir())
            self.assertTrue(evals.is_dir())
            self.assertEqual(snap.name, "iter-0003")
            self.assertEqual(evals.name, "iter-0003")
            self.assertEqual(snap.parent.name, "snapshots")
            self.assertEqual(evals.parent.name, "eval")


# ---------------------------------------------------------------------------
# _copy_code_snapshot
# ---------------------------------------------------------------------------


class TestCopyCodeSnapshot(unittest.TestCase):
    def test_copies_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "workdir"
            src.mkdir()
            (src / "agent.py").write_text("code", encoding="utf-8")
            dst = Path(tmp) / "snapshot"
            mod._copy_code_snapshot(workdir_root=src, target_dir=dst)
            self.assertEqual((dst / "agent.py").read_text(encoding="utf-8"), "code")

    def test_ignores_snapshot_ignores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "workdir"
            src.mkdir()
            (src / "agent.py").write_text("code", encoding="utf-8")
            (src / "__pycache__").mkdir()
            (src / "__pycache__" / "mod.pyc").write_bytes(b"\x00")
            (src / "debug.log").write_text("log", encoding="utf-8")
            dst = Path(tmp) / "snapshot"
            mod._copy_code_snapshot(workdir_root=src, target_dir=dst)
            self.assertTrue((dst / "agent.py").exists())
            self.assertFalse((dst / "__pycache__").exists())
            self.assertFalse((dst / "debug.log").exists())


# ---------------------------------------------------------------------------
# _benchmark_lock
# ---------------------------------------------------------------------------


class TestBenchmarkLock(unittest.TestCase):
    def test_creates_and_removes_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jobs = Path(tmp) / "jobs"
            lock_path = jobs / ".agent_evolve_benchmark.lock"
            with mod._benchmark_lock(jobs_root=jobs):
                self.assertTrue(lock_path.exists())
                stored = lock_path.read_text(encoding="utf-8").strip()
                self.assertEqual(stored, str(os.getpid()))

            self.assertFalse(lock_path.exists())

    def test_raises_when_held_by_live_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jobs = Path(tmp) / "jobs"
            jobs.mkdir(parents=True)
            lock_path = jobs / ".agent_evolve_benchmark.lock"
            lock_path.write_text("99999", encoding="utf-8")
            with patch.object(mod, "_is_pid_alive", return_value=True):
                with self.assertRaises(RuntimeError):
                    with mod._benchmark_lock(jobs_root=jobs):
                        pass

    def test_cleans_stale_lock_and_proceeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jobs = Path(tmp) / "jobs"
            jobs.mkdir(parents=True)
            lock_path = jobs / ".agent_evolve_benchmark.lock"
            lock_path.write_text("2000000000", encoding="utf-8")
            with mod._benchmark_lock(jobs_root=jobs):
                stored = lock_path.read_text(encoding="utf-8").strip()
                self.assertEqual(stored, str(os.getpid()))

            self.assertFalse(lock_path.exists())


# ---------------------------------------------------------------------------
# _load_run_summary
# ---------------------------------------------------------------------------


class TestLoadRunSummary(unittest.TestCase):
    def test_returns_summary_with_reward_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "job"
            trial = job_dir / "trial-1"
            trial.mkdir(parents=True)
            data = {
                "reward_stats": {"mean": 0.7},
                "n_trials": 10,
                "n_evals": 10,
            }
            (trial / "result.json").write_text(json.dumps(data), encoding="utf-8")
            result = mod._load_run_summary(harbor_job_dir=job_dir)
            self.assertIn("reward_stats", result)

    def test_returns_empty_when_no_reward_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "job"
            trial = job_dir / "trial-1"
            trial.mkdir(parents=True)
            (trial / "result.json").write_text(
                json.dumps({"other": "data"}), encoding="utf-8"
            )
            result = mod._load_run_summary(harbor_job_dir=job_dir)
            self.assertEqual(result, {})

    def test_returns_empty_for_empty_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "job"
            job_dir.mkdir()
            result = mod._load_run_summary(harbor_job_dir=job_dir)
            self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
