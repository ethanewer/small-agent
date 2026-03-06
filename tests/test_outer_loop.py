from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "agent_evolve" / "run_outer_loop.py"

spec = importlib.util.spec_from_file_location("run_outer_loop", MODULE_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["run_outer_loop"] = mod
spec.loader.exec_module(mod)


# ---------------------------------------------------------------------------
# _should_skip_step
# ---------------------------------------------------------------------------


class TestShouldSkipStep(unittest.TestCase):
    def test_returns_false_when_last_completed_is_none(self) -> None:
        self.assertFalse(mod._should_skip_step("dev_benchmark", None))

    def test_returns_true_when_step_equals_last_completed(self) -> None:
        self.assertTrue(mod._should_skip_step("cursor", "cursor"))

    def test_returns_true_when_step_precedes_last_completed(self) -> None:
        self.assertTrue(mod._should_skip_step("dev_benchmark", "validation"))

    def test_returns_false_when_step_follows_last_completed(self) -> None:
        self.assertFalse(mod._should_skip_step("eval", "cursor"))

    def test_returns_false_for_unknown_step(self) -> None:
        self.assertFalse(mod._should_skip_step("unknown_step", "cursor"))

    def test_returns_false_for_unknown_last_completed(self) -> None:
        self.assertFalse(mod._should_skip_step("cursor", "unknown_step"))


# ---------------------------------------------------------------------------
# _is_transient_cursor_error
# ---------------------------------------------------------------------------


class TestIsTransientCursorError(unittest.TestCase):
    def _completed(
        self, *, returncode: int, stdout: str = "", stderr: str = ""
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def test_returns_false_when_returncode_zero(self) -> None:
        cp = self._completed(returncode=0, stderr="http/2 stream closed")
        self.assertFalse(mod._is_transient_cursor_error(completed=cp))

    def test_detects_http2_stream_closed(self) -> None:
        cp = self._completed(
            returncode=1, stderr="Error: HTTP/2 stream closed unexpectedly"
        )
        self.assertTrue(mod._is_transient_cursor_error(completed=cp))

    def test_detects_error_code_cancel(self) -> None:
        cp = self._completed(returncode=1, stdout="error code cancel received")
        self.assertTrue(mod._is_transient_cursor_error(completed=cp))

    def test_detects_connection_reset(self) -> None:
        cp = self._completed(returncode=1, stderr="Connection reset by peer")
        self.assertTrue(mod._is_transient_cursor_error(completed=cp))

    def test_detects_stream_closed_with_error_code_cancel(self) -> None:
        cp = self._completed(
            returncode=1,
            stderr="stream closed with error code cancel",
        )
        self.assertTrue(mod._is_transient_cursor_error(completed=cp))

    def test_returns_false_for_non_transient_error(self) -> None:
        cp = self._completed(returncode=1, stderr="SyntaxError: invalid syntax")
        self.assertFalse(mod._is_transient_cursor_error(completed=cp))


# ---------------------------------------------------------------------------
# _build_state
# ---------------------------------------------------------------------------


class TestBuildState(unittest.TestCase):
    def _make_args(self) -> object:
        return mod.parse_args(
            [
                "--iterations",
                "5",
                "--agent-key",
                "test-agent",
            ]
        )

    def test_produces_all_expected_keys(self) -> None:
        args = self._make_args()
        state = mod._build_state(
            stop_state=mod.StopState(),
            args=args,
            run_root=Path("/tmp/run"),
            current_iteration=2,
            last_completed_step="cursor",
            eval_score=0.75,
            last_eval_agent_hash="abc123",
        )
        self.assertEqual(state["current_iteration"], 2)
        self.assertEqual(state["last_completed_step"], "cursor")
        self.assertEqual(state["eval_score"], 0.75)
        self.assertEqual(state["last_eval_agent_hash"], "abc123")
        self.assertFalse(state["dev_benchmark_failed"])
        self.assertFalse(state["eval_benchmark_failed"])
        self.assertIn("updated_at_utc", state)

    def test_includes_extra_keys(self) -> None:
        args = self._make_args()
        state = mod._build_state(
            stop_state=mod.StopState(),
            args=args,
            run_root=Path("/tmp/run"),
            current_iteration=1,
            last_completed_step=None,
            extra={"stopped_gracefully": True},
        )
        self.assertTrue(state["stopped_gracefully"])

    def test_handles_none_optional_fields(self) -> None:
        args = self._make_args()
        state = mod._build_state(
            stop_state=mod.StopState(),
            args=args,
            run_root=Path("/tmp/run"),
            current_iteration=1,
            last_completed_step=None,
        )
        self.assertIsNone(state["eval_score"])
        self.assertIsNone(state["last_eval_agent_hash"])
        self.assertIsNone(state["last_completed_step"])


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs(unittest.TestCase):
    def test_defaults(self) -> None:
        args = mod.parse_args([])
        self.assertEqual(args.iterations, 25)
        self.assertEqual(args.start_iteration, 1)
        self.assertEqual(args.agent_key, "terminus-2")
        self.assertIsNone(args.model_key)
        self.assertIsNone(args.cursor_model)
        self.assertIsNone(args.resume)
        self.assertFalse(args.skip_initial_benchmark)

    def test_resume_sets_path(self) -> None:
        args = mod.parse_args(["--resume", "/tmp/run-dir"])
        self.assertEqual(args.resume, Path("/tmp/run-dir"))

    def test_skip_initial_benchmark_flag(self) -> None:
        args = mod.parse_args(["--skip-initial-benchmark"])
        self.assertTrue(args.skip_initial_benchmark)

    def test_explicit_values_propagate(self) -> None:
        args = mod.parse_args(
            [
                "--iterations",
                "10",
                "--model-key",
                "gpt-5",
                "--cursor-model",
                "claude-4",
                "--agent-key",
                "qwen",
            ]
        )
        self.assertEqual(args.iterations, 10)
        self.assertEqual(args.model_key, "gpt-5")
        self.assertEqual(args.cursor_model, "claude-4")
        self.assertEqual(args.agent_key, "qwen")


# ---------------------------------------------------------------------------
# _create_run_root
# ---------------------------------------------------------------------------


class TestCreateRunRoot(unittest.TestCase):
    def test_creates_timestamped_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = mod._create_run_root(outputs_root=Path(tmp))
            self.assertTrue(root.exists())
            self.assertTrue(root.name.startswith("run-"))

    def test_handles_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = mod._create_run_root(outputs_root=Path(tmp))
            (Path(tmp) / first.name).mkdir(exist_ok=True)
            second = mod._create_run_root(outputs_root=Path(tmp))
            self.assertNotEqual(first, second)
            self.assertTrue(second.exists())

    def test_creates_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            deep = Path(tmp) / "a" / "b" / "c"
            root = mod._create_run_root(outputs_root=deep)
            self.assertTrue(root.exists())


# ---------------------------------------------------------------------------
# _seed_run_workdir
# ---------------------------------------------------------------------------


class TestSeedRunWorkdir(unittest.TestCase):
    def test_copies_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "template"
            src.mkdir()
            (src / "agent.py").write_text("code", encoding="utf-8")
            dst = Path(tmp) / "workdir"
            mod._seed_run_workdir(template_root=src, run_workdir=dst)
            self.assertTrue((dst / "agent.py").exists())
            self.assertEqual((dst / "agent.py").read_text(encoding="utf-8"), "code")

    def test_ignores_pycache_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "template"
            src.mkdir()
            (src / "agent.py").write_text("code", encoding="utf-8")
            (src / "__pycache__").mkdir()
            (src / "__pycache__" / "mod.pyc").write_bytes(b"\x00")
            (src / "run.log").write_text("log", encoding="utf-8")
            dst = Path(tmp) / "workdir"
            mod._seed_run_workdir(template_root=src, run_workdir=dst)
            self.assertTrue((dst / "agent.py").exists())
            self.assertFalse((dst / "__pycache__").exists())
            self.assertFalse((dst / "run.log").exists())


# ---------------------------------------------------------------------------
# _load_state / _save_state
# ---------------------------------------------------------------------------


class TestLoadSaveState(unittest.TestCase):
    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            payload = {"iteration": 3, "score": 0.5}
            mod._save_state(state_path=path, payload=payload)
            loaded = mod._load_state(state_path=path)
            self.assertEqual(loaded["iteration"], 3)
            self.assertEqual(loaded["score"], 0.5)

    def test_missing_file_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.json"
            self.assertEqual(mod._load_state(state_path=path), {})

    def test_non_dict_json_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text("[1, 2, 3]", encoding="utf-8")
            self.assertEqual(mod._load_state(state_path=path), {})


# ---------------------------------------------------------------------------
# _latest_eval_for_iteration
# ---------------------------------------------------------------------------


class TestLatestEvalForIteration(unittest.TestCase):
    def test_returns_none_when_iter_dir_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = mod._latest_eval_for_iteration(
                workdir_root=Path(tmp),
                iteration=1,
            )
            self.assertIsNone(result)

    def test_returns_none_when_no_run_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "eval" / "iter-0001"
            iter_dir.mkdir(parents=True)
            result = mod._latest_eval_for_iteration(
                workdir_root=Path(tmp),
                iteration=1,
            )
            self.assertIsNone(result)

    def test_returns_last_sorted_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "eval" / "iter-0002"
            (iter_dir / "run-0001").mkdir(parents=True)
            (iter_dir / "run-0002").mkdir(parents=True)
            result = mod._latest_eval_for_iteration(
                workdir_root=Path(tmp),
                iteration=2,
            )
            assert result is not None
            self.assertEqual(result.parent.name, "run-0002")
            self.assertEqual(result.name, "eval_summary.json")

    def test_respects_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = Path(tmp) / "eval" / "iter-0001"
            (iter_dir / "run-0001").mkdir(parents=True)
            (iter_dir / "eval-0001").mkdir(parents=True)
            result = mod._latest_eval_for_iteration(
                workdir_root=Path(tmp),
                iteration=1,
                label="eval",
            )
            assert result is not None
            self.assertEqual(result.parent.name, "eval-0001")


# ---------------------------------------------------------------------------
# _resolve_context_length
# ---------------------------------------------------------------------------


class TestResolveContextLength(unittest.TestCase):
    def _write_config(self, tmp: str, data: dict[str, object]) -> Path:
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")
        return path

    def test_returns_context_length_for_model_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_config(
                tmp,
                {
                    "models": {"m1": {"context_length": 32768}},
                },
            )
            self.assertEqual(
                mod._resolve_context_length(config_path=path, model_key="m1"),
                32768,
            )

    def test_returns_zero_when_model_key_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_config(
                tmp,
                {
                    "models": {"m1": {"context_length": 32768}},
                },
            )
            self.assertEqual(
                mod._resolve_context_length(config_path=path, model_key="missing"),
                0,
            )

    def test_returns_zero_when_config_missing(self) -> None:
        self.assertEqual(
            mod._resolve_context_length(
                config_path=Path("/nonexistent/config.json"),
                model_key="m1",
            ),
            0,
        )

    def test_uses_default_model_when_key_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_config(
                tmp,
                {
                    "default_model": "m2",
                    "models": {
                        "m1": {"context_length": 1000},
                        "m2": {"context_length": 65536},
                    },
                },
            )
            self.assertEqual(
                mod._resolve_context_length(config_path=path, model_key=None),
                65536,
            )


# ---------------------------------------------------------------------------
# _file_hash
# ---------------------------------------------------------------------------


class TestFileHash(unittest.TestCase):
    def test_returns_correct_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "file.txt"
            content = b"hello world"
            path.write_bytes(content)
            expected = hashlib.sha256(content).hexdigest()
            self.assertEqual(mod._file_hash(path), expected)


# ---------------------------------------------------------------------------
# _read_eval_summary_fields
# ---------------------------------------------------------------------------


class TestReadEvalSummaryFields(unittest.TestCase):
    def test_returns_formatted_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(
                json.dumps({"reward_mean": 0.6, "n_trials": 10}),
                encoding="utf-8",
            )
            score, trials = mod._read_eval_summary_fields(path)
            self.assertEqual(score, "0.60")
            self.assertEqual(trials, "10")

    def test_returns_na_for_missing_file(self) -> None:
        score, trials = mod._read_eval_summary_fields(Path("/nonexistent/summary.json"))
        self.assertEqual(score, "N/A")
        self.assertEqual(trials, "N/A")

    def test_returns_na_for_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text("{bad json", encoding="utf-8")
            score, trials = mod._read_eval_summary_fields(path)
            self.assertEqual(score, "N/A")
            self.assertEqual(trials, "N/A")


# ---------------------------------------------------------------------------
# _render_prompt
# ---------------------------------------------------------------------------


class TestRenderPrompt(unittest.TestCase):
    def test_all_placeholders_substituted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            template = Path(tmp) / "template.md"
            template.write_text(
                "Score: {dev_score} Trials: {dev_trials} "
                "Iter: {iteration} Padded: {iteration_padded} "
                "Root: {run_root} Workdir: {workdir_root} "
                "Eval: {eval_root} Snap: {snapshot_root} "
                "Summary: {eval_summary_path} "
                "Artifacts: {eval_artifacts_path} "
                "Ctx: {context_length}",
                encoding="utf-8",
            )
            summary = Path(tmp) / "eval" / "summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                json.dumps({"reward_mean": 0.5, "n_trials": 8}),
                encoding="utf-8",
            )
            result = mod._render_prompt(
                template_path=template,
                iteration=3,
                run_root=Path("/run"),
                workdir_root=Path("/run/agent"),
                eval_root=Path("/run/eval"),
                snapshot_root=Path("/run/snap"),
                eval_summary_path=summary,
                context_length=65536,
            )
            self.assertNotIn("{", result)
            self.assertIn("0.50", result)
            self.assertIn("8", result)
            self.assertIn("65536", result)

    def test_reads_score_from_eval_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            template = Path(tmp) / "t.md"
            template.write_text(
                "{dev_score}|{dev_trials}|{iteration}|{iteration_padded}"
                "|{run_root}|{workdir_root}|{eval_root}|{snapshot_root}"
                "|{eval_summary_path}|{eval_artifacts_path}|{context_length}",
                encoding="utf-8",
            )
            summary = Path(tmp) / "s.json"
            summary.write_text(
                json.dumps({"reward_mean": 0.9, "n_trials": 20}),
                encoding="utf-8",
            )
            result = mod._render_prompt(
                template_path=template,
                iteration=1,
                run_root=Path("/r"),
                workdir_root=Path("/w"),
                eval_root=Path("/e"),
                snapshot_root=Path("/s"),
                eval_summary_path=summary,
                context_length=0,
            )
            self.assertTrue(result.startswith("0.90|20|"))


# ---------------------------------------------------------------------------
# _record_step_output
# ---------------------------------------------------------------------------


class TestRecordStepOutput(unittest.TestCase):
    def test_writes_correct_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.json"
            cp = subprocess.CompletedProcess(
                args=[], returncode=42, stdout="out", stderr="err"
            )
            mod._record_step_output(target_path=path, completed=cp)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["returncode"], 42)
            self.assertEqual(data["stdout"], "out")
            self.assertEqual(data["stderr"], "err")


if __name__ == "__main__":
    unittest.main()
