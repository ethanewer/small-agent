"""Tests for scoreboard generation, prompt rendering, and NOTES.md seeding.

Runs against the existing completed run at
agent_evolve/outputs/run-20260307T233728Z.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from agent_evolve.run_outer_loop import (
    _build_scoreboard_text,
    _build_snapshot_index,
    _extract_architecture_label,
    _extract_notes_summary,
    _render_prompt,
    collect_iteration_records,
    seed_notes_with_history,
    update_scoreboard,
)

REAL_RUN_ROOT = Path(__file__).resolve().parent / "outputs" / "run-20260307T233728Z"

BASELINE_EVAL_SCORE = 0.40

EXPECTED_EVAL_SCORES: dict[int, float] = {
    1: 0.133,
    2: 0.333,
    3: 0.400,
    4: 0.467,
    5: 0.267,
    6: 0.333,
    7: 0.267,
}


def _has_real_run() -> bool:
    return REAL_RUN_ROOT.exists() and (REAL_RUN_ROOT / "eval").exists()


@unittest.skipUnless(_has_real_run(), "real run data not available")
class TestCollectIterationRecords(unittest.TestCase):
    def test_returns_seven_records(self) -> None:
        records = collect_iteration_records(run_root=REAL_RUN_ROOT, up_to_iteration=7)
        self.assertEqual(len(records), 7)

    def test_eval_scores_match(self) -> None:
        records = collect_iteration_records(run_root=REAL_RUN_ROOT, up_to_iteration=7)
        for rec in records:
            expected = EXPECTED_EVAL_SCORES.get(rec.iteration)
            if expected is not None and rec.eval_score is not None:
                self.assertAlmostEqual(
                    rec.eval_score,
                    expected,
                    places=2,
                    msg=f"iter {rec.iteration}",
                )

    def test_line_counts_reasonable(self) -> None:
        records = collect_iteration_records(run_root=REAL_RUN_ROOT, up_to_iteration=7)
        for rec in records:
            if rec.line_count is not None:
                self.assertGreater(
                    rec.line_count,
                    500,
                    msg=f"iter {rec.iteration} too few lines",
                )
                self.assertLess(
                    rec.line_count,
                    5000,
                    msg=f"iter {rec.iteration} too many lines",
                )

    def test_snapshot_paths_exist(self) -> None:
        records = collect_iteration_records(run_root=REAL_RUN_ROOT, up_to_iteration=7)
        for rec in records:
            if rec.snapshot_agent_path is not None:
                self.assertTrue(
                    Path(rec.snapshot_agent_path).exists(),
                    msg=f"iter {rec.iteration} snapshot missing: {rec.snapshot_agent_path}",
                )


@unittest.skipUnless(_has_real_run(), "real run data not available")
class TestScoreboardGeneration(unittest.TestCase):
    tmpdir: str = ""
    tmp_root: Path = Path()

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.tmp_root = Path(self.tmpdir)
        (self.tmp_root / "eval").symlink_to(REAL_RUN_ROOT / "eval")
        (self.tmp_root / "snapshots").symlink_to(REAL_RUN_ROOT / "snapshots")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)

    def test_scoreboard_files_created(self) -> None:
        update_scoreboard(
            run_root=self.tmp_root,
            up_to_iteration=7,
            baseline_eval_score=BASELINE_EVAL_SCORE,
        )
        self.assertTrue((self.tmp_root / "SCOREBOARD.md").exists())
        self.assertTrue((self.tmp_root / "scoreboard.json").exists())

    def test_scoreboard_md_has_correct_rows(self) -> None:
        update_scoreboard(
            run_root=self.tmp_root,
            up_to_iteration=7,
            baseline_eval_score=BASELINE_EVAL_SCORE,
        )
        md = (self.tmp_root / "SCOREBOARD.md").read_text(encoding="utf-8")
        table_rows = [
            line
            for line in md.splitlines()
            if line.startswith("| ") and not line.startswith("|--")
        ]
        header = table_rows[0]
        data_rows = table_rows[1:]
        self.assertIn("Iter", header)
        self.assertEqual(len(data_rows), 8, "baseline + 7 iterations")

    def test_scoreboard_json_valid(self) -> None:
        update_scoreboard(
            run_root=self.tmp_root,
            up_to_iteration=7,
            baseline_eval_score=BASELINE_EVAL_SCORE,
        )
        data = json.loads(
            (self.tmp_root / "scoreboard.json").read_text(encoding="utf-8")
        )
        self.assertIn("iterations", data)
        self.assertEqual(len(data["iterations"]), 8)

    def test_scoreboard_json_scores_match(self) -> None:
        update_scoreboard(
            run_root=self.tmp_root,
            up_to_iteration=7,
            baseline_eval_score=BASELINE_EVAL_SCORE,
        )
        data = json.loads(
            (self.tmp_root / "scoreboard.json").read_text(encoding="utf-8")
        )
        for entry in data["iterations"]:
            it = entry["iteration"]
            if it == 0:
                self.assertAlmostEqual(entry["eval_score"], 0.40, places=2)
            else:
                expected = EXPECTED_EVAL_SCORES.get(it)
                if expected is not None and entry["eval_score"] is not None:
                    self.assertAlmostEqual(
                        entry["eval_score"],
                        expected,
                        places=2,
                        msg=f"iter {it}",
                    )

    def test_best_iteration_is_4(self) -> None:
        update_scoreboard(
            run_root=self.tmp_root,
            up_to_iteration=7,
            baseline_eval_score=BASELINE_EVAL_SCORE,
        )
        md = (self.tmp_root / "SCOREBOARD.md").read_text(encoding="utf-8")
        self.assertIn("Iter 4", md)
        self.assertIn("0.47", md)

    def test_no_eval_artifact_paths(self) -> None:
        update_scoreboard(
            run_root=self.tmp_root,
            up_to_iteration=7,
            baseline_eval_score=BASELINE_EVAL_SCORE,
        )
        md = (self.tmp_root / "SCOREBOARD.md").read_text(encoding="utf-8")
        self.assertNotIn("harbor_job", md)
        self.assertNotIn("eval-0001/harbor_job", md)

    def test_idempotent(self) -> None:
        update_scoreboard(
            run_root=self.tmp_root,
            up_to_iteration=7,
            baseline_eval_score=BASELINE_EVAL_SCORE,
        )
        md1 = (self.tmp_root / "SCOREBOARD.md").read_text(encoding="utf-8")
        json1 = json.loads(
            (self.tmp_root / "scoreboard.json").read_text(encoding="utf-8")
        )

        update_scoreboard(
            run_root=self.tmp_root,
            up_to_iteration=7,
            baseline_eval_score=BASELINE_EVAL_SCORE,
        )
        md2 = (self.tmp_root / "SCOREBOARD.md").read_text(encoding="utf-8")
        json2 = json.loads(
            (self.tmp_root / "scoreboard.json").read_text(encoding="utf-8")
        )

        self.assertEqual(md1, md2)
        self.assertEqual(json1["iterations"], json2["iterations"])

    def test_regression_warning_present(self) -> None:
        update_scoreboard(
            run_root=self.tmp_root,
            up_to_iteration=7,
            baseline_eval_score=BASELINE_EVAL_SCORE,
        )
        md = (self.tmp_root / "SCOREBOARD.md").read_text(encoding="utf-8")
        self.assertIn("WARNING", md)
        self.assertIn("declined", md)


@unittest.skipUnless(_has_real_run(), "real run data not available")
class TestPromptRendering(unittest.TestCase):
    def _render_for_iteration(self, iteration: int) -> str:
        template_path = (
            Path(__file__).resolve().parent / "headless_inner_loop_prompt.md"
        )
        eval_root = REAL_RUN_ROOT / "eval"
        snapshot_root = REAL_RUN_ROOT / "snapshots"
        workdir_root = REAL_RUN_ROOT / "agent_evolve"

        eval_summary = eval_root / "latest_run.json"

        return _render_prompt(
            template_path=template_path,
            iteration=iteration,
            workdir_root=workdir_root,
            eval_root=eval_root,
            snapshot_root=snapshot_root,
            eval_summary_path=eval_summary,
            context_length=262144,
            run_root=REAL_RUN_ROOT,
            baseline_eval_score=BASELINE_EVAL_SCORE,
        )

    def test_scoreboard_populated(self) -> None:
        rendered = self._render_for_iteration(8)
        self.assertIn("Eval score history", rendered)
        self.assertIn("Iter 4", rendered)
        self.assertIn("0.47", rendered)

    def test_snapshot_index_populated(self) -> None:
        rendered = self._render_for_iteration(8)
        self.assertIn("Prior agent snapshots", rendered)
        self.assertIn("agent.py", rendered)

    def test_snapshot_paths_exist_on_disk(self) -> None:
        rendered = self._render_for_iteration(8)
        for line in rendered.splitlines():
            if "snapshots/iter-" in line and "agent.py" in line:
                parts = line.split("`")
                for part in parts:
                    if "snapshots/iter-" in part and "agent.py" in part:
                        path = Path(part.strip())
                        if path.is_absolute():
                            self.assertTrue(
                                path.exists(),
                                msg=f"snapshot path missing: {path}",
                            )

    def test_regression_warning_at_iter_8(self) -> None:
        rendered = self._render_for_iteration(8)
        self.assertIn("WARNING", rendered)

    def test_no_unresolved_placeholders(self) -> None:
        rendered = self._render_for_iteration(8)
        self.assertNotIn("{scoreboard}", rendered)
        self.assertNotIn("{snapshot_index}", rendered)
        self.assertNotIn("{dev_score}", rendered)
        self.assertNotIn("{dev_trials}", rendered)
        self.assertNotIn("{context_length}", rendered)
        self.assertNotIn("{workdir_root}", rendered)

    def test_no_eval_trajectory_paths(self) -> None:
        rendered = self._render_for_iteration(8)
        self.assertNotIn("harbor_job", rendered)

    def test_existing_placeholders_resolved(self) -> None:
        rendered = self._render_for_iteration(8)
        self.assertIn("262144", rendered)
        self.assertIn("agent_evolve", rendered)


@unittest.skipUnless(_has_real_run(), "real run data not available")
class TestNoRegressionWarningAtBest(unittest.TestCase):
    def test_no_warning_at_iter_4(self) -> None:
        """At iteration 4 (the best), no regression warning should appear."""
        scoreboard = _build_scoreboard_text(
            run_root=REAL_RUN_ROOT,
            up_to_iteration=4,
            baseline_eval_score=BASELINE_EVAL_SCORE,
        )
        self.assertNotIn("WARNING", scoreboard)


@unittest.skipUnless(_has_real_run(), "real run data not available")
class TestNotesSeeding(unittest.TestCase):
    tmpdir: str = ""
    workdir: Path = Path()
    tmp_root: Path = Path()

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.workdir = Path(self.tmpdir) / "agent_evolve"
        self.workdir.mkdir()
        self.tmp_root = Path(self.tmpdir)
        (self.tmp_root / "eval").symlink_to(REAL_RUN_ROOT / "eval")
        (self.tmp_root / "snapshots").symlink_to(REAL_RUN_ROOT / "snapshots")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)

    def test_history_prepended(self) -> None:
        original = "# My existing notes\n\nSome content here.\n"
        (self.workdir / "NOTES.md").write_text(original, encoding="utf-8")

        seed_notes_with_history(
            workdir_root=self.workdir,
            run_root=self.tmp_root,
            up_to_iteration=7,
            baseline_eval_score=BASELINE_EVAL_SCORE,
        )

        result = (self.workdir / "NOTES.md").read_text(encoding="utf-8")
        self.assertIn("BEGIN ITERATION HISTORY", result)
        self.assertIn("END ITERATION HISTORY", result)

    def test_existing_content_preserved(self) -> None:
        original = "# My existing notes\n\nSome content here.\n"
        (self.workdir / "NOTES.md").write_text(original, encoding="utf-8")

        seed_notes_with_history(
            workdir_root=self.workdir,
            run_root=self.tmp_root,
            up_to_iteration=7,
            baseline_eval_score=BASELINE_EVAL_SCORE,
        )

        result = (self.workdir / "NOTES.md").read_text(encoding="utf-8")
        self.assertIn("# My existing notes", result)
        self.assertIn("Some content here.", result)

    def test_scoreboard_table_present(self) -> None:
        (self.workdir / "NOTES.md").write_text("", encoding="utf-8")

        seed_notes_with_history(
            workdir_root=self.workdir,
            run_root=self.tmp_root,
            up_to_iteration=7,
            baseline_eval_score=BASELINE_EVAL_SCORE,
        )

        result = (self.workdir / "NOTES.md").read_text(encoding="utf-8")
        self.assertIn("Eval score history", result)
        self.assertIn("| Iter |", result)

    def test_per_iteration_summaries(self) -> None:
        (self.workdir / "NOTES.md").write_text("", encoding="utf-8")

        seed_notes_with_history(
            workdir_root=self.workdir,
            run_root=self.tmp_root,
            up_to_iteration=7,
            baseline_eval_score=BASELINE_EVAL_SCORE,
        )

        result = (self.workdir / "NOTES.md").read_text(encoding="utf-8")
        self.assertIn("Per-iteration summaries", result)
        self.assertIn("### Iter 1", result)
        self.assertIn("### Iter 7", result)

    def test_idempotent_seeding(self) -> None:
        original = "# My notes\n"
        (self.workdir / "NOTES.md").write_text(original, encoding="utf-8")

        seed_notes_with_history(
            workdir_root=self.workdir,
            run_root=self.tmp_root,
            up_to_iteration=7,
            baseline_eval_score=BASELINE_EVAL_SCORE,
        )
        result1 = (self.workdir / "NOTES.md").read_text(encoding="utf-8")

        seed_notes_with_history(
            workdir_root=self.workdir,
            run_root=self.tmp_root,
            up_to_iteration=7,
            baseline_eval_score=BASELINE_EVAL_SCORE,
        )
        result2 = (self.workdir / "NOTES.md").read_text(encoding="utf-8")

        self.assertEqual(result1, result2)
        count = result2.count("BEGIN ITERATION HISTORY")
        self.assertEqual(count, 1)


class TestEdgeCases(unittest.TestCase):
    tmpdir: str = ""
    run_root: Path = Path()

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.run_root = Path(self.tmpdir) / "run"
        self.run_root.mkdir()
        (self.run_root / "eval").mkdir()
        (self.run_root / "snapshots").mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)

    def test_iteration_1_no_history(self) -> None:
        scoreboard = _build_scoreboard_text(
            run_root=self.run_root,
            up_to_iteration=0,
        )
        self.assertIn("Eval score history", scoreboard)
        table_rows = [
            line
            for line in scoreboard.splitlines()
            if line.startswith("| ") and not line.startswith("|--")
        ]
        data_rows = table_rows[1:]
        self.assertEqual(len(data_rows), 1, "only baseline row")

    def test_iteration_1_snapshot_index_empty(self) -> None:
        index = _build_snapshot_index(
            run_root=self.run_root,
            up_to_iteration=0,
        )
        table_rows = [
            line
            for line in index.splitlines()
            if line.startswith("| ") and not line.startswith("|--")
        ]
        data_rows = table_rows[1:]
        self.assertEqual(len(data_rows), 0)

    def test_iteration_1_no_regression_warning(self) -> None:
        scoreboard = _build_scoreboard_text(
            run_root=self.run_root,
            up_to_iteration=0,
        )
        self.assertNotIn("WARNING", scoreboard)

    def test_missing_eval_data(self) -> None:
        (self.run_root / "eval" / "iter-0001").mkdir(parents=True)
        records = collect_iteration_records(run_root=self.run_root, up_to_iteration=1)
        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0].eval_score)

    def test_missing_eval_shows_na(self) -> None:
        (self.run_root / "eval" / "iter-0001").mkdir(parents=True)
        scoreboard = _build_scoreboard_text(run_root=self.run_root, up_to_iteration=1)
        self.assertIn("N/A", scoreboard)

    def test_missing_snapshot(self) -> None:
        index = _build_snapshot_index(run_root=self.run_root, up_to_iteration=1)
        self.assertNotIn("iter-0001", index)

    def test_empty_notes_md(self) -> None:
        workdir = self.run_root / "agent_evolve"
        workdir.mkdir()
        (workdir / "NOTES.md").write_text("", encoding="utf-8")

        seed_notes_with_history(
            workdir_root=workdir,
            run_root=self.run_root,
            up_to_iteration=0,
        )
        result = (workdir / "NOTES.md").read_text(encoding="utf-8")
        self.assertIn("BEGIN ITERATION HISTORY", result)

    def test_missing_notes_md(self) -> None:
        workdir = self.run_root / "agent_evolve"
        workdir.mkdir()

        seed_notes_with_history(
            workdir_root=workdir,
            run_root=self.run_root,
            up_to_iteration=0,
        )
        result = (workdir / "NOTES.md").read_text(encoding="utf-8")
        self.assertIn("BEGIN ITERATION HISTORY", result)

    def test_missing_notes_in_snapshot(self) -> None:
        snapshot_dir = self.run_root / "snapshots" / "iter-0001" / "eval-0001"
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "agent.py").write_text("# empty agent\n", encoding="utf-8")

        summary = _extract_notes_summary(
            snapshot_root=self.run_root / "snapshots", iteration=1
        )
        self.assertEqual(summary, "No notes available.")

    def test_missing_notes_architecture_label(self) -> None:
        label = _extract_architecture_label(
            snapshot_root=self.run_root / "snapshots", iteration=1
        )
        self.assertEqual(label, "")


@unittest.skipUnless(_has_real_run(), "real run data not available")
class TestArchitectureExtraction(unittest.TestCase):
    def test_iter_4_has_label(self) -> None:
        label = _extract_architecture_label(
            snapshot_root=REAL_RUN_ROOT / "snapshots", iteration=4
        )
        self.assertTrue(len(label) > 0, "iter 4 should have an architecture label")

    def test_labels_dont_contain_current(self) -> None:
        for it in range(1, 8):
            label = _extract_architecture_label(
                snapshot_root=REAL_RUN_ROOT / "snapshots", iteration=it
            )
            self.assertNotIn(
                "(current)",
                label.lower(),
                msg=f"iter {it} label should not contain '(current)'",
            )


if __name__ == "__main__":
    unittest.main()
