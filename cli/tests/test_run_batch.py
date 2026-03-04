from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import benchmark.run_batch as run_batch  # noqa: E402
from cli import ConfigModelEntry, LoadedConfig  # noqa: E402


def _loaded_config() -> LoadedConfig:
    return LoadedConfig(
        default_model="qwen3-coder-next",
        models={
            "qwen3-coder-next": ConfigModelEntry(
                model="qwen/qwen3-coder-next",
                api_base="https://openrouter.ai/api/v1",
                api_key="OPENROUTER_API_KEY",
                temperature=None,
            )
        },
        default_agent="terminus-2",
        agents={"terminus-2": {"final_message": False}},
        verbosity=0,
        max_turns=50,
        max_wait_seconds=60.0,
    )


class TestRunBatchHelpers(unittest.TestCase):
    def test_load_rows_parses_valid_jsonl_and_skips_empty_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input.jsonl"
            input_path.write_text(
                '{"task_id":"t-1","instruction":"echo hi"}\n\n{"id":"t-2","prompt":"ls"}\n',
                encoding="utf-8",
            )
            rows = run_batch._load_rows(input_jsonl=input_path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["task_id"], "t-1")
        self.assertEqual(rows[1]["id"], "t-2")

    def test_load_rows_raises_on_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input.jsonl"
            input_path.write_text('{"task_id":"ok"}\n{not-json}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid JSON on line 2"):
                run_batch._load_rows(input_jsonl=input_path)

    def test_load_rows_raises_when_row_not_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input.jsonl"
            input_path.write_text('"not-an-object"\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected JSON object"):
                run_batch._load_rows(input_jsonl=input_path)

    def test_build_runtime_cfg_resolves_api_key_and_injects_defaults(self) -> None:
        cfg = _loaded_config()
        with patch(
            "benchmark.run_batch.cli_module.resolve_api_key", return_value="resolved"
        ):
            runtime_cfg = run_batch._build_runtime_cfg(
                cfg=cfg,
                agent_key="terminus-2",
                model_key="qwen3-coder-next",
            )
        self.assertEqual(runtime_cfg.model.api_key, "resolved")
        self.assertEqual(runtime_cfg.agent_config["verbosity"], 0)
        self.assertEqual(runtime_cfg.agent_config["max_turns"], 50)
        self.assertEqual(runtime_cfg.agent_config["max_wait_seconds"], 60.0)
        self.assertFalse(runtime_cfg.agent_config["final_message"])

    def test_build_runtime_cfg_raises_when_api_key_missing(self) -> None:
        cfg = _loaded_config()
        with patch("benchmark.run_batch.cli_module.resolve_api_key", return_value=None):
            with self.assertRaisesRegex(ValueError, "Missing API key"):
                run_batch._build_runtime_cfg(
                    cfg=cfg,
                    agent_key="terminus-2",
                    model_key="qwen3-coder-next",
                )


class TestRunBatchMain(unittest.TestCase):
    def test_main_wires_config_and_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input.jsonl"
            output_path = Path(tmp_dir) / "output.jsonl"
            config_path = Path(tmp_dir) / "config.json"
            input_path.write_text(
                '{"task_id":"tb-1","instruction":"echo hi"}\n',
                encoding="utf-8",
            )
            config_path.write_text("{}", encoding="utf-8")
            args = argparse.Namespace(
                config=config_path,
                agent="terminus-2",
                model="qwen3-coder-next",
                input_jsonl=input_path,
                output_jsonl=output_path,
            )

            with (
                patch("benchmark.run_batch.parse_args", return_value=args),
                patch(
                    "benchmark.run_batch.cli_module.load_config",
                    return_value=_loaded_config(),
                ),
                patch(
                    "benchmark.run_batch.cli_module.resolve_api_key",
                    return_value="resolved",
                ),
                patch("benchmark.run_batch.get_agent", return_value=object()),
                patch("benchmark.run_batch.TerminalBenchTBAdapter") as tb_cls,
            ):
                tb_instance = tb_cls.return_value
                run_batch.main()
                tb_instance.run_samples_to_jsonl.assert_called_once()
                call_kwargs = tb_instance.run_samples_to_jsonl.call_args.kwargs

        self.assertEqual(call_kwargs["output_path"], output_path)
        self.assertEqual(call_kwargs["samples"][0]["task_id"], "tb-1")


if __name__ == "__main__":
    unittest.main()
