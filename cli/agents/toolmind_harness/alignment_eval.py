#!/usr/bin/env python3
"""Alignment evaluator for ToolMind-style harness outputs.

Runs static and (optionally) model-in-the-loop tests to quantify how close
generated trajectories are to sampled ToolMind-Web-QA trajectories.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import re
import sys
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_HARNESS_PATH = BASE_DIR / "harness.py"
DEFAULT_REFERENCE_PATHS = [
    BASE_DIR / "conversation_example_newid10.json",
    BASE_DIR / "conversation_example_newid9.json",
]
DEFAULT_GENERATION_DIR = BASE_DIR / "generated_trajectories_eval"
DEFAULT_REPORT_PATH = BASE_DIR / "alignment_report.md"


@dataclass
class TrajMetrics:
    turns: int
    assistant_turns: int
    think_ratio: float
    tool_ratio: float
    protocol_repair_msgs: int
    one_tool_ratio: float
    mean_assistant_len: float
    server_counts: Dict[str, int]
    tool_counts: Dict[str, int]


def normalize_system_prompt(text: str) -> str:
    lines = text.splitlines()
    out: List[str] = []
    for ln in lines:
        if "Today is:" in ln:
            ln = re.sub(r"Today is:\s*.*$", "Today is: DATE", ln)
        out.append(ln.rstrip())
    return "\n".join(out).strip()


def load_harness_module(harness_path: Path):
    module_name = "harness_mod_eval"
    spec = importlib.util.spec_from_file_location(module_name, str(harness_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def calc_metrics(path: Path) -> TrajMetrics:
    data = json.loads(path.read_text(encoding="utf-8"))
    conversations = data["conversations"]
    assistant = [t for t in conversations if t["role"] == "assistant"]
    if not assistant:
        return TrajMetrics(
            turns=len(conversations),
            assistant_turns=0,
            think_ratio=0.0,
            tool_ratio=0.0,
            protocol_repair_msgs=0,
            one_tool_ratio=0.0,
            mean_assistant_len=0.0,
            server_counts={},
            tool_counts={},
        )

    think = 0
    has_tool = 0
    one_tool = 0
    repairs = 0
    lengths: List[int] = []
    server_counts: Dict[str, int] = {}
    tool_counts: Dict[str, int] = {}

    for t in conversations:
        txt = t.get("content", "")
        if "ProtocolError:" in txt:
            repairs += 1
        if t["role"] != "assistant":
            continue
        lengths.append(len(txt))
        if "<think>" in txt:
            think += 1
        blocks = re.findall(r"<use_mcp_tool>(.*?)</use_mcp_tool>", txt, flags=re.S)
        if blocks:
            has_tool += 1
        if len(blocks) == 1:
            one_tool += 1
        for block in blocks:
            sm = re.search(r"<server_name>(.*?)</server_name>", block, re.S)
            tm = re.search(r"<tool_name>(.*?)</tool_name>", block, re.S)
            if sm:
                k = sm.group(1).strip()
                server_counts[k] = server_counts.get(k, 0) + 1
            if tm:
                k = tm.group(1).strip()
                tool_counts[k] = tool_counts.get(k, 0) + 1

    n = len(assistant)
    return TrajMetrics(
        turns=len(conversations),
        assistant_turns=n,
        think_ratio=think / n,
        tool_ratio=has_tool / n,
        protocol_repair_msgs=repairs,
        one_tool_ratio=one_tool / n,
        mean_assistant_len=statistics.mean(lengths),
        server_counts=server_counts,
        tool_counts=tool_counts,
    )


def aggregate(metrics: List[TrajMetrics]) -> Dict[str, float]:
    return {
        "count": float(len(metrics)),
        "avg_turns": statistics.mean(m.turns for m in metrics),
        "avg_assistant_turns": statistics.mean(m.assistant_turns for m in metrics),
        "avg_think_ratio": statistics.mean(m.think_ratio for m in metrics),
        "avg_tool_ratio": statistics.mean(m.tool_ratio for m in metrics),
        "avg_one_tool_ratio": statistics.mean(m.one_tool_ratio for m in metrics),
        "avg_protocol_repairs": statistics.mean(
            m.protocol_repair_msgs for m in metrics
        ),
        "avg_assistant_chars": statistics.mean(m.mean_assistant_len for m in metrics),
    }


def run_generation_if_requested(
    harness_path: Path,
    out_dir: Path,
    model: str,
    max_assistant_turns: int,
) -> List[Path]:
    if not os.getenv("OPENAI_API_KEY"):
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = [
        (
            "eval_q1.json",
            "What is the founding year of the oldest constituent college of the university where Stephen Hawking held the Lucasian Chair of Mathematics? Wrap final answer in \\boxed{}.",
        ),
        (
            "eval_q2.json",
            "The fourth-tier football league in Upper Austria is part of a third-tier league that includes a division for the largest Austrian state by area. Within this third-tier league, identify the state division that was incorporated into a Nazi administrative region established in 1938, which took its name from the Danube River. This same state was historically ruled by a monarch born in 1443 who founded a renowned Renaissance library. What is the name of this monarch? Wrap final answer in \\boxed{}.",
        ),
    ]
    generated: List[Path] = []
    for filename, question in prompts:
        out = out_dir / filename
        cmd = [
            "python3",
            str(harness_path),
            "--question",
            question,
            "--output",
            str(out),
            "--model",
            model,
            "--max-assistant-turns",
            str(max_assistant_turns),
            "--strict-protocol",
            "--min-tool-turns",
            "6",
            "--repair-attempts",
            "4",
            "--allow-fallback-search",
            "--request-reasoning",
            "--internal-protocol-retry",
            "--max-internal-protocol-retries",
            "2",
            "--no-record-protocol-repairs",
        ]
        subprocess.run(cmd, check=True)
        generated.append(out)
    return generated


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate harness alignment against reference trajectories."
    )
    p.add_argument("--harness", default=str(DEFAULT_HARNESS_PATH))
    p.add_argument(
        "--reference",
        nargs="+",
        default=[str(path) for path in DEFAULT_REFERENCE_PATHS],
    )
    p.add_argument("--candidate", nargs="*", default=[])
    p.add_argument("--run-generation", action="store_true")
    p.add_argument(
        "--generation-dir",
        default=str(DEFAULT_GENERATION_DIR),
    )
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", "qwen/qwen3.5-35b-a3b"))
    p.add_argument("--max-assistant-turns", type=int, default=24)
    p.add_argument(
        "--report",
        default=str(DEFAULT_REPORT_PATH),
    )
    args = p.parse_args()

    harness_path = Path(args.harness).resolve()
    harness_mod = load_harness_module(harness_path)

    ref_paths = [Path(x).resolve() for x in args.reference]
    reference_metrics = [calc_metrics(pth) for pth in ref_paths]
    ref_agg = aggregate(reference_metrics)

    # Prompt fidelity
    ref_system = json.loads(ref_paths[0].read_text(encoding="utf-8"))["conversations"][
        0
    ]["content"]
    cand_system = harness_mod.build_system_prompt(today=dt.date.today().isoformat())
    ref_norm = normalize_system_prompt(ref_system)
    cand_norm = normalize_system_prompt(cand_system)
    import difflib

    prompt_similarity = difflib.SequenceMatcher(None, ref_norm, cand_norm).ratio()

    # Parser round-trip on references
    total_assistant = 0
    parseable = 0
    for pth in ref_paths:
        conv = json.loads(pth.read_text(encoding="utf-8"))["conversations"]
        for turn in conv:
            if turn["role"] != "assistant":
                continue
            total_assistant += 1
            content = turn.get("content", "")
            # Count valid as either no tool call OR parseable call blocks
            blocks = re.findall(
                r"<use_mcp_tool>(.*?)</use_mcp_tool>", content, flags=re.S
            )
            if not blocks:
                parseable += 1
                continue
            ok = True
            for b in blocks:
                s = re.search(r"<server_name>(.*?)</server_name>", b, re.S)
                t = re.search(r"<tool_name>(.*?)</tool_name>", b, re.S)
                a = re.search(r"<arguments>(.*?)</arguments>", b, re.S)
                if not (s and t and a):
                    ok = False
                    break
            if ok:
                parseable += 1
    parser_roundtrip_ratio = parseable / max(1, total_assistant)

    candidate_paths = [Path(x).resolve() for x in args.candidate]
    if args.run_generation:
        generated = run_generation_if_requested(
            harness_path=harness_path,
            out_dir=Path(args.generation_dir).resolve(),
            model=args.model,
            max_assistant_turns=args.max_assistant_turns,
        )
        candidate_paths.extend(generated)

    candidate_metrics = (
        [calc_metrics(pth) for pth in candidate_paths] if candidate_paths else []
    )
    cand_agg = aggregate(candidate_metrics) if candidate_metrics else {}

    lines: List[str] = []
    lines.append("# Harness Alignment Report")
    lines.append("")
    lines.append("## Static Checks")
    lines.append(
        f"- Prompt similarity (normalized, date-agnostic): `{prompt_similarity:.3f}`"
    )
    lines.append(
        f"- Parser round-trip success on reference assistant turns: `{parser_roundtrip_ratio:.3f}`"
    )
    lines.append("")
    lines.append("## Reference Aggregate")
    for k, v in ref_agg.items():
        lines.append(f"- {k}: `{v:.3f}`")
    lines.append("")
    lines.append("## Candidate Aggregate")
    if cand_agg:
        for k, v in cand_agg.items():
            lines.append(f"- {k}: `{v:.3f}`")
    else:
        lines.append("- No candidate trajectories provided/generated.")
    lines.append("")
    if candidate_metrics:
        lines.append("## Candidate Per-File")
        for pth, m in zip(candidate_paths, candidate_metrics):
            lines.append(f"- `{pth}`")
            lines.append(
                f"  - turns={m.turns}, assistant={m.assistant_turns}, think_ratio={m.think_ratio:.3f}, "
                f"tool_ratio={m.tool_ratio:.3f}, one_tool_ratio={m.one_tool_ratio:.3f}, repairs={m.protocol_repair_msgs}"
            )
            lines.append(f"  - servers={m.server_counts}")
            lines.append(f"  - tools={m.tool_counts}")
        lines.append("")

    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
